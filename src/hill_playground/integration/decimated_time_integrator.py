r"""``dolfin_mech.core.TimeIntegrator`` with throttled XDMF output and an optional plateau exit.

The base ``integrate`` writes one XDMF frame per converged substep (thousands on a long adaptive
hold). Here, only a subset is kept, according to a write period given per step (or one for all):

* ``int`` :math:`p`: keep substep :math:`k` iff :math:`(k - 1) \bmod p = 0`. For a ~constant ``dt``.
* :class:`TimeInterval` :math:`\Delta`: keep :math:`t` iff :math:`t - t_{last} \ge \Delta`. For an adaptive ``dt``.

Times (:math:`t`, ``dt``, :class:`TimeInterval`, ``min_simu_seconds``) are in solve units: s, or :math:`t_0` for
a scaled solve :math:`\hat t = t/t_0`. The period state resets at each step. Always kept, whatever the period:

* the first converged substep of a step (``int``: :math:`k = 1`; :class:`TimeInterval`: first frame of the step),
* the last substep of a step (:math:`t = t_{fin}`),
* the last substep of a plateau exit (below),
* the last converged substep before a solver failure (:math:`dt < dt_{min}`).

Early exit on a plateaued step:
    Opt-in per step (``step.min_simu_seconds is not None``, set on the ``dolfin_mech`` step by the
    main script). Once :math:`t - t_{ini} \ge` ``min_simu_seconds``, the step ends if every monitored
    subsol (``sol_tol is not None``) satisfies

    .. math::
        \frac{\lVert u - u_{old} \rVert}{\max(\lVert u \rVert, \lVert u_{old} \rVert)} < \varepsilon,

    with :math:`\varepsilon` = ``stagnation_plateau_relative_error``. Not divided by ``dt``: the
    verdict depends on the current ``dt``.

Additional features:
    * ``functions_to_write``: fields written in the XDMF (default: subsols, ``*_old``, FOIs).
    * ``root_only_logs``: ``.out``/``.sta``/``.dat`` written on rank 0 only (otherwise ranks
      overwrite each other under ``mpirun``; logged values are already MPI-reduced).

Note:
    Only the XDMF time series is throttled; logs, QOIs and final checkpoints are not.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import dolfin
import dolfin_mech
from dolfin_mech.core import compute_error, write_VTU_file
from dolfin_mech.core.xdmffile import XDMFFile
from mpi4py import MPI

_STEP_END_ABS_TOL = 1e-9

_DEFAULT_PLATEAU_RELATIVE_ERROR = 1e-9
r"""Default plateau threshold :math:`\varepsilon` (also the zero tolerance of ``compute_error``)."""


@dataclass(frozen=True)
class TimeInterval:
	r"""Write period in simulated time: keep :math:`t` iff :math:`t - t_{last} \ge` ``seconds``.

	Attributes:
	    seconds (float): Minimum simulated time between two kept frames (> 0, solve units).
	"""

	seconds: float

	def __post_init__(self) -> None:
		"""Validate ``seconds`` is strictly positive.

		Raises:
		    ValueError: If ``seconds`` is not strictly positive.
		"""
		if self.seconds <= 0:
			raise ValueError(f"TimeInterval.seconds must be strictly positive, got {self.seconds}.")


_StepPeriod = Union[int, TimeInterval]


def should_write_at_call(call_index: int, period: int) -> bool:
	r"""Integer-period rule: write call :math:`k` iff :math:`k = 1` or :math:`(k - 1) \bmod p = 0`.

	Step ends are handled by :class:`_DecimatedXDMFFile`.

	Args:
	    call_index (int): 1-based call count :math:`k`, current call included.
	    period (int): Period :math:`p \ge 1`.

	Returns:
	    bool: True if this call writes.

	Raises:
	    ValueError: If ``period < 1``.
	"""
	if period < 1:
		raise ValueError(f"period must be a positive integer, got {period}.")
	return call_index == 1 or (call_index - 1) % period == 0


def is_subsol_stagnant(
	delta_norm: float,
	scale_norm: float,
	min_relative_error: float = _DEFAULT_PLATEAU_RELATIVE_ERROR,
) -> bool:
	r"""Stagnation test of one subsol over one substep: :math:`e < \varepsilon`.

	With :math:`e = \lVert u - u_{old} \rVert / \max(\lVert u \rVert, \lVert u_{old} \rVert)`
	(``dolfin_mech.core.compute_error``; scale :math:`< 10^{-9}`: :math:`e = 0` if the change is also
	:math:`< 10^{-9}`, else :math:`e = 1`). The norms come from :meth:`DecimatedTimeIntegrator._step_is_stagnant`.

	Args:
	    delta_norm (float): :math:`\lVert u - u_{old} \rVert_2`.
	    scale_norm (float): :math:`\max(\lVert u \rVert_2, \lVert u_{old} \rVert_2)`.
	    min_relative_error (float): Threshold :math:`\varepsilon > 0`. Defaults to ``1e-9``.

	Returns:
	    bool: True if the subsol is stagnant.

	Raises:
	    ValueError: If :math:`\varepsilon \le 0`.
	"""
	if min_relative_error <= 0:
		raise ValueError(f"min_relative_error must be strictly positive, got {min_relative_error}.")
	return compute_error(val=delta_norm, ref=scale_norm) < min_relative_error


class _DecimatedXDMFFile:
	r"""Proxy of an XDMF writer that drops frames according to a per-step write period.

	A call is forwarded iff ``force``, or :math:`t = t_{fin}` of the current step, or the period
	allows it (module docstring), and never twice for the same :math:`t`. The period state (call count,
	first frame of the step) resets when the active step changes; the last written :math:`t` does not
	(initially 0: the wrapped file already holds :math:`t = 0`).
	"""

	def __init__(
		self, xdmf_file: Any, write_period: Union[_StepPeriod, Sequence[_StepPeriod]], time_integrator: Any
	) -> None:
		"""Wrap an opened XDMF writer.

		Args:
		    xdmf_file (Any): Wrapped writer (already holds :math:`t = 0`).
		    write_period (Union[_StepPeriod, Sequence[_StepPeriod]]): One period, or one per step of
		        ``problem.steps``.
		    time_integrator (Any): Owner, read for ``step`` and ``problem.steps``.
		"""
		self._xdmf_file = xdmf_file
		self._time_integrator = time_integrator
		n_steps = len(time_integrator.problem.steps)
		self._periods_by_step: List[_StepPeriod] = (
			[write_period] * n_steps if isinstance(write_period, (int, TimeInterval)) else list(write_period)
		)
		self._call_count = 0
		self._step_written = False
		self._last_written_t = 0.0  # the wrapped file already holds t = 0
		self._last_step_index: Optional[int] = None

	def _current_step_index(self) -> int:
		"""Position, by identity, of the currently active step within ``problem.steps``."""
		return self._time_integrator.problem.steps.index(self._time_integrator.step)

	def write(self, t: float, force: bool = False) -> None:
		"""Write ``t`` if forced, at the step end, or allowed by the period.

		Args:
		    t (float): Simulated time.
		    force (bool): Write regardless of the period. Defaults to False.

		Note:
		    No-op if ``t`` equals the last written time (``force`` included, also across a step change,
		    e.g. a failure on the first substep of a step): a forced write never duplicates a frame.
		"""
		step_index = self._current_step_index()
		if step_index != self._last_step_index:
			self._call_count = 0
			self._step_written = False
			self._last_step_index = step_index
		self._call_count += 1

		if abs(t - self._last_written_t) <= _STEP_END_ABS_TOL:
			return

		current_step = self._time_integrator.step
		is_last_substep_of_step = abs(t - current_step.t_fin) <= _STEP_END_ABS_TOL
		period = self._periods_by_step[step_index]

		if isinstance(period, TimeInterval):
			should_write = not self._step_written or (t - self._last_written_t) >= period.seconds
		else:
			should_write = should_write_at_call(self._call_count, period)

		if force or is_last_substep_of_step or should_write:
			self._xdmf_file.write(t)
			self._last_written_t = t
			self._step_written = True

	def close(self) -> None:
		"""Close the wrapped file."""
		self._xdmf_file.close()


class DecimatedTimeIntegrator(dolfin_mech.core.TimeIntegrator):
	r"""``dolfin_mech.core.TimeIntegrator`` with throttled XDMF output and an optional plateau exit.

	See the module docstring. ``write_period=1`` writes every substep, as the base class.

	Args:
	    *args (Any): Forwarded to ``dolfin_mech.core.TimeIntegrator.__init__``.
	    write_period (Union[_StepPeriod, Sequence[_StepPeriod]]): ``int`` (substeps) or
	        :class:`TimeInterval` (simulated time); one value, or one per step of
	        ``problem.steps``. Defaults to 1.
	    stagnation_plateau_relative_error (Union[float, Sequence[float]]): Plateau threshold
	        :math:`\varepsilon` (:func:`is_subsol_stagnant`); one value, or one per subsol of
	        ``problem.subsols``. Defaults to ``1e-9``.
	    functions_to_write (Optional[Sequence[Any]]): Fields of each XDMF frame.
	        ``None`` keeps the base list. Defaults to None.
	    root_only_logs (bool):  Logs on rank 0 only. Defaults to True.
	    **kwargs (Any): Forwarded to ``dolfin_mech.core.TimeIntegrator.__init__``.

	Raises:
	    TypeError: If a per-step / per-subsol entry has the wrong type.
	    ValueError: If a sequence has the wrong length, or a period / threshold is not positive.
	"""

	def __init__(
		self,
		*args: Any,
		write_period: Union[_StepPeriod, Sequence[_StepPeriod]] = 1,
		stagnation_plateau_relative_error: Union[float, Sequence[float]] = _DEFAULT_PLATEAU_RELATIVE_ERROR,
		functions_to_write: Optional[Sequence[Any]] = None,
		root_only_logs: bool = True,
		**kwargs: Any,
	) -> None:
		"""Build the base integrator, then wrap its XDMF file writer if throttling is requested."""
		super().__init__(*args, **kwargs)

		# rank-0 logs, then the selected XDMF fields (before any decimation proxy).
		if root_only_logs and MPI.COMM_WORLD.rank != 0:
			self._silence_logs()
		if functions_to_write is not None and self.write_sol:
			self._reopen_xdmf(functions_to_write)

		self.xdmf_file_sol: Any

		if isinstance(stagnation_plateau_relative_error, (int, float)):
			if stagnation_plateau_relative_error <= 0:
				raise ValueError(
					f"stagnation_plateau_relative_error must be strictly positive, got "
					f"{stagnation_plateau_relative_error}."
				)
		else:
			relative_errors: List[Any] = list(stagnation_plateau_relative_error)
			n_subsols = len(self.problem.subsols)
			if len(relative_errors) != n_subsols:
				raise ValueError(
					f"stagnation_plateau_relative_error, given as a per-subsol sequence, must have exactly "
					f"one entry per registered subsol (len(problem.subsols)={n_subsols}); got "
					f"{len(relative_errors)} entries."
				)
			if not all(isinstance(relative_error, (int, float)) for relative_error in relative_errors):
				raise TypeError(
					f"Every per-subsol stagnation_plateau_relative_error entry must be an int or float; "
					f"got {relative_errors}."
				)
			if any(relative_error <= 0 for relative_error in relative_errors):
				raise ValueError(
					f"Every per-subsol stagnation_plateau_relative_error entry must be strictly positive; "
					f"got {relative_errors}."
				)
			stagnation_plateau_relative_error = relative_errors
		self.stagnation_plateau_relative_error = stagnation_plateau_relative_error

		if isinstance(write_period, (int, TimeInterval)):
			if isinstance(write_period, int) and write_period < 1:
				raise ValueError(f"write_period must be a positive integer, got {write_period}.")
			periods: List[Any] = [write_period]
		else:
			periods = list(write_period)
			n_steps = len(self.problem.steps)
			if len(periods) != n_steps:
				raise ValueError(
					f"write_period, given as a per-step sequence, must have exactly one entry per "
					f"registered load step (len(problem.steps)={n_steps}); got {len(periods)} entries."
				)
			if not all(isinstance(period, (int, TimeInterval)) for period in periods):
				raise TypeError(f"Every per-step write_period entry must be an int or a TimeInterval; got {periods}.")
			if any(isinstance(period, int) and period < 1 for period in periods):
				raise ValueError(
					f"Every per-step integer write_period entry must be a positive integer; got {periods}."
				)

		if self.write_sol and any(isinstance(period, TimeInterval) or period > 1 for period in periods):
			single = isinstance(write_period, (int, TimeInterval))
			self.xdmf_file_sol = _DecimatedXDMFFile(self.xdmf_file_sol, write_period if single else periods, self)

	def _silence_logs(self) -> None:
		"""Redirect this rank's ``.out``, ``.sta`` and ``.dat`` writers to ``os.devnull``."""
		for printer in (self.printer, self.table_printer):
			if printer.must_close:
				printer.output.close()
			printer.output = open(os.devnull, "w")  # noqa: SIM115 (closed by the printer)
			printer.must_close = True
		if self.write_qois:
			self.qoi_printer.file.close()
			self.qoi_printer.file = open(os.devnull, "w")  # noqa: SIM115 (closed by the printer)

	def _reopen_xdmf(self, functions_to_write: Sequence[Any]) -> None:
		"""Reopen the results XDMF on ``functions_to_write`` and rewrite :math:`t = 0`.

		Args:
		    functions_to_write: Functions written at each kept frame.
		"""
		self.xdmf_file_sol.close()
		self.functions_to_write = list(functions_to_write)
		self.xdmf_file_sol = XDMFFile(filename=self.write_sol_filebasename + ".xdmf", functions=self.functions_to_write)
		self.xdmf_file_sol.write(0.0)

	def _step_is_stagnant(self) -> bool:
		"""True if every monitored subsol (``sol_tol is not None``) is stagnant (:func:`is_subsol_stagnant`).

		Compares ``subsol.func`` to ``subsol.func_old``; MPI-reduced (``MIN``) like the other substep decisions.

		Returns:
		    bool: True if the step has plateaued.
		"""
		local_stagnant = 1
		for k_subsol, subsol in enumerate(self.problem.subsols):
			sol_tol = self.solver.sol_tol[k_subsol]
			if sol_tol is None:
				continue
			min_relative_error = (
				self.stagnation_plateau_relative_error
				if isinstance(self.stagnation_plateau_relative_error, (int, float))
				else self.stagnation_plateau_relative_error[k_subsol]
			)
			delta_norm = (subsol.func.vector() - subsol.func_old.vector()).norm("l2")
			scale_norm = max(subsol.func.vector().norm("l2"), subsol.func_old.vector().norm("l2"))
			if not is_subsol_stagnant(delta_norm, scale_norm, min_relative_error):
				local_stagnant = 0
				break
		global_stagnant = MPI.COMM_WORLD.allreduce(local_stagnant, op=MPI.MIN)
		return bool(global_stagnant)

	def integrate(self) -> bool:
		"""Base time loop, plus the plateau exit and the forced writes of the module docstring.

		Returns:
		    bool: True if every step completed (at :math:`t_{fin}` or by plateau exit); False if
		    :math:`dt < dt_{min}` (the last converged state is still written).
		"""
		k_t_tot = 0
		n_iter_tot = 0
		self.printer.inc()
		for k_step in range(1, len(self.problem.steps) + 1):
			self.printer.print_var("k_step", k_step, -1)

			self.step = self.problem.steps[k_step - 1]
			# this step's own optional plateau-exit floor; None means "feature off".
			min_simu_seconds = getattr(self.step, "min_simu_seconds", None)

			t = self.step.t_ini
			dt = self.step.dt_ini

			self.problem.set_variational_formulation(k_step=k_step - 1)

			self.solver.constraints = []
			self.solver.constraints += self.problem.constraints
			self.solver.constraints += self.step.constraints

			k_t = 0
			self.printer.inc()
			while True:
				k_t += 1
				k_t_tot += 1
				self.printer.print_var("k_t", k_t, -1)

				if t + dt > self.step.t_fin:
					dt = self.step.t_fin - t
				self.printer.print_var("dt", dt)

				t += dt
				self.printer.print_var("t", t)

				t_step = (t - self.step.t_ini) / (self.step.t_fin - self.step.t_ini)
				self.printer.print_var("t_step", t_step)

				for operator in self.step.operators:
					operator.set_value_at_t_step(t_step)
					operator.set_dt(dt)

				for constraint in self.step.constraints:
					constraint.set_value_at_t_step(t_step)

				for inelastic_behavior in self.problem.inelastic_behaviors_internal:
					inelastic_behavior.update_internal_variables_at_t(t)

				self.problem.sol_old_func.vector()[:] = self.problem.sol_func.vector()[:]
				if len(self.problem.subsols) > 1:
					dolfin.assign(self.problem.get_subsols_func_old_lst(), self.problem.sol_old_func)
				solver_success, n_iter = self.solver.solve(k_step, k_t, dt, t)

				local_success = 1 if solver_success else 0
				global_success = MPI.COMM_WORLD.allreduce(local_success, op=MPI.MIN)
				solver_success = bool(global_success)

				# Synchronize n_iter to prevent dt divergence across ranks
				n_iter = MPI.COMM_WORLD.allreduce(n_iter, op=MPI.MAX)

				self.table_printer.write_line([k_step, k_t, dt, t, t_step, n_iter, solver_success])

				if solver_success:
					n_iter_tot += n_iter

					self.problem.update_fois()

					local_end = 1 if dolfin.near(t, self.step.t_fin, eps=_STEP_END_ABS_TOL) else 0
					global_end = MPI.COMM_WORLD.allreduce(local_end, op=MPI.MAX)

					# plateau test once t - t_ini >= min_simu_seconds, if t != t_fin.
					global_stagnant = False
					if not global_end and min_simu_seconds is not None:
						if (t - self.step.t_ini) >= min_simu_seconds:
							global_stagnant = self._step_is_stagnant()
					step_is_ending = bool(global_end or global_stagnant)

					if self.write_sol:
						# force the step's last frame (needed for a plateau exit, t != t_fin).
						if isinstance(self.xdmf_file_sol, _DecimatedXDMFFile):
							self.xdmf_file_sol.write(t, force=step_is_ending)
						else:
							self.xdmf_file_sol.write(t)

						# write_vtus/write_xmls not covered with test. Inherited but never used here.
						if self.write_vtus:  # pragma: no cover
							write_VTU_file(
								filebasename=self.write_sol_filebasename,
								function=self.problem.displacement_subsol.subfunc,
								time=k_t_tot,
								preserve_connectivity=self.write_vtus_with_preserved_connectivity,
							)

						if self.write_xmls:  # pragma: no cover
							_ = (
								dolfin.File(self.write_sol_filebasename + "_" + str(k_t_tot).zfill(3) + ".xml")
								<< self.problem.displacement_subsol.subfunc
							)

					if self.write_qois:
						self.problem.update_qois(dt, k_step)
						self.qoi_printer.write_line([t] + [qoi.value for qoi in self.problem.qois])

					if step_is_ending:
						if global_stagnant and not global_end:
							self.printer.print_str(
								f"Step plateaued after {t - self.step.t_ini:.6g} of simulated time "
								f"(>= min_simu_seconds={min_simu_seconds:.6g}) -- exiting step early."
							)
						self.success = True
						break
					else:
						if n_iter <= self.n_iter_for_accel:
							dt *= self.accel_coeff
							if dt > self.step.dt_max:
								dt = self.step.dt_max
						elif n_iter >= self.n_iter_for_decel:
							dt /= self.decel_coeff
							if dt < self.step.dt_min:
								dt = self.step.dt_min
				else:
					self.problem.sol_func.vector()[:] = self.problem.sol_old_func.vector()[:]
					if len(self.problem.subsols) > 1:
						dolfin.assign(self.problem.get_subsols_func_lst(), self.problem.sol_func)

					for inelastic_behavior in self.problem.inelastic_behaviors_internal:
						inelastic_behavior.restore_old_value()

					for constraint in self.step.constraints:
						constraint.restore_old_value()

					k_t -= 1
					k_t_tot -= 1
					t -= dt
					dt /= self.decel_coeff

					local_fail = 1 if (dt < self.step.dt_min) else 0
					global_fail = MPI.COMM_WORLD.allreduce(local_fail, op=MPI.MAX)

					if global_fail:
						self.printer.print_str("Warning! Time integrator failed to move forward!")
						if self.write_sol and isinstance(self.xdmf_file_sol, _DecimatedXDMFFile):
							self.xdmf_file_sol.write(t, force=True)
						self.success = False
						break

			self.printer.dec()

			if not (self.success):
				break

		self.printer.dec()

		return self.success
