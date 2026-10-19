"""Throttled-XDMF-output variant of ``dolfin_mech.core.TimeIntegrator``, with an optional early exit.

Output options (blocks tagged ``CHANGE (playground)``):

    1. ``functions_to_write``: explicit list of the functions written in the results XDMF (e.g. SI
       fields only), instead of every subsol + every ``*_old`` copy + every FOI.
    2. Logs on rank 0 only (``root_only_logs=True``): ``dolfin_mech``/``myPythonLibrary`` printers
       open the ``.out``, ``.sta`` and ``.dat`` files on *every* MPI rank with independent buffers and
       offsets, so under ``mpirun`` the ranks would overwrite each other once the buffers flush.
       Non-root ranks write to ``os.devnull``; every logged
       value is already MPI-reduced (norms, QOIs), so nothing is lost.
       The Newton lines are covered too: the base constructor sets ``solver.printer = self.printer``.

Why this exists (write throttling): ``dolfin_mech.core.TimeIntegrator.integrate`` writes one
XDMF/HDF5 frame (``self.xdmf_file_sol.write(t)``) on *every* converged substep, unconditionally --
there is no writing-period option anywhere in ``dolfin_mech`` itself.
For a long adaptive-time-stepping hold at a conservative, non-accelerating ``dt`` (see
``fenicsoft.io.Step.max_dt``'s own docstring for why acceleration can silently fail to kick in),
that is one frame roughly every ``init_dt`` seconds of simulated time for the whole run --
thousands of frames, and a correspondingly huge results ``.xdmf``/``.h5`` pair.

:class:`DecimatedTimeIntegrator` builds the exact same base ``TimeIntegrator``, then wraps *only*
its XDMF field-solution writer (``self.xdmf_file_sol``) in a small proxy
(:class:`_DecimatedXDMFFile`) that forwards a subset of calls and silently drops the rest.

``write_period`` can be given, per step, as either of two fundamentally different kinds of unit:

* A plain ``int`` -- keep one frame out of every ``write_period`` *converged substeps*. This is
  the natural choice for a step where ``dt`` is (or is expected to be) roughly constant across
  the whole step -- e.g. a step with no ``max_dt`` set at all (module docstring of
  ``fenicsoft.io.Step``: ``dt`` then never accelerates past ``init_dt``, so a fixed substep count
  really does correspond to a fixed amount of simulated time).
* A :class:`TimeInterval` -- keep one frame at least every ``TimeInterval.seconds`` of *simulated*
  time, regardless of how many (or how few) substeps that takes. This is the right choice for any
  step where ``dt`` itself is *not* held constant -- typically one under
  ``dolfin_mech.core.TimeIntegrator``'s own adaptive acceleration/deceleration -- where a fixed
  substep-count period corresponds to a different, and potentially wildly different, amount of
  simulated time depending on where in the accel/decel cycle a given substep happens to fall: a
  substep-count period of 100 with ``dt`` ranging from 30 s (start of the step) to 500 s (fully
  accelerated) keeps one frame roughly every 3000 s early on but roughly every 50000 s later --
  more than a 16x swing in the actual temporal resolution of the saved time series, for the exact
  same nominal "period". :class:`TimeInterval` fixes that by working in simulated seconds
  directly, independent of however many substeps ``dt`` acceleration/deceleration happens to take
  to cover that time.

Either kind of period can be given as a single value (applied uniformly to every registered load
step) or as a ``Sequence`` with exactly one entry per step (:attr:`dolfin_mech.problems.Problem.steps`,
in the same order) -- and the two kinds of period can be freely mixed across steps in the same
sequence: e.g. ``write_period=(1, TimeInterval(900.0))`` for a short, constant-``dt`` ramp kept in
full followed by a long, accelerating hold sampled every 900 simulated seconds. The decimation
*state* (call counter for an ``int`` period, last-written time for a :class:`TimeInterval`) is
tracked per proxy call and reset at every step boundary, so a step's own period always starts
counting fresh -- it is never phase-shifted by whatever state the previous step happened to leave
behind.

Whatever the applicable period (and whichever kind), a step's own *last* converged substep (``t``
matching ``self.step.t_fin``, the same comparison ``dolfin_mech.core.TimeIntegrator.integrate``
itself uses internally to know a step has finished) is always written, never decimated away.
Without this, a coarse period on a long hold could easily miss that step's own final state by
chance (neither an integer period's substep count nor a ``TimeInterval``'s elapsed-time count need
divide the step's own total exactly), which matters in practice: it is the one frame that shows
the state the *next* step actually continues from.

Note:
    This module throttles only the per-substep XDMF time series. A checkpoint of the final state
    written by the caller (separate write path) is not affected by ``write_period``.

Early exit on a plateaued step (``min_simu_seconds``):
    A step can optionally exit *before* reaching its own ``t_fin``, once every monitored subsol's
    own substep-to-substep relative change has dropped below ``stagnation_plateau_relative_error``
    -- the intuition being that once nothing is
    changing any further (within the solver's own noise floor), continuing to burn simulated time
    (and wall-clock time) to reach the step's originally-planned ``t_fin`` adds nothing. This is
    opt-in and per step: it activates only for a step whose own ``fenicsoft.io.Step.min_simu_seconds``
    is not ``None`` (see that field's own docstring for exactly how it reaches the runtime
    ``dolfin_mech.core.step.Step`` object this integrator actually reads -- ``dolfin_mech`` itself
    has no such concept, so a demo script must set it as a plain post-construction attribute).

    A step with ``min_simu_seconds=None`` (the default -- see ``fenicsoft.io.Step``) takes the
    *exact* same path through this override as it would through the unmodified base-class loop:
    the plateau check is skipped entirely (never even evaluated).

    **The criterion itself is a direct relative-error threshold, not a rate:** see
    :func:`is_subsol_stagnant`'s own docstring for the full reasoning and its known trade-off --
    in short, a subsol is judged stagnant once its own substep-to-substep relative change
    (:func:`_relative_error`) drops below ``stagnation_plateau_relative_error``, with no
    normalization by ``dt``: the threshold is a plain relative error, at the price of a verdict
    that depends on the current adaptive ``dt``.

    **Interaction with write throttling:** the step's own natural end (``t == t_fin``) is already
    always kept by :class:`_DecimatedXDMFFile`, unassisted, because ``t`` itself matches ``t_fin``.
    A plateau exit is different: ``t`` generally does *not* match ``t_fin`` (the step stopped
    early), so left alone the period rule could silently decimate away the one frame showing the
    state everything downstream (the next step, a restart, ...) actually continues
    from. To prevent that, :meth:`_DecimatedXDMFFile.write` accepts an explicit ``force``
    flag, and this override's own write call passes ``force=True`` whenever the current substep
    ends the step for *either* reason (natural completion or a plateau exit) -- redundant, but
    harmless, for the natural-completion case (already forced by the ``t == t_fin`` check on its
    own), and the only thing that saves the plateau-exit case's own final frame from decimation.

Guaranteed last frame on a hard failure:
    A throttled write period creates one more gap besides the plateau-exit case above: if the
    solver eventually fails to move forward at all (even at the step's own ``dt_min`` --
    ``dolfin_mech.core.TimeIntegrator.integrate``'s own final failure mode), the *last
    successfully converged* substep before that failure might itself have been skipped by the
    period at the time it was reached (nothing marked it as special then -- the integrator was
    still trying to continue). Left unaddressed, the persisted XDMF trajectory could be several
    substeps stale relative to the actual last state the solver ever reached, right when that
    state is most valuable (restart, or post-mortem inspection of what the solver was converging
    towards just before it gave up). This override's own failure branch therefore also issues a
    ``write(t, force=True)`` call -- at the ``t`` the failure branch has already rewound back to
    (the last converged substep's own time, since ``problem.sol_func``/the subsols have likewise
    already been restored to that same state a few lines above, before this call) -- immediately
    before returning ``False``. This is a no-op when ``write_sol`` is disabled, and effectively a
    no-op when the writer was never throttled in the first place (every substep was already
    written unconditionally as it happened, so there is nothing new to force); the case it
    actually fixes is a throttled writer whose period had not yet reached that particular substep.
    :meth:`_DecimatedXDMFFile.write`'s own dedup guard (its own docstring) additionally makes this
    call harmless even when that exact ``t`` had, in fact, already been written.
"""

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import dolfin
import dolfin_mech
from dolfin_mech.core import write_VTU_file
from dolfin_mech.core.xdmffile import XDMFFile
from mpi4py import MPI

_STEP_END_ABS_TOL = 1e-9

_RELATIVE_ERROR_ZERO_EPS = 1e-9


@dataclass(frozen=True)
class TimeInterval:
	"""A write period expressed in simulated seconds, rather than converged-substep count.

	Pass this (instead of a plain ``int``) wherever ``write_period`` is accepted, for any step
	where ``dt`` itself is not held (roughly) constant -- typically a step under
	``dolfin_mech.core.TimeIntegrator``'s own adaptive acceleration/deceleration (see
	``fenicsoft.io.Step.max_dt``). There, a fixed *substep-count* period corresponds to a
	different amount of *simulated* time depending on where in the accel/decel cycle a given
	substep falls; a fixed number of *simulated seconds* does not have that problem, whatever
	``dt`` happens to be doing at the time. See this module's own docstring for the full
	rationale and a worked numeric example.

	Attributes:
	    seconds (float): Minimum simulated-time gap, in seconds, between two kept frames. A
	        frame is still always kept regardless once it lands on the currently active step's
	        own last converged substep (see :class:`_DecimatedXDMFFile`).
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
	"""Decide whether the ``call_index``-th call to a periodic writer should actually write.

	Dolfin-free, pure-integer logic for the substep-*count*-based case only (a
	:class:`TimeInterval` period is decided directly in :class:`_DecimatedXDMFFile.write`, by
	comparing elapsed simulated time instead), factored out here so it can be unit-tested with no
	FEniCS/``dolfin_mech`` install at all. Always keeps the very first call (``call_index == 1``:
	the initial :math:`t=0` frame, written by ``dolfin_mech.core.TimeIntegrator.__init__`` itself
	-- before :class:`DecimatedTimeIntegrator` even attaches its wrapper, so it is never subject to
	throttling in the first place, but this function still keeps it if ever called directly with
	``call_index=1``), then one call every ``period`` calls after that. This is purely the
	*periodic* decimation rule -- it knows nothing about a step's own last substep, which
	:class:`_DecimatedXDMFFile` enforces separately (see that class's own docstring).

	Args:
	    call_index (int): 1-based count of calls made so far (the value *after* incrementing for
	        the current call).
	    period (int): Write every ``period``-th call. Must be a positive integer.

	Returns:
	    bool: True if this call should actually write.

	Raises:
	    ValueError: If ``period`` is not a positive integer.
	"""
	if period < 1:
		raise ValueError(f"period must be a positive integer, got {period}.")
	return call_index == 1 or (call_index - 1) % period == 0


def _relative_error(val: float, ref: float) -> float:
	r"""Dolfin-free reimplementation of ``dolfin_mech.core.compute_error``'s own ``val/ref`` ratio.

	Deliberately duplicated rather than imported: importing ``dolfin_mech.core.compute_error``
	would pull in a real ``dolfin.near`` call (and therefore a FEniCS install) just to guard
	against a near-zero reference, purely to keep :func:`is_subsol_stagnant` unit-testable with no
	FEniCS/``dolfin_mech`` install at all -- the same rationale already used for
	:func:`should_write_at_call` (see that function's own docstring). Matches
	``dolfin_mech.core.nonlinearsolver.NonlinearSolver.exit_test``'s own error convention exactly
	(the same ``val/ref`` ratio, with the same near-zero special case), so a threshold expressed in
	terms of ``sol_tol`` compares like with like.

	Args:
	    val (float): The value to compare (an increment norm).
	    ref (float): The reference to compare against (a scale norm).

	Returns:
	    float: ``0.0`` if both are near zero, ``1.0`` if only ``ref`` is near zero, else ``val/ref``.
	"""
	if abs(ref) < _RELATIVE_ERROR_ZERO_EPS:
		return 0.0 if abs(val) < _RELATIVE_ERROR_ZERO_EPS else 1.0
	return val / ref


def is_subsol_stagnant(
	delta_norm: float,
	scale_norm: float,
	min_relative_error: float = _RELATIVE_ERROR_ZERO_EPS,
) -> bool:
	r"""Decide whether one subsol's own change over one converged substep signals a physical plateau.

	Dolfin-free, pure-float logic (the actual norms are computed by
	:meth:`DecimatedTimeIntegrator._step_is_stagnant`, which does need real ``dolfin`` vectors --
	factored out here, exactly like :func:`should_write_at_call`, purely so the *decision* itself
	is unit-testable with no FEniCS install at all).

	The criterion is a direct comparison: this subsol's own substep-to-substep relative change
	(:func:`_relative_error`) against a fixed ``min_relative_error`` threshold.

	Args:
	    delta_norm (float): :math:`\lVert u - u_{old}\rVert` (l2 norm of the converged-substep
	        increment for this subsol).
	    scale_norm (float): :math:`\max(\lVert u\rVert, \lVert u_{old}\rVert)` (the same
	        normalization ``dolfin_mech.core.nonlinearsolver.NonlinearSolver.compute_sol_err``
	        itself uses for its own, per-Newton-iteration error).
	    min_relative_error (float): The relative-error threshold below which this subsol's own
	        change over one substep is considered negligible. Defaults to
	        :data:`_RELATIVE_ERROR_ZERO_EPS` (``1e-9``) -- the same near-zero epsilon
	        :func:`_relative_error` itself already uses for its own zero-reference guard, chosen
	        so that, unconfigured, a subsol is only ever judged stagnant once it has effectively
	        stopped changing at all, rather than at some arbitrary looser default.

	Returns:
	    bool: True if this subsol's own relative change over the substep is below
	        ``min_relative_error``.

	Raises:
	    ValueError: If ``min_relative_error`` is not strictly positive.
	"""
	if min_relative_error <= 0:
		raise ValueError(f"min_relative_error must be strictly positive, got {min_relative_error}.")
	return _relative_error(delta_norm, scale_norm) < min_relative_error


class _DecimatedXDMFFile:
	"""Proxy around a ``dolfin_mech.core.xdmffile.XDMFFile``-like writer.

	Applies a write period that is either uniform (a single ``int``/:class:`TimeInterval`, the
	same for every registered load step) or per-step (a ``Sequence``, one entry per step -- ints
	and :class:`TimeInterval` freely mixed -- in the same order as
	``time_integrator.problem.steps``) -- see this module's own docstring. Whichever period
	applies, this proxy additionally *always* forwards a call explicitly marked ``force=True``, or
	one that lands on the currently active step's own last converged substep (``t`` matching
	``step.t_fin`` within :data:`_STEP_END_ABS_TOL`), regardless of what the period-based decision
	would otherwise say -- so a coarse period can never silently drop a step's own final frame,
	whether that frame is the step's natural end or (via ``force=True``) an early plateau exit
	(see this module's own docstring, "Early exit on a plateaued step").

	The decimation state resets whenever the currently active step changes (detected by identity
	against ``time_integrator.problem.steps``): the substep counter resets to zero for an ``int``
	period, and the last-written time resets to "never" for a :class:`TimeInterval` period -- so
	each step's own period always starts counting fresh from its own first substep, never
	influenced by whatever state the previous step left behind.
	"""

	def __init__(
		self, xdmf_file: Any, write_period: Union[_StepPeriod, Sequence[_StepPeriod]], time_integrator: Any
	) -> None:
		"""Wrap an already-constructed XDMF file writer.

		Args:
		    xdmf_file (Any): The real writer (``TimeIntegrator.xdmf_file_sol``, already opened
		        and already holding its own initial :math:`t=0` write).
		    write_period (Union[_StepPeriod, Sequence[_StepPeriod]]): A single period applied to
		        every step, or one period per step (``len(time_integrator.problem.steps)``
		        entries, same order) -- each entry an ``int`` or a :class:`TimeInterval`.
		    time_integrator (Any): The owning integrator -- read-only, consulted to discover which
		        of ``problem.steps`` is currently active (``time_integrator.step``) and that
		        step's own ``t_fin``.
		"""
		self._xdmf_file = xdmf_file
		self._time_integrator = time_integrator
		n_steps = len(time_integrator.problem.steps)
		self._periods_by_step: List[_StepPeriod] = (
			[write_period] * n_steps if isinstance(write_period, (int, TimeInterval)) else list(write_period)
		)
		self._call_count = 0
		self._last_written_t: Optional[float] = None
		self._last_step_index: Optional[int] = None

	def _current_step_index(self) -> int:
		"""Position, by identity, of the currently active step within ``problem.steps``."""
		return self._time_integrator.problem.steps.index(self._time_integrator.step)

	def write(self, t: float, force: bool = False) -> None:
		"""Forward to the wrapped file's own ``write``, throttled by the current step's own period.

		Always forwards regardless of the period when ``force`` is True, or when ``t`` is the
		currently active step's own last converged substep (module docstring).

		Args:
		    t (float): The current simulation time, forwarded unchanged whenever this call is
		        actually written through.
		    force (bool): If True, write unconditionally regardless of the period -- used by
		        :meth:`DecimatedTimeIntegrator.integrate` to guarantee a plateau-triggered early
		        exit's own last substep, and a hard-failure exit's own last successfully converged
		        substep, are never decimated away (module docstring, "Early exit on a plateaued
		        step" and "Guaranteed last frame on a hard failure"). Defaults to False.

		Note:
		    A call whose own ``t`` exactly matches (within :data:`_STEP_END_ABS_TOL`) the last
		    time actually forwarded to the wrapped file is always a no-op, ``force=True`` included
		    -- this is what lets :meth:`DecimatedTimeIntegrator.integrate` call ``write(t,
		    force=True)`` defensively (e.g. once more on a hard-failure exit, on top of whatever
		    this same substep's own earlier, ordinary write call already decided) without ever
		    risking a duplicate frame at the same simulated time in the XDMF/HDF5 time series.
		"""
		step_index = self._current_step_index()
		if step_index != self._last_step_index:
			self._call_count = 0
			self._last_written_t = None
			self._last_step_index = step_index
		self._call_count += 1

		if self._last_written_t is not None and abs(t - self._last_written_t) <= _STEP_END_ABS_TOL:
			return

		current_step = self._time_integrator.step
		is_last_substep_of_step = abs(t - current_step.t_fin) <= _STEP_END_ABS_TOL
		period = self._periods_by_step[step_index]

		if isinstance(period, TimeInterval):
			should_write = self._last_written_t is None or (t - self._last_written_t) >= period.seconds
		else:
			should_write = should_write_at_call(self._call_count, period)

		if force or is_last_substep_of_step or should_write:
			self._xdmf_file.write(t)
			self._last_written_t = t

	def close(self) -> None:
		"""Close the wrapped file."""
		self._xdmf_file.close()


class DecimatedTimeIntegrator(dolfin_mech.core.TimeIntegrator):
	"""``dolfin_mech.core.TimeIntegrator`` with a throttled XDMF writer and an optional plateau exit.

	Every constructor argument is forwarded unchanged to
	``dolfin_mech.core.TimeIntegrator.__init__`` -- the log file, ``.sta`` statistics table,
	QOI ``.dat`` file, and VTU/XML output (if enabled) are entirely unaffected; only the XDMF
	field-solution file (``write_sol``) is throttled. ``write_period=1`` (the default)
	reproduces the base class's own writing behavior exactly, so swapping
	``dolfin_mech.core.TimeIntegrator`` for this class is a no-op until ``write_period`` is
	actually changed -- in the default case every substep, including trivially every step's own
	last one, is already kept, so :class:`_DecimatedXDMFFile` is not even constructed. Likewise, a
	step whose own ``min_simu_seconds`` is left at its default (``None``, see ``fenicsoft.io.Step``)
	never even evaluates the plateau check -- see this module's own docstring, "Early exit on a
	plateaued step", for the full feature and why it required overriding ``integrate`` itself.

	Args:
	    *args (Any): Forwarded to ``dolfin_mech.core.TimeIntegrator.__init__``.
	    write_period (Union[_StepPeriod, Sequence[_StepPeriod]]): Either a single value -- an
	        ``int`` (keep one frame out of every ``write_period`` converged substeps, uniformly
	        for the whole run) or a :class:`TimeInterval` (keep one frame at least every
	        ``TimeInterval.seconds`` of simulated time, uniformly for the whole run) -- or a
	        ``Sequence`` with exactly one entry per registered load step (``problem.steps``, same
	        order), ints and :class:`TimeInterval` freely mixed, letting e.g. a constant-``dt``
	        step be kept by substep count while an accelerating/decelerating one is kept by
	        simulated time instead. Either way, a step's own last converged substep is always kept
	        regardless of the applicable period (see :class:`_DecimatedXDMFFile`, and this module's
	        own docstring for when to use which kind of period).
	    stagnation_plateau_relative_error (Union[float, Sequence[float]]): The relative-error
	        threshold below which a subsol's own substep-to-substep change is considered
	        negligible, for the plateau-exit feature (see :func:`is_subsol_stagnant`). Either a
	        single value, applied uniformly to every monitored subsol (the same convention
	        ``write_period`` uses for a scalar), or a ``Sequence`` with exactly one entry per
	        registered subsol (``problem.subsols``, same order -- the same convention ``sol_tol``
	        itself already uses), letting different subsols plateau at different thresholds --
	        useful for a subsol whose relative rate of change settles at a constant, nonzero value
	        (e.g. a field decaying purely exponentially towards zero: see
	        :func:`is_subsol_stagnant`'s own docstring) and so needs a looser threshold than one
	        that genuinely converges towards zero. This is entirely independent of
	        ``solver_parameters["sol_tol"]``, which is only ever consulted to decide *whether* a
	        subsol is monitored at all (``sol_tol is not None``), never as an input to the
	        threshold itself (see :func:`is_subsol_stagnant`'s own docstring for the dt-dependence
	        trade-off this implies). Defaults to :data:`_RELATIVE_ERROR_ZERO_EPS` (``1e-9``). Only
	        consulted for a step whose own ``min_simu_seconds`` is not ``None``.
	    functions_to_write (Optional[Sequence[Any]]): CHANGE (playground). Functions written at each
	        kept XDMF frame, in this order. ``None`` (default) keeps the base list (subsols,
	        ``*_old``, FOIs).
	    root_only_logs (bool): CHANGE (playground). Write ``.out``/``.sta``/``.dat`` on rank 0 only.
	        Defaults to True.
	    **kwargs (Any): Forwarded to ``dolfin_mech.core.TimeIntegrator.__init__``.

	Raises:
	    TypeError: If ``write_period`` is given as a sequence containing an entry that is
	        neither an ``int`` nor a :class:`TimeInterval`, or if ``stagnation_plateau_relative_error`` is
	        given as a sequence containing an entry that is not an ``int``/``float``.
	    ValueError: If ``write_period`` is given as a sequence whose length does not match
	        ``len(problem.steps)``, or that contains a non-positive integer entry; or if
	        ``stagnation_plateau_relative_error`` is not strictly positive (scalar form), or is given as a
	        sequence whose length does not match ``len(problem.subsols)`` or that contains a
	        non-positive entry.
	"""

	def __init__(
		self,
		*args: Any,
		write_period: Union[_StepPeriod, Sequence[_StepPeriod]] = 1,
		stagnation_plateau_relative_error: Union[float, Sequence[float]] = _RELATIVE_ERROR_ZERO_EPS,
		functions_to_write: Optional[Sequence[Any]] = None,
		root_only_logs: bool = True,
		**kwargs: Any,
	) -> None:
		"""Build the base integrator, then wrap its XDMF file writer if throttling is requested."""
		super().__init__(*args, **kwargs)

		# CHANGE (playground): rank-0 logs, then the selected XDMF fields (before any decimation proxy).
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

		if not self.write_sol:
			return

		if isinstance(write_period, (int, TimeInterval)):
			if isinstance(write_period, int) and write_period <= 1:
				return
			self.xdmf_file_sol = _DecimatedXDMFFile(self.xdmf_file_sol, write_period, self)
			return

		periods: List[Any] = list(write_period)
		n_steps = len(self.problem.steps)
		if len(periods) != n_steps:
			raise ValueError(
				f"write_period, given as a per-step sequence, must have exactly one entry per "
				f"registered load step (len(problem.steps)={n_steps}); got {len(periods)} entries."
			)
		if not all(isinstance(period, (int, TimeInterval)) for period in periods):
			raise TypeError(f"Every per-step write_period entry must be an int or a TimeInterval; got {periods}.")
		if any(isinstance(period, int) and period < 1 for period in periods):
			raise ValueError(f"Every per-step integer write_period entry must be a positive integer; got {periods}.")
		if any(isinstance(period, TimeInterval) or period > 1 for period in periods):
			self.xdmf_file_sol = _DecimatedXDMFFile(self.xdmf_file_sol, periods, self)

	def _silence_logs(self) -> None:
		"""CHANGE (playground). Redirect this rank's ``.out``, ``.sta`` and ``.dat`` writers to ``os.devnull``."""
		for printer in (self.printer, self.table_printer):
			if printer.must_close:
				printer.output.close()
			printer.output = open(os.devnull, "w")  # noqa: SIM115 (closed by the printer)
			printer.must_close = True
		if self.write_qois:
			self.qoi_printer.file.close()
			self.qoi_printer.file = open(os.devnull, "w")  # noqa: SIM115 (closed by the printer)

	def _reopen_xdmf(self, functions_to_write: Sequence[Any]) -> None:
		"""CHANGE (playground). Reopen the results XDMF on ``functions_to_write`` and rewrite :math:`t = 0`.

		Args:
		    functions_to_write: Functions written at each kept frame.
		"""
		self.xdmf_file_sol.close()
		self.functions_to_write = list(functions_to_write)
		self.xdmf_file_sol = XDMFFile(filename=self.write_sol_filebasename + ".xdmf", functions=self.functions_to_write)
		self.xdmf_file_sol.write(0.0)

	def _step_is_stagnant(self) -> bool:
		"""True if every monitored subsol's own rate of change has plateaued for the current step.

		"Monitored" means every subsol whose own ``solver_parameters["sol_tol"]`` entry is not
		``None`` -- matching ``dolfin_mech.core.nonlinearsolver.NonlinearSolver.exit_test``'s own
		convention for which subsols actually gate convergence. ``sol_tol`` is consulted only for
		this monitored/unmonitored gate here -- it plays no role in the stagnation threshold itself
		(see :func:`is_subsol_stagnant`). Reads
		``subsol.func``/``subsol.func_old`` (the converged values at the end of this substep and the
		end of the previous one, respectively -- exactly what ``NonlinearSolver.compute_sol_norm``
		itself reads for its own per-iteration norms), and takes the vector difference the same way
		``NonlinearSolver.assemble_linear_system`` already does for ``dres_vec``
		(``self.res_vec - self.res_old_vec``).

		Wrapped in an ``MPI.COMM_WORLD.allreduce(..., op=MPI.MIN)`` for defensive parallel
		consistency, matching this integrator's own established pattern for every other
		substep-level boolean decision (``solver_success``, ``local_end``) -- even though the
		underlying PETSc vector norms are themselves already collective (identical on every rank
		without it).

		Returns:
		    bool: True if every monitored subsol is judged stagnant (see
		        :func:`is_subsol_stagnant`).
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
		"""Executes the time integration loop, with an optional early exit once a step has plateaued.

		This OVERRIDES ``dolfin_mech.core.TimeIntegrator.integrate`` in full -- see this module's
		own docstring, "Early exit on a plateaued step" and "Guaranteed last frame on a hard
		failure", for why a full copy (not a call to ``super().integrate()`` plus a hook) was
		necessary, and for the exact caveat about staying in sync with ``dolfin_mech`` upstream.
		Every line below not explicitly marked "-- new --" is copied verbatim from
		``dolfin_mech.core.timeintegrator.TimeIntegrator.integrate``.

		Returns:
		    bool: True if the simulation completed -- naturally (every step reached its own
		        ``t_fin``) or via an early plateau exit on any step where one was requested and
		        triggered; False if the solver failed to move forward even at a step's own
		        ``dt_min`` -- in which case the last successfully converged state is still
		        guaranteed to have been written to the XDMF trajectory before returning (module
		        docstring, "Guaranteed last frame on a hard failure"), regardless of what
		        ``write_period`` would otherwise have decided for that particular substep.
		"""
		k_t_tot = 0
		n_iter_tot = 0
		self.printer.inc()
		for k_step in range(1, len(self.problem.steps) + 1):
			self.printer.print_var("k_step", k_step, -1)

			self.step = self.problem.steps[k_step - 1]
			# -- new: this step's own optional plateau-exit floor; None means "feature off". --
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

					# -- new: only even consider a plateau exit once min_simu_seconds have
					# actually elapsed for this step, and only if it did not already reach its
					# own natural end this same substep. --
					global_stagnant = False
					if not global_end and min_simu_seconds is not None:
						if (t - self.step.t_ini) >= min_simu_seconds:
							global_stagnant = self._step_is_stagnant()
					step_is_ending = bool(global_end or global_stagnant)  # -- new --

					if self.write_sol:
						# -- new: force through the write-throttling proxy (if any) whenever this
						# substep ends the step, for either reason. Natural completion already
						# matches _DecimatedXDMFFile's own t==t_fin check unassisted; force=True is
						# what saves a plateau exit's own final frame (t != t_fin) from being
						# silently decimated away. --
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
						if global_stagnant and not global_end:  # -- new --
							self.printer.print_str(
								f"Step plateaued after {t - self.step.t_ini:.6g}s of simulated time "
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
