"""Unit tests of the decimation helpers and of ``DecimatedTimeIntegrator``.

``dolfin_mech.core.TimeIntegrator.__init__`` is monkeypatched: :func:`_patch_base_time_integrator_init` for the
``__init__`` checks, :func:`_patch_full_time_integrator_init` to run ``integrate()`` with scripted fakes.
"""

from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import dolfin
import dolfin_mech
import pytest

from hill_playground.integration.decimated_time_integrator import (
	_DEFAULT_PLATEAU_RELATIVE_ERROR,  # type: ignore
	DecimatedTimeIntegrator,
	TimeInterval,
	_DecimatedXDMFFile,  # type: ignore
	is_subsol_stagnant,
	should_write_at_call,
)


class _FakeXdmfFile:
	"""Minimal stand-in for ``dolfin_mech.core.xdmffile.XDMFFile``: records calls, writes nothing."""

	def __init__(self) -> None:
		"""Start with an empty call log."""
		self.written_times: list[Any] = []
		self.closed = False

	def write(self, t: float) -> None:
		"""Record the time this call was made with."""
		self.written_times.append(t)

	def close(self) -> None:
		"""Record that this file was closed."""
		self.closed = True


class _FakeStep:
	"""Stand-in for ``dolfin_mech.core.step.Step``; defaults let ``_FakeStep(t_fin=...)`` suffice."""

	def __init__(
		self,
		t_fin: float,
		t_ini: float = 0.0,
		dt_ini: float = 1.0,
		dt_min: float = 1e-6,
		dt_max: Optional[float] = None,
		operators: Optional[list[Any]] = None,
		constraints: Optional[list[Any]] = None,
		min_simu_seconds: Optional[float] = None,
	) -> None:
		"""Store the step timing bounds, operators and constraints.

		Args:
		    t_fin (float): Final simulated time.
		    t_ini (float): Initial simulated time. Defaults to ``0.0``.
		    dt_ini (float): Initial substep size. Defaults to ``1.0``.
		    dt_min (float): Smallest allowed ``dt``. Defaults to ``1e-6``.
		    dt_max (Optional[float]): Largest allowed ``dt``. Defaults to ``dt_ini`` (no acceleration headroom).
		    operators (Optional[list]): Time-dependent operators. Defaults to ``[]``.
		    constraints (Optional[list]): Step constraints. Defaults to ``[]``.
		    min_simu_seconds (Optional[float]): Plateau-exit floor. Defaults to ``None`` (plateau exit off).
		"""
		self.t_fin = t_fin
		self.t_ini = t_ini
		self.dt_ini = dt_ini
		self.dt_min = dt_min
		self.dt_max = dt_max if dt_max is not None else dt_ini
		self.operators = operators if operators is not None else []
		self.constraints = constraints if constraints is not None else []
		self.min_simu_seconds = min_simu_seconds


class _FakeVector(list[Any]):
	"""Stand-in for a PETSc vector: a ``list`` (slice read/assignment), mutated in place."""


class _FakeFunc:
	"""Minimal stand-in for a ``dolfin.Function``: exposes only ``.vector()``."""

	def __init__(self, values: list[Any]) -> None:
		"""Wrap ``values`` in one :class:`_FakeVector`, returned by every ``.vector()`` call."""
		self._vector = _FakeVector(values)

	def vector(self) -> "_FakeVector":
		"""Return the same vector at every call, so mutations persist."""
		return self._vector


class _FakeInelasticBehavior:
	"""Stand-in for an internal inelastic behavior: records calls, changes nothing."""

	def __init__(self) -> None:
		"""Start with empty call logs."""
		self.updated_at: list[Any] = []
		self.restored = 0

	def update_internal_variables_at_t(self, t: float) -> None:
		"""Record the simulated time ``t``."""
		self.updated_at.append(t)

	def restore_old_value(self) -> None:
		"""Count a restore (failed substep retried)."""
		self.restored += 1


class _FakeConstraint:
	"""Stand-in for a step constraint: records calls, changes nothing."""

	def __init__(self) -> None:
		"""Start with empty call logs."""
		self.set_at: list[Any] = []
		self.restored = 0

	def set_value_at_t_step(self, t_step: float) -> None:
		"""Record the step progress ``t_step``."""
		self.set_at.append(t_step)

	def restore_old_value(self) -> None:
		"""Count a restore (failed substep retried)."""
		self.restored += 1


class _FakeSolver:
	"""Stand-in for ``dolfin_mech.core.NonlinearSolver``: ``.solve`` returns scripted ``(success, n_iter)``."""

	def __init__(self, outcomes: Sequence[Tuple[bool, int]]) -> None:
		"""Store the scripted outcomes.

		Args:
		    outcomes (Sequence[Tuple[bool, int]]): One ``(success, n_iter)`` per ``.solve`` call; the last
		        one repeats.
		"""
		self._outcomes = list(outcomes)
		self._n_calls = 0
		self.constraints: list[Any] = []

	def solve(self, k_step: int, k_t: int, dt: float, t: float) -> Tuple[bool, int]:
		"""Return the next scripted outcome (the last one once exhausted)."""
		index = min(self._n_calls, len(self._outcomes) - 1)
		self._n_calls += 1
		return self._outcomes[index]


class _FakePrinter:
	"""Minimal stand-in for ``myPythonLibrary.Printer``: records depth changes and printed strings."""

	def __init__(self) -> None:
		"""Start at depth 0 with an empty message log."""
		self.depth = 0
		self.messages: list[Any] = []

	def inc(self) -> None:
		"""Increase the nesting depth."""
		self.depth += 1

	def dec(self) -> None:
		"""Decrease the nesting depth."""
		self.depth -= 1

	def print_var(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
		"""No-op: tests assert on outcomes, not on the substep log."""

	def print_str(self, text: str) -> None:
		"""Record a status message (e.g. plateau exit, solver failure)."""
		self.messages.append(text)


class _FakeTablePrinter:
	"""Minimal stand-in for ``myPythonLibrary.TablePrinter``: records every row written to it."""

	def __init__(self) -> None:
		"""Start with an empty row log."""
		self.rows: list[Any] = []

	def write_line(self, row: list[Any]) -> None:
		"""Record one ``[k_step, k_t, dt, t, t_step, n_iter, success]`` row."""
		self.rows.append(row)


class _FakeProblem:
	"""Stand-in exposing what :meth:`DecimatedTimeIntegrator.integrate` reads on ``self.problem``.

	``subsols`` has 0 or 1 entry (no ``dolfin.assign``), except in
	:func:`test_integrate_multi_subsol_failure_branch_calls_dolfin_assign`.
	"""

	def __init__(
		self,
		steps: list[Any],
		subsols: Optional[list[Any]] = None,
		constraints: Optional[list[Any]] = None,
		inelastic_behaviors_internal: Optional[list[Any]] = None,
		qois: Optional[list[Any]] = None,
		sol_func: Optional["_FakeFunc"] = None,
		sol_old_func: Optional["_FakeFunc"] = None,
	) -> None:
		"""Store the collaborators read by ``integrate()``.

		Args:
		    steps (list): Ordered step objects.
		    subsols (Optional[list]): Subsol placeholders. Defaults to ``[]``.
		    constraints (Optional[list]): Step-independent constraints. Defaults to ``[]``.
		    inelastic_behaviors_internal (Optional[list]): Updated per converged substep, restored per failed one.
		        Defaults to ``[]``.
		    qois (Optional[list]): Quantities of interest. Defaults to ``[]`` (tests use ``write_qois=False``).
		    sol_func (Optional[_FakeFunc]): Current solution. Defaults to ``_FakeFunc([0.0])``.
		    sol_old_func (Optional[_FakeFunc]): Previous solution. Defaults to ``_FakeFunc([0.0])``.
		"""
		self.steps = steps
		self.subsols = subsols if subsols is not None else []
		self.constraints = constraints if constraints is not None else []
		self.inelastic_behaviors_internal = (
			inelastic_behaviors_internal if inelastic_behaviors_internal is not None else []
		)
		self.qois = qois if qois is not None else []
		self.sol_func = sol_func if sol_func is not None else _FakeFunc([0.0])
		self.sol_old_func = sol_old_func if sol_old_func is not None else _FakeFunc([0.0])

	def set_variational_formulation(self, k_step: int) -> None:
		"""No-op: no variational formulation."""

	def update_fois(self) -> None:
		"""No-op: no FOI."""

	def update_qois(self, dt: float, k_step: int) -> None:
		"""No-op (tests use ``write_qois=False``)."""

	def get_subsols_func_old_lst(self) -> list[Any]:
		"""Return the subsols (read only when ``len(subsols) > 1``)."""
		return list(self.subsols)

	def get_subsols_func_lst(self) -> list[Any]:
		"""Return the subsols (failure-branch counterpart of :meth:`get_subsols_func_old_lst`)."""
		return list(self.subsols)


class _FakeTimeIntegrator:
	"""Stand-in for the owning ``TimeIntegrator``: ``problem.steps`` and a settable current ``step``."""

	def __init__(self, steps: list[Any]) -> None:
		"""Register the fixed step list and start with the first one active."""
		self.problem = _FakeProblem(steps)
		self.step = steps[0]


def test_should_write_at_call_always_keeps_the_first_call() -> None:
	"""Call index 1 is always kept, regardless of period."""
	assert should_write_at_call(call_index=1, period=5) is True
	assert should_write_at_call(call_index=1, period=1) is True


def test_should_write_at_call_period_one_keeps_every_call() -> None:
	"""``period=1`` keeps every call (base-class behavior)."""
	assert [should_write_at_call(call_index=i, period=1) for i in range(1, 8)] == [True] * 7


def test_should_write_at_call_decimates_by_period() -> None:
	"""``period=3`` keeps calls 1, 4, 7, 10."""
	kept = [i for i in range(1, 13) if should_write_at_call(call_index=i, period=3)]
	assert kept == [1, 4, 7, 10]


def test_should_write_at_call_rejects_non_positive_period() -> None:
	"""``period=0`` raises ``ValueError``."""
	with pytest.raises(ValueError):
		should_write_at_call(call_index=1, period=0)


def test_time_interval_rejects_non_positive_seconds() -> None:
	"""``seconds <= 0`` raises ``ValueError``."""
	with pytest.raises(ValueError):
		TimeInterval(seconds=0.0)
	with pytest.raises(ValueError):
		TimeInterval(seconds=-5.0)


def test_time_interval_accepts_strictly_positive_seconds() -> None:
	"""A strictly positive value is stored unchanged."""
	assert TimeInterval(seconds=900.0).seconds == 900.0


def test_periodic_xdmf_file_forwards_first_and_every_period_write() -> None:
	"""``period=3`` keeps the first write and every third one (``t_fin`` never reached)."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=3, time_integrator=fake_integrator)

	for t in range(1, 11):  # 10 calls: t=1..10, mimicking 10 converged substeps.
		proxy.write(float(t))

	assert fake_file.written_times == [1.0, 4.0, 7.0, 10.0]


def test_periodic_xdmf_file_period_one_writes_every_call() -> None:
	"""``period=1`` forwards every call."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=1, time_integrator=fake_integrator)

	for t in range(1, 4):
		proxy.write(float(t))

	assert fake_file.written_times == [1.0, 2.0, 3.0]


def test_periodic_xdmf_file_close_delegates_to_wrapped_file() -> None:
	"""``close()`` closes the wrapped file."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=5, time_integrator=fake_integrator)

	proxy.close()

	assert fake_file.closed is True


def test_periodic_xdmf_file_always_writes_the_last_substep_of_a_step_even_if_the_period_would_skip_it() -> None:
	"""Step end :math:`t = t_{fin}` is kept off-period: ``period=3``, 8 calls -> 1, 4, 7, 8."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=8.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=3, time_integrator=fake_integrator)

	for t in range(1, 9):  # 8 calls: t=1..8, t=8.0 == t_fin.
		proxy.write(float(t))

	assert fake_file.written_times == [1.0, 4.0, 7.0, 8.0]


def test_periodic_xdmf_file_uses_current_steps_own_period_and_forces_each_steps_own_last_substep() -> None:
	"""Per-step periods ``[1, 5]``: each step uses its period and keeps its step end."""
	fake_file = _FakeXdmfFile()
	step_a = _FakeStep(t_fin=5.0)
	step_b = _FakeStep(t_fin=17.0)
	fake_integrator = _FakeTimeIntegrator([step_a, step_b])
	proxy = _DecimatedXDMFFile(fake_file, write_period=[1, 5], time_integrator=fake_integrator)

	for t in range(1, 6):  # step A: 5 substeps, period=1 -> every call kept.
		proxy.write(float(t))

	fake_integrator.step = step_b
	for t in range(6, 18):  # step B: 12 substeps (t=6..17), period=5.
		proxy.write(float(t))

	# Step A: [1, 2, 3, 4, 5] (all kept). Step B: naturally kept at its own call indices 1, 6, 11
	# (t=6, 11, 16); call index 12 (t=17, the step's own last substep) is forced through too.
	assert fake_file.written_times == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 16.0, 17.0]


def test_periodic_xdmf_file_resets_call_count_at_step_boundary() -> None:
	"""The call counter resets at a step change: the first substep of the new step is kept."""
	fake_file = _FakeXdmfFile()
	step_a = _FakeStep(t_fin=3.0)
	step_b = _FakeStep(t_fin=6.0)
	fake_integrator = _FakeTimeIntegrator([step_a, step_b])
	proxy = _DecimatedXDMFFile(fake_file, write_period=[5, 5], time_integrator=fake_integrator)

	for t in range(1, 4):  # 3 calls in step A at period=5: call index 1 (t=1) kept by period,
		proxy.write(float(t))  # call index 3 (t=3) forced as step A's own last substep.
	fake_integrator.step = step_b
	for t in range(4, 7):  # 3 more calls, now step B: counter restarts, so t=4 is kept too.
		proxy.write(float(t))

	assert fake_file.written_times == [1.0, 3.0, 4.0, 6.0]


def test_periodic_xdmf_file_rejects_mismatched_per_step_length() -> None:
	"""A per-step sequence shorter than ``problem.steps`` raises ``IndexError``."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	with pytest.raises(IndexError):
		proxy = _DecimatedXDMFFile(fake_file, write_period=[3], time_integrator=fake_integrator)
		fake_integrator.step = fake_integrator.problem.steps[1]
		proxy.write(2.0)


def test_periodic_xdmf_file_time_interval_decimates_by_elapsed_simulated_time_under_uneven_dt() -> None:
	r"""``TimeInterval(100)`` under uneven ``dt``: keep iff :math:`t - t_{last} \ge 100`, plus the step end."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=500.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=TimeInterval(100.0), time_integrator=fake_integrator)

	for t in (30.0, 90.0, 200.0, 350.0, 380.0, 460.0, 500.0):
		proxy.write(t)

	# t=30: first call, always kept. t=90: only 60s elapsed, skipped. t=200: 170s elapsed, kept.
	# t=350: 150s elapsed, kept. t=380: 30s elapsed, skipped. t=460: 110s elapsed, kept.
	# t=500: only 40s elapsed (would be skipped), but it is the step's own last substep -> forced.
	assert fake_file.written_times == [30.0, 200.0, 350.0, 460.0, 500.0]


def test_periodic_xdmf_file_mixes_int_and_time_interval_periods_across_steps() -> None:
	"""Per-step ``[1, TimeInterval(50)]``: each step has its own state, reset at the boundary."""
	fake_file = _FakeXdmfFile()
	step_a = _FakeStep(t_fin=3.0)
	step_b = _FakeStep(t_fin=200.0)
	fake_integrator = _FakeTimeIntegrator([step_a, step_b])
	proxy = _DecimatedXDMFFile(fake_file, write_period=[1, TimeInterval(50.0)], time_integrator=fake_integrator)

	for t in (1.0, 2.0, 3.0):  # step A: period=1, every call kept.
		proxy.write(t)

	fake_integrator.step = step_b
	for t in (10.0, 40.0, 70.0, 120.0, 190.0, 200.0):  # step B: TimeInterval(50.0).
		proxy.write(t)

	# Step A: [1, 2, 3] (all kept). Step B: t=10 (first, kept), t=40 (30s elapsed, skipped),
	# t=70 (60s elapsed, kept), t=120 (50s elapsed, kept), t=190 (70s elapsed, kept),
	# t=200 (only 10s elapsed, but the step's own last substep -> forced).
	assert fake_file.written_times == [1.0, 2.0, 3.0, 10.0, 70.0, 120.0, 190.0, 200.0]


def test_periodic_xdmf_file_force_writes_through_even_when_the_period_would_skip_it() -> None:
	"""``force=True`` writes off-period (used for the plateau-exit frame)."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=5, time_integrator=fake_integrator)

	proxy.write(1.0)  # call index 1: kept anyway (always-keep-first-call rule).
	proxy.write(2.0, force=True)  # call index 2: period alone would skip this -- forced through.
	proxy.write(3.0)  # call index 3: period alone skips this, and it is not forced -- dropped.

	assert fake_file.written_times == [1.0, 2.0]


def test_periodic_xdmf_file_force_defaults_to_false_preserving_prior_signature() -> None:
	"""Without ``force``, the period rule applies."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=5, time_integrator=fake_integrator)

	proxy.write(1.0)
	proxy.write(2.0)

	assert fake_file.written_times == [1.0]


def test_periodic_xdmf_file_force_at_an_already_written_t_is_a_harmless_no_op() -> None:
	"""``force=True`` at the last written :math:`t` writes nothing (no duplicate frame).

	Lets ``integrate()`` force the last converged substep before a solver failure ("Always kept" list of
	:mod:`hill_playground.integration.decimated_time_integrator`).
	"""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=1, time_integrator=fake_integrator)

	proxy.write(5.0)  # ordinary write (period=1 keeps every call).
	proxy.write(5.0, force=True)  # same t again, forced -- must not duplicate.

	assert fake_file.written_times == [5.0]


def test_periodic_xdmf_file_force_at_a_new_t_after_an_already_written_one_still_writes() -> None:
	"""The duplicate guard only drops an exact repeat of :math:`t`."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=5, time_integrator=fake_integrator)

	proxy.write(1.0)  # call index 1: always kept.
	proxy.write(2.0, force=True)  # a genuinely new t, forced -- must go through.
	proxy.write(2.0, force=True)  # exact repeat of the previous t -- must be suppressed.

	assert fake_file.written_times == [1.0, 2.0]


def _patch_base_time_integrator_init(
	monkeypatch: pytest.MonkeyPatch, xdmf_file: "_FakeXdmfFile", write_sol: bool = True
) -> None:
	"""Stub ``dolfin_mech.core.TimeIntegrator.__init__`` to set only ``problem``, ``write_sol``, ``xdmf_file_sol``.

	Args:
	    monkeypatch (pytest.MonkeyPatch): Patch/restore fixture.
	    xdmf_file (_FakeXdmfFile): Writer assigned to ``xdmf_file_sol`` (kept by the caller for assertions).
	    write_sol (bool): Value of ``write_sol``. Defaults to ``True``.
	"""

	def fake_init(self: Any, problem: Any, **kwargs: Any) -> None:
		self.problem = problem
		self.write_sol = write_sol
		self.xdmf_file_sol = xdmf_file

	monkeypatch.setattr(dolfin_mech.core.TimeIntegrator, "__init__", fake_init)


def _patch_full_time_integrator_init(
	monkeypatch: pytest.MonkeyPatch,
	xdmf_file: "_FakeXdmfFile",
	solver: "_FakeSolver",
	parameters: Optional[Dict[str, Any]] = None,
	write_sol: bool = True,
) -> None:
	"""Stub ``dolfin_mech.core.TimeIntegrator.__init__`` with every attribute ``integrate()`` reads.

	``write_qois`` is ``False``; ``write_vtus`` and ``write_xmls`` too (set only when ``write_sol``).

	Args:
	    monkeypatch (pytest.MonkeyPatch): Patch/restore fixture.
	    xdmf_file (_FakeXdmfFile): Writer assigned to ``xdmf_file_sol`` if ``write_sol``.
	    solver (_FakeSolver): Scripted solver.
	    parameters (Optional[Dict[str, Any]]): ``n_iter_for_accel``, ``n_iter_for_decel``, ``accel_coeff``,
	        ``decel_coeff``, with the ``dolfin_mech`` defaults (4, 16, 2, 2). Defaults to ``{}``.
	    write_sol (bool): Value of ``write_sol``. Defaults to ``True``.
	"""
	resolved_parameters = parameters if parameters is not None else {}

	def fake_init(self: Any, problem: Any, **_kwargs: Any) -> None:
		self.problem = problem
		self.solver = solver
		self.solver.constraints = []

		self.n_iter_for_accel = resolved_parameters.get("n_iter_for_accel", 4)
		self.n_iter_for_decel = resolved_parameters.get("n_iter_for_decel", 16)
		self.accel_coeff = resolved_parameters.get("accel_coeff", 2)
		self.decel_coeff = resolved_parameters.get("decel_coeff", 2)

		self.printer = _FakePrinter()
		self.table_printer = _FakeTablePrinter()
		self.write_qois = False

		self.write_sol = bool(write_sol)
		if self.write_sol:
			self.xdmf_file_sol = xdmf_file
			self.write_vtus = False
			self.write_vtus_with_preserved_connectivity = False
			self.write_xmls = False

	monkeypatch.setattr(dolfin_mech.core.TimeIntegrator, "__init__", fake_init)


def test_decimated_time_integrator_default_write_period_does_not_wrap(monkeypatch: pytest.MonkeyPatch) -> None:
	"""``write_period=1`` (default) leaves the writer unwrapped."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=2.0)])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.xdmf_file_sol is fake_file


def test_decimated_time_integrator_scalar_period_above_one_throttles_writes(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Scalar ``write_period=3`` wraps the writer and keeps calls 1, 4, 7."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	step = _FakeStep(t_fin=1000.0)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=3)
	integrator.step = step
	for t in range(1, 8):
		integrator.xdmf_file_sol.write(float(t))

	assert fake_file.written_times == [1.0, 4.0, 7.0]


def test_decimated_time_integrator_scalar_time_interval_throttles_writes(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Scalar :class:`TimeInterval` always wraps the writer and decimates by elapsed time."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	step = _FakeStep(t_fin=500.0)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=TimeInterval(100.0))
	integrator.step = step
	for t in (30.0, 90.0, 200.0, 350.0, 380.0, 460.0, 500.0):
		integrator.xdmf_file_sol.write(t)

	assert fake_file.written_times == [30.0, 200.0, 350.0, 460.0, 500.0]


def test_decimated_time_integrator_per_step_sequence_applies_ramp_then_plateau(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Per-step ``[1, 5]`` (ramp, hold): ramp kept in full, hold decimated, hold end kept."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	step_ramp = _FakeStep(t_fin=5.0)
	step_hold = _FakeStep(t_fin=13.0)
	problem = _FakeProblem([step_ramp, step_hold])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=[1, 5])
	integrator.step = step_ramp
	for t in range(1, 6):  # ramp: 5 substeps, period=1 -> every call kept.
		integrator.xdmf_file_sol.write(float(t))
	integrator.step = step_hold
	for t in range(6, 14):  # hold: 8 substeps (t=6..13), period=5.
		integrator.xdmf_file_sol.write(float(t))

	# Ramp: all 5 kept. Hold: naturally kept at its own call indices 1, 6 (t=6, 11); call index 8
	# (t=13, the hold's own last substep) is forced through even though period=5 would skip it.
	assert fake_file.written_times == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 13.0]


def test_decimated_time_integrator_per_step_sequence_mixes_int_and_time_interval(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Per-step ``[1, TimeInterval(50)]`` (constant-``dt`` ramp, adaptive hold) is applied step by step."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	step_ramp = _FakeStep(t_fin=3.0)
	step_hold = _FakeStep(t_fin=200.0)
	problem = _FakeProblem([step_ramp, step_hold])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=[1, TimeInterval(50.0)])
	integrator.step = step_ramp
	for t in (1.0, 2.0, 3.0):
		integrator.xdmf_file_sol.write(t)
	integrator.step = step_hold
	for t in (10.0, 40.0, 70.0, 120.0, 190.0, 200.0):
		integrator.xdmf_file_sol.write(t)

	assert fake_file.written_times == [1.0, 2.0, 3.0, 10.0, 70.0, 120.0, 190.0, 200.0]


def test_decimated_time_integrator_per_step_sequence_all_ones_does_not_wrap(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Per-step ``[1, 1]`` leaves the writer unwrapped."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=[1, 1])

	assert integrator.xdmf_file_sol is fake_file


def test_decimated_time_integrator_rejects_mismatched_per_step_length(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A per-step sequence of length ``!= len(problem.steps)`` raises ``ValueError``."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	with pytest.raises(ValueError, match="one entry per"):
		DecimatedTimeIntegrator(problem=problem, write_period=[1, 2, 3])


def test_decimated_time_integrator_rejects_non_positive_per_step_entry(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A per-step entry ``0`` raises ``ValueError``."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	with pytest.raises(ValueError, match="positive integer"):
		DecimatedTimeIntegrator(problem=problem, write_period=[1, 0])


@pytest.mark.parametrize("write_sol", [True, False])
def test_decimated_time_integrator_rejects_non_positive_scalar_period(
	monkeypatch: pytest.MonkeyPatch, write_sol: bool
) -> None:
	"""A scalar ``write_period`` ``0`` raises ``ValueError``, with or without XDMF output."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file, write_sol=write_sol)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)])

	with pytest.raises(ValueError, match="positive integer"):
		DecimatedTimeIntegrator(problem=problem, write_period=0)


def test_decimated_time_integrator_rejects_invalid_per_step_entry_type(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A per-step entry neither ``int`` nor :class:`TimeInterval` raises ``TypeError``."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	with pytest.raises(TypeError, match="int or a TimeInterval"):
		DecimatedTimeIntegrator(problem=problem, write_period=[1, "every 100 seconds"])  # type: ignore


def test_decimated_time_integrator_skips_wrapping_when_write_sol_is_falsy(monkeypatch: pytest.MonkeyPatch) -> None:
	"""``write_sol`` falsy: nothing to throttle, the writer is unwrapped."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file, write_sol=False)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=50)

	assert integrator.xdmf_file_sol is fake_file


def test_decimated_time_integrator_rejects_non_positive_stagnation_plateau_relative_error(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	r"""``stagnation_plateau_relative_error`` :math:`\varepsilon = 0` raises ``ValueError``."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)])

	with pytest.raises(ValueError, match="stagnation_plateau_relative_error"):
		DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=0.0)


def test_decimated_time_integrator_default_stagnation_plateau_relative_error_is_the_default_threshold(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	r"""Default :math:`\varepsilon` is ``_DEFAULT_PLATEAU_RELATIVE_ERROR`` (1e-9): stagnant only once ~unchanged."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.stagnation_plateau_relative_error == _DEFAULT_PLATEAU_RELATIVE_ERROR


def test_decimated_time_integrator_accepts_per_subsol_stagnation_plateau_relative_error_sequence(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	r"""A per-subsol :math:`\varepsilon` sequence (one entry per subsol) is stored as given.

	A constant nonzero relative change (e.g. pure exponential decay) only plateaus with a larger :math:`\varepsilon`.
	"""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)], subsols=[object(), object()])

	integrator = DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=[3.0, 150000.0])

	assert integrator.stagnation_plateau_relative_error == [3.0, 150000.0]


def test_decimated_time_integrator_rejects_mismatched_stagnation_plateau_relative_error_length(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	r"""A per-subsol :math:`\varepsilon` sequence of wrong length raises ``ValueError``."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)], subsols=[object(), object()])

	with pytest.raises(ValueError, match="one entry per registered subsol"):
		DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=[3.0, 150000.0, 1.0])


def test_decimated_time_integrator_rejects_non_positive_stagnation_plateau_relative_error_entry(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	r"""A per-subsol :math:`\varepsilon` entry ``0`` raises ``ValueError``."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)], subsols=[object(), object()])

	with pytest.raises(ValueError, match="strictly positive"):
		DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=[3.0, 0.0])


def test_decimated_time_integrator_rejects_invalid_stagnation_plateau_relative_error_entry_type(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	r"""A per-subsol :math:`\varepsilon` entry neither ``int`` nor ``float`` raises ``TypeError``."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)], subsols=[object(), object()])

	with pytest.raises(TypeError, match="int or float"):
		DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=[3.0, "loose"])  # type: ignore


def test_step_is_stagnant_treats_every_subsol_unmonitored_as_trivially_stagnant(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Every subsol unmonitored (``sol_tol=None``): all skipped, the step is trivially stagnant."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	step = _FakeStep(t_fin=1.0)
	problem = _FakeProblem([step], subsols=[object(), object()])

	integrator = DecimatedTimeIntegrator(problem=problem)
	integrator.step = step
	integrator.solver = SimpleNamespace(sol_tol=[None, None])

	assert integrator._step_is_stagnant() is True  # type: ignore


def test_periodic_xdmf_file_force_at_previous_step_end_after_step_change_is_a_no_op() -> None:
	"""Failure on the first substep of a step: forcing :math:`t_{ini}` (= previous :math:`t_{fin}`) writes nothing."""
	fake_file = _FakeXdmfFile()
	step_a, step_b = _FakeStep(t_fin=2.0), _FakeStep(t_fin=5.0)
	fake_integrator = _FakeTimeIntegrator([step_a, step_b])
	proxy = _DecimatedXDMFFile(fake_file, write_period=[1, 3], time_integrator=fake_integrator)

	proxy.write(1.0)
	proxy.write(2.0)  # step A end.
	fake_integrator.step = step_b
	proxy.write(2.0, force=True)  # dt < dt_min in step B: last converged state = step A end, already written.

	assert fake_file.written_times == [1.0, 2.0]


def test_periodic_xdmf_file_force_at_initial_time_is_a_no_op() -> None:
	"""The wrapped file already holds :math:`t = 0`: forcing it again writes nothing."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=TimeInterval(0.5), time_integrator=fake_integrator)

	proxy.write(0.0, force=True)

	assert fake_file.written_times == []


def test_integrate_zero_subsol_step_runs_inelastic_behaviors_and_wraps_the_writer(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Successful run with 0 subsols, an inelastic behavior and a wrapped writer (``write_period=2``)."""
	fake_file = _FakeXdmfFile()
	solver = _FakeSolver(outcomes=[(True, 2)])
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver)
	inelastic_behavior = _FakeInelasticBehavior()
	step = _FakeStep(t_fin=1.0, dt_ini=1.0)
	problem = _FakeProblem([step], inelastic_behaviors_internal=[inelastic_behavior])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=2, stagnation_plateau_relative_error=3.0)

	assert integrator.integrate() is True
	assert inelastic_behavior.updated_at == [1.0]
	assert isinstance(integrator.xdmf_file_sol, _DecimatedXDMFFile)
	assert fake_file.written_times == [1.0]


def test_integrate_covers_uncapped_acceleration_and_deceleration_arcs(monkeypatch: pytest.MonkeyPatch) -> None:
	"""``dt`` accelerates below ``dt_max`` (no clamp), then a high ``n_iter`` decelerates it."""
	fake_file = _FakeXdmfFile()
	# n_iter=1 accelerates dt to 2.0 (< dt_max, no clamp); n_iter=20 decelerates it; later calls re-accelerate.
	solver = _FakeSolver(outcomes=[(True, 1), (True, 20)] + [(True, 2)] * 20)
	parameters = {"n_iter_for_accel": 4, "n_iter_for_decel": 8, "accel_coeff": 2.0, "decel_coeff": 10.0}
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver, parameters=parameters, write_sol=False)
	step = _FakeStep(t_fin=20.0, dt_ini=1.0, dt_min=0.05, dt_max=50.0)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.integrate() is True
	n_iters = [row[5] for row in integrator.table_printer.rows]
	assert n_iters[0] == 1  # the uncapped-acceleration substep.
	assert 20 in n_iters  # the deceleration substep.


def test_integrate_transient_solver_failure_restores_state_and_recovers_on_retry(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A failed substep restores inelastic behaviors and constraints; the retry at :math:`dt/2` succeeds."""
	fake_file = _FakeXdmfFile()
	solver = _FakeSolver(outcomes=[(False, 5), (True, 2)] + [(True, 2)] * 10)
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver, write_sol=False)
	inelastic_behavior = _FakeInelasticBehavior()
	constraint = _FakeConstraint()
	step = _FakeStep(t_fin=5.0, dt_ini=1.0, dt_min=0.05, constraints=[constraint])
	problem = _FakeProblem([step], inelastic_behaviors_internal=[inelastic_behavior])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.integrate() is True
	assert inelastic_behavior.restored == 1
	assert constraint.restored == 1
	successes = [row[6] for row in integrator.table_printer.rows]
	assert successes[0] is False  # the transient failure was actually recorded before the retry.


def test_integrate_permanent_solver_failure_ends_the_step_unsuccessfully(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Failures until :math:`dt < dt_{min}`: ``integrate()`` returns ``False``; the last converged frame is kept.

	Here the initial state :math:`t = 0`, already in the file: the forced write does not duplicate it
	("Always kept" list of :mod:`hill_playground.integration.decimated_time_integrator`).
	"""
	fake_file = _FakeXdmfFile()
	solver = _FakeSolver(outcomes=[(False, 5)])
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver)
	step = _FakeStep(t_fin=5.0, dt_ini=1.0, dt_min=0.9)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=2)

	assert integrator.integrate() is False
	assert any("failed to move forward" in message for message in integrator.printer.messages)
	assert isinstance(integrator.xdmf_file_sol, _DecimatedXDMFFile)
	assert fake_file.written_times == []  # last converged frame = t = 0, already in the file: not rewritten.


def test_integrate_skips_solution_and_qoi_output_when_both_are_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
	"""``write_sol=False`` and ``write_qois=False``: nothing is written."""
	fake_file = _FakeXdmfFile()
	solver = _FakeSolver(outcomes=[(True, 2)])
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver, write_sol=False)
	step = _FakeStep(t_fin=1.0, dt_ini=1.0)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.integrate() is True
	assert fake_file.written_times == []  # never touched: write_sol was False throughout.


def test_integrate_deceleration_clamps_dt_to_dt_min_and_the_loop_keeps_going(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A deceleration below ``dt_min`` is clamped to ``dt_min`` and the run continues."""
	fake_file = _FakeXdmfFile()
	# n_iter=20 decelerates dt 1.0 -> 0.5 < dt_min=0.6: clamped; later n_iter=2 re-accelerate, capped by dt_max=1.0.
	solver = _FakeSolver(outcomes=[(True, 20)] + [(True, 2)] * 20)
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver, write_sol=False)
	step = _FakeStep(t_fin=5.0, dt_ini=1.0, dt_min=0.6)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.integrate() is True
	dts = [row[2] for row in integrator.table_printer.rows]
	assert dts[1] == 0.6  # the clamped-to-dt_min substep.
	assert len(dts) > 2  # the loop kept going past the clamped substep, not stopping right there.


def test_integrate_deceleration_without_hitting_dt_min_still_continues(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A deceleration staying above ``dt_min`` is not clamped and the run continues."""
	fake_file = _FakeXdmfFile()
	# n_iter=20 decelerates dt 2.0 -> 1.0 > dt_min=0.5: no clamp; later n_iter=2 re-accelerate, capped by dt_max=2.0.
	solver = _FakeSolver(outcomes=[(True, 20)] + [(True, 2)] * 20)
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver, write_sol=False)
	step = _FakeStep(t_fin=20.0, dt_ini=2.0, dt_min=0.5)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.integrate() is True
	dts = [row[2] for row in integrator.table_printer.rows]
	assert dts[1] == 1.0  # the decelerated-but-unclamped substep.
	assert len(dts) > 2  # the loop kept going past it, not stopping right there.


def test_integrate_n_iter_between_accel_and_decel_thresholds_leaves_dt_unchanged(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""``n_iter_for_accel < n_iter < n_iter_for_decel``: ``dt`` is unchanged."""
	fake_file = _FakeXdmfFile()
	# n_iter=10 sits strictly between the default n_iter_for_accel=4 and n_iter_for_decel=16, on
	# every single substep, so dt is never touched: the run simply advances at a constant dt_ini.
	solver = _FakeSolver(outcomes=[(True, 10)] * 20)
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver, write_sol=False)
	step = _FakeStep(t_fin=10.0, dt_ini=1.0)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.integrate() is True
	dts = [row[2] for row in integrator.table_printer.rows]
	assert dts == [1.0] * 10  # dt never changed: neither the accel nor the decel arm ever fired.


def test_integrate_multi_subsol_failure_branch_calls_dolfin_assign(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Failed substep with 2 subsols: the patched ``dolfin.assign`` is called pre-solve and on failure."""
	assign_calls: list[Any] = []

	def fake_assign(target: Any, source: Any) -> None:
		assign_calls.append((target, source))

	monkeypatch.setattr(dolfin, "assign", fake_assign)

	fake_file = _FakeXdmfFile()
	solver = _FakeSolver(outcomes=[(False, 5)])
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver)
	step = _FakeStep(t_fin=5.0, dt_ini=1.0, dt_min=0.9)
	problem = _FakeProblem([step], subsols=[object(), object()])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=2)

	assert integrator.integrate() is False
	assert len(assign_calls) == 2
	assert assign_calls[0][0] == problem.get_subsols_func_old_lst()
	assert assign_calls[0][1] is problem.sol_old_func
	assert assign_calls[1][0] == problem.get_subsols_func_lst()
	assert assign_calls[1][1] is problem.sol_func


def test_integrate_permanent_solver_failure_skips_forced_write_when_write_sol_is_false(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Solver failure with ``write_sol=False``: no forced write (``xdmf_file_sol`` never read)."""
	fake_file = _FakeXdmfFile()
	solver = _FakeSolver(outcomes=[(False, 5)])
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver, write_sol=False)
	step = _FakeStep(t_fin=5.0, dt_ini=1.0, dt_min=0.9)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.integrate() is False
	assert any("failed to move forward" in message for message in integrator.printer.messages)
	assert fake_file.written_times == []  # never touched: write_sol was False throughout.


def test_is_subsol_stagnant_true_when_error_below_threshold() -> None:
	"""A tiny relative change is judged stagnant against a generous min_relative_error."""
	assert is_subsol_stagnant(delta_norm=1e-8, scale_norm=1.0, min_relative_error=1e-6) is True


def test_is_subsol_stagnant_false_when_error_above_threshold() -> None:
	"""A substantial relative change is judged still evolving, not stagnant."""
	assert is_subsol_stagnant(delta_norm=0.01, scale_norm=1.0, min_relative_error=1e-6) is False


def test_is_subsol_stagnant_uses_the_default_threshold() -> None:
	r"""Default :math:`\varepsilon` = ``_DEFAULT_PLATEAU_RELATIVE_ERROR`` (1e-9): 1e-6 not stagnant, 1e-10 stagnant."""
	assert is_subsol_stagnant(delta_norm=1e-6, scale_norm=1.0) is False
	assert is_subsol_stagnant(delta_norm=1e-10, scale_norm=1.0) is True


def test_is_subsol_stagnant_zero_scale() -> None:
	"""Zero scale (subsol identically 0): stagnant if unchanged, not stagnant if it moves (``compute_error``)."""
	assert is_subsol_stagnant(delta_norm=0.0, scale_norm=0.0) is True
	assert is_subsol_stagnant(delta_norm=1e-3, scale_norm=0.0, min_relative_error=1e-2) is False


def test_is_subsol_stagnant_rejects_non_positive_min_relative_error() -> None:
	"""``min_relative_error <= 0`` raises ``ValueError``."""
	with pytest.raises(ValueError):
		is_subsol_stagnant(delta_norm=1.0, scale_norm=1.0, min_relative_error=0.0)
	with pytest.raises(ValueError):
		is_subsol_stagnant(delta_norm=1.0, scale_norm=1.0, min_relative_error=-1e-6)
