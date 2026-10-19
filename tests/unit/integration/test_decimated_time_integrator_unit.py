"""Unit tests for the periodic-write XDMF throttling helpers and the plateau early-exit helpers.

Copy of fenicsoft ``tests/unit/io/test_decimated_time_integrator_unit.py`` (only the import changed),
run against the playground copy of the module; the two playground changes are tested in
``test_decimated_time_integrator.py``.

What is tested here is the pure decimation logic (:func:`should_write_at_call`), the
:class:`TimeInterval` value type, the proxy that applies either kind of period
(:class:`_DecimatedXDMFFile`, uniform or per-step, ints and :class:`TimeInterval` freely mixed,
plus the "always keep a step's own last substep" guarantee and the ``force`` override),
:class:`DecimatedTimeIntegrator.__init__`'s own write_period-to-proxy branching logic, the two
dolfin-free pure functions behind the plateau early-exit feature (:func:`_relative_error`,
:func:`is_subsol_stagnant`), and -- via a richer fake harness -- :meth:`DecimatedTimeIntegrator.integrate`
itself (the full loop it copies verbatim from ``dolfin_mech.core.TimeIntegrator.integrate``, plus
the plateau-exit lines added on top of it).

``DecimatedTimeIntegrator`` itself subclasses ``dolfin_mech.core.TimeIntegrator``, whose own real
``__init__`` needs a genuine ``Problem``/``NonlinearSolver``/mesh -- out of scope here. Rather than
skipping ``DecimatedTimeIntegrator.__init__`` entirely, the tests below monkeypatch
``dolfin_mech.core.TimeIntegrator.__init__`` itself, either with a trivial stub
(:func:`_patch_base_time_integrator_init`) that sets only the three attributes
``DecimatedTimeIntegrator.__init__`` itself reads (``problem``, ``write_sol``, ``xdmf_file_sol``) --
enough to exercise every branch of its own logic (scalar vs. per-step ``write_period``, ints and
:class:`TimeInterval` freely mixed, every ``ValueError``/``TypeError`` case, the ``write_sol``-falsy
early return, and ``stagnation_plateau_relative_error`` validation) -- or with a fuller stub
(:func:`_patch_full_time_integrator_init`) that additionally sets every attribute the *inherited*
``integrate()`` body reads (``solver``, ``printer``, ``table_printer``, the accel/decel
parameters, ...), letting that entire loop run end to end against plain-Python fakes (a scripted
``_FakeSolver`` standing in for real Newton convergence, ``_FakeInelasticBehavior``/
``_FakeConstraint`` recording their own calls). Both stubs need no FEniCS/PETSc dependency at all,
and are restored automatically at the end of each test.

"""

from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import dolfin
import dolfin_mech
import pytest

from hill_playground.integration.decimated_time_integrator import (
	_RELATIVE_ERROR_ZERO_EPS,  # type: ignore
	DecimatedTimeIntegrator,
	TimeInterval,
	_DecimatedXDMFFile,  # type: ignore
	_relative_error,  # type: ignore
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
	"""Minimal stand-in for ``dolfin_mech.core.step.Step``.

	The :class:`_DecimatedXDMFFile`-focused tests only ever read ``t_fin``; the
	:meth:`DecimatedTimeIntegrator.integrate`-focused tests further down also read every other
	attribute a real ``dolfin_mech.core.step.Step`` (or ``fenicsoft.io.Step``, for
	``min_simu_seconds``) exposes, all defaulted here so existing ``_FakeStep(t_fin=...)`` call
	sites keep working unchanged.
	"""

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
		"""Register this step's own timing/adaptivity bounds and (optionally) operators/constraints.

		Args:
		    t_fin (float): This step's own final simulated time.
		    t_ini (float): This step's own starting simulated time. Defaults to ``0.0``.
		    dt_ini (float): This step's own nominal/initial substep size. Defaults to ``1.0``.
		    dt_min (float): Floor the adaptive ``dt`` is never allowed to shrink below. Defaults to
		        a small positive value.
		    dt_max (Optional[float]): Ceiling the adaptive ``dt`` is never allowed to grow past --
		        defaults to ``dt_ini`` (matching ``fenicsoft.io.Step``'s own default: no
		        acceleration headroom unless a test explicitly asks for it).
		    operators (Optional[list]): This step's own time-dependent operators. Defaults to an
		        empty list.
		    constraints (Optional[list]): This step's own constraints. Defaults to an empty list.
		    min_simu_seconds (Optional[float]): See ``fenicsoft.io.Step``'s own docstring. Defaults
		        to ``None`` (the plateau-exit feature off for this step), matching every existing
		        test here, none of which exercises the stagnation check through a full
		        ``integrate()`` run (see this module's own docstring).
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
	"""Minimal stand-in for a PETSc vector: a plain mutable sequence supporting slice assignment.

	``self.problem.sol_old_func.vector()[:] = self.problem.sol_func.vector()[:]`` (inside
	``DecimatedTimeIntegrator.integrate``, copied verbatim from ``dolfin_mech.core.TimeIntegrator``)
	only ever needs slice read/assignment on whatever ``.vector()`` returns -- a plain ``list``
	already supports both natively. Subclassing it (rather than handing out a bare ``list``) is
	what lets ``.vector()`` return the *same* object identity on every call, matching a real PETSc
	vector's own mutate-in-place semantics -- a fresh ``list()`` copy on every call would make the
	assignment above a silent no-op.
	"""


class _FakeFunc:
	"""Minimal stand-in for a ``dolfin.Function``: exposes only ``.vector()``."""

	def __init__(self, values: list[Any]) -> None:
		"""Wrap ``values`` in a single, stable :class:`_FakeVector` returned by every ``.vector()`` call."""
		self._vector = _FakeVector(values)

	def vector(self) -> "_FakeVector":
		"""Return this function's own vector (same object every call, so mutations persist)."""
		return self._vector


class _FakeInelasticBehavior:
	"""Minimal stand-in for an internal inelastic behavior: records its own calls, changes nothing."""

	def __init__(self) -> None:
		"""Start with empty call logs."""
		self.updated_at: list[Any] = []
		self.restored = 0

	def update_internal_variables_at_t(self, t: float) -> None:
		"""Record the simulated time this call was made with."""
		self.updated_at.append(t)

	def restore_old_value(self) -> None:
		"""Record that this behavior's old value was restored (a failed substep being retried)."""
		self.restored += 1


class _FakeConstraint:
	"""Minimal stand-in for a step constraint: records its own calls, changes nothing."""

	def __init__(self) -> None:
		"""Start with empty call logs."""
		self.set_at: list[Any] = []
		self.restored = 0

	def set_value_at_t_step(self, t_step: float) -> None:
		"""Record the fractional step-progress this call was made with."""
		self.set_at.append(t_step)

	def restore_old_value(self) -> None:
		"""Record that this constraint's old value was restored (a failed substep being retried)."""
		self.restored += 1


class _FakeSolver:
	"""Minimal stand-in for ``dolfin_mech.core.NonlinearSolver``.

	``.solve`` returns a pre-scripted sequence of ``(success, n_iter)`` outcomes, one per call --
	enough to script an exact convergence history (a transient failure that recovers on retry, a
	step that never converges, an iteration count that triggers acceleration or deceleration)
	without ever running a real Newton solve.
	"""

	def __init__(self, outcomes: Sequence[Tuple[bool, int]]) -> None:
		"""Register the scripted outcome sequence.

		Args:
		    outcomes (Sequence[Tuple[bool, int]]): One ``(solver_success, n_iter)`` pair per
		        ``.solve`` call, consumed in order; the last entry then repeats forever once the
		        script is exhausted, so a test does not need to predict exactly how many substeps a
		        natural completion takes.
		"""
		self._outcomes = list(outcomes)
		self._n_calls = 0
		self.constraints: list[Any] = []

	def solve(self, k_step: int, k_t: int, dt: float, t: float) -> Tuple[bool, int]:
		"""Return this call's own scripted outcome (or the last one, once the script is exhausted)."""
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
		"""Increase the nesting depth (mirrors the real printer's own indentation bookkeeping)."""
		self.depth += 1

	def dec(self) -> None:
		"""Decrease the nesting depth."""
		self.depth -= 1

	def print_var(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
		"""No-op: the tests using this fake assert on outcomes, not on the per-substep log text."""

	def print_str(self, text: str) -> None:
		"""Record a plain status message (e.g. the plateau-exit / solver-failure notices)."""
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
	"""Minimal stand-in exposing every attribute :meth:`DecimatedTimeIntegrator.integrate` reads on ``self.problem``.

	``steps`` is the fixed list :class:`_DecimatedXDMFFile` indexes into. ``subsols`` is read by
	:class:`DecimatedTimeIntegrator.__init__`'s own per-subsol ``stagnation_plateau_relative_error`` length
	check, and by ``integrate()``'s own ``len(self.problem.subsols) > 1`` guard around
	``dolfin.assign``. Every ``integrate()``-focused test below keeps this at 0 or 1 entries --
	so that guard's own False arm is exercised -- *except*
	:func:`test_integrate_multi_subsol_failure_branch_calls_dolfin_assign`, which sets it to 2 and
	monkeypatches ``dolfin.assign`` itself (real-dolfin-only otherwise) to exercise that guard's
	True arm on both the unconditional pre-solve call and the failure-branch call in a single
	attempted substep (that test's own docstring). The same True arm on a *successfully converged*
	substep is additionally covered by the real-FEniCS integration test in
	``tests/integration/analytical_consolidation_1D/plateau_exit/``, whose own problem registers
	more than one subsol. Elsewhere, placeholder entries (e.g. plain
	``object()``) are enough wherever a non-zero length is all that is needed (see this module's
	own docstring for why the loop inside :meth:`DecimatedTimeIntegrator._step_is_stagnant` itself
	stays out of scope beyond one branch).
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
		"""Register every fixed collaborator ``integrate()`` reads, all defaulted to empty/trivial.

		Args:
		    steps (list): The fixed, ordered list of step objects.
		    subsols (Optional[list]): The fixed list of subsol placeholders, defaulting to an empty
		        list.
		    constraints (Optional[list]): This problem's own (step-independent) constraints.
		        Defaults to an empty list.
		    inelastic_behaviors_internal (Optional[list]): Internal inelastic behaviors updated once
		        per converged substep and restored once per failed one. Defaults to an empty list.
		    qois (Optional[list]): Quantities of interest. Defaults to an empty list -- fine for
		        every test here, all of which set ``write_qois=False``.
		    sol_func (Optional[_FakeFunc]): The problem's own current-solution stand-in. Defaults to
		        a fresh, trivial :class:`_FakeFunc`.
		    sol_old_func (Optional[_FakeFunc]): The problem's own previous-solution stand-in.
		        Defaults to a fresh, trivial :class:`_FakeFunc`.
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
		"""No-op: this fake problem has no real variational formulation to (re)build per step."""

	def update_fois(self) -> None:
		"""No-op: this fake problem tracks no fields-of-interest."""

	def update_qois(self, dt: float, k_step: int) -> None:
		"""No-op: never called by any test here (all set ``write_qois=False``); kept for interface parity."""

	def get_subsols_func_old_lst(self) -> list[Any]:
		"""Called only by :func:`test_integrate_multi_subsol_failure_branch_calls_dolfin_assign`.

		Every other test here keeps ``len(subsols) <= 1``, so the branch that would call this is
		never reached for them.
		"""
		return list(self.subsols)

	def get_subsols_func_lst(self) -> list[Any]:
		"""See :meth:`get_subsols_func_old_lst` -- read by the failure-branch counterpart instead."""
		return list(self.subsols)


class _FakeTimeIntegrator:
	"""Minimal stand-in for the owning ``TimeIntegrator``.

	Exposes exactly what :class:`_DecimatedXDMFFile` reads: ``problem.steps`` (the fixed, ordered
	list of macro-step objects, indexed by identity) and ``step`` (whichever one is "current"),
	the latter settable between calls to simulate
	``dolfin_mech.core.TimeIntegrator.integrate``'s own step transitions.
	"""

	def __init__(self, steps: list[Any]) -> None:
		"""Register the fixed step list and start with the first one active."""
		self.problem = _FakeProblem(steps)
		self.step = steps[0]


def test_should_write_at_call_always_keeps_the_first_call() -> None:
	"""Call index 1 is always kept, regardless of period."""
	assert should_write_at_call(call_index=1, period=5) is True
	assert should_write_at_call(call_index=1, period=1) is True


def test_should_write_at_call_period_one_keeps_every_call() -> None:
	"""period=1 reproduces the base class's own behavior exactly: every call is kept."""
	assert [should_write_at_call(call_index=i, period=1) for i in range(1, 8)] == [True] * 7


def test_should_write_at_call_decimates_by_period() -> None:
	"""period=3 keeps calls 1, 4, 7, 10, ... -- one call index out of every three, plus the first."""
	kept = [i for i in range(1, 13) if should_write_at_call(call_index=i, period=3)]
	assert kept == [1, 4, 7, 10]


def test_should_write_at_call_rejects_non_positive_period() -> None:
	"""A period below 1 is meaningless (nothing would ever repeat) and must raise."""
	with pytest.raises(ValueError):
		should_write_at_call(call_index=1, period=0)


def test_time_interval_rejects_non_positive_seconds() -> None:
	"""seconds=0 or seconds<0 is meaningless (every call would qualify as "elapsed enough")."""
	with pytest.raises(ValueError):
		TimeInterval(seconds=0.0)
	with pytest.raises(ValueError):
		TimeInterval(seconds=-5.0)


def test_time_interval_accepts_strictly_positive_seconds() -> None:
	"""A strictly positive value is stored unchanged."""
	assert TimeInterval(seconds=900.0).seconds == 900.0


def test_periodic_xdmf_file_forwards_first_and_every_period_write() -> None:
	"""Keeps the first write and every period-th one after, on a single-step fake run.

	``t_fin`` is set far beyond every ``t`` used here, so the "always keep the step's own last
	substep" guarantee never fires -- this test isolates the pure periodic behavior.
	"""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=3, time_integrator=fake_integrator)

	for t in range(1, 11):  # 10 calls: t=1..10, mimicking 10 converged substeps.
		proxy.write(float(t))

	assert fake_file.written_times == [1.0, 4.0, 7.0, 10.0]


def test_periodic_xdmf_file_period_one_writes_every_call() -> None:
	"""period=1 forwards every call unchanged -- the no-throttling default."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=1, time_integrator=fake_integrator)

	for t in range(1, 4):
		proxy.write(float(t))

	assert fake_file.written_times == [1.0, 2.0, 3.0]


def test_periodic_xdmf_file_close_delegates_to_wrapped_file() -> None:
	"""close() closes the underlying (real) file, not just the proxy."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=5, time_integrator=fake_integrator)

	proxy.close()

	assert fake_file.closed is True


def test_periodic_xdmf_file_always_writes_the_last_substep_of_a_step_even_if_the_period_would_skip_it() -> None:
	"""A step's own last converged substep is always kept, even when the period alone would drop it.

	8 substeps at period=3 would naturally keep only call indices 1, 4, 7 -- call index 8 (the
	step's own last substep, t == t_fin) would otherwise be silently dropped.
	"""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=8.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=3, time_integrator=fake_integrator)

	for t in range(1, 9):  # 8 calls: t=1..8, t=8.0 == t_fin.
		proxy.write(float(t))

	assert fake_file.written_times == [1.0, 4.0, 7.0, 8.0]


def test_periodic_xdmf_file_uses_current_steps_own_period_and_forces_each_steps_own_last_substep() -> None:
	"""Per-step periods apply correctly across a step boundary, each step's own last substep kept.

	Mimics this project's own ramp-then-hold demo pattern: a short step kept in full (period=1)
	followed by a longer step decimated by period=5, whose own last substep (call index 12, not
	naturally kept at period=5) must still be forced through.
	"""
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
	"""Switching steps resets the decimation counter.

	Without the reset, a short step following a long one could inherit an arbitrary call-count
	phase and lose its own very first substep to the previous step's decimation cycle.
	"""
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
	"""A per-step sequence not matching ``problem.steps`` in length is a caller error, not silently truncated/padded."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	with pytest.raises(IndexError):
		proxy = _DecimatedXDMFFile(fake_file, write_period=[3], time_integrator=fake_integrator)
		fake_integrator.step = fake_integrator.problem.steps[1]
		proxy.write(2.0)


def test_periodic_xdmf_file_time_interval_decimates_by_elapsed_simulated_time_under_uneven_dt() -> None:
	"""A :class:`TimeInterval` period tracks elapsed *simulated* time, not call count.

	Proven here with a genuinely uneven sequence of ``dt`` values (30, 60, 110, 150, 30, 80, 40),
	mimicking a step under acceleration/deceleration where a substep-count period would be
	meaningless. Kept whenever at least 100 s have elapsed since the last kept frame, plus the
	step's own last substep (t=500, forced through even though only 40 s elapsed since the
	previous kept frame).
	"""
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
	"""Ints and :class:`TimeInterval` freely mix in a per-step sequence, each with its own state.

	Step A (period=1, substep-count-based) keeps every call. Step B
	(period=``TimeInterval(50.0)``, simulated-time-based) decimates by elapsed time, resetting its
	own "last written" state fresh at the step boundary rather than inheriting anything from A.
	"""
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
	"""``force=True`` forwards the write unconditionally, regardless of the period or step end.

	``t_fin`` is set far beyond every ``t`` used here (so the "step's own last substep" guarantee
	never fires on its own) and ``period=5`` would naturally skip call index 2 -- ``force=True``
	is what :meth:`DecimatedTimeIntegrator.integrate` now relies on to save a plateau exit's own
	final frame (module docstring, "Early exit on a plateaued step").
	"""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=5, time_integrator=fake_integrator)

	proxy.write(1.0)  # call index 1: kept anyway (always-keep-first-call rule).
	proxy.write(2.0, force=True)  # call index 2: period alone would skip this -- forced through.
	proxy.write(3.0)  # call index 3: period alone skips this, and it is not forced -- dropped.

	assert fake_file.written_times == [1.0, 2.0]


def test_periodic_xdmf_file_force_defaults_to_false_preserving_prior_signature() -> None:
	"""Every pre-existing call site (none of which passes ``force``) keeps its exact prior behavior."""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=5, time_integrator=fake_integrator)

	proxy.write(1.0)
	proxy.write(2.0)

	assert fake_file.written_times == [1.0]


def test_periodic_xdmf_file_force_at_an_already_written_t_is_a_harmless_no_op() -> None:
	"""A ``force=True`` call at the same t as the last actual write never duplicates that frame.

	This is what lets :meth:`DecimatedTimeIntegrator.integrate` call ``write(t, force=True)``
	defensively on a hard-failure exit (module docstring, "Guaranteed last frame on a hard
	failure") without risking two entries at the same simulated time in the XDMF/HDF5 series, even
	when that exact substep had, in fact, already been written through the ordinary period rule.
	"""
	fake_file = _FakeXdmfFile()
	fake_integrator = _FakeTimeIntegrator([_FakeStep(t_fin=1000.0)])
	proxy = _DecimatedXDMFFile(fake_file, write_period=1, time_integrator=fake_integrator)

	proxy.write(5.0)  # ordinary write (period=1 keeps every call).
	proxy.write(5.0, force=True)  # same t again, forced -- must not duplicate.

	assert fake_file.written_times == [5.0]


def test_periodic_xdmf_file_force_at_a_new_t_after_an_already_written_one_still_writes() -> None:
	"""The dedup guard only ever suppresses an exact repeat -- it never swallows a genuine new t."""
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
	"""Replace ``dolfin_mech.core.TimeIntegrator.__init__`` with a trivial stub, for this test only.

	See this module's own docstring for why: it lets :class:`DecimatedTimeIntegrator`'s own
	``__init__`` branching logic run for real, against a real ``dolfin_mech.core.TimeIntegrator``
	subclass, with no FEniCS/PETSc dependency. ``monkeypatch`` restores the real ``__init__``
	automatically once the test using it returns.

	Args:
	    monkeypatch (pytest.MonkeyPatch): The test's own fixture, doing the actual patch/restore.
	    xdmf_file (_FakeXdmfFile): The fake writer ``self.xdmf_file_sol`` is set to -- kept by the
	        caller so it can assert on ``written_times``/``closed`` even after
	        :class:`DecimatedTimeIntegrator` wraps it in a :class:`_DecimatedXDMFFile` proxy.
	    write_sol (bool): Value to set ``self.write_sol`` to (True unless a test needs to exercise
	        the "no XDMF output requested at all" early return).
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
	"""Replace ``dolfin_mech.core.TimeIntegrator.__init__`` with a fuller stub, for ``integrate()`` tests only.

	Unlike :func:`_patch_base_time_integrator_init` (which only sets the three attributes
	``DecimatedTimeIntegrator.__init__`` itself reads), this sets every attribute the *inherited*
	``integrate()`` body (copied verbatim from ``dolfin_mech.core.TimeIntegrator.integrate`` except
	for the lines this project's own module docstring marks ``-- new --``) reads from ``self`` --
	enough to run that full loop end to end against plain-Python fakes, with no FEniCS/PETSc
	dependency, for every branch that does not itself require a real ``dolfin.Function`` (see this
	module's own docstring for the two branches that still do, and are covered elsewhere instead).
	``write_qois`` is always left ``False`` here: none of the branches this stub exists to cover
	need real QOI output, and lines 727-728 (the ``write_qois=True`` body itself) are already
	covered by the real integration test in
	``tests/integration/analytical_consolidation_1D/plateau_exit/``.

	Args:
	    monkeypatch (pytest.MonkeyPatch): The test's own fixture, doing the actual patch/restore.
	    xdmf_file (_FakeXdmfFile): The fake writer ``self.xdmf_file_sol`` is set to when
	        ``write_sol`` is True -- kept by the caller so it can assert on
	        ``written_times``/``closed`` even after :class:`DecimatedTimeIntegrator` wraps it in a
	        :class:`_DecimatedXDMFFile` proxy.
	    solver (_FakeSolver): The scripted fake solver ``self.solver`` is set to.
	    parameters (Optional[Dict[str, Any]]): Forwarded to derive
	        ``n_iter_for_accel``/``n_iter_for_decel``/``accel_coeff``/``decel_coeff``, matching
	        ``dolfin_mech.core.TimeIntegrator.__init__``'s own ``parameters.get(..., default)``
	        convention exactly. Defaults to an empty dict (every default applies).
	    write_sol (bool): Value to set ``self.write_sol`` to. When True, ``write_vtus``/
	        ``write_xmls`` are always left False here (see this module's own docstring).
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
	"""write_period=1 (the default) leaves the base class's own writer completely untouched."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=2.0)])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.xdmf_file_sol is fake_file


def test_decimated_time_integrator_scalar_period_above_one_throttles_writes(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A uniform int write_period above 1 wraps the writer and throttles substep writes accordingly."""
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
	"""A uniform :class:`TimeInterval` write_period wraps the writer and decimates by elapsed time.

	Unlike a scalar int, this always wraps regardless of the interval's value, since even a very
	short interval must still track elapsed time rather than call count.
	"""
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
	"""A per-step write_period sequence reproduces this project's own ramp/plateau demo pattern.

	Also confirms the plateau's own last converged substep is forced through even though it does
	not land on a period=5 boundary -- the same guarantee :class:`_DecimatedXDMFFile` provides,
	now exercised through the real :class:`DecimatedTimeIntegrator` construction path.
	"""
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
	"""A per-step sequence may mix a substep-count ``int`` with a :class:`TimeInterval`.

	Typical use: a ramp (constant ``dt``, kept in full by substep count) followed by an accel/decel
	hold (kept by elapsed simulated time).
	"""
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
	"""A per-step sequence with every entry equal to 1 needs no throttling, so the writer is left unwrapped."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=[1, 1])

	assert integrator.xdmf_file_sol is fake_file


def test_decimated_time_integrator_rejects_mismatched_per_step_length(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A per-step sequence not matching len(problem.steps) is rejected with a clear ValueError."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	with pytest.raises(ValueError, match="one entry per"):
		DecimatedTimeIntegrator(problem=problem, write_period=[1, 2, 3])


def test_decimated_time_integrator_rejects_non_positive_per_step_entry(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A per-step sequence containing a non-positive integer entry is rejected with a clear ValueError."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	with pytest.raises(ValueError, match="positive integer"):
		DecimatedTimeIntegrator(problem=problem, write_period=[1, 0])


def test_decimated_time_integrator_rejects_invalid_per_step_entry_type(monkeypatch: pytest.MonkeyPatch) -> None:
	"""A per-step sequence entry that is neither an int nor a TimeInterval is rejected with a clear TypeError."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0), _FakeStep(t_fin=2.0)])

	with pytest.raises(TypeError, match="int or a TimeInterval"):
		DecimatedTimeIntegrator(problem=problem, write_period=[1, "every 100 seconds"])  # type: ignore


def test_decimated_time_integrator_skips_wrapping_when_write_sol_is_falsy(monkeypatch: pytest.MonkeyPatch) -> None:
	"""No XDMF field-solution output at all (write_sol falsy) means nothing to throttle."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file, write_sol=False)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)])

	integrator = DecimatedTimeIntegrator(problem=problem, write_period=50)

	assert integrator.xdmf_file_sol is fake_file


def test_decimated_time_integrator_rejects_non_positive_stagnation_plateau_relative_error(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""stagnation_plateau_relative_error must be strictly positive (never zero or negative)."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)])

	with pytest.raises(ValueError, match="stagnation_plateau_relative_error"):
		DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=0.0)


def test_decimated_time_integrator_default_stagnation_plateau_relative_error_is_the_near_zero_epsilon(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""The default stagnation_plateau_relative_error is _RELATIVE_ERROR_ZERO_EPS (1e-9), not an arbitrary value.

	Unconfigured, a subsol should only ever be judged stagnant once it has effectively stopped
	changing at all -- the same near-zero epsilon :func:`_relative_error` itself already uses for
	its own zero-reference guard -- rather than at some arbitrary looser default (see
	:func:`is_subsol_stagnant`'s own docstring).
	"""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.stagnation_plateau_relative_error == _RELATIVE_ERROR_ZERO_EPS


def test_decimated_time_integrator_accepts_per_subsol_stagnation_plateau_relative_error_sequence(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A per-subsol stagnation_plateau_relative_error sequence, one entry per problem.subsols, is stored as given.

	A subsol whose relative rate of change settles at a constant, nonzero value (e.g. a field
	decaying purely exponentially -- see is_subsol_stagnant's own docstring) can only be caught by a
	larger stagnation_plateau_relative_error; sol_tol is never consulted by the threshold itself.
	"""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)], subsols=[object(), object()])

	integrator = DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=[3.0, 150000.0])

	assert integrator.stagnation_plateau_relative_error == [3.0, 150000.0]


def test_decimated_time_integrator_rejects_mismatched_stagnation_plateau_relative_error_length(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A mismatched-length stagnation_plateau_relative_error sequence is rejected with a ValueError."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)], subsols=[object(), object()])

	with pytest.raises(ValueError, match="one entry per registered subsol"):
		DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=[3.0, 150000.0, 1.0])


def test_decimated_time_integrator_rejects_non_positive_stagnation_plateau_relative_error_entry(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A stagnation_plateau_relative_error entry that is non-positive is rejected with a ValueError."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)], subsols=[object(), object()])

	with pytest.raises(ValueError, match="strictly positive"):
		DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=[3.0, 0.0])


def test_decimated_time_integrator_rejects_invalid_stagnation_plateau_relative_error_entry_type(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A stagnation_plateau_relative_error entry that is not an int or float is rejected with a TypeError."""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	problem = _FakeProblem([_FakeStep(t_fin=1.0)], subsols=[object(), object()])

	with pytest.raises(TypeError, match="int or float"):
		DecimatedTimeIntegrator(problem=problem, stagnation_plateau_relative_error=[3.0, "loose"])  # type: ignore


def test_step_is_stagnant_treats_every_subsol_unmonitored_as_trivially_stagnant(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A subsol whose own sol_tol is None is skipped entirely (continue), never touching its vector.

	Covers the one branch of :meth:`DecimatedTimeIntegrator._step_is_stagnant` this module's own
	dolfin-free unit tests can safely exercise directly (see this module's own docstring): when
	*every* registered subsol has ``sol_tol=None``, the loop hits ``continue`` on each one without
	ever reading ``subsol.func.vector()`` -- so plain placeholder objects (never actually touched)
	are enough, with no real dolfin Function needed. ``local_stagnant`` is never set to 0 in that
	case, so the step is (trivially) reported as stagnant.
	"""
	fake_file = _FakeXdmfFile()
	_patch_base_time_integrator_init(monkeypatch, fake_file)
	step = _FakeStep(t_fin=1.0)
	problem = _FakeProblem([step], subsols=[object(), object()])

	integrator = DecimatedTimeIntegrator(problem=problem)
	integrator.step = step
	integrator.solver = SimpleNamespace(sol_tol=[None, None])

	assert integrator._step_is_stagnant() is True  # type: ignore


def test_integrate_zero_subsol_step_runs_inelastic_behaviors_and_wraps_the_writer(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""A successful multi-substep run with 0 subsols, an inelastic behavior, and a throttled writer.

	Exercises three spots of :meth:`DecimatedTimeIntegrator.integrate` at once,
	all naturally part of the same "ordinary successful run" scenario: the inelastic-behavior
	update call (the real integration test's own problem registers none), the False arm of
	``len(self.problem.subsols) > 1`` (0 subsols here, so ``dolfin.assign`` -- real-dolfin-only --
	is never reached), and the throttled-write branch (``isinstance(self.xdmf_file_sol,
	_DecimatedXDMFFile)``, only True when ``write_period`` actually wraps the writer -- the real
	integration test never passes ``write_period`` at all).
	"""
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
	"""Dt genuinely grows without an immediate dt_max clamp, and a high n_iter genuinely decelerates dt.

	The real integration test's own step never sets ``max_dt`` (so ``dt_max == dt_ini`` there,
	meaning every acceleration is clamped back immediately -- the ``dt > dt_max`` check is always
	True) and never sees ``n_iter`` reach ``n_iter_for_decel`` (its own Newton solves converge in
	2-6 iterations throughout). Here ``dt_max`` has real headroom above ``dt_ini`` and the scripted
	solver returns a high ``n_iter`` once, so both the "acceleration not yet clamped" arm and the
	deceleration branch actually execute, on top of the arms the real test already covers.
	"""
	fake_file = _FakeXdmfFile()
	# n_iter=1 (<= n_iter_for_accel) accelerates dt to 2.0, still below dt_max=50.0 -- the
	# "dt > dt_max is False" arm. n_iter=20 (>= n_iter_for_decel) then
	# decelerates dt to 2.0/10.0=0.2, below dt_min=0.05 -- clamped up to dt_min, covering the
	# deceleration branch's own clamp line too. Every remaining call converges in few iterations,
	# so dt re-accelerates and the run reaches t_fin naturally well within this many substeps.
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
	"""A failed substep restores the previous state (inelastic behaviors, constraints) and retries successfully.

	The real integration test's own run never fails a single substep (every one of its 300
	converges), so ``DecimatedTimeIntegrator.integrate``'s entire "restore and cut dt" branch --
	inherited verbatim from ``dolfin_mech.core.TimeIntegrator.integrate`` -- is not exercised there.
	Here the scripted solver fails the very first call, then succeeds on the retry (at half the
	original dt, comfortably above dt_min so this is a transient failure, not a permanent one --
	see :func:`test_integrate_permanent_solver_failure_ends_the_step_unsuccessfully` for that case).
	"""
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
	"""Every retry fails until dt drops below dt_min: the step gives up, but forces a final write first.

	Covers the permanent-failure tail of the same branch as
	:func:`test_integrate_transient_solver_failure_restores_state_and_recovers_on_retry` -- also
	never reached by the real integration test, which never fails a substep at all. ``dt_min`` is
	set close to ``dt_ini`` so the very first failed retry already drops below it, matching module
	docstring, "Guaranteed last frame on a hard failure": the last successfully converged state
	(here, the initial one) is still forced through the (throttled) writer even though the step
	itself ends in failure.
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
	assert fake_file.written_times == [0.0]  # the guaranteed-last-frame forced write.


def test_integrate_skips_solution_and_qoi_output_when_both_are_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
	"""write_sol=False and write_qois=False skip their entire output blocks, not just their bodies.

	The real integration test always passes both truthy, so neither the ``write_sol`` False arm
	(everything from the throttled-write branch through the VTU/XML writers) nor the ``write_qois``
	False arm has ever executed.
	"""
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
	"""A deceleration that would push dt below dt_min is clamped up to dt_min, and the run continues.

	:func:`test_integrate_covers_uncapped_acceleration_and_deceleration_arcs` already exercises the
	deceleration branch itself, but its own scripted ``dt_min`` is low enough that the decelerated
	``dt`` never actually drops below it -- so ``dt = self.step.dt_min`` (the clamp line itself) has
	never executed. Here ``dt_min`` is set just above the post-deceleration ``dt``, forcing the
	clamp; ``t_fin`` is set well beyond what the clamped substep alone reaches, so the loop also
	genuinely goes around again afterwards (rather than the clamped substep happening to be the
	step's own last one). See
	:func:`test_integrate_deceleration_without_hitting_dt_min_still_continues` for the sibling case
	where the decelerated ``dt`` does *not* need clamping.
	"""
	fake_file = _FakeXdmfFile()
	# n_iter=20 (>= the default n_iter_for_decel=16) decelerates dt from 1.0 to 1.0/2=0.5, below
	# dt_min=0.6 -- clamped up to 0.6. Every later call converges in few iterations (n_iter=2 <=
	# the default n_iter_for_accel=4), re-accelerating dt, but dt_max=1.0 (== dt_ini, the _FakeStep
	# default) clamps it straight back down each time, keeping the remaining substeps predictable.
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
	"""A deceleration that stays above dt_min needs no clamp, and the run still continues past it.

	Sibling of :func:`test_integrate_deceleration_clamps_dt_to_dt_min_and_the_loop_keeps_going`: that
	test's own decelerated ``dt`` always falls below ``dt_min`` (forcing the clamp on every
	deceleration it scripts), so the *other* way through the same branch -- decelerate, land at or
	above ``dt_min`` (no clamp needed), and still loop around to another substep -- has never
	executed on its own. Here ``dt_min`` is set comfortably below the post-deceleration ``dt``, and
	``t_fin`` again well beyond what that one substep alone reaches.
	"""
	fake_file = _FakeXdmfFile()
	# n_iter=20 (>= the default n_iter_for_decel=16) decelerates dt from 2.0 to 2.0/2=1.0, still
	# above dt_min=0.5 -- no clamp needed. Every later call converges in few iterations (n_iter=2 <=
	# the default n_iter_for_accel=4), re-accelerating dt back up, but dt_max=2.0 (== dt_ini, the
	# _FakeStep default) clamps it straight back down each time, keeping the remaining substeps
	# predictable.
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
	"""An n_iter strictly between the accel and decel thresholds triggers neither arm: dt is untouched.

	``if n_iter <= n_iter_for_accel: ... elif n_iter >= n_iter_for_decel: ...`` has no ``else``
	clause: previous tests here each script an ``n_iter`` at or below ``n_iter_for_accel`` or at or
	above ``n_iter_for_decel`` (accelerating or decelerating every single substep), so the
	fall-through for an ``n_iter`` strictly *between* the two -- dt carried over completely unchanged
	to the next substep, and the loop still continuing afterwards -- is not exercised there. This is
	distinct from either deceleration test just above: those both take the ``elif`` branch (with or
	without its own inner clamp firing); this one takes neither branch at all.
	"""
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
	"""A failed substep with more than one subsol forwards through the real ``dolfin.assign`` call.

	Every other ``integrate()``-focused test here keeps ``len(problem.subsols) <= 1`` specifically to
	avoid this (real-dolfin-only) call (see :class:`_FakeProblem`'s own docstring); here it is
	exercised deliberately by monkeypatching ``dolfin.assign`` itself with a plain call-recording
	stand-in. Reuses the exact permanent-failure setup of
	:func:`test_integrate_permanent_solver_failure_ends_the_step_unsuccessfully`, the simplest scripted
	history that reaches the failure branch at all -- a single attempted substep, which exercises
	*both* ``dolfin.assign`` call sites at once: the unconditional pre-solve one (``sol_old_func``,
	always reached regardless of whether that substep goes on to succeed or fail) and the
	failure-branch one (``sol_func``, this test's own actual target) right after it fails.
	"""
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
	"""The hard-failure forced write (module docstring) is itself gated on ``write_sol``, not unconditional.

	:func:`test_integrate_permanent_solver_failure_ends_the_step_unsuccessfully` always passes
	``write_sol=True``, so ``if self.write_sol and isinstance(...)`` has only ever been exercised
	with its first operand True; with ``write_sol=False`` here, short-circuit evaluation skips the
	``isinstance`` check (and therefore the forced ``write`` call) entirely, without ever touching
	``self.xdmf_file_sol`` -- which this stub does not even set when ``write_sol`` is False (see
	:func:`_patch_full_time_integrator_init`'s own docstring), so reaching that ``isinstance`` check
	at all would raise ``AttributeError`` rather than just being skipped.
	"""
	fake_file = _FakeXdmfFile()
	solver = _FakeSolver(outcomes=[(False, 5)])
	_patch_full_time_integrator_init(monkeypatch, fake_file, solver, write_sol=False)
	step = _FakeStep(t_fin=5.0, dt_ini=1.0, dt_min=0.9)
	problem = _FakeProblem([step])

	integrator = DecimatedTimeIntegrator(problem=problem)

	assert integrator.integrate() is False
	assert any("failed to move forward" in message for message in integrator.printer.messages)
	assert fake_file.written_times == []  # never touched: write_sol was False throughout.


def test_relative_error_both_near_zero_returns_zero() -> None:
	"""Both val and ref within the near-zero guard: defined as no error at all, not a division by ~0."""
	assert _relative_error(val=1e-10, ref=1e-10) == 0.0


def test_relative_error_ref_near_zero_val_not_returns_one() -> None:
	"""Ref alone near zero: the same "maximal disagreement" convention as ``dolfin_mech.core.compute_error``."""
	assert _relative_error(val=0.5, ref=0.0) == 1.0


def test_relative_error_normal_ratio() -> None:
	"""Away from the near-zero guard, this is a plain val/ref ratio."""
	assert _relative_error(val=4.0, ref=2.0) == 2.0


def test_is_subsol_stagnant_true_when_error_below_threshold() -> None:
	"""A tiny relative change is judged stagnant against a generous min_relative_error."""
	assert is_subsol_stagnant(delta_norm=1e-8, scale_norm=1.0, min_relative_error=1e-6) is True


def test_is_subsol_stagnant_false_when_error_above_threshold() -> None:
	"""A substantial relative change is judged still evolving, not stagnant."""
	assert is_subsol_stagnant(delta_norm=0.01, scale_norm=1.0, min_relative_error=1e-6) is False


def test_is_subsol_stagnant_uses_the_near_zero_epsilon_by_default() -> None:
	"""With no explicit min_relative_error, only an (near-)exactly unchanged subsol is stagnant.

	A relative change of 1e-6 comfortably clears :data:`_RELATIVE_ERROR_ZERO_EPS` (``1e-9``, the
	default) and so is judged still evolving, although a looser explicit threshold would catch it:
	the *default* is deliberately strict.
	"""
	assert is_subsol_stagnant(delta_norm=1e-6, scale_norm=1.0) is False
	assert is_subsol_stagnant(delta_norm=1e-10, scale_norm=1.0) is True


def test_is_subsol_stagnant_rejects_non_positive_min_relative_error() -> None:
	"""min_relative_error=0 or <0 would make the threshold undefined/meaningless."""
	with pytest.raises(ValueError):
		is_subsol_stagnant(delta_norm=1.0, scale_norm=1.0, min_relative_error=0.0)
	with pytest.raises(ValueError):
		is_subsol_stagnant(delta_norm=1.0, scale_norm=1.0, min_relative_error=-1e-6)
