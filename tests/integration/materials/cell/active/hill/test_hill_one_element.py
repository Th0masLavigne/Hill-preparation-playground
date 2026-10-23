r"""Integration: Hill chain in the solid, one homogeneous element (monolithic :math:`(R_u, R_a)`) vs the 0D reference.

Checked: :math:`\varepsilon_a(t_n)` and :math:`P_{xx}` (isometric), isometric equilibrium at the end of the hold,
traction-free state (free faces). Figure and error table in ``outputs_hill_one_element/``
(``hill_one_element.png``, ``errors.csv``); the table is printed in the pytest summary.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hill_0d_reference as ref  # noqa: E402
from hill_one_element import OneElementRun, errors, plot, run, stress_check  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "outputs_hill_one_element"

ISOMETRIC_STRETCHES = (1.0, 1.04)
ROWS = OneElementRun(
	kinematics="free",
	quadrature_degrees={"displacement": 1, "eps_a": 0},
	normalizations={"displacement": OneElementRun().model.mu, "eps_a": OneElementRun().model.f_max},
)
"""Free faces through ``ResidualRows``: other degree per row and row normalizations.

Homogeneous state, CG1 displacement: every integrand is constant per cell, so any degree is exact and
the row normalizations do not change Newton's iterates: same solution as the default free run. The wiring
(1/N_k and q_k reaching the operators) is checked on ``attrs["wiring"]``.
"""


@pytest.fixture(scope="module")
def isometric_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[float, pd.DataFrame]:
	r"""One isometric run per imposed stretch :math:`\lambda`."""
	return {
		lmbda: run(OneElementRun(kinematics="isometric", lmbda=lmbda), tmp_path_factory.mktemp(f"iso_{lmbda}"))
		for lmbda in ISOMETRIC_STRETCHES
	}


@pytest.fixture(scope="module")
def free_run(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
	"""Run with traction-free faces."""
	return run(OneElementRun(kinematics="free"), tmp_path_factory.mktemp("free"))


@pytest.fixture(scope="module")
def rows_run(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
	"""Free faces with per-row degrees and normalizations (:data:`ROWS`)."""
	return run(ROWS, tmp_path_factory.mktemp("rows"))


def test_activation_follows_the_ramp(isometric_runs: dict[float, pd.DataFrame], free_run: pd.DataFrame) -> None:
	r""":math:`\alpha(t) = \min(t/t_{ramp}, 1)`: the FE schedule the 0D reference takes as input."""
	ramp = OneElementRun().ramp_duration
	for table in (*isometric_runs.values(), free_run):
		expected = np.minimum(table["t"].to_numpy() / ramp, 1.0)
		np.testing.assert_allclose(table["alpha"].to_numpy(), expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("lmbda", ISOMETRIC_STRETCHES)
def test_isometric_eps_a_matches_0d(isometric_runs: dict[float, pd.DataFrame], lmbda: float) -> None:
	r""":math:`\varepsilon_a^{FE}(t_n) = \varepsilon_a^{0D}(t_n)` to round-off."""
	table = isometric_runs[lmbda]
	expected = ref.isometric_history(OneElementRun().model, lmbda, table["t"], table["alpha"])
	np.testing.assert_allclose(table["eps_a"], expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("lmbda", ISOMETRIC_STRETCHES)
def test_isometric_stress_matches_0d(isometric_runs: dict[float, pd.DataFrame], lmbda: float) -> None:
	r""":math:`P_{xx}^{FE} = \lambda(S^{gel}_{xx} + \Phi_c T_e / I_4)`."""
	table = isometric_runs[lmbda]
	np.testing.assert_allclose(table["pk1_xx"], stress_check(OneElementRun().model, table), rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("lmbda", ISOMETRIC_STRETCHES)
def test_isometric_reaches_equilibrium(isometric_runs: dict[float, pd.DataFrame], lmbda: float) -> None:
	r"""End of hold: :math:`D_a \to 0` and :math:`f_{max} f_{FL} + T_p = T_e`."""
	eps_end = isometric_runs[lmbda]["eps_a"].iloc[-1]
	assert eps_end == pytest.approx(ref.isometric_equilibrium(OneElementRun().model, lmbda, 1.0), abs=1e-8)


def test_isometric_activation_contracts_ce(isometric_runs: dict[float, pd.DataFrame]) -> None:
	r"""The CE shortens monotonically during the ramp (:math:`\varepsilon_a \downarrow`, :math:`T_e \uparrow`)."""
	table = isometric_runs[1.0]
	ramp = table[table["t"] <= OneElementRun().ramp_duration]
	assert np.all(np.diff(ramp["eps_a"]) < 0.0)
	assert np.all(np.diff(ramp["tension_see"]) > 0.0)


def test_free_state_matches_0d(free_run: pd.DataFrame) -> None:
	r"""FE state :math:`(\lambda, \lambda_t, \varepsilon_a)` equals the 0D one (coupled :math:`R_u`-:math:`R_a`)."""
	expected = ref.free_history(OneElementRun().model, free_run["t"], free_run["alpha"])
	np.testing.assert_allclose(free_run[["lambda", "lambda_t", "eps_a"]].to_numpy(), expected, rtol=0.0, atol=1e-10)


def test_free_faces_are_traction_free(free_run: pd.DataFrame) -> None:
	r""":math:`P_{xx} = P_{yy} = 0` (Pa, relative to :math:`\Phi_c T_e`)."""
	scale = OneElementRun().model.phi_c * free_run["tension_see"].abs().max()
	assert free_run[["pk1_xx", "pk1_yy"]].abs().to_numpy().max() < 1e-10 * scale


def test_free_cells_shorten_gel(free_run: pd.DataFrame) -> None:
	r"""Active cells compact the gel along :math:`\mathbf f_0`: :math:`\lambda < 0.9` (0D: 0.86), :math:`J < 1`."""
	last = free_run.iloc[-1]
	assert last["lambda"] < 0.9
	assert last["lambda"] * last["lambda_t"] ** 2 < 1.0


@pytest.fixture(scope="module")
def report(isometric_runs: dict[float, pd.DataFrame], free_run: pd.DataFrame, rows_run: pd.DataFrame) -> pd.DataFrame:
	"""Error table and figure of the four runs."""
	runs = {
		f"isometric lambda={lmbda:g}": (OneElementRun(kinematics="isometric", lmbda=lmbda), isometric_runs[lmbda])
		for lmbda in ISOMETRIC_STRETCHES
	}
	runs["free faces"] = (OneElementRun(kinematics="free"), free_run)
	runs["free faces, rows"] = (ROWS, rows_run)
	model = OneElementRun().model
	table = pd.DataFrame([{"case": label, **errors(model, settings, data)} for label, (settings, data) in runs.items()])
	OUTPUT.mkdir(parents=True, exist_ok=True)
	table.to_csv(OUTPUT / "errors.csv", index=False)
	plot(model, runs, OUTPUT / "hill_one_element.png")
	return table


def test_report_errors_and_figure(report: pd.DataFrame) -> None:
	"""Every case below round-off tolerances; figure and table written."""
	assert (report[["eps_a", "stretch"]].to_numpy() < 1e-10).all()
	assert (report[["tension_see", "pk1_xx"]].to_numpy() < 1e-10).all()
	assert (OUTPUT / "hill_one_element.png").stat().st_size > 0


def test_rows_free_state_matches_0d(rows_run: pd.DataFrame) -> None:
	r"""With per-row degrees and normalizations: still the 0D state :math:`(\lambda, \lambda_t, \varepsilon_a)`."""
	expected = ref.free_history(OneElementRun().model, rows_run["t"], rows_run["alpha"])
	np.testing.assert_allclose(rows_run[["lambda", "lambda_t", "eps_a"]].to_numpy(), expected, rtol=0.0, atol=1e-10)


def test_rows_are_wired_to_the_operators(rows_run: pd.DataFrame, free_run: pd.DataFrame) -> None:
	r"""Operators built with :math:`1/\mathcal N_k` and :math:`q_k` of their row (the solution cannot show it)."""
	model = OneElementRun().model
	(sf_u, q_u), (sf_a, q_a) = rows_run.attrs["wiring"]["displacement"], rows_run.attrs["wiring"]["eps_a"]
	assert sf_u == pytest.approx(1.0 / model.mu, rel=1e-15)
	assert sf_a == pytest.approx(1.0 / model.f_max, rel=1e-15)
	assert (q_u, q_a) == (1, 0)
	assert [sf for sf, _ in free_run.attrs["wiring"].values()] == [1.0, 1.0]


def test_rows_do_not_change_the_solution(rows_run: pd.DataFrame, free_run: pd.DataFrame) -> None:
	"""Row normalizations and exact per-row degrees: same history as the default free run."""
	cols = ["lambda", "lambda_t", "eps_a", "tension_see"]
	np.testing.assert_allclose(rows_run[cols].to_numpy(), free_run[cols].to_numpy(), rtol=1e-10, atol=1e-12)
