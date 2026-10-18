r"""Integration: the Hill chain inside the solid, one homogeneous element vs the 0D reference.

Checks, on the assembled monolithic :math:`(R_u, R_a)` problem (``TimeIntegrator`` + ``NonlinearSolver``):

* :math:`\varepsilon_a(t_n)` equals the 0D backward-Euler solution (same :math:`\Delta t`, same :math:`\alpha(t_n)`);
* :math:`P_{xx}` equals :math:`\lambda(S^{gel}_{xx} + \Phi_c T_e/I_4)` on the computed state;
* at the end of the hold, :math:`\varepsilon_a` is the isometric equilibrium :math:`f_{max} f_{FL} + T_p = T_e`;
* traction-free faces: :math:`P_{xx} = P_{yy} = 0` and the cell shortens the gel (:math:`\lambda < 1`).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hill_0d_reference as ref  # noqa: E402
from hill_one_element import OneElementRun, run, stress_check  # noqa: E402

ISOMETRIC_STRETCHES = (1.0, 1.04)


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
	r"""Active cells compact the gel along :math:`\mathbf f_0`: :math:`\lambda < 1`, :math:`J < 1`."""
	last = free_run.iloc[-1]
	assert last["lambda"] < 0.9
	assert last["lambda"] * last["lambda_t"] ** 2 < 1.0
