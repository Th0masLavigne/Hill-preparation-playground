r"""Integration: Terzaghi consolidation, SI vs operator ``scale_factor`` vs dimensionless parameters.

Checks that both nondimensionalization methods (i) give the SI solution to round-off and
(ii) match the analytical solution. Figures and error table are written next to this file in
``outputs_terzaghi_nondim/`` (``terzaghi_nondim.png``, ``errors.csv``); run ``pytest -s`` to see
the table in the terminal.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import terzaghi_nondim as tz  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "outputs_terzaghi_nondim"
INPUTS = tz.TerzaghiInputs(n_cells=30, n_steps=150)
ROW_DEGREES = {"displacement": 3, "liquid_pressure": 2}
"""Per-row quadrature (``ResidualRows``) of the ``"<mode>+rows"`` variants; the default degree is 4."""
CASES = [*tz.MODES, *(f"{mode}+rows" for mode in tz.MODES)]
PRESSURE_TOL, SETTLEMENT_TOL = 1e-2, 1e-2


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
	"""One run per mode, then figure and error table."""
	results = {mode: tz.solve(INPUTS, mode, OUTPUT) for mode in tz.MODES}
	results |= {f"{mode}+rows": tz.solve(INPUTS, mode, OUTPUT, ROW_DEGREES) for mode in tz.MODES}
	rows = [{"mode": mode, **vars(tz.errors(INPUTS, table))} for mode, table in results.items()]
	pd.DataFrame(rows).to_csv(OUTPUT / "errors.csv", index=False)
	tz.plot(INPUTS, results, OUTPUT / "terzaghi_nondim.png")
	print("\nTerzaghi relative errors vs analytical (figure: %s)" % (OUTPUT / "terzaghi_nondim.png"))
	print(pd.DataFrame(rows).to_string(index=False, float_format="%.3e"))
	return results


@pytest.mark.parametrize("mode", CASES)
def test_matches_analytical(tables: dict[str, pd.DataFrame], mode: str) -> None:
	r"""Pressure and settlement within 1 % of Terzaghi (:math:`T \ge 0.05`)."""
	err = tz.errors(INPUTS, tables[mode])
	assert err.pressure < PRESSURE_TOL, f"{mode}: pressure error {err.pressure:.3e}"
	assert err.settlement < SETTLEMENT_TOL, f"{mode}: settlement error {err.settlement:.3e}"


@pytest.mark.parametrize(
	("mode", "reference_mode"), [(m, "none") for m in tz.MODES[1:]] + [(f"{m}+rows", "none+rows") for m in tz.MODES[1:]]
)
def test_same_solution_as_si(tables: dict[str, pd.DataFrame], mode: str, reference_mode: str) -> None:
	"""Row scaling does not change Newton's iterates: same times, settlement and pressures as SI (same rows)."""
	reference, scaled = tables[reference_mode], tables[mode]
	assert len(scaled) == len(reference)
	np.testing.assert_allclose(scaled["t"], reference["t"], rtol=1e-12)
	np.testing.assert_allclose(scaled["u0"], reference["u0"], rtol=1e-8, atol=1e-8 * reference["u0"].abs().max())
	p_cols = [c for c in reference.columns if c.startswith("p_")]
	np.testing.assert_allclose(scaled[p_cols], reference[p_cols], rtol=0.0, atol=1e-8 * INPUTS.load)


def test_outputs_written(tables: dict[str, pd.DataFrame]) -> None:
	"""Figure and error table exist."""
	assert (OUTPUT / "terzaghi_nondim.png").stat().st_size > 0
	assert len(pd.read_csv(OUTPUT / "errors.csv")) == len(CASES)


def test_row_degrees_are_applied(tables: dict[str, pd.DataFrame]) -> None:
	"""Per-row degrees change the result (they are applied) but only at discretization level (< 1e-3)."""
	p_cols = [c for c in tables["none"].columns if c.startswith("p_")]
	diff = (tables["none+rows"][p_cols] - tables["none"][p_cols]).abs().to_numpy().max() / INPUTS.load
	assert 1e-12 < diff < 1e-3
