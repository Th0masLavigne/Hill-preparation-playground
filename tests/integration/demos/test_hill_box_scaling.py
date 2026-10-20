r"""Integration: the Hill box demo assembles the same Jacobian in ``"operator"`` and ``"parameters"`` modes.

``"operator"`` keeps the coefficients in SI and scales each term by :math:`s\,v/\mathcal N_{row}`
(``ResidualRows``); ``"parameters"`` converts the coefficients. Both must give the same dimensionless
system, hence the same matrix, well conditioned compared to SI.
Uses ``hill_box_modes/jacobian_scaling.py`` on a small box (about 2000 dofs, assembly only, no solve).
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "hill_box_modes"))

import jacobian_scaling as js  # noqa: E402


@pytest.fixture(scope="module")
def jacobians(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
	"""Jacobian and dofs per mode."""
	return {
		mode: js.assemble_jacobian(js.small_box(mode, tmp_path_factory.mktemp(mode)))
		for mode in ("none", "operator", "parameters")
	}


def test_operator_equals_parameters(jacobians: dict[str, Any]) -> None:
	"""Same matrix, entry by entry (relative to its largest entry)."""
	operator, parameters = jacobians["operator"][0], jacobians["parameters"][0]
	assert abs(operator - parameters).max() <= 1e-10 * abs(parameters).max()


def test_blocks_are_order_one(jacobians: dict[str, Any]) -> None:
	"""Mechanics and pressure rows O(1); DG0 rows scale with the cell volume (about 2e-3 here)."""
	medians = js.block_diagonals(*jacobians["operator"])
	assert 0.1 < medians["displacement"] < 10.0
	assert 0.1 < medians["liquid_pressure"] < 10.0
	assert 0.1 < medians["eps_a"] < 10.0
	for name in ("theta", "phi", "cell_fraction"):
		assert 1e-4 < medians[name] < 1e-1


def test_conditioning_improved(jacobians: dict[str, Any]) -> None:
	"""cond2: SI is useless (> 1e15); both scaled modes below 1e6 and equal."""
	cond = {mode: js.condition_number(matrix) for mode, (matrix, _) in jacobians.items()}
	print("\ncond2:", {mode: f"{value:.2e}" for mode, value in cond.items()})
	assert cond["none"] > 1e15
	assert cond["operator"] < 1e6
	assert np.isclose(cond["operator"], cond["parameters"], rtol=1e-6)
