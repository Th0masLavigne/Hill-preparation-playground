r"""Integration: the Hill box assembles the same Jacobian in ``"operator"`` and ``"parameters"`` modes.

Both solve the same dimensionless system (``"operator"``: SI coefficients, rows scaled by
:math:`s\,v/\mathcal N_{row}`; ``"parameters"``: converted coefficients), better conditioned than SI.
Small box (about 2000 dofs), assembly only, via ``hill_box_modes/jacobian_scaling.py``.
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
	r"""Stiffness-led rows (``displacement``, ``liquid_pressure``) and ``eps_a`` O(1); mass-led rows smaller.

	Mass-led rows (``theta``, ``phi``, ``cell_fraction``) scale with :math:`(t_0/\Delta t)\,|\hat\Omega_e|`
	(:math:`|\hat\Omega_e| \approx \hat h^3/6 \approx 0.02`, :math:`t_0/\Delta t \approx 0.2`: about 1e-3 here).
	``eps_a`` is DG0 too, but its diagonal, dominated by :math:`(k_e + k_p)/\mu_0\,|\hat\Omega_e| \approx 2.5`, is O(1).
	``substrate_smoothed`` scales with :math:`|\hat\Omega_e| \approx 0.02` (no :math:`t_0/\Delta t` factor).
	"""
	medians = js.block_diagonals(*jacobians["operator"])
	assert 0.1 < medians["displacement"] < 10.0
	assert 0.1 < medians["liquid_pressure"] < 10.0
	assert 0.1 < medians["eps_a"] < 10.0
	for name in ("theta", "phi", "cell_fraction"):
		assert 1e-4 < medians[name] < 1e-1
	assert 1e-3 < medians["substrate_smoothed"] < 1.0


def test_conditioning_improved(jacobians: dict[str, Any]) -> None:
	"""cond2: SI is useless (> 1e15); both scaled modes below 1e6 and equal."""
	cond = {mode: js.condition_number(matrix) for mode, (matrix, _) in jacobians.items()}
	print("\ncond2:", {mode: f"{value:.2e}" for mode, value in cond.items()})
	assert cond["none"] > 1e15
	assert cond["operator"] < 1e6
	assert np.isclose(cond["operator"], cond["parameters"], rtol=1e-6, atol=1e-8)
