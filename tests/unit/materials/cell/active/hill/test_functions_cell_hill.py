"""Unit tests of the Hill-chain UFL functions: SEE, PEE, force--length and force--velocity laws.

Each UFL expression is evaluated on constants and compared with the scalar formula.
"""

import math
from typing import Callable

import dolfin
import pytest

from hill_playground.materials.cell.active.hill import functions as hill

K_E, K_P = 200e3, 5e3
A, B, F_ECC = 0.25, 0.05, 1.8


@pytest.mark.parametrize("variant", hill.SEE_VARIANTS)
def test_see_natural_state(variant: str, mean: Callable[..., float]) -> None:
	"""T_e = Psi_e = 0 at I4e = 1, for every variant."""
	k_2e = 1.0 if variant == "exponential" else 0.0
	one = dolfin.Constant(1.0)
	assert abs(mean(hill.see_tension(one, K_E, variant, k_2e))) < 1e-9
	assert abs(mean(hill.see_energy(one, K_E, variant, k_2e))) < 1e-9


@pytest.mark.parametrize("variant", hill.SEE_VARIANTS)
def test_see_tension_is_energy_derivative(variant: str, mean: Callable[..., float]) -> None:
	"""T_e = 2 I4e dPsi_e/dI4e, checked by UFL differentiation at I4e = 1.2."""
	k_2e = 0.5 if variant == "exponential" else 0.0
	inv4e = dolfin.variable(dolfin.Constant(1.2))
	derivative = 2.0 * inv4e * dolfin.diff(hill.see_energy(inv4e, K_E, variant, k_2e), inv4e)
	assert math.isclose(mean(derivative), mean(hill.see_tension(inv4e, K_E, variant, k_2e)), rel_tol=1e-10)


def test_see_slack_in_compression(mean: Callable[..., float]) -> None:
	"""Tension-only SEE: T_e = 0 for I4e < 1."""
	assert mean(hill.see_tension(dolfin.Constant(0.8), K_E, "tension_only")) == 0.0


def test_pee_tension_is_energy_derivative(mean: Callable[..., float]) -> None:
	"""T_p = lambda_a dPsi_p/dlambda_a = dPsi_p/deps_a."""
	eps_a = dolfin.variable(dolfin.Constant(-0.1))
	derivative = dolfin.diff(hill.pee_energy(eps_a, K_P), eps_a)
	assert math.isclose(mean(derivative), mean(hill.pee_tension(eps_a, K_P)), rel_tol=1e-10)


def test_pee_initial_stiffness(mean: Callable[..., float]) -> None:
	"""dT_p/deps_a = k_p at rest (k_p is the initial stiffness)."""
	eps_a = dolfin.variable(dolfin.Constant(0.0))
	assert math.isclose(mean(dolfin.diff(hill.pee_tension(eps_a, K_P), eps_a)), K_P, rel_tol=1e-12)


def test_force_length_peak(mean: Callable[..., float]) -> None:
	"""f_FL(lambda_opt) = 1."""
	assert math.isclose(mean(hill.force_length(dolfin.Constant(math.log(1.05)), 1.05, 0.3)), 1.0, rel_tol=1e-12)


@pytest.mark.parametrize(
	("rate", "expected"),
	[
		(0.0, 1.0),  # isometric
		(-B / A, 0.0),  # v_max = b/A
		(-0.02, (1 + A) * B / (B + 0.02) - A),  # hyperbola
		(1e6, F_ECC),  # eccentric saturation
	],
)
def test_force_velocity_values(rate: float, expected: float, mean: Callable[..., float]) -> None:
	"""h(0) = 1, h(-b/A) = 0, hyperbola for D_a < 0, h -> f_ecc."""
	assert math.isclose(mean(hill.force_velocity(dolfin.Constant(rate), A, B, F_ECC)), expected, abs_tol=1e-5)


def test_force_velocity_c1_at_zero(mean: Callable[..., float]) -> None:
	"""h'(0-) = h'(0+) = (1 + A)/b (C1 junction of the concentric and eccentric branches)."""
	slope = (1 + A) / B
	step = 1e-7

	def h(rate: float) -> float:
		return mean(hill.force_velocity(dolfin.Constant(rate), A, B, F_ECC))

	assert math.isclose((h(0.0) - h(-step)) / step, slope, rel_tol=1e-4)
	assert math.isclose((h(step) - h(0.0)) / step, slope, rel_tol=1e-4)


def test_unknown_variant() -> None:
	"""Unknown SEE variant raises, for the tension and the energy."""
	with pytest.raises(ValueError):
		hill.see_tension(dolfin.Constant(1.0), K_E, "nope")
	with pytest.raises(ValueError):
		hill.see_energy(dolfin.Constant(1.0), K_E, "nope")
