"""Unit tests of the saturated-mixture relations (J_min, Phi_l, phi_l, phi_s)."""

import pytest

from hill_playground.materials import saturated_mixture

PHI_S0, PHI_C0, JAC = 0.25, 0.05, 0.8


def test_compaction_jacobian() -> None:
	"""J_min = Phi_s0 + Phi_c."""
	assert saturated_mixture.compaction_jacobian(PHI_S0, PHI_C0) == pytest.approx(0.3, rel=1e-12)


def test_fluid_fractions() -> None:
	"""Phi_l = J - J_min and phi_l = Phi_l / J."""
	assert saturated_mixture.lagrangian_fluid_fraction(JAC, PHI_S0, PHI_C0) == pytest.approx(0.5, rel=1e-12)
	assert saturated_mixture.eulerian_porosity(JAC, PHI_S0, PHI_C0) == pytest.approx(0.5 / JAC, rel=1e-12)


def test_solid_fraction() -> None:
	"""phi_s = Phi_s0 / J."""
	assert saturated_mixture.solid_fraction(JAC, PHI_S0) == pytest.approx(PHI_S0 / JAC, rel=1e-12)


def test_volume_fractions_sum_to_one() -> None:
	"""phi_s + Phi_c/J + phi_l = 1 (saturation)."""
	total = (
		saturated_mixture.solid_fraction(JAC, PHI_S0)
		+ PHI_C0 / JAC
		+ saturated_mixture.eulerian_porosity(JAC, PHI_S0, PHI_C0)
	)
	assert total == pytest.approx(1.0, rel=1e-12, abs=1e-12)


def test_two_phase_mixture_is_the_default() -> None:
	"""Phi_c omitted: J_min = Phi_s0 and phi_l = 1 - phi_s (solid + liquid)."""
	assert saturated_mixture.compaction_jacobian(PHI_S0) == pytest.approx(PHI_S0, rel=1e-12)
	assert saturated_mixture.lagrangian_fluid_fraction(JAC, PHI_S0) == pytest.approx(JAC - PHI_S0, rel=1e-12)
	porosity = saturated_mixture.eulerian_porosity(JAC, PHI_S0)
	assert porosity == pytest.approx(1.0 - saturated_mixture.solid_fraction(JAC, PHI_S0), rel=1e-12)
