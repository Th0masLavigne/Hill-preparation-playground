"""Unit tests of the Hill-chain stress law (equations.tex, Eqs. S-h, P-h)."""

import math
from typing import Any

import dolfin
import pytest

from hill_playground.materials.cell.active.hill import law as hill_law

K_E, K_P, PHI_C = 200e3, 5e3, 0.05


def test_stress_free_at_rest(hill_setup: Any) -> None:
	"""Lambda = lambda_a = 1: S_h = 0."""
	assert abs(hill_setup.mean(hill_setup.law.pk2[0, 0])) < 1e-10


def test_pk2_uses_phi_c_not_phi_c_over_j(hill_setup: Any) -> None:
	"""Uniaxial stretch 1.1 along f0, eps_a = 0: S_11 = Phi_c T_e / I4 (no 1/J)."""
	hill_setup.u.interpolate(dolfin.Expression(("0.1*x[0]", "0.0", "0.0"), degree=1))
	i4 = 1.1**2
	expected = PHI_C * (K_E / 4.0) * (i4 - 1.0 / i4) / i4
	assert math.isclose(hill_setup.mean(hill_setup.law.pk2[0, 0]), expected, rel_tol=1e-10)


def test_contracted_ce_pulls(hill_setup: Any) -> None:
	"""Lambda = 1, lambda_a = 0.9: I4e = 1/0.81, the SEE is stretched, T_e > 0, T_p < 0."""
	hill_setup.eps_a.vector()[:] = math.log(0.9)
	i4e = 1.0 / 0.81
	assert math.isclose(hill_setup.mean(hill_setup.law.tension_see), K_E / 4.0 * (i4e - 1.0 / i4e), rel_tol=1e-10)
	assert math.isclose(hill_setup.mean(hill_setup.law.tension_pee), K_P / 4.0 * (0.81 - 1.0 / 0.81), rel_tol=1e-10)


def test_ce_stretch_and_tension(hill_setup: Any) -> None:
	"""lambda_a = exp(eps_a) and T_CE = T_e - T_p (series equilibrium)."""
	hill_setup.eps_a.vector()[:] = math.log(0.9)
	law = hill_setup.law
	assert math.isclose(hill_setup.mean(law.lmbda_a), 0.9, rel_tol=1e-12)
	expected = hill_setup.mean(law.tension_see) - hill_setup.mean(law.tension_pee)
	assert math.isclose(hill_setup.mean(law.tension_ce), expected, rel_tol=1e-12)


def test_cauchy_along_current_fibre(hill_setup: Any) -> None:
	"""sigma_h = phi_c T_e f x f (Eq. P-h): with F = diag(1.1, 1, 1), sigma_11 = (Phi_c/J) T_e."""
	hill_setup.u.interpolate(dolfin.Expression(("0.1*x[0]", "0.0", "0.0"), degree=1))
	i4 = 1.1**2
	expected = PHI_C / 1.1 * (K_E / 4.0) * (i4 - 1.0 / i4)
	assert math.isclose(hill_setup.mean(hill_setup.law.cauchy[0, 0]), expected, rel_tol=1e-10)


def test_pk2_is_energy_derivative(hill_setup: Any) -> None:
	"""S_h = 2 dW/dC at fixed eps_a (Eq. hill-overview), checked along a stretch path.

	With F = diag(s, 1, 1): dW/ds = P_11 = s S_11.
	"""
	hill_setup.eps_a.vector()[:] = math.log(0.95)
	step = 1e-6

	def energy(stretch: float) -> float:
		hill_setup.u.interpolate(dolfin.Expression((f"{stretch - 1.0}*x[0]", "0.0", "0.0"), degree=1))
		return hill_setup.mean(hill_setup.law.ener)

	numerical = (energy(1.05 + step) - energy(1.05 - step)) / (2 * step)
	energy(1.05)
	assert math.isclose(numerical, hill_setup.mean(hill_setup.law.pk1[0, 0]), rel_tol=1e-5)


def test_bad_variant_rejected(make_hill_setup: Any) -> None:
	"""Unknown SEE variant, or exponential with k_2e = 0, raise."""
	with pytest.raises(ValueError):
		make_hill_setup(see_variant="nope")
	with pytest.raises(ValueError):
		make_hill_setup(see_variant="exponential", k_2e=0.0)


def test_parameters_are_aniso_parameters() -> None:
	"""The Hill parameters extend the aniso ones (activation machinery reused)."""
	from fenicsoft.materials.cell.active.aniso import law as aniso_law

	assert issubclass(hill_law.Parameters, aniso_law.Parameters)
	assert issubclass(hill_law.Constitutive, aniso_law.Constitutive)
