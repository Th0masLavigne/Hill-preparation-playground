"""Unit tests of the Hill-chain stress law: S_h = Phi_c T_e/I4 f0 x f0 and its push-forwards."""

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


def test_ce_stretch(hill_setup: Any) -> None:
	"""lambda_a = exp(eps_a)."""
	hill_setup.eps_a.vector()[:] = math.log(0.9)
	assert math.isclose(hill_setup.mean(hill_setup.law.lmbda_a), 0.9, rel_tol=1e-12)


def test_cauchy_along_current_fibre(hill_setup: Any) -> None:
	"""sigma_h = phi_c T_e f x f: with F = diag(1.1, 1, 1), sigma_11 = (Phi_c/J) T_e."""
	hill_setup.u.interpolate(dolfin.Expression(("0.1*x[0]", "0.0", "0.0"), degree=1))
	i4 = 1.1**2
	expected = PHI_C / 1.1 * (K_E / 4.0) * (i4 - 1.0 / i4)
	assert math.isclose(hill_setup.mean(hill_setup.law.cauchy[0, 0]), expected, rel_tol=1e-10)


def test_pk2_is_energy_derivative(hill_setup: Any) -> None:
	"""S_h = 2 dW/dC at fixed eps_a, checked along a stretch path.

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


@pytest.mark.parametrize("override", [{"hill_b": 0.0}, {"hill_a": -1.0}, {"f_ecc": 1.0}])
def test_bad_force_velocity_rejected(make_hill_setup: Any, override: dict[str, float]) -> None:
	"""Singular h (b <= 0, 1 + A <= 0 or f_ecc <= 1) raises."""
	with pytest.raises(ValueError):
		make_hill_setup(**override)


def test_law_is_independent_of_other_laws() -> None:
	"""The Hill law derives from the fenicsoft base classes only (no other material in its MRO or imports)."""
	import inspect

	from fenicsoft.materials.materials import ConstitutiveLaw, MaterialParameters

	assert hill_law.Parameters.__mro__[1] is MaterialParameters
	assert hill_law.Constitutive.__mro__[1] is ConstitutiveLaw
	source = inspect.getsource(hill_law)
	assert "cell.active.aniso" not in source and "cell.active.iso" not in source


def test_activation_follows_steps(make_hill_setup: Any) -> None:
	"""p_act: ramp 0 -> 1 on step 0 (value at t_step), constant 0.5 on step 1."""
	from unittest.mock import MagicMock

	from fenicsoft.materials.cell.active.activation import ActivePowerConfig

	steps = [MagicMock(t_ini=0.0, t_fin=1.0), MagicMock(t_ini=1.0, t_fin=2.0)]
	configs = [ActivePowerConfig.ramp(k_step=0, ini=0.0, fin=1.0), ActivePowerConfig.constant(k_step=1, value=0.5)]
	law = make_hill_setup(activation_configs=configs, simulation_steps=steps).law
	law.set_k_step(0)
	law.update_time(0.25)
	assert law.current_activation == pytest.approx(0.25, rel=1e-12)
	assert float(law.p_act) == pytest.approx(0.25, rel=1e-12)
	law.set_k_step(1)
	law.update_time(0.7)
	assert law.current_activation == pytest.approx(0.5, rel=1e-12)
	assert law.current_time == pytest.approx(1.7, rel=1e-12)  # t_ini + t_step (t_fin - t_ini) of step 1


def test_uniqueness_margin_of_defaults() -> None:
	"""Nominal chain (k_e = 200 kPa, k_p = 5 kPa, f_max = 20 kPa, lambda_opt = 1.05, W = 0.3): unique."""
	assert hill_law.check_uniqueness(K_E, K_P, 20e3, 1.05, 0.3) > 1.0


def test_uniqueness_violated_without_see() -> None:
	"""k_e = 0 (free cell): k_p alone cannot dominate the descending limb."""
	with pytest.raises(ValueError, match="not unique"):
		hill_law.check_uniqueness(0.0, K_P, 20e3, 1.05, 0.3)


def test_uniqueness_wide_gaussian() -> None:
	r"""W = 2.5, :math:`\lambda_{opt} = 1`: :math:`\max|f_{FL}'\lambda| = 1.042` at :math:`\lambda \approx 3.3`.

	Missed by a :math:`\lambda \le 3` sampling window.
	"""
	with pytest.raises(ValueError, match="not unique"):
		hill_law.check_uniqueness(1.03, 0.0, 1.0, 1.0, 2.5)


def test_uniqueness_passive_cell() -> None:
	"""f_max = 0 (no active tension): always unique, infinite margin."""
	assert hill_law.check_uniqueness(K_E, K_P, 0.0, 1.05, 0.3) == math.inf
