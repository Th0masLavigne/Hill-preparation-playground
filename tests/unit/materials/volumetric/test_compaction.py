"""Unit tests of the compaction volumetric energy W = kappa (1-J_min)^2 [x - ln x - 1], x = (J-J_min)/(1-J_min)."""

import math
from typing import Any, Callable

import dolfin
import pytest
from dolfin_mech.kinematics import Kinematics

from hill_playground.materials.volumetric import compaction

KAPPA, JC = 1417.0, 0.3


def _law(state: tuple[Kinematics, Any], x_floor: float = 1e-8) -> tuple[compaction.CompactionVolumetricPenalty, Any]:
	kin, mesh = state
	law = compaction.CompactionVolumetricPenalty(
		kin, compaction.Parameters(kappa=KAPPA, solid_jacobian=dolfin.Constant(JC), x_floor=x_floor)
	)
	return law, mesh


def test_stress_free_reference(mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]) -> None:
	"""W(1) = 0 and S(1) = 0."""
	law, mesh = _law(dilate(1.0))
	assert abs(mean(law.ener, mesh)) < 1e-12
	assert abs(mean(law.pk2[0, 0], mesh)) < 1e-10


def test_small_strain_bulk_modulus_is_kappa(
	mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""Mean Cauchy stress ~ kappa (J - 1) near J = 1 (same K_d as kappa/2 (J-1)^2)."""
	jac = 1.0 + 1e-6
	law, mesh = _law(dilate(jac))
	assert mean(law.cauchy[0, 0], mesh) == pytest.approx(KAPPA * (jac - 1.0), rel=1e-5)


def test_closed_form_and_barrier(mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]) -> None:
	"""S_ii = kappa J (J-1)/x / C_ii; W grows without bound as J -> J_min."""
	jac = 0.5
	law, mesh = _law(dilate(jac))
	x = (jac - JC) / (1.0 - JC)
	assert mean(law.pk2[0, 0], mesh) == pytest.approx(KAPPA * jac * (jac - 1.0) / x / jac ** (2.0 / 3.0), rel=1e-10)
	assert mean(law.ener, mesh) == pytest.approx(KAPPA * (1 - JC) ** 2 * (x - math.log(x) - 1.0), rel=1e-10)
	near, mesh_near = _law(dilate(JC + 1e-4))
	assert mean(near.ener, mesh_near) > 10.0 * mean(law.ener, mesh)


def test_stiffer_than_quadratic_in_compaction(
	mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	r"""Below :math:`J = 1`, :math:`|S|` exceeds :math:`\kappa J (J-1) C^{-1}` (the barrier)."""
	jac = 0.5
	law, mesh = _law(dilate(jac))
	quadratic = KAPPA * jac * (jac - 1.0) / jac ** (2.0 / 3.0)
	assert abs(mean(law.pk2[0, 0], mesh)) > abs(quadratic)


def test_stress_is_derivative_of_energy(
	mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""S_ii C_ii / J = dW/dJ (finite difference), above and below the floor."""
	for jac, x_floor in ((0.6, 1e-8), (0.32, 0.1)):
		h = 1e-6
		w_plus, m_plus = _law(dilate(jac + h), x_floor)
		w_minus, m_minus = _law(dilate(jac - h), x_floor)
		d_ener = (mean(w_plus.ener, m_plus) - mean(w_minus.ener, m_minus)) / (2.0 * h)
		law, mesh = _law(dilate(jac), x_floor)
		assert mean(law.pk2[0, 0], mesh) * jac ** (2.0 / 3.0) / jac == pytest.approx(d_ener, rel=1e-5)


def test_floor_keeps_stress_finite(
	mean: Callable[..., float], dilate: Callable[[float], tuple[Kinematics, Any]]
) -> None:
	"""Below x_f the stress is frozen at its floor value (linear continuation of W)."""
	x_floor = 0.1
	inside, mesh_in = _law(dilate(JC + 0.5 * x_floor * (1.0 - JC)), x_floor)
	at, mesh_at = _law(dilate(JC + x_floor * (1.0 - JC)), x_floor)
	s_floor = KAPPA * (1.0 - JC) * (1.0 - 1.0 / x_floor)
	assert mean(at.cauchy[0, 0], mesh_at) == pytest.approx(s_floor, rel=1e-8)
	assert mean(inside.cauchy[0, 0], mesh_in) == pytest.approx(s_floor, rel=1e-8)
