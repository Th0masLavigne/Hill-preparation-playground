"""Unit tests of the compaction volumetric energy (equations.tex, Eqs. W-vol-compaction, S-vol-compaction)."""

import math
from typing import Any

import dolfin
import pytest
from dolfin_mech.kinematics import Kinematics

from hill_playground.materials.volumetric import compaction

KAPPA, JC = 1417.0, 0.3


def _law(jac: float, x_floor: float = 1e-8) -> tuple[compaction.CompactionVolumetricPenalty, Any]:
	mesh = dolfin.UnitCubeMesh(1, 1, 1)
	u = dolfin.Function(dolfin.VectorFunctionSpace(mesh, "CG", 1))
	s = jac ** (1.0 / 3.0) - 1.0
	u.interpolate(dolfin.Expression(("s*x[0]", "s*x[1]", "s*x[2]"), s=s, degree=1))
	law = compaction.CompactionVolumetricPenalty(
		Kinematics(U=u), compaction.Parameters(kappa=KAPPA, solid_jacobian=dolfin.Constant(JC), x_floor=x_floor)
	)
	return law, mesh


def _mean(expr: Any, mesh: Any) -> float:
	return dolfin.assemble(expr * dolfin.dx(domain=mesh))


def test_stress_free_reference() -> None:
	"""W(1) = 0 and S(1) = 0."""
	law, mesh = _law(1.0)
	assert abs(_mean(law.ener, mesh)) < 1e-12
	assert abs(_mean(law.pk2[0, 0], mesh)) < 1e-10


def test_small_strain_bulk_modulus_is_kappa() -> None:
	"""Mean Cauchy stress ~ kappa (J - 1) near J = 1 (same K_d as kappa/2 (J-1)^2)."""
	jac = 1.0 + 1e-6
	law, mesh = _law(jac)
	assert _mean(law.cauchy[0, 0], mesh) == pytest.approx(KAPPA * (jac - 1.0), rel=1e-5)


def test_closed_form_and_barrier() -> None:
	"""S_ii = kappa J (J-1)/x / C_ii; W grows without bound as J -> J_c."""
	jac = 0.5
	law, mesh = _law(jac)
	x = (jac - JC) / (1.0 - JC)
	assert _mean(law.pk2[0, 0], mesh) == pytest.approx(KAPPA * jac * (jac - 1.0) / x / jac ** (2.0 / 3.0), rel=1e-10)
	assert _mean(law.ener, mesh) == pytest.approx(KAPPA * (1 - JC) ** 2 * (x - math.log(x) - 1.0), rel=1e-10)
	near, mesh_near = _law(JC + 1e-4)
	assert _mean(near.ener, mesh_near) > 10.0 * _mean(law.ener, mesh)


def test_stiffer_than_quadratic_in_compaction() -> None:
	r"""Below :math:`J = 1`, :math:`|S|` exceeds :math:`\kappa J (J-1) C^{-1}` (the barrier)."""
	jac = 0.5
	law, mesh = _law(jac)
	quadratic = KAPPA * jac * (jac - 1.0) / jac ** (2.0 / 3.0)
	assert abs(_mean(law.pk2[0, 0], mesh)) > abs(quadratic)


def test_stress_is_derivative_of_energy() -> None:
	"""S_ii C_ii / J = dW/dJ (finite difference), above and below the floor."""
	for jac, x_floor in ((0.6, 1e-8), (0.32, 0.1)):
		h = 1e-6
		w_plus, m_plus = _law(jac + h, x_floor)
		w_minus, m_minus = _law(jac - h, x_floor)
		d_ener = (_mean(w_plus.ener, m_plus) - _mean(w_minus.ener, m_minus)) / (2.0 * h)
		law, mesh = _law(jac, x_floor)
		assert _mean(law.pk2[0, 0], mesh) * jac ** (2.0 / 3.0) / jac == pytest.approx(d_ener, rel=1e-5)


def test_floor_keeps_stress_finite() -> None:
	"""Below x_f the stress is frozen at its floor value (linear continuation of W)."""
	x_floor = 0.1
	inside, mesh_in = _law(JC + 0.5 * x_floor * (1.0 - JC), x_floor)
	at, mesh_at = _law(JC + x_floor * (1.0 - JC), x_floor)
	s_floor = KAPPA * (1.0 - JC) * (1.0 - 1.0 / x_floor)
	assert _mean(at.cauchy[0, 0], mesh_at) == pytest.approx(s_floor, rel=1e-8)
	assert _mean(inside.cauchy[0, 0], mesh_in) == pytest.approx(s_floor, rel=1e-8)
