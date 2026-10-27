"""Shared fixtures: unit-cube average, homogeneous dilatation (Hill fixtures: fenicsoft `tests/conftest.py`)."""

from typing import Any, Callable

import dolfin
import pytest
from dolfin_mech.kinematics import Kinematics


def cube_mean(expr: Any, mesh: Any = None) -> float:
	r"""Volume average over the unit cube, :math:`\int_\Omega e\,\mathrm dV` (:math:`|\Omega| = 1`).

	Args:
	    expr: UFL scalar.
	    mesh: Unit cube carrying the coefficients of ``expr``; ``None`` builds one (constants only).
	        Defaults to None.

	Returns:
	    The average.
	"""
	if mesh is None:
		mesh = dolfin.UnitCubeMesh(1, 1, 1)
	return dolfin.assemble(expr * dolfin.dx(domain=mesh))


def dilated_kinematics(jac: float) -> tuple[Kinematics, Any]:
	r"""Homogeneous dilatation :math:`\mathbf F = J^{1/3}\mathbf I` of the unit cube.

	Args:
	    jac: :math:`J`.

	Returns:
	    Kinematics and mesh.
	"""
	mesh = dolfin.UnitCubeMesh(1, 1, 1)
	u = dolfin.Function(dolfin.VectorFunctionSpace(mesh, "CG", 1))
	stretch = jac ** (1.0 / 3.0) - 1.0
	u.interpolate(dolfin.Expression(("s*x[0]", "s*x[1]", "s*x[2]"), s=stretch, degree=1))
	return Kinematics(U=u), mesh


@pytest.fixture
def mean() -> Callable[..., float]:
	"""Volume average over the unit cube.

	Returns:
	    :func:`cube_mean`.
	"""
	return cube_mean


@pytest.fixture
def dilate() -> Callable[[float], tuple[Kinematics, Any]]:
	"""Kinematics of a homogeneous dilatation.

	Returns:
	    :func:`dilated_kinematics`.
	"""
	return dilated_kinematics
