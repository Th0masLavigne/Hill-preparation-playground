"""Unit tests of the clipped SaturatedProliferation: r <1 - Phi_c/Phi_c,max>_+ Phi_c."""

import math

import dolfin
import pytest
from fenicsoft.operators.cell import SaturatedProliferation as FenicsoftSaturatedProliferation

from hill_playground.operators.cell import SaturatedProliferation

R, PHI_C_MAX = 2.0, 0.1


def _residual(cls: type, phi_c_value: float) -> float:
	"""Assembled residual on the unit cube for a uniform Phi_c.

	Args:
	    cls: Operator class.
	    phi_c_value: Uniform Phi_c.

	Returns:
	    Sum of the residual vector.
	"""
	mesh = dolfin.UnitCubeMesh(1, 1, 1)
	space = dolfin.FunctionSpace(mesh, "CG", 1)
	phi_c = dolfin.interpolate(dolfin.Constant(phi_c_value), space)
	op = cls(phi_c, R, PHI_C_MAX, dolfin.TestFunction(space), dolfin.dx(domain=mesh))
	return float(dolfin.assemble(op.res_form).sum())


@pytest.mark.parametrize("phi_c", [0.0, 0.05, 0.099])
def test_same_as_fenicsoft_below_max(phi_c: float) -> None:
	"""Phi_c <= Phi_c,max: identical to fenicsoft."""
	assert math.isclose(
		_residual(SaturatedProliferation, phi_c), _residual(FenicsoftSaturatedProliferation, phi_c), abs_tol=1e-14
	)


def test_no_sink_above_max() -> None:
	"""Phi_c > Phi_c,max: zero source (fenicsoft's version becomes a sink)."""
	assert _residual(SaturatedProliferation, 0.15) == 0.0
	assert _residual(FenicsoftSaturatedProliferation, 0.15) > 0.0
