"""Unit tests of the shared UFL guards (floored, macaulay)."""

import dolfin
import pytest

from hill_playground.materials.ufl_functions import floored, macaulay


def _value(expr: object) -> float:
	mesh = dolfin.UnitCubeMesh(1, 1, 1)
	return dolfin.assemble(expr * dolfin.dx(domain=mesh))


@pytest.mark.parametrize(("x", "expected"), [(-1.0, 0.5), (0.5, 0.5), (2.0, 2.0)])
def test_floored(x: float, expected: float) -> None:
	"""max(x, x_f) with x_f = 0.5."""
	assert _value(floored(dolfin.Constant(x), 0.5)) == pytest.approx(expected)


@pytest.mark.parametrize(("x", "expected"), [(-2.0, 0.0), (0.0, 0.0), (3.0, 3.0)])
def test_macaulay(x: float, expected: float) -> None:
	"""<x>_+ = max(x, 0)."""
	assert _value(macaulay(dolfin.Constant(x))) == pytest.approx(expected)
