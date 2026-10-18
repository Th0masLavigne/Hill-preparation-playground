"""Unit tests of the orientation diagnostics (tilt, axis angle, order parameter, current direction)."""

import math
from typing import Any

import dolfin
import pytest
from dolfin_mech.kinematics import Kinematics

from hill_playground.materials.orientation import display

MESH = dolfin.UnitCubeMesh(1, 1, 1)
EZ, EX = dolfin.Constant((0.0, 0.0, 1.0)), dolfin.Constant((1.0, 0.0, 0.0))


def _mean(expr: Any) -> float:
	return dolfin.assemble(expr * dolfin.dx(domain=MESH))


def _unit(angle_from_xy_deg: float, sign: float = 1.0) -> Any:
	a = math.radians(angle_from_xy_deg)
	return dolfin.Constant((sign * math.cos(a), 0.0, sign * math.sin(a)))


@pytest.mark.parametrize("tilt", [0.0, 30.0, 90.0])
def test_tilt_and_axis_angle(tilt: float) -> None:
	"""Fibre in the xz-plane at angle a from xy: tilt = a, angle to x = a."""
	f = _unit(tilt)
	assert _mean(display.tilt_degrees(f, EZ)) == pytest.approx(tilt, abs=1e-10)
	assert _mean(display.axis_angle_degrees(f, EX)) == pytest.approx(tilt, abs=1e-10)


def test_head_tail_symmetry() -> None:
	"""Directions f and -f give the same tilt, angle and order."""
	f, g = _unit(30.0), _unit(30.0, sign=-1.0)
	for func in (display.tilt_degrees, display.axis_angle_degrees, display.order_parameter):
		assert _mean(func(f, EZ)) == pytest.approx(_mean(func(g, EZ)), abs=1e-12)


def test_order_parameter_bounds() -> None:
	"""S = 1 aligned, -1/2 orthogonal."""
	assert _mean(display.order_parameter(EX, EX)) == pytest.approx(1.0)
	assert _mean(display.order_parameter(EZ, EX)) == pytest.approx(-0.5)


def test_current_direction_follows_shear() -> None:
	"""Simple shear u = (g y, 0, 0): F e_y = (g, 1, 0)/norm."""
	u = dolfin.Function(dolfin.VectorFunctionSpace(MESH, "CG", 1))
	u.interpolate(dolfin.Expression(("0.5*x[1]", "0.0", "0.0"), degree=1))
	f = display.current_direction(Kinematics(U=u), dolfin.Constant((0.0, 1.0, 0.0)))
	norm = math.sqrt(1.25)
	assert _mean(f[0]) == pytest.approx(0.5 / norm)
	assert _mean(f[1]) == pytest.approx(1.0 / norm)
