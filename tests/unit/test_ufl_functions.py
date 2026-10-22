"""Unit tests of the shared UFL guards (piecewise, floored, capped, macaulay, macaulay_negative)."""

from typing import Callable

import dolfin
import pytest

from hill_playground.ufl_functions import capped, floored, macaulay, macaulay_negative, piecewise


@pytest.mark.parametrize(("x", "expected"), [(-1.0, 0.5), (0.5, 0.5), (2.0, 2.0)])
def test_floored(x: float, expected: float, mean: Callable[..., float]) -> None:
	"""max(x, x_f) with x_f = 0.5."""
	assert mean(floored(dolfin.Constant(x), 0.5)) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("x", "expected"), [(-2.0, 0.0), (0.0, 0.0), (3.0, 3.0)])
def test_macaulay(x: float, expected: float, mean: Callable[..., float]) -> None:
	"""<x>_+ = max(x, 0)."""
	assert mean(macaulay(dolfin.Constant(x))) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("x", "expected"), [(-1.0, -1.0), (0.5, 0.5), (2.0, 0.5)])
def test_capped(x: float, expected: float, mean: Callable[..., float]) -> None:
	"""min(x, x_c) with x_c = 0.5."""
	assert mean(capped(dolfin.Constant(x), 0.5)) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("x", "expected"), [(-2.0, -2.0), (0.0, 0.0), (3.0, 0.0)])
def test_macaulay_negative(x: float, expected: float, mean: Callable[..., float]) -> None:
	"""<x>_- = min(x, 0)."""
	assert mean(macaulay_negative(dolfin.Constant(x))) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("x", [-2.0, 0.0, 3.0])
def test_macaulay_split(x: float, mean: Callable[..., float]) -> None:
	"""X = <x>_+ + <x>_-."""
	value = dolfin.Constant(x)
	assert mean(macaulay(value) + macaulay_negative(value)) == pytest.approx(x, rel=1e-12, abs=1e-12)


def test_macaulay_split_derivative_at_zero(mean: Callable[..., float]) -> None:
	"""At x = 0: d<x>_+/dx = 0, d<x>_-/dx = 1 (one-sided UFL tangents, summing to 1)."""
	value = dolfin.Constant(0.0)
	assert mean(dolfin.diff(macaulay(value), value)) == pytest.approx(0.0, rel=1e-12, abs=1e-12)
	assert mean(dolfin.diff(macaulay_negative(value), value)) == pytest.approx(1.0, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("x", "expected"), [(-1.0, 10.0), (0.5, 10.0), (2.0, 20.0)])
def test_piecewise(x: float, expected: float, mean: Callable[..., float]) -> None:
	"""10 if x <= 0.5 (tie included), 20 above."""
	assert mean(piecewise(dolfin.Constant(x), 0.5, 10.0, 20.0)) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_piecewise_tangent_at_tie(mean: Callable[..., float]) -> None:
	"""At x = x_0 the UFL tangent is the lower branch's: d/dx piecewise(x, 0, 2x, 3x) = 2."""
	value = dolfin.Constant(0.0)
	tangent = dolfin.diff(piecewise(value, 0.0, 2.0 * value, 3.0 * value), value)
	assert mean(tangent) == pytest.approx(2.0, rel=1e-12, abs=1e-12)
