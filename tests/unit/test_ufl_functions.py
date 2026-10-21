"""Unit tests of the shared UFL guards (floored, macaulay)."""

from typing import Callable

import dolfin
import pytest

from hill_playground.ufl_functions import floored, macaulay


@pytest.mark.parametrize(("x", "expected"), [(-1.0, 0.5), (0.5, 0.5), (2.0, 2.0)])
def test_floored(x: float, expected: float, mean: Callable[..., float]) -> None:
	"""max(x, x_f) with x_f = 0.5."""
	assert mean(floored(dolfin.Constant(x), 0.5)) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(("x", "expected"), [(-2.0, 0.0), (0.0, 0.0), (3.0, 3.0)])
def test_macaulay(x: float, expected: float, mean: Callable[..., float]) -> None:
	"""<x>_+ = max(x, 0)."""
	assert mean(macaulay(dolfin.Constant(x))) == pytest.approx(expected, rel=1e-12, abs=1e-12)
