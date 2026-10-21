"""Unit tests of ``ResidualRows``: per-row quadrature degree and normalization."""

from types import SimpleNamespace
from typing import Any

import dolfin
import pytest

from hill_playground.operators import ResidualRows

ROWS = ("displacement", "liquid_pressure", "eps_a")


def test_scale_factor_is_test_scale_times_value_over_normalization() -> None:
	"""scale_factor = s_k v / N_k; s_k = N_k = 1 for a row without settings."""
	rows = ResidualRows(ROWS, test_scales={"liquid_pressure": 3.0}, normalizations={"liquid_pressure": 4.0})
	assert rows.scale_factor("liquid_pressure", 2.0) == pytest.approx(1.5, rel=1e-12)
	assert rows.scale_factor("liquid_pressure") == pytest.approx(0.75, rel=1e-12)
	assert rows.scale_factor("eps_a", 3.0) == pytest.approx(3.0, rel=1e-12)


def test_measure_degree_overrides_default_for_its_row_only() -> None:
	"""Residual (rank 1): degree 2 on its row only, default elsewhere; the summed residual stays additive."""
	mesh = dolfin.UnitCubeMesh(2, 2, 2)
	x = dolfin.SpatialCoordinate(mesh)[0]
	test = dolfin.TestFunction(dolfin.FunctionSpace(mesh, "CG", 2))
	dv = dolfin.Measure("dx", domain=mesh, metadata={"quadrature_degree": 8})
	rows = ResidualRows(ROWS, quadrature_degrees={"eps_a": 2})
	assert rows.measure(dv, "displacement") is dv
	exact = dolfin.assemble(x**6 * test * rows.measure(dv, "displacement")).get_local()
	low = dolfin.assemble(x**6 * test * rows.measure(dv, "eps_a")).get_local()
	assert abs(exact - low).max() > 1e-6
	both = dolfin.assemble(x**6 * test * rows.measure(dv, "displacement") + x**8 * test * rows.measure(dv, "eps_a"))
	low8 = dolfin.assemble(x**8 * test * rows.measure(dv, "eps_a")).get_local()
	assert abs(both.get_local() - exact - low8).max() < 1e-14


def test_from_problem_uses_subsol_names() -> None:
	"""Rows = names of problem.subsols, in order."""
	problem = SimpleNamespace(subsols=[SimpleNamespace(name=name) for name in ROWS])
	rows = ResidualRows.from_problem(
		problem,
		quadrature_degrees={"eps_a": 2},
		test_scales={"displacement": 5.0},
		normalizations={"displacement": 2.0},
	)
	assert rows.rows == ROWS
	assert rows.quadrature_degrees["eps_a"] == 2
	assert rows.scale_factor("displacement") == pytest.approx(2.5, rel=1e-12)
	assert ResidualRows.from_problem(problem).normalizations == {}
	assert ResidualRows.from_problem(problem).test_scales == {}


@pytest.mark.parametrize(
	"kwargs",
	[
		{"quadrature_degrees": {"pressure": 2}},
		{"normalizations": {"pressure": 1.0}},
		{"quadrature_degrees": {"eps_a": -1}},
		{"quadrature_degrees": {"eps_a": 2.0}},
		{"quadrature_degrees": {"eps_a": True}},
		{"normalizations": {"eps_a": 0.0}},
		{"test_scales": {"pressure": 1.0}},
		{"test_scales": {"eps_a": -1.0}},
	],
)
def test_invalid_settings_raise(kwargs: dict[str, Any]) -> None:
	"""Unknown row, negative/non-integer degree, non-positive normalization."""
	with pytest.raises(ValueError):
		ResidualRows(ROWS, **kwargs)


def test_unknown_row_at_call_raises() -> None:
	"""A typo in a row name at call time is an error, not a silent default."""
	rows = ResidualRows(ROWS)
	with pytest.raises(ValueError):
		rows.scale_factor("pressure")
	with pytest.raises(ValueError):
		rows.measure(dolfin.dx, "pressure")


def test_settings_are_frozen() -> None:
	"""The mappings cannot be modified after construction."""
	settings = {"eps_a": 2}
	rows = ResidualRows(ROWS, quadrature_degrees=settings)
	settings["eps_a"] = 4
	assert rows.quadrature_degrees["eps_a"] == 2
	with pytest.raises(TypeError):
		rows.quadrature_degrees["eps_a"] = 3  # type: ignore[index]
