"""Unit tests of the subdomain indicators (DG0 cell indicator, CG1 interior-vertex indicator)."""

import dolfin
import numpy as np
import pytest

from hill_playground.materials import domain_fields

INSIDE, OUTSIDE, N = 1, 2, 4


@pytest.fixture(name="tagged")
def fixture_tagged() -> tuple[dolfin.Mesh, dolfin.MeshFunction]:
	"""Unit cube, cells with x_mid < 1/2 tagged INSIDE, the others OUTSIDE."""
	mesh = dolfin.UnitCubeMesh(N, N, N)
	domains = dolfin.MeshFunction("size_t", mesh, 3, OUTSIDE)
	for cell in dolfin.cells(mesh):
		if cell.midpoint().x() < 0.5:
			domains.set_value(cell.index(), INSIDE)
	return mesh, domains


def test_cell_indicator_measures_the_tagged_volume(tagged: tuple[dolfin.Mesh, dolfin.MeshFunction]) -> None:
	"""The integral of chi_T is the tagged volume 1/2; values in {0, 1}."""
	mesh, domains = tagged
	chi = domain_fields.cell_indicator(mesh, domains, [INSIDE])
	assert dolfin.assemble(chi * dolfin.dx(domain=mesh)) == pytest.approx(0.5, rel=1e-12)
	assert set(np.unique(chi.vector().get_local())) <= {0.0, 1.0}
	assert chi(0.1, 0.5, 0.5) == pytest.approx(1.0, abs=1e-12)
	assert chi(0.9, 0.5, 0.5) == pytest.approx(0.0, abs=1e-12)


def test_interior_vertex_indicator_is_zero_on_the_interface(tagged: tuple[dolfin.Mesh, dolfin.MeshFunction]) -> None:
	"""chi_T^int = 1 for x < 1/2, 0 on the interface x = 1/2 and beyond."""
	mesh, domains = tagged
	chi = domain_fields.interior_vertex_indicator(mesh, domains, [INSIDE])
	x = chi.function_space().tabulate_dof_coordinates()[:, 0]
	values = chi.vector().get_local()
	np.testing.assert_allclose(values, (x < 0.5 - 1e-12).astype(float), rtol=0.0, atol=1e-12)
