"""Environment check: fenicsoft stack + playground are importable.

The generic stack (dolfin, nlopt, scipy, ...) is checked by fenicsoft ``tests/environment``.
"""

import importlib
import importlib.metadata
from pathlib import Path

import pytest

STAMP = Path.home() / ".fenicsoft_installed"


@pytest.mark.parametrize("mod_name", ["dolfin", "dolfin_mech", "fenicsoft", "hill_playground"])
def test_import(mod_name: str) -> None:
	"""Each module imports and exposes a non-empty version (printed).

	Args:
		mod_name: Importable module name.
	"""
	try:
		lib = importlib.import_module(mod_name)
	except ImportError as e:
		pytest.fail(f"{mod_name} cannot be imported: {e}")
	version = getattr(lib, "__version__", None) or importlib.metadata.version(mod_name)
	assert version, f"{mod_name} has no version"
	print(f"  {mod_name} version: {version}")


def test_fenicsoft_origin() -> None:
	"""Prints which fenicsoft ref/commit is installed (written by install_fenicsoft.sh)."""
	if not STAMP.exists():
		pytest.skip("fenicsoft not installed through .devcontainer/install_fenicsoft.sh")
	print("\n" + STAMP.read_text())


NEW_BRANCHES: tuple[str, ...] = ()
"""Packages that do not exist yet in fenicsoft (transfer plan); everything else must already exist."""


def test_layout_mirrors_fenicsoft() -> None:
	"""Every playground package exists in fenicsoft, or lies in a declared new branch.

	``hill_playground.<path>`` must map to an importable ``fenicsoft.<path>``, unless ``<path>``
	starts with one of :data:`NEW_BRANCHES` (whose parent must then exist in fenicsoft).
	"""
	import pkgutil

	import hill_playground

	for branch in NEW_BRANCHES:
		parent = branch.rpartition(".")[0]
		importlib.import_module(".".join(filter(None, ["fenicsoft", parent])))

	for info in pkgutil.walk_packages(hill_playground.__path__, "hill_playground."):
		if not info.ispkg:
			continue
		path = info.name.split(".", 1)[1]
		if any(path == b or path.startswith(b + ".") for b in NEW_BRANCHES):
			continue
		importlib.import_module(f"fenicsoft.{path}")


def test_quadrature_degree_per_integral() -> None:
	r"""FFC honours ``dx(degree=q)`` per integral in residuals and Jacobians (``ResidualRows``).

	A summed residual and its Jacobian, with one term at degree 8 and one at degree 2, equal the sums of
	the separately assembled terms. Not true for a scalar functional (rank 0), see ``ResidualRows``.
	"""
	import dolfin

	mesh = dolfin.UnitCubeMesh(2, 2, 2)
	x = dolfin.SpatialCoordinate(mesh)[0]  # type:ignore
	space = dolfin.FunctionSpace(mesh, "CG", 2)
	u, test = dolfin.Function(space), dolfin.TestFunction(space)
	u.interpolate(dolfin.Expression("1.0 + x[0]*x[1]", degree=2))
	high = x**4 * u**3 * test * dolfin.dx(domain=mesh, degree=8)
	low = x**6 * u * test * dolfin.dx(domain=mesh, degree=2)
	split = dolfin.assemble(high).get_local() + dolfin.assemble(low).get_local()
	exact_low = x**6 * u * test * dolfin.dx(domain=mesh, degree=8)
	assert abs(dolfin.assemble(low).get_local() - dolfin.assemble(exact_low).get_local()).max() > 1e-8  # degree acts
	assert abs(dolfin.assemble(high + low).get_local() - split).max() < 1e-14
	jac = dolfin.assemble(dolfin.derivative(high, u)).array() + dolfin.assemble(dolfin.derivative(low, u)).array()
	assert abs(dolfin.assemble(dolfin.derivative(high + low, u)).array() - jac).max() < 1e-14
