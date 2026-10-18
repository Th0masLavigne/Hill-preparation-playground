"""Environment check: fenicsoft stack + playground are importable.

The generic stack (dolfin, nlopt, scipy, ...) is checked by fenicsoft's own
``tests/environment``; here we only check what this workspace adds.
"""

import importlib
import importlib.metadata
from pathlib import Path

import pytest

STAMP = Path.home() / ".fenicsoft_installed"


@pytest.mark.parametrize("mod_name", ["dolfin", "dolfin_mech", "fenicsoft", "hill_playground"])
def test_import(mod_name: str) -> None:
	"""Each module imports and exposes a version.

	Args:
		mod_name: Importable module name.
	"""
	try:
		lib = importlib.import_module(mod_name)
	except ImportError as e:
		pytest.fail(f"{mod_name} cannot be imported: {e}")
	version = getattr(lib, "__version__", None) or importlib.metadata.version(mod_name)
	print(f"  {mod_name} version: {version}")


def test_fenicsoft_origin() -> None:
	"""Prints which fenicsoft ref/commit is installed (written by install_fenicsoft.sh)."""
	if not STAMP.exists():
		pytest.skip("fenicsoft not installed through .devcontainer/install_fenicsoft.sh")
	print("\n" + STAMP.read_text())


NEW_BRANCHES = ("materials.cell.active.hill", "materials.permeability", "operators.cell.hill", "scaling")
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
	r"""FFC honours ``dx(degree=q)`` per integral, also inside a summed form (demo ``quadrature_degrees``).

	:math:`\int_0^1 x^8` with degree 2 is inexact (0.0748 instead of 1/9); summed with an exact degree-8
	integral it must stay inexact, i.e. it is not promoted to the form's maximal degree.
	"""
	import dolfin

	mesh = dolfin.UnitIntervalMesh(1)
	x = dolfin.SpatialCoordinate(mesh)[0]
	low = x**8 * dolfin.dx(domain=mesh, degree=2)
	high = x * dolfin.dx(domain=mesh, degree=8)
	alone = dolfin.assemble(low)
	assert abs(alone - 1.0 / 9.0) > 1e-2
	assert abs(dolfin.assemble(low + high) - 0.5 - alone) < 1e-12
