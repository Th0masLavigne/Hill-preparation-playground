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


def test_layout_mirrors_fenicsoft() -> None:
	"""Every playground sub-package parent exists in fenicsoft (1:1 transfer).

	For ``hill_playground.a.b.c``, ``fenicsoft.a.b`` must be importable.
	"""
	import pkgutil

	import hill_playground

	for info in pkgutil.walk_packages(hill_playground.__path__, "hill_playground."):
		parts = info.name.split(".")[1:-1]
		if parts:
			importlib.import_module(".".join(["fenicsoft", *parts]))
