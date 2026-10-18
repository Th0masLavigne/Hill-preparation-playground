"""Unit tests of the post-run redimensionalization of XDMF geometry and time."""

from pathlib import Path
from typing import Any

import dolfin
import h5py
import numpy as np
import pytest

from hill_playground.io import xdmf_units


def _write(path: Path) -> None:
	mesh = dolfin.UnitCubeMesh(1, 1, 1)
	func = dolfin.Function(dolfin.FunctionSpace(mesh, "CG", 1), name="f")
	with dolfin.XDMFFile(str(path)) as xdmf:
		xdmf.parameters["functions_share_mesh"] = True
		for t in (0.0, 0.5):
			xdmf.write(func, t)


def _geometry_max(path: Path) -> float:
	values: list[float] = []
	with h5py.File(path.with_suffix(".h5"), "r") as h5:

		def collect(name: str, obj: Any) -> None:
			if name.endswith("mesh/geometry"):
				values.append(float(np.max(obj[...])))

		h5.visititems(collect)
	return max(values)


def test_scales_geometry_and_times_once(tmp_path: Path) -> None:
	"""X <- L0 X, t <- t0 t; a second call is a no-op."""
	xdmf = tmp_path / "res.xdmf"
	_write(xdmf)
	assert xdmf_units.redimensionalize_results(xdmf, length=50e-6, time=2.0)
	assert _geometry_max(xdmf) == pytest.approx(50e-6)
	assert 'Time Value="1.000000000000000e+00"' in xdmf.read_text()
	assert not xdmf_units.redimensionalize_results(xdmf, length=50e-6, time=2.0)
	assert _geometry_max(xdmf) == pytest.approx(50e-6)


def test_identity_does_nothing(tmp_path: Path) -> None:
	"""L0 = t0 = 1: files untouched."""
	xdmf = tmp_path / "res.xdmf"
	_write(xdmf)
	assert not xdmf_units.redimensionalize_results(xdmf, length=1.0, time=1.0)
	assert _geometry_max(xdmf) == pytest.approx(1.0)


def test_geometry_only_keeps_times(tmp_path: Path) -> None:
	"""Unit time scale: geometry scaled, time stamps unchanged."""
	xdmf = tmp_path / "res.xdmf"
	_write(xdmf)
	assert xdmf_units.redimensionalize_results(xdmf, length=2.0)
	assert _geometry_max(xdmf) == pytest.approx(2.0)
	assert 'Time Value="5.000000000000000e-01"' in xdmf.read_text()
