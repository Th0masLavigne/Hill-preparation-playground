"""Unit tests of the post-run redimensionalization of XDMF geometry and time."""

from pathlib import Path
from typing import Any

import dolfin
import h5py
import numpy as np
import pytest

from hill_playground.io import xdmf_units
from hill_playground.io.xdmf_reading import read_time_stamps


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
	assert not xdmf_units.is_redimensionalized(xdmf.with_suffix(".h5"))
	assert xdmf_units.redimensionalize_results(xdmf, length=50e-6, time=2.0)
	assert xdmf_units.is_redimensionalized(xdmf.with_suffix(".h5"))
	assert _geometry_max(xdmf) == pytest.approx(50e-6, rel=1e-12)
	assert read_time_stamps(xdmf) == pytest.approx([0.0, 1.0], rel=1e-12, abs=1e-15)
	assert not xdmf_units.redimensionalize_results(xdmf, length=50e-6, time=2.0)
	assert _geometry_max(xdmf) == pytest.approx(50e-6, rel=1e-12)
	assert read_time_stamps(xdmf) == pytest.approx([0.0, 1.0], rel=1e-12, abs=1e-15)


def test_identity_does_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
	"""L0 = t0 = 1: files untouched, no stamp, and a warning says so."""
	xdmf = tmp_path / "res.xdmf"
	_write(xdmf)
	with caplog.at_level("WARNING", logger=xdmf_units.__name__):
		assert not xdmf_units.redimensionalize_results(xdmf, length=1.0, time=1.0)
	assert "no stamp written" in caplog.text
	assert not xdmf_units.is_redimensionalized(xdmf.with_suffix(".h5"))
	assert _geometry_max(xdmf) == pytest.approx(1.0, rel=1e-12)
	assert read_time_stamps(xdmf) == pytest.approx([0.0, 0.5], rel=1e-12, abs=1e-15)


def test_geometry_only_keeps_times(tmp_path: Path) -> None:
	"""Unit time scale: geometry scaled, time stamps unchanged."""
	xdmf = tmp_path / "res.xdmf"
	_write(xdmf)
	assert xdmf_units.redimensionalize_results(xdmf, length=2.0)
	assert xdmf_units.is_redimensionalized(xdmf.with_suffix(".h5"))
	assert not xdmf_units.redimensionalize_results(xdmf, length=2.0, time=3.0)
	assert _geometry_max(xdmf) == pytest.approx(2.0, rel=1e-12)
	assert read_time_stamps(xdmf) == pytest.approx([0.0, 0.5], rel=1e-12, abs=1e-15)


def test_time_only_keeps_geometry(tmp_path: Path) -> None:
	"""Unit length scale: time stamps scaled, geometry unchanged; a second call is a no-op."""
	xdmf = tmp_path / "res.xdmf"
	_write(xdmf)
	assert xdmf_units.redimensionalize_results(xdmf, length=1.0, time=4.0)
	assert not xdmf_units.redimensionalize_results(xdmf, length=1.0, time=4.0)
	assert _geometry_max(xdmf) == pytest.approx(1.0, rel=1e-12)
	assert read_time_stamps(xdmf) == pytest.approx([0.0, 2.0], rel=1e-12, abs=1e-15)


def test_resumes_after_interrupted_call(tmp_path: Path) -> None:
	"""Geometry already stamped, times not (interrupted run): only the times are scaled."""
	xdmf = tmp_path / "res.xdmf"
	_write(xdmf)
	assert xdmf_units.redimensionalize_results(xdmf, length=3.0, time=2.0)
	with h5py.File(xdmf.with_suffix(".h5"), "r+") as h5:
		del h5.attrs["time_scale"]
	xdmf.write_text(xdmf.read_text().replace('Time Value="1.000000000000000e+00"', 'Time Value="0.5"'))
	assert not xdmf_units.is_redimensionalized(xdmf.with_suffix(".h5"))
	assert xdmf_units.redimensionalize_results(xdmf, length=3.0, time=2.0)
	assert _geometry_max(xdmf) == pytest.approx(3.0, rel=1e-12)
	assert read_time_stamps(xdmf) == pytest.approx([0.0, 1.0], rel=1e-12, abs=1e-15)
	assert not (tmp_path / "res.xdmf.tmp").exists()


def test_unreadable_h5_is_not_redimensionalized(tmp_path: Path) -> None:
	"""Missing (or not yet readable) ``.h5``: False, no exception."""
	assert not xdmf_units.is_redimensionalized(tmp_path / "missing.h5")
