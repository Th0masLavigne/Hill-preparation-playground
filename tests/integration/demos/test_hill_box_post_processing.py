r"""Integration: orientation post-processing of the Hill box on known direction sets (no ``dolfin``).

Order tensor :math:`\mathbf Q = \langle \tfrac32 \mathbf f\otimes\mathbf f - \tfrac12\mathbf I\rangle` on three
reference sets (isotropic, all along :math:`x`, isotropic in the :math:`xy` plane), equal-area property of the
projection, and a full run of ``main`` on a two-frame synthetic results pair.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray

DEMO = Path(__file__).resolve().parents[3] / "demos" / "hill_box"
sys.path.insert(0, str(DEMO))

import post_processing as pp  # noqa: E402

N = 4000


def _fibonacci_sphere(n: int) -> NDArray[np.float64]:
	"""Quasi-uniform unit vectors on the sphere."""
	k = np.arange(n) + 0.5
	z = 1.0 - 2.0 * k / n
	angle = np.pi * (1.0 + 5.0**0.5) * k
	r = np.sqrt(1.0 - z**2)
	return np.stack((r * np.cos(angle), r * np.sin(angle), z), axis=1)


def _uniform(n: int) -> NDArray[np.float64]:
	return np.full(n, 1.0 / n)


def test_isotropic_set() -> None:
	"""Uniform on the sphere: Q = 0, isotropic, mean tilt out of xy = pi/2 - 1 rad = 32.70 deg."""
	m = pp.order_measures(_fibonacci_sphere(N), _uniform(N))
	assert m.state == "isotropic"
	assert np.allclose(m.q, 0.0, rtol=0.0, atol=2e-3)
	assert m.mean_tilt_xy_deg == pytest.approx(90.0 - 180.0 / np.pi, abs=0.05)


def test_aligned_set() -> None:
	"""All along +-x: prolate, S = Q_xx = 1, Q_zz = -1/2, director e_x."""
	f = np.zeros((N, 3))
	f[:, 0] = np.where(np.arange(N) % 2, 1.0, -1.0)
	m = pp.order_measures(f, _uniform(N))
	assert m.state == "prolate"
	assert m.order == pytest.approx(1.0, rel=1e-12)
	assert (m.q_xx, m.q_zz) == pytest.approx((1.0, -0.5), rel=1e-12)
	assert np.allclose(m.director, [1.0, 0.0, 0.0], rtol=0.0, atol=1e-10)


def test_planar_set() -> None:
	"""Isotropic in the xy plane: oblate, S = -2 lambda_3 = 1, director e_z, no tilt."""
	angle = np.linspace(0.0, np.pi, N, endpoint=False)
	f = np.stack((np.cos(angle), np.sin(angle), np.zeros(N)), axis=1)
	m = pp.order_measures(f, _uniform(N))
	assert m.state == "oblate"
	assert m.order == pytest.approx(1.0, rel=1e-12)
	assert np.allclose(m.director, [0.0, 0.0, 1.0], rtol=0.0, atol=1e-10)
	assert m.q_zz == pytest.approx(-0.5, rel=1e-12)
	assert m.mean_tilt_xy_deg == pytest.approx(0.0, abs=1e-12)


def test_head_tail_invariance() -> None:
	"""Flipping any subset of the directions leaves Q unchanged."""
	f = _fibonacci_sphere(N)[:, [2, 0, 1]] * np.array([1.0, 0.3, 0.3])
	f = pp.unit_directions(f)
	flipped = np.where((np.arange(N) % 3 == 0)[:, None], -f, f)
	assert np.allclose(pp.order_tensor(f, _uniform(N)), pp.order_tensor(flipped, _uniform(N)), rtol=0.0, atol=1e-14)


def test_lambert_projection_is_equal_area() -> None:
	"""Isotropic set: uniform on the unit disc, so the mean of r^2 is 1/2; e_z at the centre, xy on the rim."""
	xy = pp.lambert_upper_hemisphere(_fibonacci_sphere(N))
	r2 = np.sum(xy**2, axis=1)
	assert r2.max() <= 1.0 + 1e-12
	assert r2.mean() == pytest.approx(0.5, abs=2e-3)
	ends = pp.lambert_upper_hemisphere(np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]))
	assert np.allclose(ends, [[0.0, 0.0], [0.0, 1.0]], rtol=0.0, atol=1e-12)


def _write_results(folder: Path, frames: list[NDArray[np.float64]]) -> Path:
	"""Two-tetrahedron mesh, one ``fibre`` and ``fibre_current`` field per frame, dolfin's XDMF layout."""
	geometry = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
	topology = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.uint64)
	grids = []
	with h5py.File(folder / "res.h5", "w") as h5:
		h5["/Mesh/0/mesh/geometry"], h5["/Mesh/0/mesh/topology"] = geometry, topology
		for k, f in enumerate(frames):
			h5[f"/VisualisationVector/{2 * k}"], h5[f"/VisualisationVector/{2 * k + 1}"] = f, f
			attributes = "".join(
				f'<Attribute Name="{name}" AttributeType="Vector" Center="Cell"><DataItem Format="HDF">'
				f"res.h5:/VisualisationVector/{2 * k + j}</DataItem></Attribute>"
				for j, name in enumerate(("fibre", "fibre_current"))
			)
			grids.append(
				'<Grid Name="mesh" GridType="Uniform"><Topology TopologyType="Tetrahedron">'
				'<DataItem Format="HDF">res.h5:/Mesh/0/mesh/topology</DataItem></Topology><Geometry GeometryType="XYZ">'
				f'<DataItem Format="HDF">res.h5:/Mesh/0/mesh/geometry</DataItem></Geometry><Time Value="{3600.0 * k}"/>'
				f"{attributes}</Grid>"
			)
	xdmf = folder / "res.xdmf"
	xdmf.write_text(
		'<?xml version="1.0"?><Xdmf Version="3.0"><Domain><Grid Name="TimeSeries" GridType="Collection" '
		f'CollectionType="Temporal">{"".join(grids)}</Grid></Domain></Xdmf>'
	)
	return xdmf


def test_main_on_synthetic_results(tmp_path: Path) -> None:
	"""Tetrahedra of volumes 1/6 and 1/3 (weights 1/3 and 2/3), measures at both frames, three figures written."""
	xdmf = _write_results(tmp_path, [np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), np.array([[2.0, 0, 0], [1, 0, 0]])])
	series = pp.ResultsSeries(xdmf)
	assert series.times == [0.0, 3600.0]
	assert series.cell_weights() == pytest.approx([1.0 / 3.0, 2.0 / 3.0], rel=1e-12)
	paths = pp.main(["--results", str(xdmf), "--output-dir", str(tmp_path / "figures")])
	assert all(path.stat().st_size > 0 for path in paths.values())
