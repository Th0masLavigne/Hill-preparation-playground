r"""Bring a results XDMF/HDF5 pair written on a scaled mesh back to SI geometry and time.

When the solve uses :math:`\hat{\mathbf X} = \mathbf X/L_0` (and :math:`\hat t = t/t_0`), dolfin writes
the mesh, hence the ParaView geometry, in :math:`L_0` units and the time stamps in :math:`t_0`
units. After the run, on rank 0:

.. math::
	\mathbf X \leftarrow L_0\, \hat{\mathbf X} \quad (\texttt{Mesh/*/mesh/geometry}), \qquad
	t \leftarrow t_0\, \hat t \quad (\texttt{<Time Value>})

Fields are untouched (export the SI ones, e.g. ``displacement_si``). Idempotent, step by step: the HDF5
root gets ``geometry_scale`` :math:`= L_0` (set with the geometry) and ``time_scale`` :math:`= t_0` (set
right after the XDMF is replaced), both recorded even when equal to 1. A second call, even after an
interrupted first one, never scales twice; :func:`is_redimensionalized` is True once both are set.
"""

import os
from pathlib import Path
from typing import Any

import h5py

from .xdmf_reading import TIME_VALUE

_GEOMETRY_STAMP = "geometry_scale"
_TIME_STAMP = "time_scale"


def is_redimensionalized(h5_path: Path) -> bool:
	"""True once both geometry and time stamps are in SI (both attributes set).

	Args:
	    h5_path: Results ``.h5``.

	Returns:
	    False also when the file cannot be opened (e.g. still written by the solver).
	"""
	try:
		with h5py.File(h5_path, "r") as h5:
			return _GEOMETRY_STAMP in h5.attrs and _TIME_STAMP in h5.attrs
	except OSError:
		return False


def _scale_geometry(h5: h5py.File, length: float) -> None:
	r""":math:`\mathbf X \leftarrow L_0 \mathbf X` on every ``mesh/geometry`` dataset."""
	geometries: list[str] = []

	def collect(name: str, _obj: Any) -> None:
		if name.endswith("mesh/geometry"):
			geometries.append(name)

	h5.visititems(collect)
	for name in geometries:
		dataset = h5[name]
		assert isinstance(dataset, h5py.Dataset)
		dataset[...] = dataset[...] * length


def _scale_times(xdmf_path: Path, time: float) -> None:
	r""":math:`t \leftarrow t_0 t` on every ``<Time Value>``; the file is replaced atomically."""
	text = TIME_VALUE.sub(lambda m: f"{m.group(1)}{float(m.group(2)) * time:.15e}{m.group(3)}", xdmf_path.read_text())
	tmp = xdmf_path.with_name(xdmf_path.name + ".tmp")
	tmp.write_text(text)
	os.replace(tmp, xdmf_path)


def redimensionalize_results(xdmf_path: Path, length: float, time: float = 1.0) -> bool:
	r"""Scale geometry by :math:`L_0` and time stamps by :math:`t_0`, in place.

	Args:
	    xdmf_path: Results ``.xdmf`` (its ``.h5`` must sit next to it, same stem).
	    length: :math:`L_0` (m); 1 leaves the geometry unchanged.
	    time: :math:`t_0` (s); 1 leaves the times unchanged.

	Returns:
	    True if the files were changed, False if already done or :math:`L_0 = t_0 = 1` (files untouched).
	"""
	if length == 1.0 and time == 1.0:
		return False
	xdmf_path = Path(xdmf_path)
	changed = False
	with h5py.File(xdmf_path.with_suffix(".h5"), "r+") as h5:
		if _GEOMETRY_STAMP not in h5.attrs:
			if length != 1.0:
				_scale_geometry(h5, length)
			h5.attrs[_GEOMETRY_STAMP] = length
			h5.flush()
			changed = True
		if _TIME_STAMP not in h5.attrs:
			if time != 1.0:
				_scale_times(xdmf_path, time)
			h5.attrs[_TIME_STAMP] = time
			h5.flush()
			changed = True
	return changed
