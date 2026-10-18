r"""Bring a results XDMF/HDF5 pair written on a scaled mesh back to SI geometry and time.

When the solve uses :math:`\hat{\mathbf X} = \mathbf X/L_0` (and :math:`\hat t = t/t_0`), dolfin writes
the mesh, hence the ParaView geometry, in :math:`L_0` units and the time stamps in :math:`t_0`
units. After the run, on rank 0:

.. math::
	\mathbf X \leftarrow L_0\, \hat{\mathbf X} \quad (\texttt{Mesh/*/mesh/geometry}), \qquad
	t \leftarrow t_0\, \hat t \quad (\texttt{<Time Value>})

Fields are untouched (export the SI ones, e.g. ``displacement_si``). Idempotent: the HDF5 root gets
the attribute ``geometry_scale`` and a second call does nothing.
"""

import re
from pathlib import Path
from typing import Any

import h5py

_TIME = re.compile(r'(<Time Value=")([^"]+)(")')
_STAMP = "geometry_scale"


def redimensionalize_results(xdmf_path: Path, length: float, time: float = 1.0) -> bool:
	r"""Scale geometry by :math:`L_0` and time stamps by :math:`t_0`, in place.

	Args:
	    xdmf_path: Results ``.xdmf`` (its ``.h5`` must sit next to it, same stem).
	    length: :math:`L_0` (m).
	    time: :math:`t_0` (s); 1 leaves the times unchanged.

	Returns:
	    True if the files were changed, False if already done or nothing to do.
	"""
	if length == 1.0 and time == 1.0:
		return False
	h5_path = Path(xdmf_path).with_suffix(".h5")
	with h5py.File(h5_path, "r+") as h5:
		if _STAMP in h5.attrs:
			return False
		geometries: list[str] = []

		def collect(name: str, _obj: Any) -> None:
			if name.endswith("mesh/geometry"):
				geometries.append(name)

		h5.visititems(collect)
		for name in geometries:
			dataset = h5[name]
			assert isinstance(dataset, h5py.Dataset)
			dataset[...] = dataset[...] * length
		h5.attrs[_STAMP] = length
	if time != 1.0:
		text = Path(xdmf_path).read_text()
		text = _TIME.sub(lambda m: f"{m.group(1)}{float(m.group(2)) * time:.15e}{m.group(3)}", text)
		Path(xdmf_path).write_text(text)
	return True
