r"""Post-processing of the microtissue between pillars: cell orientation, :math:`\Phi_c` and :math:`p` in a region.

Reads the results XDMF/HDF5 pair and the QOI file of stage A or B (``--stage``, or ``--results``; ``numpy``,
``h5py``, ``matplotlib`` only, no ``dolfin``) and restricts every measure to a region of interest (ROI):

* ROI: gel cells (``gel_indicator`` = 1) whose centroid lies in the square of side ``--roi-side-um`` (default
  130 :math:`\mu m`, the central patch of Mostert et al.) centred on the gel (``--center-x-um``/``--center-y-um``,
  default the centre of the gel bounding box, i.e. of the core patch); ``--whole-domain``: every gel cell.
* Orientation: two unit directions per cell (DG0 fields), the reference fibre ``fibre``
  :math:`\mathbf f_0(\theta, \varphi)` (changed by remodelling only) and the current fibre ``fibre_current``
  :math:`\mathbf f = \mathbf F\mathbf f_0/\|\mathbf F\mathbf f_0\|` (also follows the deformation), cells weighted
  by :math:`w_e\,\mathbb 1_{ROI}/\sum`.
* :math:`\Phi_c` (``cell_fraction``) and :math:`p` (``pressure_si``): mean, min, max over the ROI nodes (vertices
  of the ROI cells; cell values for a DG0 field) at every frame.

Order tensor
------------
Volume-weighted average over the cells (:math:`w_e = |\Omega_e|/|\Omega|`, reference volumes):

.. math::
	\mathbf Q = \sum_e w_e \Big(\tfrac32\, \mathbf f_e \otimes \mathbf f_e - \tfrac12 \mathbf I\Big)

:math:`\mathbf Q` is symmetric, traceless and invariant under :math:`\mathbf f \to -\mathbf f` (no
head/tail ambiguity). With its eigenvalues :math:`\lambda_1 \ge \lambda_2 \ge \lambda_3`:

* ``isotropic`` if :math:`\lambda_1 <` ``isotropic_tol``;
* else ``biaxial`` if :math:`\beta \ge` ``biaxial_tol``: director :math:`\mathbf n` = eigenvector of
  :math:`\lambda_1`, :math:`S = \lambda_1`;
* else ``prolate`` (rods) if :math:`\lambda_1 - \lambda_2 \ge \lambda_2 - \lambda_3`: :math:`\mathbf n` = eigenvector
  of :math:`\lambda_1`, :math:`S = \lambda_1` (1: all along :math:`\mathbf n`);
* else ``oblate`` (disc): :math:`\mathbf n` = eigenvector of :math:`\lambda_3`, :math:`S = -2\lambda_3`
  (1: all in the plane normal to :math:`\mathbf n`);

with :math:`S \in [0, 1]`, 0 isotropic, and the biaxiality
:math:`\beta^2 = 1 - 6(\mathrm{tr}\,\mathbf Q^3)^2/(\mathrm{tr}\,\mathbf Q^2)^3`, 0 for any
uniaxial (prolate or oblate) state. The lab components read the tissue directly:
:math:`Q_{xx} \to 1` aligned with :math:`x` (axis of the core pillar rows), :math:`Q_{zz} \to -\frac12` all in the
:math:`xy` plane. For :math:`\mathbf f`, :math:`Q_{xx}`, :math:`Q_{zz}` and the mean tilt out of :math:`xy` are
the QOIs ``order_x``, ``order_z`` and ``mean_tilt_xy_deg``, averaged over the whole gel and drawn over the XDMF
values as a check (equal up to the cell-wise projection of :math:`\mathbf f` with ``--whole-domain`` only).

Figures (``--output-dir``, default ``<results>/../post_processing_roi`` or ``_gel``)
------------------------------------------------------------------------------------
* ``orientation_histograms.png``: first vs last frame, rows :math:`\mathbf f_0` and :math:`\mathbf f`, volume-weighted
  densities of the in-plane angle :math:`\chi = \operatorname{atan2}(f_y, f_x) \bmod 180°` (0: along :math:`x`,
  weight :math:`w_e (f_x^2 + f_y^2)`), of the tilt out of :math:`xy`, :math:`\beta_z = \arcsin|f_z|`, and of the
  angle :math:`\psi` to the director of the frame; the dotted line is the isotropic density.
* ``orientation_sphere.png``: equal-area (Lambert) projection of :math:`\pm\mathbf f` on the upper hemisphere,
  seen from :math:`+z`: centre = along :math:`z`, rim = in the :math:`xy` plane, :math:`x` to the
  right. Colour: density relative to the isotropic one (log scale, white = isotropic).
* ``orientation_time.png``: :math:`S`, :math:`Q_{xx}`, :math:`Q_{zz}` and the mean tilt out of :math:`xy` over the
  XDMF frames.

* ``phi_pressure_time.png``: mean and min--max band of :math:`\Phi_c` and :math:`p` over the region.

The XDMF times and geometry are SI once the run is over. During a run they are the solver times
:math:`\hat t`, converted with the QOI file (columns ``t`` and ``time_si``), and the coordinates
:math:`\hat{\mathbf X} = \mathbf X/L_0`, so the ROI is converted to :math:`L_0` units (``--length-scale-um``,
default :math:`L_0 = T = 50` µm); frames not yet flushed to the HDF5 file are skipped.

Usage::

	python3 pillars_post_processing.py [--stage a|b] [--results path.xdmf] [--whole-domain] [--roi-side-um 130]
		[--length-scale-um 50]
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # read while the solver writes

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from hill_playground.io import xdmf_units  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = {
	"a": HERE / "outputs_pillars_biaxial" / "results" / "pillars_biaxial.xdmf",
	"b": HERE / "outputs_pillars_uniaxial_release" / "results" / "pillars_uniaxial_release.xdmf",
}
"""Default results of each stage (``project_prefix`` and ``stem_name`` of their ``UserInputs``)."""
SCALARS: Dict[str, str] = {"cell_fraction": r"$\Phi_c$", "pressure_si": r"$p$ (Pa)"}
"""Scalar fields summarised over the ROI: XDMF name -> label."""
DIRECTIONS: Dict[str, str] = {"reference": "fibre", "current": "fibre_current"}
"""Analysed directions: label -> XDMF field (DG0 vector)."""
LABELS: Dict[str, str] = {"reference": r"reference $\mathbf{f}_0$", "current": r"current $\mathbf{f}$"}
COLORS: Dict[str, str] = {"first": "#2a78d6", "last": "#eb6834", "reference": "#2a78d6", "current": "#eb6834"}
MUTED = "#8c8b85"
DENSITY_CMAP = LinearSegmentedColormap.from_list("isotropic_ratio", ["#184f95", "#f0efec", "#e34948"])
"""Diverging map of the density relative to the isotropic one: blue below, white at 1, red above."""
_EPS = 1e-12


# --------------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------------


class ResultsSeries:
	"""Time series of a ``dolfin.XDMFFile`` results pair (``.xdmf`` + ``.h5``), read without ``dolfin``.

	Args:
	    xdmf_path: Results ``.xdmf``; its HDF5 file is the one named in the ``DataItem`` entries.

	Attributes:
	    xdmf_path: Results ``.xdmf``.
	    times: Time of each frame, ascending.

	Raises:
	    FileNotFoundError: No ``xdmf_path``.
	    ValueError: Unreadable XDMF (e.g. caught while the solver rewrites it: retry).
	"""

	def __init__(self, xdmf_path: Path) -> None:
		"""Parse the frames."""
		self.xdmf_path = Path(xdmf_path)
		if not self.xdmf_path.exists():
			raise FileNotFoundError(
				f"Results not found: {self.xdmf_path}. Run pillars_biaxial.py (or pillars_uniaxial_release.py) first."
			)
		try:
			root = ET.parse(self.xdmf_path).getroot()
		except ET.ParseError as error:
			raise ValueError(f"Unreadable {self.xdmf_path} (being rewritten by the solver?): retry.") from error
		grids = [g for g in root.iter("Grid") if g.find("Time") is not None]
		frames = [(float(g.find("Time").get("Value")), g) for g in grids]  # type: ignore[union-attr, arg-type]
		frames.sort(key=lambda frame: frame[0])
		self.times: List[float] = [t for t, _ in frames]
		self._grids: List[ET.Element] = [g for _, g in frames]

	@staticmethod
	def _reference(element: Optional[ET.Element]) -> Optional[Tuple[str, str]]:
		"""``(h5 file, dataset)`` of the ``DataItem`` of ``element``, or None."""
		item = None if element is None else element.find("DataItem")
		if item is None or not item.text:
			return None
		h5_name, _, dataset = item.text.strip().partition(":")
		return h5_name, dataset

	def _read(self, reference: Optional[Tuple[str, str]]) -> Optional[NDArray]:  # type: ignore
		"""Dataset of ``reference``, or None if absent (not yet flushed)."""
		if reference is None:
			return None
		try:
			with h5py.File(self.xdmf_path.parent / reference[0], "r") as h5:
				return np.asarray(h5[reference[1]])
		except (KeyError, OSError):
			return None

	def field(self, name: str, frame: int) -> Optional[NDArray[np.float64]]:
		"""Values of the field ``name`` at ``frame``.

		Args:
		    name: XDMF attribute name.
		    frame: Frame index (negative from the end).

		Returns:
		    Array ``(n_cells or n_nodes, n_components)``, or None if absent at that frame.
		"""
		grid = self._grids[frame]
		attribute = next((a for a in grid.findall("Attribute") if a.get("Name") == name), None)
		values = self._read(self._reference(attribute))
		return None if values is None else values.astype(np.float64).reshape(values.shape[0], -1)

	def is_redimensionalized(self) -> bool:
		"""True once ``io.xdmf_units`` has put the geometry and times back in SI."""
		reference = self._reference(self._grids[0].find("Geometry"))
		return reference is not None and xdmf_units.is_redimensionalized(self.xdmf_path.parent / reference[0])

	def mesh(self) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
		"""Mesh of the first frame.

		Returns:
		    ``(topology, geometry)``: vertex indices per cell ``(n_cells, 4)`` and coordinates ``(n_nodes, 3)``.

		Raises:
		    ValueError: No readable topology or geometry.
		"""
		grid = self._grids[0]
		topology = self._read(self._reference(grid.find("Topology")))
		geometry = self._read(self._reference(grid.find("Geometry")))
		if topology is None or geometry is None:
			raise ValueError(f"No mesh in the first frame of {self.xdmf_path}.")
		return topology.astype(np.int64), geometry.astype(np.float64)

	def cell_weights(self) -> NDArray[np.float64]:
		r"""Volume fractions :math:`w_e = |\Omega_e|/|\Omega|` of the tetrahedra (mesh of the first frame).

		Returns:
		    Shape ``(n_cells,)``, summing to 1 (independent of the length unit of the geometry).

		Raises:
		    ValueError: No readable topology or geometry.
		"""
		topology, geometry = self.mesh()
		vertices = geometry[topology]
		edges = vertices[:, 1:, :] - vertices[:, :1, :]
		volumes = np.abs(np.linalg.det(edges)) / 6.0
		return volumes / volumes.sum()


def read_qois(dat_path: Path) -> Optional[Dict[str, NDArray[np.float64]]]:
	"""QOI columns of the ``.dat`` file (header line starting with ``#``).

	Args:
	    dat_path: QOI file.

	Returns:
	    Column name -> values, or None if the file is missing or empty.
	"""
	if not dat_path.exists():
		return None
	with dat_path.open() as file:
		names = file.readline().lstrip("#").split()
	data = np.loadtxt(dat_path, comments="#", ndmin=2)
	if data.shape[0] == 0:
		return None
	return {name: data[:, k] for k, name in enumerate(names[: data.shape[1]])}


# --------------------------------------------------------------------------------------------------
# Orientation measures
# --------------------------------------------------------------------------------------------------


def unit_directions(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
	r""":math:`\mathbf f / \|\mathbf f\|` (a cell-wise projection of a unit field is slightly shorter than 1).

	Args:
	    vectors: Shape ``(n, 3)``.

	Returns:
	    Shape ``(n, 3)``; a zero vector stays zero.
	"""
	norms = np.linalg.norm(vectors, axis=1, keepdims=True)
	return vectors / np.maximum(norms, _EPS)


def order_tensor(directions: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
	r""":math:`\mathbf Q = \sum_e w_e (\tfrac32 \mathbf f_e\otimes\mathbf f_e - \tfrac12\mathbf I)`.

	Args:
	    directions: Unit vectors, shape ``(n, 3)``.
	    weights: :math:`w_e`, shape ``(n,)``, summing to 1.

	Returns:
	    Shape ``(3, 3)``, symmetric and traceless.
	"""
	return 1.5 * np.einsum("n,ni,nj->ij", weights, directions, directions) - 0.5 * np.eye(3)


def biaxiality(eigenvalues: NDArray[np.float64]) -> float:
	r""":math:`\beta = \sqrt{1 - 6(\mathrm{tr}\,\mathbf Q^3)^2/(\mathrm{tr}\,\mathbf Q^2)^3} \in [0, 1]`.

	Args:
	    eigenvalues: Eigenvalues of :math:`\mathbf Q`.

	Returns:
	    0 for a uniaxial state (and for :math:`\mathbf Q = 0`), 1 for the most biaxial one.
	"""
	tr_q2, tr_q3 = float(np.sum(eigenvalues**2)), float(np.sum(eigenvalues**3))
	if tr_q2 < _EPS:
		return 0.0
	return float(np.sqrt(max(0.0, 1.0 - 6.0 * tr_q3**2 / tr_q2**3)))


@dataclass(frozen=True)
class OrderMeasures:
	r"""Orientation measures of one set of directions (module docstring, "Order tensor").

	Attributes:
	    q: :math:`\mathbf Q`.
	    eigenvalues: :math:`\lambda_1 \ge \lambda_2 \ge \lambda_3`.
	    state: ``"isotropic"``, ``"prolate"``, ``"oblate"`` or ``"biaxial"``.
	    order: :math:`S \in [0, 1]`.
	    director: :math:`\mathbf n` (unit, :math:`n_z \ge 0`, then :math:`n_x \ge 0`, then :math:`n_y \ge 0`).
	    biaxiality: :math:`\beta` (0 when ``"isotropic"``).
	    mean_tilt_xy_deg: :math:`\sum_e w_e \beta_{z,e}` (degrees).
	"""

	q: NDArray[np.float64]
	eigenvalues: NDArray[np.float64]
	state: str
	order: float
	director: NDArray[np.float64]
	biaxiality: float
	mean_tilt_xy_deg: float

	@property
	def q_xx(self) -> float:
		r""":math:`Q_{xx}`: 1 along :math:`x`, 0 isotropic, :math:`-\frac12` normal to it."""
		return float(self.q[0, 0])

	@property
	def q_zz(self) -> float:
		r""":math:`Q_{zz}`: :math:`-\frac12` for fibres all in the :math:`xy` plane."""
		return float(self.q[2, 2])


def order_measures(
	directions: NDArray[np.float64],
	weights: NDArray[np.float64],
	isotropic_tol: float = 0.05,
	biaxial_tol: float = 0.15,
) -> OrderMeasures:
	r"""Order tensor, state, :math:`S`, director and mean tilt (module docstring, "Order tensor").

	Args:
	    directions: Unit vectors, shape ``(n, 3)``.
	    weights: :math:`w_e`, summing to 1.
	    isotropic_tol: Below this :math:`\lambda_1`, ``"isotropic"`` (:math:`\beta` is then noise).
	    biaxial_tol: From this :math:`\beta` on, ``"biaxial"``.

	Returns:
	    The measures.
	"""
	q = order_tensor(directions, weights)
	values, vectors = np.linalg.eigh(q)
	values, vectors = values[::-1], vectors[:, ::-1]
	beta = biaxiality(values)
	if values[0] < isotropic_tol:
		state, beta = "isotropic", 0.0  # beta, a ratio, is only noise when Q -> 0
	elif beta >= biaxial_tol:
		state = "biaxial"
	else:
		state = "prolate" if values[0] - values[1] >= values[1] - values[2] else "oblate"
	if state == "oblate":
		director, order = vectors[:, 2], float(-2.0 * values[2])
	else:
		director, order = vectors[:, 0], float(values[0])
	for k in (2, 0, 1):  # sign convention only
		if abs(director[k]) > _EPS:
			director = director if director[k] > 0.0 else -director
			break
	return OrderMeasures(
		q=q,
		eigenvalues=values,
		state=state,
		order=order,
		director=director,
		biaxiality=beta,
		mean_tilt_xy_deg=float(np.sum(weights * tilt_xy_deg(directions))),
	)


def in_plane_angle_deg(directions: NDArray[np.float64]) -> NDArray[np.float64]:
	r""":math:`\chi = \operatorname{atan2}(f_y, f_x) \bmod 180°`, in :math:`[0, 180)` (0: along :math:`x`)."""
	return np.mod(np.degrees(np.arctan2(directions[:, 1], directions[:, 0])), 180.0)


def tilt_xy_deg(directions: NDArray[np.float64]) -> NDArray[np.float64]:
	r""":math:`\beta_z = \arcsin|f_z| \in [0, 90°]` (0: in the :math:`xy` plane)."""
	return np.degrees(np.arcsin(np.clip(np.abs(directions[:, 2]), 0.0, 1.0)))


def angle_to_deg(directions: NDArray[np.float64], axis: NDArray[np.float64]) -> NDArray[np.float64]:
	r""":math:`\psi = \arccos|\mathbf f\cdot\mathbf n| \in [0, 90°]`."""
	return np.degrees(np.arccos(np.clip(np.abs(directions @ axis), 0.0, 1.0)))


def lambert_upper_hemisphere(directions: NDArray[np.float64]) -> NDArray[np.float64]:
	r"""Equal-area projection of :math:`\pm\mathbf f` (sign with :math:`f_z \ge 0`) onto the unit disc.

	.. math::
		(X, Y) = \frac{(f_x, f_y)}{\sqrt{1 + f_z}}, \qquad X^2 + Y^2 = 1 - f_z

	Centre: :math:`\mathbf f = \mathbf e_z`; rim: :math:`\mathbf f` in the :math:`xy` plane. An isotropic set
	has a uniform density on the disc.

	Args:
	    directions: Unit vectors, shape ``(n, 3)``.

	Returns:
	    Shape ``(n, 2)``.
	"""
	upper = np.where(directions[:, 2:3] < 0.0, -directions, directions)
	return upper[:, :2] / np.sqrt(1.0 + upper[:, 2:3])


# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------


def format_time(seconds: float) -> str:
	"""``t`` in s, min or h, whichever reads best."""
	if seconds < 600.0:
		return f"{seconds:.0f} s"
	if seconds < 3 * 3600.0:
		return f"{seconds / 60.0:.0f} min"
	return f"{seconds / 3600.0:.1f} h"


def _style(ax: plt.Axes) -> None:
	"""Recessive frame and grid."""
	ax.grid(True, color="#e4e3df", linewidth=0.6)
	ax.set_axisbelow(True)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(MUTED)
	ax.tick_params(colors="#55544f", labelsize=9)


def plot_histograms(
	frames: Dict[str, Tuple[float, Dict[str, NDArray[np.float64]], Dict[str, OrderMeasures]]],
	weights: NDArray[np.float64],
	output: Path,
	bins: int,
) -> None:
	r"""First vs last frame: densities of :math:`\chi`, :math:`\beta_z` and :math:`\psi` (module docstring).

	Args:
	    frames: ``"first"``/``"last"`` -> (time, directions per label, measures per label).
	    weights: :math:`w_e`.
	    output: PNG path.
	    bins: Number of bins of each histogram.
	"""
	fig, axes = plt.subplots(len(DIRECTIONS), 3, figsize=(14, 7.5), squeeze=False)
	for row, label in enumerate(DIRECTIONS):
		for col in range(3):
			ax = axes[row, col]
			_style(ax)
			span = 180.0 if col == 0 else 90.0
			edges = np.linspace(0.0, span, bins + 1)
			grid = np.linspace(0.0, span, 200)
			for key, (time, directions, measures) in frames.items():
				f, m = directions[label], measures[label]
				if col == 0:
					values, w = in_plane_angle_deg(f), weights * (f[:, 0] ** 2 + f[:, 1] ** 2)
				elif col == 1:
					values, w = tilt_xy_deg(f), weights
				else:
					values, w = angle_to_deg(f, m.director), weights
				text = f"{key}, t = {format_time(time)}"
				if col == 2:
					text += f"  (S = {m.order:.2f}, {m.state})"
				ax.hist(
					values,
					bins=edges,
					weights=w / max(w.sum(), _EPS),
					density=True,
					histtype="step",
					linewidth=2.0,
					color=COLORS[key],
					label=text,
				)
			isotropic = (
				np.full_like(grid, 1.0 / 180.0)
				if col == 0
				else np.radians(1.0) * (np.cos(np.radians(grid)) if col == 1 else np.sin(np.radians(grid)))
			)
			ax.plot(grid, isotropic, color=MUTED, linestyle=":", linewidth=1.5, label="isotropic")
			ax.set_xlim(0.0, span)
			ax.set_xticks(np.arange(0.0, span + 1.0, 30.0 if col == 0 else 15.0))
			ax.set_ylim(bottom=0.0)
			if col == 0:
				ax.set_ylabel(f"{LABELS[label]}\ndensity (1/deg)")
			if row == len(DIRECTIONS) - 1:
				ax.set_xlabel(
					(
						r"in-plane angle $\chi$ to $x$ (deg)",
						r"tilt $\beta_z$ out of the $xy$ plane (deg)",
						r"angle $\psi$ to the director (deg)",
					)[col]
				)
			ax.legend(fontsize=8, frameon=False, loc="upper right")
	fig.suptitle("Cell orientation: first vs last frame (volume-weighted)", fontsize=12)
	fig.tight_layout()
	fig.savefig(output, dpi=150)  # type: ignore
	plt.close(fig)


def plot_sphere(
	frames: Dict[str, Tuple[float, Dict[str, NDArray[np.float64]], Dict[str, OrderMeasures]]],
	weights: NDArray[np.float64],
	output: Path,
	bins: int,
) -> None:
	r"""Equal-area projections (:func:`lambert_upper_hemisphere`), density relative to the isotropic one.

	Each bin's weight is divided by its area inside the disc; bins less than 25 % inside are masked.

	Args:
	    frames: ``"first"``/``"last"`` -> (time, directions per label, measures per label).
	    weights: :math:`w_e`.
	    output: PNG path.
	    bins: Number of bins along each axis of the disc.
	"""
	edges = np.linspace(-1.0, 1.0, bins + 1)
	sub = np.linspace(-1.0, 1.0, 16 * bins + 1)
	sub = 0.5 * (sub[1:] + sub[:-1])  # 16 x 16 sub-samples per bin
	inside = (sub[None, :] ** 2 + sub[:, None] ** 2) <= 1.0
	coverage = inside.reshape(bins, 16, bins, 16).mean(axis=(1, 3))  # fraction of each bin in the disc
	outside = coverage < 0.25  # bins mostly outside the disc: too few samples
	bin_area = (edges[1] - edges[0]) ** 2 * np.maximum(coverage, 0.25)  # in-disc area of each bin
	norm = LogNorm(vmin=0.1, vmax=10.0)  # type: ignore  # matplotlib < 3.8: no stubs, decorated signature
	fig, axes = plt.subplots(len(DIRECTIONS), len(frames), figsize=(4.2 * len(frames) + 1.2, 8.4), squeeze=False)
	image = None
	circle = np.linspace(0.0, 2.0 * np.pi, 361)
	for row, label in enumerate(DIRECTIONS):
		for col, (key, (time, directions, measures)) in enumerate(frames.items()):
			ax = axes[row, col]
			xy = lambert_upper_hemisphere(directions[label])
			counts, _, _ = np.histogram2d(
				xy[:, 1], xy[:, 0], bins=bins, range=[[-1.0, 1.0], [-1.0, 1.0]], weights=weights
			)
			ratio = np.ma.masked_where(outside, np.maximum(counts, 1e-3 * bin_area / np.pi) * np.pi / bin_area)
			image = ax.pcolormesh(edges, edges, ratio, cmap=DENSITY_CMAP, norm=norm, shading="flat")
			ax.plot(np.cos(circle), np.sin(circle), color="#55544f", linewidth=1.0)
			ring = np.sqrt(1.0 - np.sin(np.radians(45.0)))  # tilt 45 deg out of xy
			ax.plot(ring * np.cos(circle), ring * np.sin(circle), color=MUTED, linewidth=0.6, linestyle=":")
			ax.plot([-1, 1], [0, 0], color=MUTED, linewidth=0.6, linestyle=":")
			ax.plot([0, 0], [-1, 1], color=MUTED, linewidth=0.6, linestyle=":")
			for text, (x, y), align in (
				("x", (1.06, 0.0), "left"),
				("y", (0.0, 1.06), "center"),
				("z", (0.06, -0.1), "left"),
			):
				box = {"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.8} if text == "z" else None
				ax.text(
					x,
					y,
					text,
					ha=align,
					va="bottom" if text == "y" else "center",
					fontsize=10,
					color="#55544f",
					bbox=box,
				)
			m = measures[label]
			ax.set_title(
				f"{LABELS[label]}, {key}: t = {format_time(time)}\nS = {m.order:.2f} ({m.state}), "
				f"$Q_{{xx}}$ = {m.q_xx:.2f}, $Q_{{zz}}$ = {m.q_zz:.2f}",
				fontsize=9,
			)
			ax.set_aspect("equal")
			ax.set_xlim(-1.15, 1.2)
			ax.set_ylim(-1.15, 1.2)
			ax.axis("off")
	bar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
	bar.set_label("density / isotropic density")
	fig.suptitle(
		"Equal-area projection of the fibre axes seen from +z (centre: along z, rim: in the xy plane, dotted: 45°)",
		fontsize=11,
	)
	fig.savefig(output, dpi=150, bbox_inches="tight")  # type: ignore
	plt.close(fig)


def plot_time(
	times: Sequence[float],
	history: Dict[str, List[OrderMeasures]],
	qois: Optional[Dict[str, NDArray[np.float64]]],
	output: Path,
) -> None:
	r""":math:`S`, :math:`Q_{xx}`, :math:`Q_{zz}`, mean tilt per frame; QOIs of :math:`\mathbf f` (whole gel) in gray.

	Args:
	    times: Frame times (s).
	    history: Label -> measures per frame.
	    qois: Columns of the ``.dat`` file, or None.
	    output: PNG path.
	"""
	hours = np.asarray(times) / 3600.0
	fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
	panels = (
		("order", "scalar order S", None),
		("q_xx", r"$Q_{xx}$ (1: along x)", "order_x"),
		("q_zz", r"$Q_{zz}$ ($-\frac{1}{2}$: all in the xy plane)", "order_z"),
		("mean_tilt_xy_deg", "mean tilt out of xy (deg)", "mean_tilt_xy_deg"),
	)
	qoi_hours = None
	if qois is not None and "time_si" in qois:
		qoi_hours = qois["time_si"] / 3600.0
	for ax, (attribute, title, qoi) in zip(axes, panels):
		_style(ax)
		for label, measures in history.items():
			ax.plot(
				hours,
				[getattr(m, attribute) for m in measures],
				color=COLORS[label],
				linewidth=2.0,
				marker="o",
				markersize=3,
				label=LABELS[label],
			)
		if qoi is not None and qoi_hours is not None and qoi in qois:  # type: ignore[operator]
			ax.plot(
				qoi_hours,
				qois[qoi],  # type: ignore
				color=MUTED,
				linewidth=1.0,
				linestyle="--",  # type: ignore[index]
				label=f"QOI {qoi} (current, whole gel)",
			)
		ax.set_title(title, fontsize=10)
		ax.set_xlabel("time (h)")
		ax.legend(fontsize=8, frameon=False)
	fig.tight_layout()
	fig.savefig(output, dpi=150)  # type: ignore
	plt.close(fig)


# --------------------------------------------------------------------------------------------------
# Region of interest
# --------------------------------------------------------------------------------------------------


def roi_cells(
	series: ResultsSeries,
	center_xy: Optional[Tuple[float, float]],
	side: Optional[float],
) -> NDArray[np.bool_]:
	"""Gel cells in the ROI (module docstring).

	Args:
	    series: Results.
	    center_xy: ROI centre, in the length unit of the geometry (m, or :math:`L_0` during a run); default the
	        centre of the bounding box of the gel cells.
	    side: ROI side, same unit; None: every gel cell.

	Returns:
	    Mask ``(n_cells,)``.

	Raises:
	    ValueError: No ``gel_indicator`` field, or an empty ROI.
	"""
	gel = series.field("gel_indicator", 0)
	if gel is None:
		raise ValueError(f"No gel_indicator in {series.xdmf_path} (written by pillars_biaxial.py).")
	in_gel = gel[:, 0] > 0.5
	topology, geometry = series.mesh()
	centroids = geometry[topology].mean(axis=1)
	mask = in_gel
	if side is not None:
		if center_xy is None:
			nodes = geometry[np.unique(topology[in_gel])]
			center_xy = tuple(0.5 * (nodes[:, :2].min(axis=0) + nodes[:, :2].max(axis=0)))  # type: ignore[assignment]
		half = side / 2.0
		inside = np.all(np.abs(centroids[:, :2] - np.asarray(center_xy)) <= half, axis=1)
		mask = in_gel & inside
	if not mask.any():
		raise ValueError("Empty region of interest.")
	return mask


def scalar_history(
	series: ResultsSeries, cells: NDArray[np.bool_], name: str
) -> Tuple[List[float], NDArray[np.float64]]:
	"""Mean, min and max of the field ``name`` over the ROI at every frame holding it.

	Args:
	    series: Results.
	    cells: ROI mask of the cells.
	    name: Scalar field (node or cell data).

	Returns:
	    ``(times, stats)`` with ``stats`` of shape ``(n_frames, 3)`` (mean, min, max).
	"""
	topology, geometry = series.mesh()
	nodes = np.zeros(geometry.shape[0], dtype=bool)
	nodes[np.unique(topology[cells])] = True
	times, stats = [], []
	for frame, time in enumerate(series.times):
		values = series.field(name, frame)
		if values is None:
			continue
		selected = values[nodes, 0] if values.shape[0] == nodes.size else values[cells, 0]
		times.append(time)
		stats.append((selected.mean(), selected.min(), selected.max()))
	return times, np.asarray(stats).reshape(-1, 3)


def plot_scalars(histories: Dict[str, Tuple[List[float], NDArray[np.float64]]], output: Path) -> None:
	"""Mean (line) and min--max band (shaded) of each scalar over time.

	Args:
	    histories: Field name -> (times, stats).
	    output: PNG path.
	"""
	fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4), squeeze=False)
	for ax, (name, (times, stats)) in zip(axes[0], histories.items()):
		_style(ax)
		hours = np.asarray(times) / 3600.0
		ax.fill_between(hours, stats[:, 1], stats[:, 2], color=COLORS["reference"], alpha=0.2, label="min--max")
		ax.plot(hours, stats[:, 0], color=COLORS["reference"], linewidth=2.0, label="mean")
		ax.set_xlabel("time (h)")
		ax.set_title(SCALARS[name], fontsize=10)
		ax.legend(fontsize=8, frameon=False)
	fig.tight_layout()
	fig.savefig(output, dpi=150)  # type: ignore
	plt.close(fig)


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------


def _report(label: str, time: float, measures: OrderMeasures) -> str:
	"""One table line."""
	n = measures.director
	return (
		f"{label:28s} {format_time(time):>9s}  {measures.state:9s} {measures.order:6.3f} {measures.biaxiality:6.3f} "
		f"{measures.q_xx:7.3f} {measures.q_zz:7.3f} {measures.mean_tilt_xy_deg:7.2f}   "
		f"({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f})"
	)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
	"""Read the results, restrict to the ROI, print the measures and write the figures.

	Args:
	    argv: Command-line arguments (``sys.argv[1:]`` when None).

	Returns:
	    Figure name -> path.

	Raises:
	    FileNotFoundError: No results file.
	    ValueError: Unreadable results, no ``gel_indicator``, empty region of interest, or no frame holds both
	        direction fields.
	    SystemExit: Only one of ``--center-x-um``, ``--center-y-um``.
	"""
	parser = argparse.ArgumentParser(description="ROI post-processing of the microtissue between pillars.")
	parser.add_argument("--stage", choices=sorted(RESULTS), default="a", help="Stage whose default results to read.")
	parser.add_argument("--results", type=Path, default=None, help="Results .xdmf (overrides --stage).")
	parser.add_argument("--output-dir", type=Path, default=None, help="Figure folder.")
	parser.add_argument("--whole-domain", action="store_true", help="Every gel cell (no square ROI).")
	parser.add_argument("--roi-side-um", type=float, default=130.0, help="ROI side (um).")
	parser.add_argument("--center-x-um", type=float, default=None, help="ROI centre x (um), default gel centre.")
	parser.add_argument("--center-y-um", type=float, default=None, help="ROI centre y (um), default gel centre.")
	parser.add_argument("--bins", type=int, default=36, help="Bins of the histograms.")
	parser.add_argument("--sphere-bins", type=int, default=30, help="Bins per axis of the projections.")
	parser.add_argument(
		"--length-scale-um", type=float, default=50.0, help="L0 (um) of a run in progress (geometry in L0 units)."
	)
	args = parser.parse_args(argv)
	if (args.center_x_um is None) != (args.center_y_um is None):
		parser.error("--center-x-um and --center-y-um go together.")

	results = args.results or RESULTS[args.stage]
	series = ResultsSeries(results)
	qois = read_qois(results.with_suffix(".dat"))
	in_progress = not series.is_redimensionalized()
	if in_progress and qois is not None and {"t", "time_si"} <= set(qois):
		series.times = list(np.interp(series.times, qois["t"], qois["time_si"]))  # run in progress: t_hat -> s
	unit = args.length_scale_um * 1e-6 if in_progress else 1.0  # geometry length unit: L0 during a run, else 1 m
	center = None
	if args.center_x_um is not None and args.center_y_um is not None:
		center = (args.center_x_um * 1e-6 / unit, args.center_y_um * 1e-6 / unit)
	cells = roi_cells(series, center, None if args.whole_domain else args.roi_side_um * 1e-6 / unit)
	weights = series.cell_weights() * cells
	weights /= weights.sum()

	history: Dict[str, List[OrderMeasures]] = {label: [] for label in DIRECTIONS}
	times: List[float] = []
	snapshots: Dict[int, Tuple[float, Dict[str, NDArray[np.float64]], Dict[str, OrderMeasures]]] = {}
	for frame, time in enumerate(series.times):
		directions = {label: series.field(name, frame) for label, name in DIRECTIONS.items()}
		if any(v is None or v.shape != (weights.size, 3) for v in directions.values()):
			continue  # not written (or not yet flushed) at this frame
		units = {label: unit_directions(v) for label, v in directions.items()}  # type: ignore[arg-type]
		measures = {label: order_measures(f, weights) for label, f in units.items()}
		times.append(time)
		for label in DIRECTIONS:
			history[label].append(measures[label])
		snapshots[frame] = (time, units, measures)
	if not snapshots:
		raise ValueError(f"No frame of {results} holds {sorted(DIRECTIONS.values())}.")
	frames = {"first": snapshots[min(snapshots)], "last": snapshots[max(snapshots)]}

	region = "gel" if args.whole_domain else "roi"
	output_dir = args.output_dir or results.parent.parent / f"post_processing_{region}"
	output_dir.mkdir(parents=True, exist_ok=True)
	paths = {
		"histograms": output_dir / "orientation_histograms.png",
		"sphere": output_dir / "orientation_sphere.png",
		"time": output_dir / "orientation_time.png",
	}
	plot_histograms(frames, weights, paths["histograms"], args.bins)
	plot_sphere(frames, weights, paths["sphere"], args.sphere_bins)
	plot_time(times, history, qois, paths["time"])

	print(f"{results}: {len(times)} frames, {int(np.count_nonzero(cells))} cells in the {region} (volume-weighted).")
	print(f"{'':28s} {'t':>9s}  {'state':9s} {'S':>6s} {'beta':>6s} {'Q_xx':>7s} {'Q_zz':>7s} {'tilt':>7s}   director")
	for key, (time, _, measures) in frames.items():
		for label in DIRECTIONS:
			print(_report(f"{key} {label}", time, measures[label]))

	histories = {name: scalar_history(series, cells, name) for name in SCALARS}
	histories = {name: h for name, h in histories.items() if h[0]}
	if histories:
		paths["phi_pressure"] = output_dir / "phi_pressure_time.png"
		plot_scalars(histories, paths["phi_pressure"])
		for name, (frame_times, stats) in histories.items():
			print(
				f"{name:14s} {region}: first mean {stats[0, 0]:.4g}, last mean {stats[-1, 0]:.4g} "
				f"(min {stats[-1, 1]:.4g}, max {stats[-1, 2]:.4g}) at t = {format_time(frame_times[-1])}"
			)
	for path in paths.values():
		print(f"Saved {path}")
	return paths


if __name__ == "__main__":
	main()
