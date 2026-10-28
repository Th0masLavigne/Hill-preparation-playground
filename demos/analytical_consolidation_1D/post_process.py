r"""Terzaghi demos: read the QOIs, compare with the analytical solution, plot (no ``dolfin``).

The demos write, at every converged time step, the settlement :math:`u_x(0)` (QOI ``u0``) and the pressure at
:math:`n` probes :math:`x_i` on the axis of the column (QOIs ``p_0`` ... ``p_{n-1}``), all in SI. Errors per
time step (:class:`~terzaghi_analytics.TerzaghiCase` gives the exact solution):

.. math::
	e_p(t) = \frac{1}{p_0}\sqrt{\frac{1}{n}\sum_i \big(p_h(x_i, t) - p(x_i, t)\big)^2}, \qquad
	e_u(t) = \frac{|u_h(0, t) - u(0, t)|}{u_\infty}

The summary keeps :math:`T \ge 0.05`: earlier, the pressure drop at :math:`x = 0` is a boundary layer thinner
than one element, which no mesh of the demos resolves.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from terzaghi_analytics import TerzaghiCase  # noqa: E402

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6")
"""One colour per plotted time, fixed order."""
T_PLOTTED = (0.02, 0.1, 0.3, 0.6, 1.0)
"""Dimensionless times :math:`T` of the pressure profiles."""


def read_qois(path: Path, n_probes: int, time_scale: float = 1.0) -> pd.DataFrame:
	r"""QOI file of a demo, time in s.

	Args:
	    path: ``*-qois.dat`` written by ``dolfin_mech.core.TimeIntegrator``.
	    n_probes: Number of pressure probes.
	    time_scale: Factor bringing the solver time to s (:math:`t_0` if the solve runs in :math:`\hat t`).

	Returns:
	    Columns ``t`` (s), ``u0`` (m), ``p_0`` ... (Pa); the initial row :math:`t = 0` is dropped.
	"""
	names = ["t", "u0", *(f"p_{i}" for i in range(n_probes))]
	table = pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=names)
	table["t"] *= time_scale
	return table[table["t"] > 0.0].reset_index(drop=True)


@dataclass(frozen=True)
class Errors:
	r"""Errors against the analytical solution.

	Attributes:
	    history: Columns ``t`` (s), ``T``, ``pressure`` (:math:`e_p`), ``settlement`` (:math:`e_u`).
	    pressure: :math:`\max e_p` over :math:`T \ge T_{min}`.
	    settlement: :math:`\max e_u` over :math:`T \ge T_{min}`.
	    t_factor_min: :math:`T_{min}`.
	"""

	history: pd.DataFrame
	pressure: float
	settlement: float
	t_factor_min: float

	def __str__(self) -> str:
		"""One line for the console."""
		return (
			f"max error vs Terzaghi (T >= {self.t_factor_min:g}): pressure {self.pressure:.2e} p0, "
			f"settlement {self.settlement:.2e} u_inf"
		)


def compute_errors(
	case: TerzaghiCase, table: pd.DataFrame, probe_x: NDArray[np.float64], t_factor_min: float = 0.05
) -> Errors:
	r""":math:`e_p(t)`, :math:`e_u(t)` and their maxima over :math:`T \ge T_{min}`.

	Args:
	    case: Analytical solution.
	    table: Output of :func:`read_qois`.
	    probe_x: Probe abscissae (m), in the order of the ``p_i`` columns.
	    t_factor_min: :math:`T_{min}`.

	Returns:
	    The errors.
	"""
	columns = [f"p_{i}" for i in range(len(probe_x))]
	times = table["t"].to_numpy()
	pressure = [
		float(np.sqrt(np.mean((row - case.pressure(probe_x, t)) ** 2))) / case.load
		for t, row in zip(times, table[columns].to_numpy())
	]
	settlement = np.abs(table["u0"].to_numpy() - case.settlement(times)) / case.final_settlement
	history = pd.DataFrame({"t": times, "T": case.time_factor(times), "pressure": pressure, "settlement": settlement})
	kept = history[history["T"] >= t_factor_min]
	return Errors(
		history=history,
		pressure=float(kept["pressure"].max()),
		settlement=float(kept["settlement"].max()),
		t_factor_min=t_factor_min,
	)


def plot_results(
	case: TerzaghiCase, table: pd.DataFrame, probe_x: NDArray[np.float64], errors: Errors, path: Path, title: str
) -> Path:
	"""Pressure profiles, settlement and errors: FE (markers) against Terzaghi (lines).

	Args:
	    case: Analytical solution.
	    table: Output of :func:`read_qois`.
	    probe_x: Probe abscissae (m).
	    errors: Output of :func:`compute_errors`.
	    path: PNG file.
	    title: Figure title.

	Returns:
	    ``path``.
	"""
	x_um, x_fine = np.asarray(probe_x) * 1e6, np.linspace(0.0, case.length, 200)
	columns = [f"p_{i}" for i in range(len(probe_x))]
	fig, (ax_p, ax_u, ax_e) = plt.subplots(1, 3, figsize=(16, 4.6))

	for color, big_t in zip(COLORS, T_PLOTTED):
		row = table.iloc[int((table["t"] - case.t_start - big_t * case.consolidation_time).abs().argmin())]
		t_row = float(row["t"])
		ax_p.plot(x_fine * 1e6, case.pressure(x_fine, t_row), color=color, linewidth=1.5)
		ax_p.plot(x_um, row[columns].to_numpy(dtype=float), "o", color=color, markersize=5, markerfacecolor="none",
				  label=f"T = {float(case.time_factor(t_row)):.2f}")  # fmt: skip
	ax_p.set(xlabel="x (µm), drained and loaded at x = 0", ylabel="p (Pa)", title="Pressure profiles")

	t_fine = np.linspace(case.t_start, table["t"].max(), 400)
	ax_u.plot(t_fine, case.settlement(t_fine) * 1e9, color="#55544f", linewidth=1.5, label="Terzaghi")
	ax_u.plot(table["t"], table["u0"] * 1e9, "o", color=COLORS[0], markersize=4, markerfacecolor="none", label="FE")
	ax_u.set(xlabel="t (s)", ylabel=r"$u_x(0)$ (nm)", title="Settlement of the loaded face")

	history = errors.history
	ax_e.semilogy(history["T"], history["pressure"], color=COLORS[0], linewidth=1.5, label=r"pressure $e_p$")
	ax_e.semilogy(history["T"], history["settlement"], color=COLORS[1], linewidth=1.5, label=r"settlement $e_u$")
	ax_e.axvline(errors.t_factor_min, color="#8c8b85", linestyle=":", linewidth=1.0)
	ax_e.set(xlabel=r"$T = c_v (t - t_s)/L^2$", ylabel="relative error", title="Error vs Terzaghi")

	for ax in (ax_p, ax_u, ax_e):
		ax.grid(True, color="#e4e3df", linewidth=0.6)
		ax.legend(fontsize=8, frameon=False)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
	fig.suptitle(f"{title}\n{errors}", fontsize=11)
	fig.tight_layout()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=150)  # type:ignore
	plt.close(fig)
	return path
