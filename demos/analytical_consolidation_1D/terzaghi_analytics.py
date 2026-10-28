r"""Terzaghi's 1D consolidation: analytical pressure and settlement (numpy only).

Column :math:`x \in [0, L]`, drained (:math:`p = 0`) and loaded by :math:`p_0` at :math:`x = 0`, impermeable and
clamped at :math:`x = L`, :math:`p(x, t_s) = p_0` (undrained response to the load applied at :math:`t_s`).
With :math:`M` the oedometric modulus, :math:`c_v = (k/\eta) M`, :math:`T = c_v (t - t_s)/L^2` and
:math:`m_j = (2j - 1)\pi/2`:

.. math::
	p(x, t) = \sum_{j \ge 1} \frac{2 p_0}{m_j} \sin\Big(m_j \frac{x}{L}\Big) e^{-m_j^2 T}, \qquad
	u_x(0, t) = \frac{p_0 L}{M}\Big[1 - \sum_{j \ge 1} \frac{2}{m_j^2} e^{-m_j^2 T}\Big]

:math:`T` is the dimensionless time of the problem: with :math:`t_0 = L^2/c_v`, :math:`T = \hat t - \hat t_s`.
Small strains only (:math:`p_0 \ll M`).
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class TerzaghiCase:
	r"""Data of the analytical solution (SI).

	Attributes:
	    length: Drainage length :math:`L` (m).
	    load: Applied load :math:`p_0` (Pa), also the initial pressure.
	    oedometric_modulus: :math:`M` (Pa).
	    hydraulic_conductivity: :math:`k/\eta` (m^2/(Pa.s)).
	    t_start: :math:`t_s` (s), time at which the drainage starts.
	    n_terms: Number of terms of the series.
	"""

	length: float
	load: float
	oedometric_modulus: float
	hydraulic_conductivity: float
	t_start: float = 0.0
	n_terms: int = 400

	@property
	def consolidation_coefficient(self) -> float:
		r""":math:`c_v = (k/\eta) M` (m^2/s)."""
		return self.hydraulic_conductivity * self.oedometric_modulus

	@property
	def consolidation_time(self) -> float:
		r""":math:`L^2/c_v` (s): :math:`T = 1` there."""
		return self.length**2 / self.consolidation_coefficient

	@property
	def final_settlement(self) -> float:
		r""":math:`u_\infty = p_0 L / M` (m)."""
		return self.load * self.length / self.oedometric_modulus

	def time_factor(self, t: ArrayLike) -> NDArray[np.float64]:
		r""":math:`T = c_v (t - t_s)/L^2`, floored at 0.

		Args:
		    t: Times (s).

		Returns:
		    :math:`T` per time.
		"""
		return np.maximum(np.asarray(t, dtype=float) - self.t_start, 0.0) / self.consolidation_time

	def _modes(self) -> NDArray[np.float64]:
		return (2.0 * np.arange(1, self.n_terms + 1) - 1.0) * math.pi / 2.0

	def pressure(self, x: ArrayLike, t: float) -> NDArray[np.float64]:
		r""":math:`p(x, t)` (Pa); :math:`p_0` everywhere for :math:`t \le t_s`.

		Args:
		    x: Abscissae (m), drained at 0.
		    t: Time (s).

		Returns:
		    Pressure at each ``x``.
		"""
		x = np.asarray(x, dtype=float)
		big_t = float(self.time_factor(t))
		if big_t <= 0.0:
			return np.full_like(x, self.load)
		m = self._modes()
		terms = 2.0 / m * np.sin(m * x[:, None] / self.length) * np.exp(-(m**2) * big_t)
		return self.load * terms.sum(axis=1)

	def settlement(self, t: ArrayLike) -> NDArray[np.float64]:
		r""":math:`u_x(0, t)` (m), displacement of the loaded face; 0 for :math:`t \le t_s`.

		Args:
		    t: Times (s).

		Returns:
		    Settlement at each time.
		"""
		big_t = np.atleast_1d(self.time_factor(t))
		m = self._modes()
		series = (2.0 / m**2 * np.exp(-(m**2) * big_t[:, None])).sum(axis=1)
		return self.final_settlement * np.where(big_t > 0.0, 1.0 - series, 0.0)  # truncated series != 0 at T = 0
