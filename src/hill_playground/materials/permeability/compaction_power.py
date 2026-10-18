r"""Compaction power law: permeability written on the Lagrangian fluid content :math:`\Phi_\ell = J - J_c`.

.. math::
	\frac{k}{k_0} = \Big(\frac{\Phi_\ell}{\Phi_{\ell,0}}\Big)^n = \Big(\frac{J - J_c}{1 - J_{c,0}}\Big)^n,
	\qquad J_c = \Phi_{s,0} + \Phi_c, \qquad n \ge 1

Why this form (equations.tex, Remark rem:kphi). Written on :math:`(J, \Phi_c)` it vanishes exactly at
the compaction point :math:`\phi_\ell = 0 \Leftrightarrow J = J_c`, with a controlled order :math:`n`.
Near :math:`J_c`, the drainage rate behaves like :math:`\dot J \sim -(J - J_c)^n`, so :math:`J = J_c`
is reached in finite time if and only if :math:`n < 1`: :math:`n \ge 1` keeps :math:`J > J_c` and
:math:`\int_0^\infty \dot J\,\mathrm dt` finite (kinetic barrier; the energetic one is
``materials.volumetric.compaction``). Holmes--Mow with :math:`M = 0` is the same law.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from . import porosity


@dataclass(slots=True, frozen=True)
class Parameters:
	r"""Parameters.

	Attributes:
	    conductivity_0: :math:`k_0/\eta` (m^2/(Pa.s), or :math:`\hat K`).
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c` (unknown or data).
	    phi_c0: :math:`\Phi_{c,0}` of the reference state (float).
	    exponent: :math:`n \ge 1`.
	    fluid_floor: Lower bound of :math:`\Phi_\ell/\Phi_{\ell,0}` (Newton safeguard).
	"""

	conductivity_0: Any
	phi_s0: Any
	phi_c: Any
	phi_c0: float
	exponent: float = 1.0
	fluid_floor: float = 1e-6

	def __post_init__(self) -> None:
		r"""Check :math:`n \ge 1`.

		Raises:
		    ValueError: ``exponent < 1`` (compaction reached in finite time).
		"""
		if self.exponent < 1.0:
			raise ValueError(f"exponent = {self.exponent} < 1: J = J_c would be reached in finite time.")


class Permeability:
	r""":math:`k = k_0 (\Phi_\ell/\Phi_{\ell,0})^n`.

	Args:
	    kinematics: Object exposing ``J`` and ``C_inv``.
	    parameters: :class:`Parameters`.
	"""

	def __init__(self, kinematics: Any, parameters: Parameters) -> None:
		"""Store the kinematics and the parameters."""
		self._kinematics = kinematics
		self._parameters = parameters

	@property
	def parameters(self) -> Parameters:
		"""Parameters."""
		return self._parameters

	@cached_property
	def porosity(self) -> Any:
		r""":math:`\phi_\ell = 1 - J_c/J`."""
		p = self._parameters
		return porosity.eulerian_porosity(self._kinematics.J, p.phi_s0, p.phi_c)

	@cached_property
	def ratio(self) -> Any:
		r""":math:`k/k_0 = (\Phi_\ell/\Phi_{\ell,0})^n`."""
		p = self._parameters
		fluid_0 = 1.0 - float(p.phi_s0) - p.phi_c0
		fluid = porosity.lagrangian_fluid_fraction(self._kinematics.J, p.phi_s0, p.phi_c)
		return porosity.floored(fluid / fluid_0, p.fluid_floor) ** p.exponent

	@cached_property
	def hydraulic_conductivity(self) -> Any:
		r""":math:`k/\eta = (k_0/\eta)\, k/k_0`."""
		return self._parameters.conductivity_0 * self.ratio

	@cached_property
	def tensor(self) -> Any:
		r""":math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`."""
		return porosity.pulled_back_conductivity(self.hydraulic_conductivity, self._kinematics)
