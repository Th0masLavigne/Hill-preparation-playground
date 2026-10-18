r"""Kozeny--Carman permeability (equations.tex, Remark rem:kphi; Carman 1937).

.. math::
	\frac{k}{k_0} = \frac{\phi_\ell^3/(1-\phi_\ell)^2}{\phi_{\ell,0}^3/(1-\phi_{\ell,0})^2},
	\qquad \phi_\ell = 1 - \frac{\Phi_{s,0}+\Phi_c}{J}, \qquad \phi_{\ell,0} = 1 - \Phi_{s,0} - \Phi_{c,0}

:math:`k \sim \phi_\ell^3 \to 0` at the compaction point :math:`J = J_c`.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from . import porosity


@dataclass(slots=True, frozen=True)
class Parameters:
	r"""Parameters.

	Attributes:
	    conductivity_0: :math:`k_0/\eta` at :math:`\phi_{\ell,0}` (m^2/(Pa.s), or :math:`\hat K`).
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c` (unknown or data).
	    phi_c0: :math:`\Phi_{c,0}` of the reference state (float).
	    porosity_floor: Lower bound of :math:`\phi_\ell` in the formula (Newton safeguard).
	"""

	conductivity_0: Any
	phi_s0: Any
	phi_c: Any
	phi_c0: float
	porosity_floor: float = 1e-6


class Permeability:
	r""":math:`k(\phi_\ell)` of Kozeny--Carman.

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
		r""":math:`k/k_0`."""
		p = self._parameters
		phi = porosity.floored(self.porosity, p.porosity_floor)
		phi_0 = 1.0 - float(p.phi_s0) - p.phi_c0
		return (phi**3 / (1.0 - phi) ** 2) / (phi_0**3 / (1.0 - phi_0) ** 2)

	@cached_property
	def hydraulic_conductivity(self) -> Any:
		r""":math:`k/\eta = (k_0/\eta)\, k/k_0`."""
		return self._parameters.conductivity_0 * self.ratio

	@cached_property
	def tensor(self) -> Any:
		r""":math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`."""
		return porosity.pulled_back_conductivity(self.hydraulic_conductivity, self._kinematics)
