r"""Holmes--Mow permeability for large deformations of hydrated tissues (equations.tex, Remark rem:kphi; Holmes 1990).

.. math::
	\frac{k}{k_0} = \Big(\frac{J - J_c}{1 - J_{c,0}}\Big)^{\kappa_{HM}} \exp\Big(\frac{M}{2}(J^2 - 1)\Big),
	\qquad J_c = \Phi_{s,0} + \Phi_c, \quad J_{c,0} = \Phi_{s,0} + \Phi_{c,0}

:math:`(J-J_c)/(1-J_{c,0}) = \Phi_\ell/\Phi_{\ell,0}`: :math:`k = 0` exactly when :math:`\Phi_\ell = 0`.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import dolfin

from . import porosity


@dataclass(slots=True, frozen=True)
class Parameters:
	r"""Parameters.

	Attributes:
	    conductivity_0: :math:`k_0/\eta` (m^2/(Pa.s), or :math:`\hat K`).
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c` (unknown or data).
	    phi_c0: :math:`\Phi_{c,0}` of the reference state (float).
	    exponent: :math:`\kappa_{HM} \ge 0`.
	    stiffening: :math:`M \ge 0`.
	    fluid_floor: Lower bound of :math:`\Phi_\ell/\Phi_{\ell,0}` (Newton safeguard).
	"""

	conductivity_0: Any
	phi_s0: Any
	phi_c: Any
	phi_c0: float
	exponent: float = 2.0
	stiffening: float = 0.0
	fluid_floor: float = 1e-6


class Permeability:
	r""":math:`k(J, \Phi_c)` of Holmes--Mow.

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
		jac = self._kinematics.J
		fluid_0 = 1.0 - float(p.phi_s0) - p.phi_c0
		relative = porosity.floored(porosity.lagrangian_fluid_fraction(jac, p.phi_s0, p.phi_c) / fluid_0, p.fluid_floor)
		return relative**p.exponent * dolfin.exp(0.5 * p.stiffening * (jac**2 - 1.0))

	@cached_property
	def hydraulic_conductivity(self) -> Any:
		r""":math:`k/\eta = (k_0/\eta)\, k/k_0`."""
		return self._parameters.conductivity_0 * self.ratio

	@cached_property
	def tensor(self) -> Any:
		r""":math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`."""
		return porosity.pulled_back_conductivity(self.hydraulic_conductivity, self._kinematics)
