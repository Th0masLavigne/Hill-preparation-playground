r"""Base classes of the permeability laws: each law only writes :math:`k/k_0` (``_ratio``).

.. math::
	\frac{k}{\eta} = \frac{k_0}{\eta}\,\frac{k}{k_0}, \qquad \mathbf K_L = \frac{k}{\eta} J \mathbf C^{-1}

Porosity-dependent laws (:class:`PorousPermeability`) also share

.. math::
	\phi_\ell = 1 - \frac{J_c}{J}, \qquad
	\frac{\Phi_\ell}{\Phi_{\ell,0}} = \frac{J - J_c}{1 - J_{c,0}}, \qquad
	\Phi_{\ell,0} = \phi_{\ell,0} = 1 - \Phi_{s,0} - \Phi_{c,0}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, TypeVar

import dolfin
from fenicsoft.materials.materials import MaterialParameters

from . import porosity


@dataclass(slots=True, frozen=True)
class PermeabilityParameters(MaterialParameters):
	r"""Parameters common to every law.

	Attributes:
	    conductivity_0: :math:`k_0/\eta` (m^2/(Pa.s), or dimensionless :math:`\hat K`).
	"""

	conductivity_0: Any


@dataclass(slots=True, frozen=True)
class PorousPermeabilityParameters(PermeabilityParameters):
	r"""Parameters of the porosity-dependent laws.

	Attributes:
	    phi_s0: :math:`\Phi_{s,0}`, uniform (float or ``dolfin.Constant``): it enters :math:`\Phi_{\ell,0}`.
	    phi_c: :math:`\Phi_c` (unknown or data).
	    phi_c0: :math:`\Phi_{c,0}` of the reference state.
	"""

	phi_s0: float | dolfin.Constant
	phi_c: Any
	phi_c0: float


T_Perm = TypeVar("T_Perm", bound=PermeabilityParameters, covariant=True)


class Permeability(ABC, Generic[T_Perm]):
	r"""Law :math:`k/k_0`; gives :math:`k/\eta` and :math:`\mathbf K_L`.

	Args:
	    kinematics: Object exposing ``J`` and ``C_inv``.
	    parameters: Parameters of the law.
	"""

	def __init__(self, kinematics: Any, parameters: T_Perm) -> None:
		"""Store the kinematics and the parameters."""
		self._kinematics = kinematics
		self._parameters = parameters

	@property
	def parameters(self) -> T_Perm:
		"""Parameters."""
		return self._parameters

	@abstractmethod
	def _ratio(self) -> Any:
		""":math:`k/k_0` of the law (UFL scalar or float)."""

	@cached_property
	def ratio(self) -> Any:
		r""":math:`k/k_0`: dimensionless factor applied to the reference conductivity :math:`k_0/\eta`."""
		return self._ratio()

	@cached_property
	def hydraulic_conductivity(self) -> Any:
		r""":math:`k/\eta = (k_0/\eta)\, k/k_0`."""
		return self._parameters.conductivity_0 * self.ratio

	@cached_property
	def tensor(self) -> Any:
		r""":math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`."""
		return porosity.pulled_back_conductivity(self.hydraulic_conductivity, self._kinematics)


T_Porous = TypeVar("T_Porous", bound=PorousPermeabilityParameters, covariant=True)


class PorousPermeability(Permeability[T_Porous]):
	r"""Law :math:`k/k_0` written on :math:`(J, \Phi_c)` through the porosity."""

	@cached_property
	def porosity(self) -> Any:
		r""":math:`\phi_\ell = 1 - J_c/J`."""
		p = self._parameters
		return porosity.eulerian_porosity(self._kinematics.J, p.phi_s0, p.phi_c)

	@cached_property
	def porosity_0(self) -> float:
		r""":math:`\phi_{\ell,0} = \Phi_{\ell,0} = 1 - \Phi_{s,0} - \Phi_{c,0}` (at :math:`J = 1`)."""
		p = self._parameters
		return 1.0 - float(p.phi_s0) - p.phi_c0

	@cached_property
	def fluid_ratio(self) -> Any:
		r""":math:`\Phi_\ell/\Phi_{\ell,0} = (J - J_c)/(1 - J_{c,0})` (not floored)."""
		p = self._parameters
		return porosity.lagrangian_fluid_fraction(self._kinematics.J, p.phi_s0, p.phi_c) / self.porosity_0
