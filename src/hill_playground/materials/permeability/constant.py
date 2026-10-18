r"""Constant intrinsic permeability (Hypothesis h:frozen).

.. math::
	k = k_0, \qquad \mathbf K_L = \frac{k_0}{\eta} J\mathbf C^{-1}
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from . import porosity


@dataclass(slots=True, frozen=True)
class Parameters:
	r"""Parameters.

	Attributes:
	    conductivity_0: :math:`k_0/\eta` (m^2/(Pa.s), or dimensionless :math:`\hat K`).
	"""

	conductivity_0: Any


class Permeability:
	r""":math:`k/k_0 = 1`.

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
	def ratio(self) -> Any:
		""":math:`k/k_0 = 1`."""
		return 1.0

	@cached_property
	def hydraulic_conductivity(self) -> Any:
		r""":math:`k/\eta`."""
		return self._parameters.conductivity_0

	@cached_property
	def tensor(self) -> Any:
		r""":math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`."""
		return porosity.pulled_back_conductivity(self.hydraulic_conductivity, self._kinematics)
