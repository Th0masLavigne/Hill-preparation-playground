r"""Constant intrinsic permeability (independent of the deformation and of the cell fraction).

.. math::
	k = k_0, \qquad \mathbf K_L = \frac{k_0}{\eta} J\mathbf C^{-1}
"""

from dataclasses import dataclass
from typing import Any

from .base import Permeability as BasePermeability
from .base import PermeabilityParameters


@dataclass(slots=True, frozen=True)
class Parameters(PermeabilityParameters):
	r"""Parameters: :math:`k_0/\eta` only (``conductivity_0``)."""


class Permeability(BasePermeability[Parameters]):
	r""":math:`k/k_0 = 1`."""

	def _ratio(self) -> Any:
		""":math:`k/k_0 = 1`."""
		return 1.0
