r"""Kozeny--Carman permeability (Carman 1937).

.. math::
	\frac{k}{k_0} = \frac{\phi_\ell^3/(1-\phi_\ell)^2}{\phi_{\ell,0}^3/(1-\phi_{\ell,0})^2},
	\qquad \phi_\ell = 1 - \frac{\Phi_{s,0}+\Phi_c}{J}, \qquad \phi_{\ell,0} = 1 - \Phi_{s,0} - \Phi_{c,0}

:math:`k \sim \phi_\ell^3 \to 0` at the compaction point :math:`J = J_c`.
"""

from dataclasses import dataclass
from typing import Any

from ..ufl_functions import floored
from .base import PorousPermeability, PorousPermeabilityParameters


@dataclass(slots=True, frozen=True)
class Parameters(PorousPermeabilityParameters):
	r"""Parameters (plus ``conductivity_0``, ``phi_s0``, ``phi_c``, ``phi_c0``).

	Attributes:
	    porosity_floor: Lower bound of :math:`\phi_\ell` in the formula (Newton safeguard).
	"""

	porosity_floor: float = 1e-6


class Permeability(PorousPermeability[Parameters]):
	r""":math:`k(\phi_\ell)` of Kozeny--Carman."""

	def _ratio(self) -> Any:
		r""":math:`k/k_0`."""
		phi = floored(self.porosity, self._parameters.porosity_floor)
		phi_0 = self.porosity_0
		return (phi**3 / (1.0 - phi) ** 2) / (phi_0**3 / (1.0 - phi_0) ** 2)
