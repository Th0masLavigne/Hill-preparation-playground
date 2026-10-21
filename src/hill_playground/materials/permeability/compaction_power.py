r"""Compaction power law: permeability written on the Lagrangian fluid content :math:`\Phi_\ell = J - J_{\min}`.

.. math::
	\frac{k}{k_0} = \Big(\frac{\Phi_\ell}{\Phi_{\ell,0}}\Big)^n = \Big(\frac{J - J_{\min}}{1 - J_{\min,0}}\Big)^n,
	\qquad J_{\min} = \Phi_{s,0} + \Phi_c, \qquad n \ge 1

:math:`k = 0` at the compaction point :math:`J = J_{\min}`. Near it :math:`\dot J \sim -(J - J_{\min})^n`:
:math:`n \ge 1` keeps :math:`J > J_{\min}` for all time (kinetic barrier; energetic one:
:mod:`hill_playground.materials.volumetric.compaction`). Equals Holmes--Mow with :math:`M = 0`.
"""

from dataclasses import dataclass
from typing import Any

from ..ufl_functions import floored
from .base import PorousPermeability, PorousPermeabilityParameters


@dataclass(slots=True, frozen=True)
class Parameters(PorousPermeabilityParameters):
	r"""Parameters (plus ``conductivity_0``, ``phi_s0``, ``phi_c``, ``phi_c0``).

	Attributes:
	    exponent: :math:`n \ge 1`.
	    fluid_floor: Lower bound of :math:`\Phi_\ell/\Phi_{\ell,0}` (Newton safeguard).
	"""

	exponent: float = 1.0
	fluid_floor: float = 1e-6

	def __post_init__(self) -> None:
		r"""Check :math:`n \ge 1`.

		Raises:
		    ValueError: ``exponent < 1`` (compaction reached in finite time).
		"""
		if self.exponent < 1.0:
			raise ValueError(f"exponent = {self.exponent} < 1: J = J_min would be reached in finite time.")


class Permeability(PorousPermeability[Parameters]):
	r""":math:`k = k_0 (\Phi_\ell/\Phi_{\ell,0})^n`."""

	def _ratio(self) -> Any:
		r""":math:`k/k_0 = (\Phi_\ell/\Phi_{\ell,0})^n`."""
		p = self._parameters
		return floored(self.fluid_ratio, p.fluid_floor) ** p.exponent
