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
			raise ValueError(f"exponent = {self.exponent} < 1: J = J_c would be reached in finite time.")


class Permeability(PorousPermeability[Parameters]):
	r""":math:`k = k_0 (\Phi_\ell/\Phi_{\ell,0})^n`."""

	def _ratio(self) -> Any:
		r""":math:`k/k_0 = (\Phi_\ell/\Phi_{\ell,0})^n`."""
		p = self._parameters
		return floored(self.fluid_ratio, p.fluid_floor) ** p.exponent
