r"""Holmes--Mow permeability for large deformations of hydrated tissues (Holmes 1990).

.. math::
	\frac{k}{k_0} = \Big(\frac{J - J_{\min}}{1 - J_{\min,0}}\Big)^{\kappa_{HM}} \exp\Big(\frac{M}{2}(J^2 - 1)\Big),
	\qquad J_{\min} = \Phi_{s,0} + \Phi_c, \quad J_{\min,0} = \Phi_{s,0} + \Phi_{c,0}

:math:`(J-J_{\min})/(1-J_{\min,0}) = \Phi_\ell/\Phi_{\ell,0}`: :math:`k \to 0` when :math:`\Phi_\ell \to 0`
for :math:`\kappa_{HM} > 0` (:math:`\kappa_{HM} = 0`: exponential stiffening only). In the code
:math:`\Phi_\ell/\Phi_{\ell,0}` is floored at :math:`x_f` = ``fluid_floor``:
:math:`k/k_0 \ge x_f^{\kappa_{HM}} e^{\frac{M}{2}(J^2-1)} > 0`.
"""

from dataclasses import dataclass
from typing import Any

import dolfin

from ...ufl_functions import floored
from .base import PorousPermeability, PorousPermeabilityParameters


@dataclass(slots=True, frozen=True)
class Parameters(PorousPermeabilityParameters):
	r"""Parameters (plus ``conductivity_0``, ``phi_s0``, ``phi_c``, ``phi_c0``).

	Attributes:
	    exponent: :math:`\kappa_{HM} \ge 0`.
	    stiffening: :math:`M \ge 0`.
	    fluid_floor: Lower bound of :math:`\Phi_\ell/\Phi_{\ell,0}` (Newton safeguard).
	"""

	exponent: float = 2.0
	stiffening: float = 0.0
	fluid_floor: float = 1e-6

	def __post_init__(self) -> None:
		r"""Check :math:`\phi_{\ell,0} > 0` (base class), :math:`\kappa_{HM} \ge 0` and :math:`M \ge 0`.

		Raises:
		    ValueError: :math:`\Phi_{s,0} + \Phi_{c,0} \ge 1`, ``exponent < 0`` or ``stiffening < 0``.
		"""
		PorousPermeabilityParameters.__post_init__(self)
		if self.exponent < 0.0:
			raise ValueError(f"exponent = {self.exponent} < 0: k would blow up at the compaction point.")
		if self.stiffening < 0.0:
			raise ValueError(f"stiffening = {self.stiffening} < 0: k would decrease when the gel swells.")


class Permeability(PorousPermeability[Parameters]):
	r""":math:`k(J, \Phi_c)` of Holmes--Mow."""

	def _ratio(self) -> Any:
		r""":math:`k/k_0`."""
		p = self._parameters
		jac = self._kinematics.J
		return floored(self.fluid_ratio, p.fluid_floor) ** p.exponent * dolfin.exp(0.5 * p.stiffening * (jac**2 - 1.0))
