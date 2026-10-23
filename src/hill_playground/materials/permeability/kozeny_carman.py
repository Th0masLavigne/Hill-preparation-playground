r"""Kozeny--Carman permeability (Carman 1937).

.. math::
	\frac{k}{k_0} = \frac{\phi_\ell^3/(1-\phi_\ell)^2}{\phi_{\ell,0}^3/(1-\phi_{\ell,0})^2},
	\qquad \phi_\ell = 1 - \frac{\Phi_{s,0}+\Phi_c}{J}, \qquad \phi_{\ell,0} = 1 - \Phi_{s,0} - \Phi_{c,0}

:math:`k \sim \phi_\ell^3 \to 0` at the compaction point :math:`J = J_{\min}`
(in the code :math:`\phi_\ell` is floored at ``porosity_floor``).
"""

from dataclasses import dataclass
from typing import Any

from ...ufl_functions import floored
from .. import saturated_mixture
from .base import PorousPermeability, PorousPermeabilityParameters


def _carman(phi: Any) -> Any:
	r""":math:`\phi^3/(1-\phi)^2`.

	Args:
	    phi: Porosity :math:`\phi`.

	Returns:
	    Scalar (UFL or float).
	"""
	return phi**3 / (1.0 - phi) ** 2


@dataclass(slots=True, frozen=True)
class Parameters(PorousPermeabilityParameters):
	r"""Parameters (plus ``conductivity_0``, ``phi_s0``, ``phi_c``, ``phi_c0``).

	Attributes:
	    porosity_floor: Lower bound of :math:`\phi_\ell` in the formula (Newton safeguard).
	"""

	porosity_floor: float = 1e-6

	def __post_init__(self) -> None:
		r"""Check :math:`0 < J_{\min,0} = \Phi_{s,0} + \Phi_{c,0} < 1` (:math:`\phi_{\ell,0} = 1`: division by 0).

		Raises:
		    ValueError: :math:`J_{\min,0} \ge 1` (base class) or :math:`J_{\min,0} \le 0`.
		"""
		PorousPermeabilityParameters.__post_init__(self)
		j_min_0 = saturated_mixture.compaction_jacobian(float(self.phi_s0), self.phi_c0)
		if j_min_0 <= 0.0:
			raise ValueError(f"phi_s0 + phi_c0 = {j_min_0} <= 0: Kozeny-Carman needs a solid phase (phi_l0 < 1).")


class Permeability(PorousPermeability[Parameters]):
	r""":math:`k(\phi_\ell)` of Kozeny--Carman."""

	def _ratio(self) -> Any:
		r""":math:`k/k_0`."""
		phi = floored(self.porosity, self._parameters.porosity_floor)
		return _carman(phi) / _carman(self.porosity_0)
