r"""Clipped saturated proliferation (equations.tex, Eq. prolif).

.. math::
	R = -\int r\, \langle 1 - \Phi_c / \Phi_{c,max} \rangle_+\, \Phi_c\, \delta c \, \mathrm{d}V

``fenicsoft``'s ``SaturatedProliferation`` uses :math:`1 - \Phi_c/\Phi_{c,max}` without the
positive part, so the source turns into a sink above :math:`\Phi_{c,max}` (code.tex, Sec.
"Saturation of proliferation"). Transfer: clip the saturation in fenicsoft's class and delete
this module.
"""

from typing import Any

import dolfin
from fenicsoft.operators.cell import SaturatedProliferation as _SaturatedProliferation


class SaturatedProliferation(_SaturatedProliferation):
	r"""``fenicsoft`` ``SaturatedProliferation`` with :math:`\mathrm{sat} = \max(0, 1 - \Phi_c/\Phi_{c,max})`.

	Args:
	    phi_c: :math:`\Phi_c`.
	    r: Proliferation rate :math:`r` (1/s).
	    phi_c_max: :math:`\Phi_{c,max}`.
	    test_c: Test function of :math:`\Phi_c`.
	    measure: Gel measure.
	    scale_factor: Multiplies the residual.
	"""

	def __init__(
		self,
		phi_c: dolfin.Function,
		r: float | dolfin.Constant | dolfin.Function,
		phi_c_max: float | dolfin.Constant | dolfin.Function,
		test_c: Any,
		measure: dolfin.Measure,
		scale_factor: float = 1.0,
	) -> None:
		"""Builds the parent form, then replaces it by the clipped one."""
		super().__init__(phi_c, r, phi_c_max, test_c, measure, scale_factor)
		saturation = dolfin.conditional(dolfin.gt(phi_c, phi_c_max), 0.0, 1.0 - phi_c / phi_c_max)
		self.res_form = -scale_factor * r * saturation * phi_c * test_c * self.measure
