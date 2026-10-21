r"""Clipped saturated proliferation.

.. math::
	R = -\int r\, \langle 1 - \Phi_c / \Phi_{c,max} \rangle_+\, \Phi_c\, \delta c \, \mathrm{d}V

The positive part keeps the term a source: it vanishes at and above :math:`\Phi_{c,max}` (never a sink).
"""

from typing import Any

import dolfin
from fenicsoft.operators.cell import SaturatedProliferation as _SaturatedProliferation

from ....materials.ufl_functions import macaulay


class SaturatedProliferation(_SaturatedProliferation):
	r"""``fenicsoft`` ``SaturatedProliferation`` with :math:`\mathrm{sat} = \langle 1 - \Phi_c/\Phi_{c,max} \rangle_+`.

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
		saturation = macaulay(1.0 - phi_c / phi_c_max)
		self.res_form = -scale_factor * r * saturation * phi_c * test_c * self.measure
