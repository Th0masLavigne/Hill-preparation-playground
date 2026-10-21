r"""Scalar UFL guards shared by the constitutive laws and the operators.

Target: ``fenicsoft.ufl_functions``.

.. math::
	\mathrm{floored}(x, x_f) = \max(x, x_f), \qquad \langle x \rangle_+ = \max(x, 0)

The floor keeps powers, logs and quotients defined on non-converged Newton iterates.
"""

from typing import Any

import dolfin


def floored(value: Any, floor: float) -> Any:
	r""":math:`\max(x, x_f)`.

	Args:
	    value: UFL scalar :math:`x`.
	    floor: :math:`x_f` (:math:`> 0` before a log, a quotient or a non-integer power).

	Returns:
	    UFL scalar.
	"""
	return dolfin.conditional(dolfin.gt(value, floor), value, floor)


def macaulay(value: Any) -> Any:
	r"""Positive part :math:`\langle x \rangle_+ = \max(x, 0)`.

	Args:
	    value: UFL scalar :math:`x`.

	Returns:
	    UFL scalar.
	"""
	return floored(value, 0.0)
