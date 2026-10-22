r"""Scalar UFL guards shared by the constitutive laws and the operators.

Target: ``fenicsoft.ufl_functions``.

.. math::
	\mathrm{piecewise}(x, x_0, f_\le, f_>) = \begin{cases} f_\le & x \le x_0 \\ f_> & x > x_0 \end{cases}

	\mathrm{floored}(x, x_f) = \max(x, x_f), \qquad \mathrm{capped}(x, x_c) = \min(x, x_c),

	\langle x \rangle_+ = \max(x, 0), \qquad \langle x \rangle_- = \min(x, 0), \qquad
	x = \langle x \rangle_+ + \langle x \rangle_-

The floor keeps powers, logs and quotients defined on non-converged Newton iterates.

All are written with :func:`piecewise`: the tie :math:`x = x_0` belongs to the lower branch. At the kink both
branches give the same value; the UFL derivative (Newton tangent) is the one of the lower branch.
``floored`` keeps the constant there (derivative 0), ``capped`` keeps :math:`x` (derivative 1), so the split
also holds for the derivatives at 0:

.. math::
	\frac{d\langle x \rangle_+}{dx}(0) = 0, \qquad \frac{d\langle x \rangle_-}{dx}(0) = 1, \qquad
	\frac{d\langle x \rangle_+}{dx} + \frac{d\langle x \rangle_-}{dx} = 1
"""

from typing import Any

import dolfin


def piecewise(value: Any, threshold: Any, at_or_below: Any, above: Any) -> Any:
	r""":math:`f_\le` if :math:`x \le x_0`, :math:`f_>` if :math:`x > x_0` (tie and tangent: lower branch).

	Args:
	    value: UFL scalar :math:`x`.
	    threshold: :math:`x_0`.
	    at_or_below: :math:`f_\le`, used for :math:`x \le x_0`.
	    above: :math:`f_>`, used for :math:`x > x_0`.

	Returns:
	    UFL scalar.
	"""
	return dolfin.conditional(dolfin.le(value, threshold), at_or_below, above)


def floored(value: Any, floor: float) -> Any:
	r""":math:`\max(x, x_f)`; derivative 0 at :math:`x = x_f`.

	Args:
	    value: UFL scalar :math:`x`.
	    floor: :math:`x_f` (:math:`> 0` before a log, a quotient or a non-integer power).

	Returns:
	    UFL scalar.
	"""
	return piecewise(value, floor, floor, value)


def capped(value: Any, cap: float) -> Any:
	r""":math:`\min(x, x_c)`; derivative 1 at :math:`x = x_c`.

	Args:
	    value: UFL scalar :math:`x`.
	    cap: :math:`x_c`.

	Returns:
	    UFL scalar.
	"""
	return piecewise(value, cap, value, cap)


def macaulay(value: Any) -> Any:
	r"""Positive part :math:`\langle x \rangle_+ = \max(x, 0)`; derivative 0 at :math:`x = 0`.

	Args:
	    value: UFL scalar :math:`x`.

	Returns:
	    UFL scalar.
	"""
	return floored(value, 0.0)


def macaulay_negative(value: Any) -> Any:
	r"""Signed negative part :math:`\langle x \rangle_-`.

	:math:`\langle x \rangle_- = \min(x, 0) \le 0`, so that
	:math:`x = \langle x \rangle_+ + \langle x \rangle_-`; derivative 1 at :math:`x = 0`.

	Args:
	    value: UFL scalar :math:`x`.

	Returns:
	    UFL scalar.
	"""
	return capped(value, 0.0)
