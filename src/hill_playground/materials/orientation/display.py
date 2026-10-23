r"""Orientation diagnostics for ParaView: current fibre, tilt out of a plane, angle to an axis, order parameters.

Complements ``materials.orientation.spherical`` (``canonical_theta_phi_degrees``,
``wrapped_theta_degrees``), which describe the *reference* angles :math:`(\theta, \varphi)`.
The angles and the order parameter are head/tail symmetric (:math:`\mathbf f \to -\mathbf f` invariant;
:math:`\mathbf f` itself changes sign with :math:`\mathbf f_0`) and written with ``atan2`` (no ``acos`` / ``asin``
domain issue at :math:`|\cdot| = 1`):

.. math::
	\mathbf f = \frac{\mathbf F \mathbf f_0}{\|\mathbf F \mathbf f_0\|}, \qquad
	\beta_{\mathbf n} = \operatorname{atan2}\big(|\mathbf f\cdot\mathbf n|,
	\|\mathbf f - (\mathbf f\cdot\mathbf n)\mathbf n\|\big),
	\qquad
	\gamma_{\mathbf a} = \operatorname{atan2}\big(\|\mathbf f \times \mathbf a\|,\ |\mathbf f\cdot\mathbf a|\big)

:math:`\beta_{\mathbf n} \in [0, 90°]` is the tilt out of the plane of normal :math:`\mathbf n`
(0: fibre in the plane, "planar orientation"); :math:`\gamma_{\mathbf a} \in [0, 90°]` is the
angle to the axis :math:`\mathbf a` (0: aligned). Order parameter along a unit axis:

.. math::
	S_{\mathbf a} = \tfrac{3}{2}(\mathbf f\cdot\mathbf a)^2 - \tfrac{1}{2} \in [-\tfrac12, 1]

averaged over a region: 1 all aligned with :math:`\mathbf a`, 0 isotropic, :math:`-\frac12` all
orthogonal to :math:`\mathbf a` (with :math:`\mathbf a = \mathbf n`: all in the plane).
"""

import math
from typing import Any

import dolfin
import ufl_legacy
from fenicsoft.materials.orientation.generic import from_vector

_TO_DEG = 180.0 / math.pi


def current_direction(kinematics: Any, f0: Any) -> Any:
	r""":math:`\mathbf f = \mathbf F\mathbf f_0 / \|\mathbf F\mathbf f_0\|` (fibre pushed to the current configuration).

	Args:
	    kinematics: Object exposing ``F``.
	    f0: Unit reference direction (UFL vector).

	Returns:
	    UFL unit vector (``materials.orientation.generic.from_vector``).
	"""
	return from_vector(kinematics.F * f0)


def tilt_degrees(direction: Any, normal: Any) -> Any:
	r"""Tilt :math:`\beta_{\mathbf n}` out of the plane of unit normal :math:`\mathbf n`, in degrees.

	Args:
	    direction: Unit vector :math:`\mathbf f`.
	    normal: Unit normal :math:`\mathbf n` (e.g. ``dolfin.Constant((0, 0, 1))``).

	Returns:
	    UFL scalar in :math:`[0, 90]`.
	"""
	normal_part = dolfin.inner(direction, normal)
	in_plane = direction - normal_part * normal
	return _TO_DEG * ufl_legacy.atan_2(abs(normal_part), dolfin.sqrt(dolfin.inner(in_plane, in_plane)))  # type: ignore


def axis_angle_degrees(direction: Any, axis: Any) -> Any:
	r"""Angle :math:`\gamma_{\mathbf a}` between the fibre axis and the unit axis :math:`\mathbf a`, in degrees.

	Args:
	    direction: Unit vector :math:`\mathbf f`.
	    axis: Unit vector :math:`\mathbf a`.

	Returns:
	    UFL scalar in :math:`[0, 90]`.
	"""
	cross = dolfin.cross(direction, axis)
	return _TO_DEG * ufl_legacy.atan_2(dolfin.sqrt(dolfin.inner(cross, cross)), abs(dolfin.inner(direction, axis)))  # type: ignore


def order_parameter(direction: Any, axis: Any) -> Any:
	r""":math:`S_{\mathbf a} = \frac32(\mathbf f\cdot\mathbf a)^2 - \frac12` (average it for the regional order).

	Args:
	    direction: Unit vector :math:`\mathbf f`.
	    axis: Unit vector :math:`\mathbf a`.

	Returns:
	    UFL scalar in :math:`[-1/2, 1]`.
	"""
	return 1.5 * dolfin.inner(direction, axis) ** 2 - 0.5
