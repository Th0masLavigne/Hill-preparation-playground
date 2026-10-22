r"""Kinematic relations of the saturated mixture (substrate, optional cells, liquid), shared by several laws.

Target: ``fenicsoft.materials.saturated_mixture``.

With :math:`\Phi_{s,0}` the substrate and :math:`\Phi_c` the cell volume per unit reference volume
(incompressible constituents):

.. math::
	J_{\min} = \Phi_{s,0} + \Phi_c, \qquad \Phi_\ell = J - J_{\min}, \qquad
	\phi_\ell = \frac{\Phi_\ell}{J} = 1 - \frac{J_{\min}}{J}, \qquad \phi_s = \frac{\Phi_{s,0}}{J}

:math:`J_{\min}` is the compaction point: :math:`\phi_\ell = 0 \Leftrightarrow J = J_{\min}`.
Two-phase mixture (solid + liquid, no cells): :math:`\Phi_c = 0`, the default, so
:math:`J_{\min} = \Phi_{s,0}` and :math:`\phi_\ell = 1 - \phi_s`.
Used by the permeability laws (:math:`\phi_\ell`, :math:`\Phi_\ell`) and the volumetric compaction energy
(:math:`J_{\min}`).
"""

from typing import Any


def compaction_jacobian(phi_s0: Any, phi_c: Any = 0.0) -> Any:
	r""":math:`J_{\min} = \Phi_{s,0} + \Phi_c` (volume of the solid phases per unit reference volume).

	Args:
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c` (0: two-phase mixture).

	Returns:
	    UFL scalar (or float).
	"""
	return phi_s0 + phi_c


def lagrangian_fluid_fraction(jacobian: Any, phi_s0: Any, phi_c: Any = 0.0) -> Any:
	r""":math:`\Phi_\ell = J - J_{\min}` (liquid volume per unit reference volume).

	Args:
	    jacobian: :math:`J`.
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c` (0: two-phase mixture).

	Returns:
	    UFL scalar (or float).
	"""
	return jacobian - compaction_jacobian(phi_s0, phi_c)


def eulerian_porosity(jacobian: Any, phi_s0: Any, phi_c: Any = 0.0) -> Any:
	r""":math:`\phi_\ell = 1 - J_{\min}/J` (liquid volume per unit current volume).

	Args:
	    jacobian: :math:`J`.
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c` (0: two-phase mixture).

	Returns:
	    UFL scalar (or float).
	"""
	return 1.0 - compaction_jacobian(phi_s0, phi_c) / jacobian


def solid_fraction(jacobian: Any, phi_s0: Any) -> Any:
	r""":math:`\phi_s = \Phi_{s,0}/J` (substrate volume per unit current volume).

	Args:
	    jacobian: :math:`J`.
	    phi_s0: :math:`\Phi_{s,0}`.

	Returns:
	    UFL scalar (or float).
	"""
	return phi_s0 / jacobian
