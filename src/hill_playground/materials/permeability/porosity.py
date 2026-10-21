r"""Kinematic relations of the saturated mixture shared by the permeability laws (not a law).

With :math:`\Phi_{s,0}` the substrate and :math:`\Phi_c` the cell volume per unit reference volume
(saturated mixture, incompressible constituents):

.. math::
	J_{\min} = \Phi_{s,0} + \Phi_c, \qquad \Phi_\ell = J - J_{\min}, \qquad
	\phi_\ell = \frac{\Phi_\ell}{J} = 1 - \frac{J_{\min}}{J},
	\qquad \mathbf K_L = \frac{k}{\eta} J \mathbf C^{-1}

:math:`J_{\min}` is the compaction point: :math:`\phi_\ell = 0 \Leftrightarrow J = J_{\min}`.
"""

from typing import Any


def compaction_jacobian(phi_s0: Any, phi_c: Any) -> Any:
	r""":math:`J_{\min} = \Phi_{s,0} + \Phi_c` (volume of the solid phases per unit reference volume).

	Args:
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c`.

	Returns:
	    UFL scalar.
	"""
	return phi_s0 + phi_c


def lagrangian_fluid_fraction(jacobian: Any, phi_s0: Any, phi_c: Any) -> Any:
	r""":math:`\Phi_\ell = J - J_{\min}`.

	Args:
	    jacobian: :math:`J`.
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c`.

	Returns:
	    UFL scalar.
	"""
	return jacobian - compaction_jacobian(phi_s0, phi_c)


def eulerian_porosity(jacobian: Any, phi_s0: Any, phi_c: Any) -> Any:
	r""":math:`\phi_\ell = 1 - J_{\min}/J`.

	Args:
	    jacobian: :math:`J`.
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c: :math:`\Phi_c`.

	Returns:
	    UFL scalar.
	"""
	return 1.0 - compaction_jacobian(phi_s0, phi_c) / jacobian


def pulled_back_conductivity(hydraulic_conductivity: Any, kinematics: Any) -> Any:
	r""":math:`\mathbf K_L = (k/\eta)\, J \mathbf C^{-1}` (Darcy conductivity in the reference configuration).

	Args:
	    hydraulic_conductivity: :math:`k/\eta` (scalar, may depend on the unknowns).
	    kinematics: Object exposing ``J`` and ``C_inv``.

	Returns:
	    UFL tensor.
	"""
	return hydraulic_conductivity * kinematics.J * kinematics.C_inv
