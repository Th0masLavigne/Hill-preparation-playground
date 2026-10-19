r"""Volumetric energy of the drained skeleton with a compaction point.

With :math:`J_c` the compaction point (solid volume per unit reference volume, Ehlers 1999):

.. math::
	W^{vol}_S = \kappa (1-J_c)^2 \big[x - \ln x - 1\big], \qquad
	x = \frac{J - J_c}{1 - J_c}, \qquad J_c = \Phi_{s,0} + \Phi_c

.. math::
	\frac{\mathrm d W^{vol}_S}{\mathrm d J} = \kappa \frac{J - 1}{x}, \qquad
	\mathbf S^{vol}_S = J \frac{\mathrm d W^{vol}_S}{\mathrm d J}\mathbf C^{-1} = \kappa \frac{J (J-1)}{x}\mathbf C^{-1}

Properties: :math:`W(1) = 0` and :math:`W''(1) = \kappa` (same small-strain :math:`K_d`, :math:`\nu_d`, :math:`t_0`
as :math:`\frac{\kappa}{2}(J-1)^2`); :math:`W \to \infty` when :math:`J \to J_c^+` (the fluid is never fully
expelled: energetic barrier). :math:`J_c \to 0` gives :math:`\kappa (J - 1 - \ln J)`.
:math:`J_c` may depend on :math:`\Phi_c`; only :math:`\mathbf S` (at fixed :math:`\Phi_c`) enters :math:`R_u`.

Newton safeguard: below :math:`x_f` (trial iterates with :math:`J \le J_c`), :math:`W` is continued linearly in
:math:`x`, so :math:`W` and :math:`\mathbf S` stay consistent and finite:

.. math::
	\bar x = \max(x, x_f), \qquad
	\frac{\mathrm d W^{vol}_S}{\mathrm d J} = \kappa (1 - J_c)\Big(1 - \frac{1}{\bar x}\Big)
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import dolfin
from fenicsoft.materials.materials import ConstitutiveLaw, MaterialParameters

from ..ufl_functions import floored


@dataclass(slots=True, frozen=True)
class Parameters(MaterialParameters):
	r"""Parameters.

	Attributes:
	    kappa: Drained bulk modulus at :math:`J = 1`, :math:`\kappa` (Pa, or :math:`\hat\kappa`).
	    solid_jacobian: :math:`J_c = \Phi_{s,0} + \Phi_c` (float or UFL), :math:`< 1`.
	    x_floor: :math:`x_f`, start of the linear continuation of :math:`W` (Newton safeguard).
	"""

	kappa: Any
	solid_jacobian: Any
	x_floor: float = 1e-8


class CompactionVolumetricPenalty(ConstitutiveLaw[Parameters]):
	r""":math:`W^{vol}_S(J; J_c)`, to sum with the isochoric gel law."""

	@cached_property
	def x(self) -> Any:
		r""":math:`x = (J - J_c)/(1 - J_c)`."""
		jc = self._parameters.solid_jacobian
		return (self._kinematics.J - jc) / (1.0 - jc)

	@cached_property
	def x_bar(self) -> Any:
		r""":math:`\bar x = \max(x, x_f)`."""
		return floored(self.x, self._parameters.x_floor)

	@cached_property
	def ener(self) -> Any:
		r""":math:`\kappa (1-J_c)^2 \big[\bar x - \ln \bar x - 1 + (1 - 1/\bar x)(x - \bar x)\big]`."""
		jc = self._parameters.solid_jacobian
		xb = self.x_bar
		return self._parameters.kappa * (1.0 - jc) ** 2 * (xb - dolfin.ln(xb) - 1.0 + (1.0 - 1.0 / xb) * (self.x - xb))

	@cached_property
	def pk2(self) -> Any:
		r""":math:`\mathbf S = J\, \kappa (1 - J_c)(1 - 1/\bar x)\, \mathbf C^{-1}`.

		Above the floor: :math:`\mathbf S = \kappa J (J-1)/x\, \mathbf C^{-1}`.
		"""
		jc = self._parameters.solid_jacobian
		d_ener = self._parameters.kappa * (1.0 - jc) * (1.0 - 1.0 / self.x_bar)
		return self._kinematics.J * d_ener * self._kinematics.C_inv

	@cached_property
	def pk1(self) -> Any:
		r""":math:`\mathbf P = \mathbf F \mathbf S`."""
		return self._kinematics.F * self.pk2

	@cached_property
	def cauchy(self) -> Any:
		r""":math:`\boldsymbol\sigma = \mathbf P \mathbf F^T / J`."""
		return self.pk1 * self._kinematics.F.T / self._kinematics.J
