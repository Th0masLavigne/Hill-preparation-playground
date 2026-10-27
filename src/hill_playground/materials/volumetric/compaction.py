r"""Volumetric energy of the drained skeleton with a compaction point.

With :math:`J_{\min}` the compaction point (solid volume per unit reference volume, Ehlers 1999):

.. math::
	W^{vol}_S = \kappa (1-J_{\min})^2 \big[x - \ln x - 1\big], \qquad
	x = \frac{J - J_{\min}}{1 - J_{\min}}, \qquad J_{\min} = \Phi_{s,0} + \Phi_c

.. math::
	\frac{\mathrm d W^{vol}_S}{\mathrm d J} = \kappa \frac{J - 1}{x}, \qquad
	\mathbf S^{vol}_S = J \frac{\mathrm d W^{vol}_S}{\mathrm d J}\mathbf C^{-1} = \kappa \frac{J (J-1)}{x}\mathbf C^{-1}

Properties: :math:`W(1) = 0` and :math:`W''(1) = \kappa` (same small-strain :math:`K_d`, :math:`\nu_d`, :math:`t_0`
as :math:`\frac{\kappa}{2}(J-1)^2`); :math:`W \to \infty` when :math:`J \to J_{\min}^+` (the fluid is never fully
expelled: energetic barrier). :math:`J_{\min} \to 0` gives :math:`\kappa (J - 1 - \ln J)`.
:math:`J_{\min}` may depend on :math:`\Phi_c`; only :math:`\mathbf S` (at fixed :math:`\Phi_c`) enters :math:`R_u`.

Newton safeguard: below :math:`x_f` (trial iterates with :math:`J \le J_{\min}`), :math:`W` is continued linearly in
:math:`x`, so :math:`W` and :math:`\mathbf S` stay consistent and finite:

.. math::
	\bar x = \max(x, x_f), \qquad
	\frac{\mathrm d W^{vol}_S}{\mathrm d J} = \kappa (1 - J_{\min})\Big(1 - \frac{1}{\bar x}\Big)
"""

import numbers
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import dolfin
from fenicsoft.materials.materials import ConstitutiveLaw
from fenicsoft.materials.volumetric.parameters import VolumetricPenaltyParameters
from fenicsoft.ufl_functions import floored


@dataclass(slots=True, frozen=True)
class Parameters(VolumetricPenaltyParameters):
	r"""Fenicsoft ``VolumetricPenaltyParameters`` (:math:`\kappa`) plus the compaction point.

	Parent: :class:`~fenicsoft.materials.volumetric.parameters.VolumetricPenaltyParameters`.

	Attributes:
	    kappa: Drained bulk modulus at :math:`J = 1`, :math:`\kappa` (Pa, or :math:`\hat\kappa`).
	    solid_jacobian: :math:`J_{\min} = \Phi_{s,0} + \Phi_c` (float or UFL), :math:`< 1` (checked for a number or
	        a ``dolfin.Constant``); :math:`\Phi_{s,0}` for a two-phase mixture
	        (``saturated_mixture.compaction_jacobian(phi_s0)``).
	    x_floor: :math:`x_f`, start of the linear continuation of :math:`W` (Newton safeguard).
	"""

	solid_jacobian: Any
	x_floor: float = 1e-8

	def __post_init__(self) -> None:
		r"""Check :math:`J_{\min} < 1` when it is a number or a ``dolfin.Constant`` (a UFL field is not checked).

		Raises:
		    ValueError: :math:`J_{\min} \ge 1` (no fluid at :math:`J = 1`, :math:`x` undefined).
		"""
		jc = self.solid_jacobian
		if isinstance(jc, (numbers.Real, dolfin.Constant)) and float(jc) >= 1.0:
			raise ValueError(f"solid_jacobian = {float(jc)} >= 1: no fluid in the reference state.")

	@classmethod
	def from_young_poisson(  # type: ignore[override]
		cls, young: Any, poisson: Any, solid_jacobian: Any, x_floor: float = 1e-8
	) -> "Parameters":
		r""":math:`\kappa = E / (3(1 - 2\nu))`, via the parent ``from_young_poisson``.

		Parent: :meth:`~fenicsoft.materials.volumetric.parameters.VolumetricPenaltyParameters.from_young_poisson`.

		Args:
		    young: Drained Young modulus :math:`E`.
		    poisson: Drained Poisson ratio :math:`\nu`.
		    solid_jacobian: :math:`J_{\min}`.
		    x_floor: :math:`x_f`. Defaults to ``1e-8``.

		Returns:
		    Parameters.
		"""
		kappa = VolumetricPenaltyParameters.from_young_poisson(young, poisson).kappa
		return cls(kappa=kappa, solid_jacobian=solid_jacobian, x_floor=x_floor)

	@classmethod
	def from_lame(  # type: ignore[override]
		cls, lame_lambda: Any, lame_mu: Any, solid_jacobian: Any, x_floor: float = 1e-8
	) -> "Parameters":
		r""":math:`\kappa = \lambda + 2\mu/3`, via the parent ``from_lame``.

		Parent: :meth:`~fenicsoft.materials.volumetric.parameters.VolumetricPenaltyParameters.from_lame`.

		Args:
		    lame_lambda: Drained :math:`\lambda`.
		    lame_mu: Drained :math:`\mu`.
		    solid_jacobian: :math:`J_{\min}`.
		    x_floor: :math:`x_f`. Defaults to ``1e-8``.

		Returns:
		    Parameters.
		"""
		kappa = VolumetricPenaltyParameters.from_lame(lame_lambda, lame_mu).kappa
		return cls(kappa=kappa, solid_jacobian=solid_jacobian, x_floor=x_floor)


class CompactionVolumetricPenalty(ConstitutiveLaw[Parameters]):
	r""":math:`W^{vol}_S(J; J_{\min})`, to sum with the isochoric gel law."""

	@cached_property
	def x(self) -> Any:
		r""":math:`x = (J - J_{\min})/(1 - J_{\min})`."""
		jc = self._parameters.solid_jacobian
		return (self._kinematics.J - jc) / (1.0 - jc)

	@cached_property
	def x_bar(self) -> Any:
		r""":math:`\bar x = \max(x, x_f)`."""
		return floored(self.x, self._parameters.x_floor)

	@cached_property
	def ener(self) -> Any:
		r""":math:`\kappa (1-J_{\min})^2 \big[\bar x - \ln \bar x - 1 + (1 - 1/\bar x)(x - \bar x)\big]`."""
		jc = self._parameters.solid_jacobian
		xb = self.x_bar
		return self._parameters.kappa * (1.0 - jc) ** 2 * (xb - dolfin.ln(xb) - 1.0 + (1.0 - 1.0 / xb) * (self.x - xb))

	@cached_property
	def pk2(self) -> Any:
		r""":math:`\mathbf S = J\, \kappa (1 - J_{\min})(1 - 1/\bar x)\, \mathbf C^{-1}`.

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
