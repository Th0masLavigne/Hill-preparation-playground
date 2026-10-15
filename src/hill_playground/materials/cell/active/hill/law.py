r"""Cell Hill chain SEE -- (CE :math:`\parallel` PEE) as a stress branch of :math:`R_u`.

equations.tex, Eqs. Psi-h, S-h, P-h. Only the SEE stress enters the momentum balance:

.. math::
	\mathbf{S}_h = \Phi_c \frac{T_e(I_{4e})}{I_4}\, \mathbf{f}_0 \otimes \mathbf{f}_0, \qquad
	I_{4e} = I_4 e^{-2\varepsilon_a}
	\qquad (\Phi_c, \text{ not } \Phi_c / J)

The CE and the PEE act on :math:`\varepsilon_a = \ln\lambda_a` through :math:`R_a`
(:class:`hill_playground.operators.cell.hill.ContractileEvolution`).

The activation :math:`\alpha(t)` (``p_act``, ``set_k_step``, ``update_time``) is inherited from
:class:`fenicsoft.materials.cell.active.aniso.law.Constitutive`: ``p_act`` is the single
``dolfin.Constant`` shared with ``OrientationChange`` and ``ContractileEvolution``, and is
refreshed by ``HyperElasticity`` because this law is summed into the gel material.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import dolfin
from fenicsoft.materials.cell.active.aniso import law as aniso_law

from . import functions as hill


@dataclass(slots=True, frozen=True, kw_only=True)
class Parameters(aniso_law.Parameters):
	r"""Hill-chain parameters, on top of ``aniso.law.Parameters``.

	Inherited and used: ``phi_c`` (:math:`\Phi_c`), ``f0``, ``activation_configs``, ``f_max``,
	``simulation_steps``, ``use_force_length``, ``lmbda_opt``, ``W``, ``inv4_floor``,
	``default_force_length``.

	Attributes:
	    eps_a: :math:`\varepsilon_a` (current iterate of the DG0 subsol).
	    eps_a_old: :math:`\varepsilon_a^n` (previous converged value, ``func_old``).
	    k_e: SEE initial stiffness :math:`k_e` (Pa).
	    k_p: PEE initial stiffness :math:`k_p` (Pa).
	    hill_a: Hill curvature :math:`A = a / T_0` (-).
	    hill_b: Hill rate constant :math:`b` (1/s); :math:`v_{max} = b / A`.
	    f_ecc: Eccentric saturation :math:`f_{ecc}` (-).
	    eta_0: Optional CE damper :math:`\eta_0` (Pa.s); 0 removes the branch.
	    see_variant: SEE law, one of :data:`functions.SEE_VARIANTS`.
	    k_2e: Stiffening exponent of the ``"exponential"`` SEE (-), > 0 for that variant.
	"""

	eps_a: Any
	eps_a_old: Any
	k_e: float | dolfin.Constant
	k_p: float | dolfin.Constant
	hill_a: float | dolfin.Constant = 0.25
	hill_b: float | dolfin.Constant = 0.05
	f_ecc: float | dolfin.Constant = 1.8
	eta_0: float | dolfin.Constant = 0.0
	see_variant: str = "mooney"
	k_2e: float | dolfin.Constant = 0.0

	def __post_init__(self) -> None:
		"""Check the SEE variant.

		Raises:
		    ValueError: Unknown ``see_variant``, or ``k_2e <= 0`` with ``"exponential"``.
		"""
		if self.see_variant not in hill.SEE_VARIANTS:
			raise ValueError(f"Unknown SEE variant '{self.see_variant}', expected one of {hill.SEE_VARIANTS}.")
		if self.see_variant == "exponential" and float(self.k_2e) <= 0.0:
			raise ValueError("The exponential SEE needs k_2e > 0.")


class Constitutive(aniso_law.Constitutive):
	r"""SEE stress of the Hill chain, weighted by :math:`\Phi_c` (sum it into the gel law with ``+``).

	``pk1`` (:math:`\mathbf{F}\mathbf{S}_h`) and ``cauchy`` (:math:`\mathbf{P}_h\mathbf{F}^T/J`)
	are inherited.
	"""

	_parameters: Parameters

	@cached_property
	def inv4(self) -> Any:
		r""":math:`I_4 = \mathbf{f}_0 \cdot \mathbf{C}\,\mathbf{f}_0` (floored)."""
		return hill.fibre_invariant(self._kinematics, self._parameters.f0, self._parameters.inv4_floor)

	@cached_property
	def inv4e(self) -> Any:
		r""":math:`I_{4e} = I_4 / \lambda_a^2`."""
		return hill.elastic_invariant(self.inv4, self._parameters.eps_a)

	@cached_property
	def lmbda_a(self) -> Any:
		r"""CE stretch :math:`\lambda_a = e^{\varepsilon_a}`."""
		return dolfin.exp(self._parameters.eps_a)

	@cached_property
	def tension_see(self) -> Any:
		""":math:`T_e` (Pa)."""
		p = self._parameters
		return hill.see_tension(self.inv4e, p.k_e, p.see_variant, p.k_2e)

	@cached_property
	def tension_pee(self) -> Any:
		""":math:`T_p` (Pa)."""
		return hill.pee_tension(self._parameters.eps_a, self._parameters.k_p)

	@cached_property
	def tension_ce(self) -> Any:
		r""":math:`T_{CE} = T_e - T_p` (series equilibrium; :math:`= T_0 h(D_a) + \eta_0 D_a` at convergence)."""
		return self.tension_see - self.tension_pee

	@cached_property
	def ener(self) -> dolfin.Form:
		r""":math:`\Phi_c (\Psi_e + \Psi_p)` per unit reference mixture volume (Eq. Psi-h)."""
		p = self._parameters
		psi_e = hill.see_energy(self.inv4e, p.k_e, p.see_variant, p.k_2e)
		return p.phi_c * (psi_e + hill.pee_energy(p.eps_a, p.k_p))

	@cached_property
	def pk2(self) -> dolfin.Form:
		r"""Second Piola stress of the chain at fixed :math:`\varepsilon_a` (Eq. S-h).

		.. math::
			\mathbf{S}_h = \Phi_c \frac{T_e}{I_4}\, \mathbf{f}_0 \otimes \mathbf{f}_0
		"""
		p = self._parameters
		return p.phi_c * self.tension_see / self.inv4 * dolfin.outer(p.f0, p.f0)
