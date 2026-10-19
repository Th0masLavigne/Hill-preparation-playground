r"""Cell Hill chain SEE -- (CE :math:`\parallel` PEE) as a stress branch of :math:`R_u`.

equations.tex, Eqs. Psi-h, S-h, P-h. Only the SEE stress enters the momentum balance:

.. math::
	\mathbf{S}_h = \Phi_c \frac{T_e(I_{4e})}{I_4}\, \mathbf{f}_0 \otimes \mathbf{f}_0, \qquad
	I_{4e} = I_4 e^{-2\varepsilon_a}
	\qquad (\Phi_c, \text{ not } \Phi_c / J)

The CE and the PEE act on :math:`\varepsilon_a = \ln\lambda_a` through :math:`R_a`
(:class:`hill_playground.operators.cell.hill.ContractileEvolution`).

Self-contained law: it derives from the fenicsoft base classes only (``MaterialParameters``,
``ConstitutiveLaw``) and owns its activation :math:`\alpha(t)` by composition
(:class:`~hill_playground.materials.cell.active.activation.ActivationSchedule`): ``p_act`` is the
single ``dolfin.Constant`` shared with ``OrientationChange`` and ``ContractileEvolution``, refreshed
through ``set_k_step``/``update_time`` by ``HyperElasticity`` (this law is summed into the gel law).
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

import dolfin
import dolfin_mech as dmech
from fenicsoft.materials.materials import ConstitutiveLaw, MaterialParameters

from ..activation import ActivationSchedule, ActivePowerConfig
from . import functions as hill


@dataclass(slots=True, frozen=True, kw_only=True)
class Parameters(MaterialParameters):
	r"""Hill-chain parameters (SI).

	Attributes:
	    phi_c: :math:`\Phi_c`, reference cell fraction (``Constant`` or ``Function``).
	    f0: Unit reference fibre direction :math:`\mathbf f_0` (UFL vector).
	    activation_configs: :math:`\alpha(t)`, one :class:`ActivePowerConfig` per step.
	    simulation_steps: ``problem.steps``.
	    f_max: Maximal isometric tension :math:`f_{max}` (Pa).
	    eps_a: :math:`\varepsilon_a` (current iterate of the DG0 subsol).
	    eps_a_old: :math:`\varepsilon_a^n` (previous converged value, ``func_old``).
	    k_e: SEE initial stiffness :math:`k_e` (Pa).
	    k_p: PEE initial stiffness :math:`k_p` (Pa).
	    use_force_length: Gaussian :math:`f_{FL}(\lambda_a)`, else ``default_force_length``.
	    lmbda_opt: :math:`\lambda_{opt}`.
	    W: Width :math:`W` of :math:`f_{FL}`.
	    inv4_floor: Lower bound of :math:`I_4` (Newton safeguard).
	    default_force_length: :math:`f_{FL}` when ``use_force_length`` is False.
	    hill_a: Hill curvature :math:`A = a / T_0` (-).
	    hill_b: Hill rate constant :math:`b` (1/s); :math:`v_{max} = b / A`.
	    f_ecc: Eccentric saturation :math:`f_{ecc}` (-).
	    eta_0: Optional CE damper :math:`\eta_0` (Pa.s); 0 removes the branch.
	    see_variant: SEE law, one of :data:`functions.SeeVariant`.
	    k_2e: Stiffening exponent of the ``"exponential"`` SEE (-), > 0 for that variant.
	"""

	phi_c: Any
	f0: Any
	activation_configs: Sequence[ActivePowerConfig]
	simulation_steps: Sequence[dmech.core.Step]
	f_max: float | dolfin.Constant
	eps_a: Any
	eps_a_old: Any
	k_e: float | dolfin.Constant
	k_p: float | dolfin.Constant
	use_force_length: bool = True
	lmbda_opt: float | dolfin.Constant = 1.05
	W: float | dolfin.Constant = 0.3
	inv4_floor: float = 1e-6
	default_force_length: float = 1.0
	hill_a: float | dolfin.Constant = 0.25
	hill_b: float | dolfin.Constant = 0.05
	f_ecc: float | dolfin.Constant = 1.8
	eta_0: float | dolfin.Constant = 0.0
	see_variant: hill.SeeVariant = "mooney"
	k_2e: float | dolfin.Constant = 0.0

	def __post_init__(self) -> None:
		r"""Check the SEE variant and the force--velocity constants.

		Raises:
		    ValueError: Unknown ``see_variant``, ``k_2e <= 0`` with ``"exponential"``, ``hill_b <= 0``,
		        ``1 + hill_a <= 0`` or ``f_ecc <= 1`` (then :math:`b_e \le 0` and :math:`h` is singular).
		"""
		if self.see_variant not in hill.SEE_VARIANTS:
			raise ValueError(f"Unknown SEE variant '{self.see_variant}', expected one of {hill.SEE_VARIANTS}.")
		if self.see_variant == "exponential" and float(self.k_2e) <= 0.0:
			raise ValueError("The exponential SEE needs k_2e > 0.")
		if float(self.hill_b) <= 0.0:
			raise ValueError("The force-velocity law needs hill_b > 0.")
		if 1.0 + float(self.hill_a) <= 0.0:
			raise ValueError("The force-velocity law needs 1 + hill_a > 0.")
		if float(self.f_ecc) <= 1.0:
			raise ValueError("The force-velocity law needs f_ecc > 1.")


class Constitutive(ConstitutiveLaw[Parameters]):
	r"""SEE stress of the Hill chain, weighted by :math:`\Phi_c` (sum it into the gel law with ``+``).

	Args:
	    kinematics: ``dolfin_mech`` kinematics.
	    parameters: :class:`Parameters`.
	    name: Optional name.

	Attributes:
	    p_act: :math:`\alpha` (``dolfin.Constant``), owned by the activation schedule.
	"""

	_parameters: Parameters

	def __init__(self, kinematics: Any, parameters: Parameters, name: str | None = None) -> None:
		"""Build the activation schedule."""
		super().__init__(kinematics, parameters, name=name)
		self._activation = ActivationSchedule(parameters.activation_configs, parameters.simulation_steps)
		self.p_act = self._activation.p_act

	def set_k_step(self, k_step: int) -> None:
		"""Select the activation of step ``k_step``.

		Args:
		    k_step: Step index.
		"""
		self._activation.set_k_step(k_step)

	def update_time(self, t_step: float) -> None:
		r"""Update :math:`\alpha` at the step fraction :math:`t_{step}`.

		Args:
		    t_step: Normalised step time in :math:`[0, 1]`.
		"""
		self._activation.update_time(t_step)

	@property
	def current_activation(self) -> float:
		r"""Current :math:`\alpha`."""
		return self._activation.value

	@property
	def current_time(self) -> float:
		"""Global time of the last ``update_time`` (solve units)."""
		return self._activation.time

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

	@cached_property
	def pk1(self) -> dolfin.Form:
		r""":math:`\mathbf P_h = \mathbf F \mathbf S_h`."""
		return self._kinematics.F * self.pk2

	@cached_property
	def cauchy(self) -> dolfin.Form:
		r""":math:`\boldsymbol\sigma_h = \mathbf P_h \mathbf F^T / J`."""
		return self.pk1 * self._kinematics.F.T / self._kinematics.J
