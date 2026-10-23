r"""Cell Hill chain SEE -- (CE :math:`\parallel` PEE) as a stress branch of :math:`R_u`.

Energy :math:`\Phi_c(\Psi_e + \Psi_p)` per unit reference mixture volume. Only the SEE stress enters the
momentum balance:

.. math::
	\mathbf{S}_h = \Phi_c \frac{T_e(I_{4e})}{I_4}\, \mathbf{f}_0 \otimes \mathbf{f}_0, \qquad
	I_{4e} = I_4 e^{-2\varepsilon_a}
	\qquad (\Phi_c, \text{ not } \Phi_c / J)

The CE and the PEE act on :math:`\varepsilon_a = \ln\lambda_a` through :math:`R_a`
(:class:`.ContractileEvolution`).

Derives from the base classes ``MaterialParameters`` and ``ConstitutiveLaw`` only; :math:`\alpha(t)`
comes from an :class:`.ActivationSchedule` (composition).
``p_act`` is the ``dolfin.Constant`` shared with ``OrientationChange`` and ``ContractileEvolution``, refreshed
by ``HyperElasticity`` via ``set_k_step``/``update_time``.

:func:`check_uniqueness` (sufficient CE-uniqueness condition) is not enforced by :class:`Parameters`:
a free cell (:math:`k_e = 0`) violates it.
"""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

import dolfin
import dolfin_mech as dmech
from fenicsoft.materials.materials import ConstitutiveLaw, MaterialParameters

from ..activation import ActivationSchedule, ActivePowerConfig
from . import functions as hill


def check_uniqueness(k_e: float, k_p: float, f_max: float, lmbda_opt: float, width: float) -> float:
	r"""Check :math:`k_e + k_p > f_{max} \max_\lambda |f_{FL}'(\lambda)\,\lambda|`.

	Sufficient (not necessary) condition for a unique isometric CE equilibrium :math:`\lambda_a`
	(:math:`D_a = 0`, :math:`\alpha \le 1`, no descending-limb instability), with
	:math:`f_{FL}(\lambda) = \exp(-((\lambda - \lambda_{opt})/W)^2)`:

	.. math::
		\frac{\partial R_a}{\partial\varepsilon_a} \ge k_e + k_p - f_{max} \max_\lambda |f_{FL}'(\lambda)\,\lambda| > 0,
		\qquad \frac{\mathrm dT_p}{\mathrm d\varepsilon_a} = k_p \cosh 2\varepsilon_a \ge k_p,
		\qquad -\frac{\mathrm dT_e}{\mathrm d\varepsilon_a} \ge k_e

	The last bound holds for the ``"mooney"`` SEE at any :math:`I_{4e}`
	(:math:`\frac{k_e}{2}(I_{4e} + I_{4e}^{-1})`), and for ``"tension_only"`` / ``"exponential"`` only while the
	SEE is taut (:math:`I_{4e} \ge 1`); a slack non-Mooney SEE adds no stiffness. The max is sampled on
	:math:`\lambda \in [0.05, 3]`. Any consistent stress unit (SI or :math:`\mu_0`).

	Args:
	    k_e: SEE stiffness.
	    k_p: PEE stiffness.
	    f_max: Maximal isometric tension.
	    lmbda_opt: :math:`\lambda_{opt}`.
	    width: :math:`W`.

	Returns:
	    Margin :math:`(k_e + k_p) / (f_{max} \max|f_{FL}'\lambda|)`, > 1 (``inf`` if the bound is 0,
	    e.g. :math:`f_{max} = 0`).

	Raises:
	    ValueError: Condition violated (descending-limb instability possible).
	"""
	n_samples = 20001
	worst = 0.0
	for i in range(n_samples):
		lam = 0.05 + 2.95 * i / (n_samples - 1)
		f_fl = math.exp(-(((lam - lmbda_opt) / width) ** 2))
		worst = max(worst, abs(-2.0 * (lam - lmbda_opt) / width**2 * f_fl * lam))
	bound = f_max * worst
	if k_e + k_p <= bound:
		raise ValueError(f"Hill chain not unique: k_e + k_p = {k_e + k_p:.3g} <= {bound:.3g}.")
	return math.inf if bound == 0.0 else (k_e + k_p) / bound


@dataclass(slots=True, frozen=True, kw_only=True)
class Parameters(MaterialParameters):
	r"""Hill-chain parameters (SI).

	Attributes:
	    phi_c: :math:`\Phi_c`, cell volume per unit reference volume (``Constant``, ``Function`` or unknown).
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
	    hill_b: Hill rate constant :math:`b` (1/s); :math:`v_{max} = b / A` (:math:`A > 0`).
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
		"""Global time of the last ``update_time`` (solve units: s, or :math:`t_0` if scaled)."""
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
	def ener(self) -> Any:
		r""":math:`\Phi_c (\Psi_e + \Psi_p)` per unit reference mixture volume."""
		p = self._parameters
		psi_e = hill.see_energy(self.inv4e, p.k_e, p.see_variant, p.k_2e)
		return p.phi_c * (psi_e + hill.pee_energy(p.eps_a, p.k_p))

	@cached_property
	def pk2(self) -> Any:
		r"""Second Piola stress of the chain at fixed :math:`\varepsilon_a`.

		.. math::
			\mathbf{S}_h = \Phi_c \frac{T_e}{I_4}\, \mathbf{f}_0 \otimes \mathbf{f}_0
		"""
		p = self._parameters
		return p.phi_c * self.tension_see / self.inv4 * dolfin.outer(p.f0, p.f0)

	@cached_property
	def pk1(self) -> Any:
		r""":math:`\mathbf P_h = \mathbf F \mathbf S_h`."""
		return self._kinematics.F * self.pk2

	@cached_property
	def cauchy(self) -> Any:
		r""":math:`\boldsymbol\sigma_h = \mathbf P_h \mathbf F^T / J`."""
		return self.pk1 * self._kinematics.F.T / self._kinematics.J
