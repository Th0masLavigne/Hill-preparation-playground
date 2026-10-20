r"""Hill box: SI inputs converted to the dimensionless coefficients of the weak form.

Scales (``UserInputs.scales``): :math:`L_0 = L_z` (drainage length), :math:`\mu_0 = \mu_{gel}`,
:math:`t_0 = L_0^2\eta/(k\,\kappa_{gel})` (consolidation time over the thickness), :math:`\rho_0 = \rho_s`.
Each coefficient is divided by the combination of scales of its unit
(``CharacteristicScales.nondimensionalize_*``):

=====================================================  ===================================
Coefficient                                            Dimensionless value
=====================================================  ===================================
stresses :math:`\mu, \kappa, f_{max}, k_e, k_p`         :math:`\hat\sigma = \sigma/\mu_0`
CE viscosity :math:`\eta_0` (Pa.s)                      :math:`\eta_0/(\mu_0 t_0)`
rates :math:`b, r` (1/s)                                :math:`b\,t_0`
times :math:`\tau`, steps, write period (s)             :math:`t/t_0`
conductivity :math:`k_0/\eta` (m^2/(Pa.s))              :math:`(k_0/\eta)\,\mu_0 t_0/L_0^2`
diffusivities :math:`D_\phi`, :math:`h_\phi\rho_s`      :math:`D\,t_0/L_0^2`
density :math:`\rho_s`                                  :math:`\rho_s/\rho_0`
=====================================================  ===================================

Dimensionless already: :math:`\Phi`'s, :math:`\lambda_{opt}`, :math:`W`, :math:`A`, :math:`f_{ecc}`, :math:`k_{2e}`,
exponents.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

from fenicsoft import io

from hill_playground import integration
from hill_playground.scaling import CharacteristicScales

if TYPE_CHECKING:
	from hill_box_uniaxial import UserInputs


@dataclass(frozen=True)
class Coefficients:
	r"""Every SI input entering the forms, divided by its scale (module docstring).

	Attributes:
	    mu: :math:`\hat\mu`.
	    kappa: :math:`\hat\kappa`.
	    f_max: :math:`\hat f_{max}`.
	    k_e: :math:`\hat k_e`.
	    k_p: :math:`\hat k_p`.
	    eta_0: :math:`\hat\eta_0`.
	    hill_b: :math:`\hat b`.
	    remodel_tau: :math:`\hat\tau`.
	    conductivity: :math:`\hat k_0/\hat\eta`.
	    cellular_diffusion: :math:`\hat D_\phi`.
	    hapt_coeff: :math:`\hat h_\phi` (times :math:`\rho_0`, a diffusivity).
	    rho_s: :math:`\hat\rho_s`.
	    prol_coeff: :math:`\hat r`.
	    steps: Steps in :math:`\hat t`.
	    xdmf_write_period: Write period, ``TimeInterval`` in :math:`\hat t`.
	"""

	mu: float
	kappa: float
	f_max: float
	k_e: float
	k_p: float
	eta_0: float
	hill_b: float
	remodel_tau: float
	conductivity: float
	cellular_diffusion: float
	hapt_coeff: float
	rho_s: float
	prol_coeff: float
	steps: Tuple[io.Step, ...]
	xdmf_write_period: Any

	@classmethod
	def from_inputs(cls, user_inputs: UserInputs) -> Coefficients:
		"""Convert the SI inputs with ``user_inputs.scales``.

		Args:
		    user_inputs: Configuration.

		Returns:
		    The coefficients.
		"""
		s, ui = user_inputs.scales, user_inputs
		return cls(
			mu=s.nondimensionalize_stress(ui.gel_shear_modulus),
			kappa=s.nondimensionalize_stress(ui.gel_kappa),
			f_max=s.nondimensionalize_stress(ui.f_max),
			k_e=s.nondimensionalize_stress(ui.k_e),
			k_p=s.nondimensionalize_stress(ui.k_p),
			eta_0=s.nondimensionalize_time(s.nondimensionalize_stress(ui.eta_0)),
			hill_b=s.nondimensionalize_rate(ui.hill_b),
			remodel_tau=s.nondimensionalize_time(ui.remodel_tau),
			conductivity=s.nondimensionalize_permeability_over_viscosity(ui.hydraulic_conductivity),
			cellular_diffusion=s.nondimensionalize_diffusivity(ui.cellular_diffusion),
			hapt_coeff=s.nondimensionalize_diffusivity(ui.hapt_coeff * s.density),
			rho_s=s.nondimensionalize_density(ui.rho_s),
			prol_coeff=s.nondimensionalize_rate(ui.prol_coeff),
			steps=tuple(_step_in_t_hat(step, s) for step in ui.steps),
			xdmf_write_period=_period_in_t_hat(ui.xdmf_write_period, s),
		)


def _step_in_t_hat(step: io.Step, scales: CharacteristicScales) -> io.Step:
	"""Every time field of ``step`` divided by :math:`t_0`."""
	fields = ("duration", "init_dt", "min_dt", "max_dt", "min_simu_seconds")
	values = {k: getattr(step, k) for k in fields if getattr(step, k, None) is not None}
	return dataclasses.replace(step, **{k: scales.nondimensionalize_time(v) for k, v in values.items()})


def _period_in_t_hat(period: Any, scales: CharacteristicScales) -> Any:
	"""``TimeInterval`` periods divided by :math:`t_0`; substep counts unchanged."""
	if isinstance(period, integration.TimeInterval):
		return integration.TimeInterval(scales.nondimensionalize_time(period.seconds))
	if isinstance(period, (list, tuple)):
		return type(period)(_period_in_t_hat(p, scales) for p in period)
	return period
