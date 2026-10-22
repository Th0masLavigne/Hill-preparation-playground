r"""Residual :math:`R_a` of the contractile element, implicit force form.

.. math::
	R_a = \int_{\Omega_0^{gel}} \Big[ T_{CE} + T_p(\lambda_a) - T_e\big(I_4 / \lambda_a^2\big) \Big]
	      \delta\varepsilon \, \mathrm{d}V, \qquad
	T_{CE} = \underbrace{f_{max}\,\alpha\, f_{FL}(\lambda_a)\, h(D_a)}_{\text{motor}}
	       + \underbrace{\eta_0 D_a}_{\text{damper}}

with :math:`T_0 = f_{max}\alpha f_{FL}` the isometric tension, :math:`h` the force--velocity law,
:math:`T_p` the PEE and :math:`T_e` the SEE tensions (:mod:`.active.hill.functions`).
:math:`T_{CE}` needs :math:`D_a`, hence :math:`\Delta t`: it lives here, not in the law.
Unknown :math:`\varepsilon_a = \ln\lambda_a` (DG0); exponential-map rate (backward Euler on
:math:`\varepsilon_a`, so :math:`\lambda_a > 0` for any step):

.. math::
	D_a = \frac{\varepsilon_a - \varepsilon_a^n}{\Delta t}

No division by :math:`T_0`: valid for :math:`\alpha = 0` (then :math:`T_e = T_p + \eta_0 D_a`).
"""

from typing import Any, Optional

import dolfin
from dolfin_mech import core, operators

from ....materials.cell.active.hill import functions as hill
from ....materials.cell.active.hill.law import Constitutive as HillLaw


class ContractileEvolution(operators.Operator):
	r"""Implicit Hill force--length--velocity evolution of :math:`\varepsilon_a`.

	Args:
	    law: Hill-chain law already summed into the gel material; provides the kinematics,
	        :math:`\mathbf{f}_0`, :math:`\varepsilon_a`, :math:`\varepsilon_a^n`, the chain
	        parameters and the shared ``p_act`` (:math:`\alpha`).
	    test: DG0 test function of the :math:`\varepsilon_a` subsol.
	    dx: Gel measure.
	    scale_factor: Multiplies the residual. :math:`1/\mu_0` for :math:`\tilde R_a = R_a/(\mu_0 L_0^3)` on a
	        mesh rescaled by :math:`L_0` with tensions in Pa (``scale_factor`` method); 1 in SI or with
	        dimensionless coefficients :math:`\hat f_{max}, \hat k_e, \hat k_p, \hat b, \hat\eta_0`.

	Attributes:
	    rate: :math:`D_a` (1/s).
	    isometric_tension: :math:`T_0 = f_{max}\,\alpha\, f_{FL}(\lambda_a)` (Pa).
	    tension_motor: :math:`T_0\, h(D_a)` (Pa).
	    tension_ce: :math:`T_{CE} = T_0\, h(D_a) + \eta_0 D_a` (Pa); :math:`= T_e - T_p` at convergence.
	    tv_dt: Time-step constant, updated by :meth:`set_dt`.
	    res_form: Residual form.
	"""

	def __init__(
		self,
		law: HillLaw,
		test: Any,  # type: ignore
		dx: Optional[dolfin.Measure] = None,
		scale_factor: float = 1.0,
	) -> None:
		"""Builds the residual form."""
		assert dx is not None, "You must provide a gel measure dx."
		self.measure = dx
		self.scale_factor = scale_factor
		self.law = law

		self.tv_dt = core.TimeVaryingConstant(0.0)
		dt = self.tv_dt.val

		p = law.parameters
		self.rate = (p.eps_a - p.eps_a_old) / dt
		if p.use_force_length:
			f_fl = hill.force_length(p.eps_a, p.lmbda_opt, p.W)
		else:
			f_fl = dolfin.Constant(p.default_force_length)
		self.isometric_tension = p.f_max * law.p_act * f_fl
		self.tension_motor = self.isometric_tension * hill.force_velocity(self.rate, p.hill_a, p.hill_b, p.f_ecc)
		self.tension_ce = self.tension_motor + p.eta_0 * self.rate

		self.res_form = scale_factor * (self.tension_ce + law.tension_pee - law.tension_see) * test * self.measure

	def set_dt(self, dt: float) -> None:
		r"""Update :math:`\Delta t` without recompilation.

		Args:
		    dt: Time step (s).
		"""
		self.tv_dt.set_value(dt)
