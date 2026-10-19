r"""0D (homogeneous) reference of the Hill chain in the gel, written from the model equations only.

Independent of the UFL implementation (numpy + scipy), used by the one-element integration tests.

Homogeneous state, fibre :math:`\mathbf f_0 = \mathbf e_x`,
:math:`\mathbf F = \mathrm{diag}(\lambda, \lambda_t, \lambda_t)`:

.. math::
	I_4 = \lambda^2, \quad J = \lambda\lambda_t^2, \quad I_1 = \lambda^2 + 2\lambda_t^2, \quad
	I_{4e} = I_4 e^{-2\varepsilon_a}

Gel (Neo-Hooke + quadratic volumetric) and chain (SEE stress), PK2 diagonal:

.. math::
	S^{gel}_{ii} = \mu J^{-2/3}\Big(1 - \frac{I_1}{3 C_{ii}}\Big) + \kappa J (J-1) \frac{1}{C_{ii}}, \qquad
	S^h_{xx} = \Phi_c \frac{T_e(I_{4e})}{I_4}

Contractile element, backward Euler on :math:`\varepsilon_a`:

.. math::
	r_a = f_{max}\alpha f_{FL}(e^{\varepsilon_a}) h(D_a) + \eta_0 D_a + T_p(\varepsilon_a) - T_e(I_{4e}) = 0,
	\qquad D_a = \frac{\varepsilon_a - \varepsilon_a^n}{\Delta t}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, cast

import numpy as np
from numpy.typing import NDArray
from scipy import optimize


@dataclass(frozen=True)
class HillChain0D:
	r"""Scalar Hill-chain parameters (SI, ``"mooney"`` SEE, Gaussian :math:`f_{FL}`).

	Attributes:
	    mu: Gel shear modulus :math:`\mu` (Pa).
	    kappa: Gel drained bulk penalty :math:`\kappa` (Pa).
	    phi_c: :math:`\Phi_c`.
	    f_max: :math:`f_{max}` (Pa).
	    k_e: :math:`k_e` (Pa).
	    k_p: :math:`k_p` (Pa).
	    lmbda_opt: :math:`\lambda_{opt}`.
	    width: :math:`W`.
	    hill_a: :math:`A`.
	    hill_b: :math:`b` (1/s).
	    f_ecc: :math:`f_{ecc}`.
	    eta_0: :math:`\eta_0` (Pa.s).
	"""

	mu: float = 1700.0
	kappa: float = 1417.0
	phi_c: float = 0.05
	f_max: float = 20e3
	k_e: float = 200e3
	k_p: float = 5e3
	lmbda_opt: float = 1.05
	width: float = 0.3
	hill_a: float = 0.25
	hill_b: float = 0.05
	f_ecc: float = 1.8
	eta_0: float = 0.0

	def tension_see(self, inv4e: float) -> float:
		r""":math:`T_e = \tfrac{k_e}{4}(I_{4e} - I_{4e}^{-1})`."""
		return 0.25 * self.k_e * (inv4e - 1.0 / inv4e)

	def tension_pee(self, eps_a: float) -> float:
		r""":math:`T_p = \tfrac{k_p}{2}\sinh 2\varepsilon_a`."""
		return 0.5 * self.k_p * math.sinh(2.0 * eps_a)

	def force_length(self, eps_a: float) -> float:
		r""":math:`f_{FL} = \exp(-((e^{\varepsilon_a} - \lambda_{opt})/W)^2)`."""
		return math.exp(-(((math.exp(eps_a) - self.lmbda_opt) / self.width) ** 2))

	def force_velocity(self, rate: float) -> float:
		r"""Hill hyperbola (:math:`D_a \le 0`) and eccentric branch (:math:`D_a > 0`)."""
		if rate <= 0.0:
			return (1.0 + self.hill_a) * self.hill_b / (self.hill_b - rate) - self.hill_a
		b_ecc = (self.f_ecc - 1.0) * self.hill_b / (1.0 + self.hill_a)
		return self.f_ecc - (self.f_ecc - 1.0) * b_ecc / (b_ecc + rate)

	def ce_residual(self, eps_a: float, eps_old: float, dt: float, alpha: float, inv4: float) -> float:
		""":math:`r_a` (Pa)."""
		rate = (eps_a - eps_old) / dt
		motor = self.f_max * alpha * self.force_length(eps_a) * self.force_velocity(rate) + self.eta_0 * rate
		return motor + self.tension_pee(eps_a) - self.tension_see(inv4 * math.exp(-2.0 * eps_a))

	def gel_pk2(self, lmbda: float, lmbda_t: float) -> tuple[float, float]:
		r"""Gel :math:`(S_{xx}, S_{yy})` (Pa)."""
		jac = lmbda * lmbda_t**2
		inv1 = lmbda**2 + 2.0 * lmbda_t**2

		def comp(c_ii: float) -> float:
			return self.mu * jac ** (-2.0 / 3.0) * (1.0 - inv1 / (3.0 * c_ii)) + self.kappa * jac * (jac - 1.0) / c_ii

		return comp(lmbda**2), comp(lmbda_t**2)

	def total_pk1_xx(self, lmbda: float, lmbda_t: float, eps_a: float) -> float:
		r""":math:`P_{xx} = \lambda (S^{gel}_{xx} + S^h_{xx})` (Pa)."""
		inv4 = lmbda**2
		s_h = self.phi_c * self.tension_see(inv4 * math.exp(-2.0 * eps_a)) / inv4
		return lmbda * (self.gel_pk2(lmbda, lmbda_t)[0] + s_h)


def isometric_history(
	model: HillChain0D, lmbda: float, times: Iterable[float], alphas: Iterable[float]
) -> NDArray[np.float64]:
	r""":math:`\varepsilon_a(t_n)` at fixed :math:`\lambda` (:math:`\lambda_t = 1`), backward Euler.

	Args:
	    model: Parameters.
	    lmbda: Imposed fibre stretch.
	    times: :math:`t_1 < t_2 < \dots` (s), :math:`t_0 = 0`.
	    alphas: :math:`\alpha(t_n)`.

	Returns:
	    :math:`\varepsilon_a(t_n)`.
	"""
	eps, t_old, out = 0.0, 0.0, []
	for t, alpha in zip(times, alphas):
		dt = t - t_old
		eps = optimize.brentq(model.ce_residual, eps - 1.0, eps + 1.0, args=(eps, dt, alpha, lmbda**2), xtol=1e-15)
		out.append(eps)
		t_old = t
	return np.array(out)


def free_history(model: HillChain0D, times: Iterable[float], alphas: Iterable[float]) -> NDArray[np.float64]:
	r""":math:`(\lambda, \lambda_t, \varepsilon_a)(t_n)` with traction-free faces (:math:`P_{xx} = P_{yy} = 0`).

	Args:
	    model: Parameters.
	    times: :math:`t_1 < t_2 < \dots` (s), :math:`t_0 = 0`.
	    alphas: :math:`\alpha(t_n)`.

	Returns:
	    Array of shape ``(n, 3)``.
	"""
	state, t_old, out = np.array([1.0, 1.0, 0.0]), 0.0, []
	for t, alpha in zip(times, alphas):
		dt, eps_old = t - t_old, state[2]

		def residual(
			x: NDArray[np.float64], dt: float = dt, eps_old: float = eps_old, alpha: float = alpha
		) -> list[float]:
			lmbda, lmbda_t, eps = x
			s_xx, s_yy = model.gel_pk2(lmbda, lmbda_t)
			inv4 = lmbda**2
			s_xx += model.phi_c * model.tension_see(inv4 * math.exp(-2.0 * eps)) / inv4
			return [s_xx / model.mu, s_yy / model.mu, model.ce_residual(eps, eps_old, dt, alpha, inv4) / model.f_max]

		state, *_ = optimize.fsolve(residual, state, xtol=1e-12, full_output=True)
		if max(abs(r) for r in residual(state)) > 1e-12:
			raise RuntimeError(f"0D free state not converged at t = {t}.")
		out.append(state.copy())
		t_old = t
	return np.array(out)


def isometric_equilibrium(model: HillChain0D, lmbda: float, alpha: float) -> float:
	r"""Steady :math:`\varepsilon_a` (:math:`D_a = 0`, :math:`h = 1`) at fixed :math:`\lambda`.

	Args:
	    model: Parameters.
	    lmbda: Imposed fibre stretch.
	    alpha: Activation.

	Returns:
	    :math:`\varepsilon_a^\infty`, root of :math:`f_{max}\alpha f_{FL} + T_p - T_e = 0`.
	"""

	def residual(eps: float) -> float:
		return model.ce_residual(eps, eps, 1.0, alpha, lmbda**2)

	return cast(float, optimize.brentq(residual, -1.0, 0.5, xtol=1e-15))
