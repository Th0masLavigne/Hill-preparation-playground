r"""Demo-side helpers of the Hill-chain demos (code.tex, Sec. "Darcy conductivity").

* :func:`pulled_back_conductivity`: :math:`\mathbf{K}_L = (k/\eta)\, J\, \mathbf{C}^{-1}` (Eq. darcy-lag).
* :func:`check_hill_uniqueness`: sufficient condition (Eq. uniqueness) for a unique CE equilibrium.
"""

import math
from typing import Any


def pulled_back_conductivity(k_l_scalar: Any, kinematics: Any) -> Any:
	r"""Lagrangian conductivity :math:`\mathbf{K}_L = (k/\eta)\, J\, \mathbf{C}^{-1}` (Eq. darcy-lag).

	Replaces the frozen :math:`(k/\eta)\,\mathbf{I}`; ``DarcyGradient`` accepts any UFL tensor.

	Args:
	    k_l_scalar: :math:`k/\eta` (m^2/(Pa.s)).
	    kinematics: Object exposing ``J`` and ``C_inv``.

	Returns:
	    UFL tensor.
	"""
	return k_l_scalar * kinematics.J * kinematics.C_inv


def check_hill_uniqueness(k_e: float, k_p: float, f_max: float, lmbda_opt: float, width: float) -> float:
	r"""Check :math:`k_e + k_p > f_{max} \max_\lambda |f_{FL}'(\lambda)\,\lambda|` (Eq. uniqueness).

	Args:
	    k_e: SEE stiffness (Pa).
	    k_p: PEE stiffness (Pa).
	    f_max: Maximal isometric tension (Pa).
	    lmbda_opt: :math:`\lambda_{opt}`.
	    width: :math:`W`.

	Returns:
	    Margin :math:`(k_e + k_p) / (f_{max} \max|f_{FL}'\lambda|)`, > 1.

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
		raise ValueError(f"Hill chain not unique: k_e + k_p = {k_e + k_p:.3g} Pa <= {bound:.3g} Pa.")
	return (k_e + k_p) / bound
