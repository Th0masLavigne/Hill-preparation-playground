r"""Demo-side helpers of the Hill-chain demos.

* :func:`check_hill_uniqueness`: sufficient condition for a unique contractile-element (CE) equilibrium.
"""

import math


def check_hill_uniqueness(k_e: float, k_p: float, f_max: float, lmbda_opt: float, width: float) -> float:
	r"""Check :math:`k_e + k_p > f_{max} \max_\lambda |f_{FL}'(\lambda)\,\lambda|`.

	With :math:`f_{FL}(\lambda) = \exp(-((\lambda - \lambda_{opt})/W)^2)`, the series (SEE, :math:`k_e`) and
	parallel (PEE, :math:`k_p`) stiffnesses then dominate the softening of the force--length law: the CE
	equilibrium :math:`\lambda_a` is unique (no descending-limb instability). The max is sampled on
	:math:`\lambda \in [0.05, 3]`.

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
