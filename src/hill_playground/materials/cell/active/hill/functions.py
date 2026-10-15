r"""UFL building blocks of the cell Hill chain SEE -- (CE :math:`\parallel` PEE).

equations.tex, Sec. "The cell Hill chain". Pure functions shared by the stress law
(:math:`R_u`) and the CE evolution operator (:math:`R_a`), so that :math:`T_e`, :math:`T_p`,
:math:`f_{FL}` and :math:`h` are written once.

Internal variable: :math:`\varepsilon_a = \ln\lambda_a` (appendix "eps_a"), hence

.. math::
	\lambda_a = e^{\varepsilon_a}, \qquad I_{4e} = I_4\, e^{-2\varepsilon_a}, \qquad
	D_a = \dot\varepsilon_a .

Every function works on SI tensions (Pa) and rates (1/s).
"""

from typing import Any

import dolfin

SEE_VARIANTS = ("mooney", "tension_only", "exponential")
"""Available SEE laws: ``"mooney"`` (baseline, Eq. springs), ``"tension_only"``, ``"exponential"``."""


def floored(value: Any, floor: float) -> Any:
	r"""Strictly positive floor :math:`\max(x, x_{min})` (same guard as ``aniso.law.f_force_length``).

	Args:
	    value: UFL scalar :math:`x`.
	    floor: :math:`x_{min} > 0`.

	Returns:
	    UFL scalar.
	"""
	return dolfin.conditional(dolfin.gt(value, floor), value, floor)


def macaulay(value: Any) -> Any:
	r"""Positive part :math:`\langle x \rangle_+ = \max(x, 0)`.

	Args:
	    value: UFL scalar :math:`x`.

	Returns:
	    UFL scalar.
	"""
	return dolfin.conditional(dolfin.gt(value, 0.0), value, 0.0)


def fibre_invariant(kinematics: Any, f0: Any, floor: float = 1e-6) -> Any:
	r""":math:`I_4 = \mathbf{f}_0 \cdot \mathbf{C}\,\mathbf{f}_0`, floored (Eq. I4).

	Args:
	    kinematics: Object exposing ``C``.
	    f0: Unit reference fibre direction.
	    floor: Lower bound of :math:`I_4`.

	Returns:
	    UFL scalar.
	"""
	return floored(dolfin.inner(kinematics.C * f0, f0), floor)


def elastic_invariant(inv4: Any, eps_a: Any) -> Any:
	r""":math:`I_{4e} = I_4 / \lambda_a^2 = I_4\, e^{-2\varepsilon_a}` (Eq. Psi-h).

	Args:
	    inv4: :math:`I_4`.
	    eps_a: :math:`\varepsilon_a`.

	Returns:
	    UFL scalar.
	"""
	return inv4 * dolfin.exp(-2.0 * eps_a)


def see_tension(inv4e: Any, k_e: Any, variant: str = "mooney", k_2e: Any = 0.0) -> Any:
	r"""SEE tension :math:`T_e = 2 I_{4e}\, \partial\Psi_e / \partial I_{4e}` (Eq. springs and variants).

	.. math::
		\text{mooney: } T_e = \tfrac{k_e}{4}(I_{4e} - I_{4e}^{-1}), \quad
		\text{tension\_only: } T_e = \tfrac{k_e}{2} I_{4e} \langle I_{4e}-1 \rangle_+, \quad
		\text{exponential: } T_e = \tfrac{k_e}{2} I_{4e} \langle I_{4e}-1 \rangle_+
		e^{k_{2e}\langle I_{4e}-1 \rangle_+^2}

	Args:
	    inv4e: :math:`I_{4e}`.
	    k_e: SEE initial stiffness (Pa).
	    variant: One of :data:`SEE_VARIANTS`.
	    k_2e: Stiffening exponent (``"exponential"`` only).

	Returns:
	    UFL scalar (Pa).

	Raises:
	    ValueError: Unknown ``variant``.
	"""
	if variant == "mooney":
		return 0.25 * k_e * (inv4e - 1.0 / inv4e)
	stretch_excess = macaulay(inv4e - 1.0)
	if variant == "tension_only":
		return 0.5 * k_e * inv4e * stretch_excess
	if variant == "exponential":
		return 0.5 * k_e * inv4e * stretch_excess * dolfin.exp(k_2e * stretch_excess**2)
	raise ValueError(f"Unknown SEE variant '{variant}', expected one of {SEE_VARIANTS}.")


def see_energy(inv4e: Any, k_e: Any, variant: str = "mooney", k_2e: Any = 0.0) -> Any:
	r"""SEE energy :math:`\Psi_e` per unit reference cell volume.

	.. math::
		\text{mooney: } \Psi_e = \tfrac{k_e}{8}(I_{4e} + I_{4e}^{-1} - 2), \quad
		\text{tension\_only: } \tfrac{k_e}{8}\langle I_{4e}-1 \rangle_+^2, \quad
		\text{exponential: } \tfrac{k_e}{8 k_{2e}}\big(e^{k_{2e}\langle I_{4e}-1 \rangle_+^2} - 1\big)

	Args:
	    inv4e: :math:`I_{4e}`.
	    k_e: SEE initial stiffness (Pa).
	    variant: One of :data:`SEE_VARIANTS`.
	    k_2e: Stiffening exponent (``"exponential"`` only, > 0).

	Returns:
	    UFL scalar (Pa).

	Raises:
	    ValueError: Unknown ``variant``.
	"""
	if variant == "mooney":
		return k_e / 8.0 * (inv4e + 1.0 / inv4e - 2.0)
	stretch_excess = macaulay(inv4e - 1.0)
	if variant == "tension_only":
		return k_e / 8.0 * stretch_excess**2
	if variant == "exponential":
		return k_e / (8.0 * k_2e) * (dolfin.exp(k_2e * stretch_excess**2) - 1.0)
	raise ValueError(f"Unknown SEE variant '{variant}', expected one of {SEE_VARIANTS}.")


def pee_tension(eps_a: Any, k_p: Any) -> Any:
	r"""PEE tension :math:`T_p = \tfrac{k_p}{4}(\lambda_a^2 - \lambda_a^{-2}) = \tfrac{k_p}{2}\sinh(2\varepsilon_a)`.

	Args:
	    eps_a: :math:`\varepsilon_a`.
	    k_p: PEE initial stiffness (Pa).

	Returns:
	    UFL scalar (Pa).
	"""
	return 0.25 * k_p * (dolfin.exp(2.0 * eps_a) - dolfin.exp(-2.0 * eps_a))


def pee_energy(eps_a: Any, k_p: Any) -> Any:
	r"""PEE energy :math:`\Psi_p = \tfrac{k_p}{8}(\lambda_a^2 + \lambda_a^{-2} - 2)`.

	Args:
	    eps_a: :math:`\varepsilon_a`.
	    k_p: PEE initial stiffness (Pa).

	Returns:
	    UFL scalar (Pa).
	"""
	return k_p / 8.0 * (dolfin.exp(2.0 * eps_a) + dolfin.exp(-2.0 * eps_a) - 2.0)


def force_length(eps_a: Any, lmbda_opt: Any, width: Any) -> Any:
	r"""Gaussian force--length law on the CE stretch (Eq. ce-law).

	.. math::
		f_{FL}(\lambda_a) = \exp\big(-((\lambda_a - \lambda_{opt})/W)^2\big), \qquad \lambda_a = e^{\varepsilon_a}

	Args:
	    eps_a: :math:`\varepsilon_a`.
	    lmbda_opt: :math:`\lambda_{opt}`.
	    width: :math:`W`.

	Returns:
	    UFL scalar in :math:`(0, 1]`.
	"""
	return dolfin.exp(-(((dolfin.exp(eps_a) - lmbda_opt) / width) ** 2))


def force_velocity(rate: Any, hill_a: Any, hill_b: Any, f_ecc: Any) -> Any:
	r"""Hill force--velocity :math:`h(D_a)` (Eqs. hill-hyperbola, hill-ecc).

	.. math::
		h = \frac{(1+A)\,b}{b - D_a} - A \quad (D_a \le 0), \qquad
		h = f_{ecc} - (f_{ecc}-1)\frac{b_e}{b_e + D_a} \quad (D_a > 0), \qquad
		b_e = \frac{(f_{ecc}-1)\,b}{1+A}

	:math:`h(0) = 1`, :math:`h'(0) = (1+A)/b` on both sides, :math:`h` strictly increasing.
	Each branch is evaluated on its own half-line, so the unused one never divides by zero.

	Args:
	    rate: :math:`D_a` (1/s).
	    hill_a: Curvature :math:`A` (-).
	    hill_b: Rate constant :math:`b` (1/s).
	    f_ecc: Eccentric saturation :math:`f_{ecc}` (-).

	Returns:
	    UFL scalar (-).
	"""
	b_ecc = (f_ecc - 1.0) * hill_b / (1.0 + hill_a)
	rate_neg = dolfin.conditional(dolfin.le(rate, 0.0), rate, 0.0)
	rate_pos = dolfin.conditional(dolfin.gt(rate, 0.0), rate, 0.0)
	concentric = (1.0 + hill_a) * hill_b / (hill_b - rate_neg) - hill_a
	eccentric = f_ecc - (f_ecc - 1.0) * b_ecc / (b_ecc + rate_pos)
	return dolfin.conditional(dolfin.le(rate, 0.0), concentric, eccentric)
