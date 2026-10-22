r"""UFL building blocks of the cell Hill chain SEE -- (CE :math:`\parallel` PEE).

Series elastic (SEE), contractile (CE: motor and optional damper) and parallel elastic (PEE) elements.
Pure functions shared by the stress law (:math:`R_u`) and the CE evolution operator (:math:`R_a`).

Rheological scheme (fibre direction :math:`\mathbf{f}_0`):

.. image:: /_static/resources/hill_rheology.svg
	:alt: Hill chain: SEE in series with CE, optional damper and PEE in parallel
	:width: 80%
	:align: center

Series equilibrium (same tension through the chain):

.. math::
	T = T_e(I_{4e}) = T_{CE} + T_p(\varepsilon_a), \qquad
	T_{CE} = T_0\, h(D_a) + \eta_0 D_a

(motor :math:`T_0 h` and optional damper :math:`\eta_0 D_a`, ``ContractileEvolution.tension_ce``).

Internal variable: the CE log-stretch :math:`\varepsilon_a = \ln\lambda_a` (multiplicative split of the
fibre stretch :math:`\lambda = \lambda_e\lambda_a`), hence

.. math::
	\lambda_a = e^{\varepsilon_a}, \qquad I_{4e} = I_4\, e^{-2\varepsilon_a}, \qquad
	D_a = \dot\varepsilon_a .

Units: tensions in Pa, rates in 1/s.
"""

from typing import Any, Literal, get_args

import dolfin

from .....ufl_functions import floored, macaulay, macaulay_negative, piecewise

SeeVariant = Literal["mooney", "tension_only", "exponential"]
"""SEE law: ``"mooney"`` (baseline), ``"tension_only"``, ``"exponential"`` (see :func:`see_tension`)."""

SEE_VARIANTS: tuple[str, ...] = get_args(SeeVariant)
"""Values of :data:`SeeVariant`, for runtime checks (``Literal`` is static only)."""


def fibre_invariant(kinematics: Any, f0: Any, floor: float = 1e-6) -> Any:
	r""":math:`I_4 = \mathbf{f}_0 \cdot \mathbf{C}\,\mathbf{f}_0`, floored (squared fibre stretch).

	Args:
	    kinematics: Object exposing ``C``.
	    f0: Unit reference fibre direction.
	    floor: Lower bound of :math:`I_4`.

	Returns:
	    UFL scalar.
	"""
	return floored(dolfin.inner(kinematics.C * f0, f0), floor)


def elastic_invariant(inv4: Any, eps_a: Any) -> Any:
	r""":math:`I_{4e} = I_4 / \lambda_a^2 = I_4\, e^{-2\varepsilon_a}` (squared SEE stretch).

	Args:
	    inv4: :math:`I_4`.
	    eps_a: :math:`\varepsilon_a`.

	Returns:
	    UFL scalar.
	"""
	return inv4 * dolfin.exp(-2.0 * eps_a)


def see_tension(inv4e: Any, k_e: Any, variant: str = "mooney", k_2e: Any = 0.0) -> Any:
	r"""SEE tension :math:`T_e = 2 I_{4e}\, \partial\Psi_e / \partial I_{4e}`.

	- ``"mooney"``: :math:`T_e = \tfrac{k_e}{4}(I_{4e} - I_{4e}^{-1})`
	- ``"tension_only"``: :math:`T_e = \tfrac{k_e}{2} I_{4e} \langle I_{4e}-1 \rangle_+`
	- ``"exponential"``:
	  :math:`T_e = \tfrac{k_e}{2} I_{4e} \langle I_{4e}-1 \rangle_+ e^{k_{2e}\langle I_{4e}-1 \rangle_+^2}`

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

	- ``"mooney"``: :math:`\Psi_e = \tfrac{k_e}{8}(I_{4e} + I_{4e}^{-1} - 2)`
	- ``"tension_only"``: :math:`\Psi_e = \tfrac{k_e}{8}\langle I_{4e}-1 \rangle_+^2`
	- ``"exponential"``: :math:`\Psi_e = \tfrac{k_e}{8 k_{2e}}\big(e^{k_{2e}\langle I_{4e}-1 \rangle_+^2} - 1\big)`

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

	Mooney law of the SEE (:func:`see_tension`) evaluated at :math:`\lambda_a^2 = e^{2\varepsilon_a}`.

	Args:
	    eps_a: :math:`\varepsilon_a`.
	    k_p: PEE initial stiffness (Pa).

	Returns:
	    UFL scalar (Pa).
	"""
	return see_tension(dolfin.exp(2.0 * eps_a), k_p, "mooney")


def pee_energy(eps_a: Any, k_p: Any) -> Any:
	r"""PEE energy :math:`\Psi_p = \tfrac{k_p}{8}(\lambda_a^2 + \lambda_a^{-2} - 2)`.

	Mooney law of the SEE (:func:`see_energy`) evaluated at :math:`\lambda_a^2 = e^{2\varepsilon_a}`.

	Args:
	    eps_a: :math:`\varepsilon_a`.
	    k_p: PEE initial stiffness (Pa).

	Returns:
	    UFL scalar (Pa).
	"""
	return see_energy(dolfin.exp(2.0 * eps_a), k_p, "mooney")


def force_length(eps_a: Any, lmbda_opt: Any, width: Any) -> Any:
	r"""Gaussian force--length law on the CE stretch.

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
	r"""Hill force--velocity :math:`h(D_a)`: Hill hyperbola when shortening, saturating when lengthening.

	.. math::
		h_{con}(D_a) = \frac{(1+A)\,b}{b - D_a} - A \quad (D_a \le 0), \qquad
		h_{ecc}(D_a) = f_{ecc} - (f_{ecc}-1)\frac{b_e}{b_e + D_a} \quad (D_a > 0), \qquad
		b_e = \frac{(f_{ecc}-1)\,b}{1+A}

	:math:`h(0) = 1`, :math:`h'(0) = (1+A)/b` on both sides, :math:`h` strictly increasing,
	:math:`h \ge 0` on :math:`D_a \ge -b/A` (below, the hyperbola is extrapolated: :math:`h < 0`).
	Each branch is evaluated on its half-line (Macaulay brackets: no division by zero in the unused one):

	.. math::
		h(D_a) = h_{con}\big(\langle D_a \rangle_-\big) \ (D_a \le 0), \qquad
		h(D_a) = h_{ecc}\big(\langle D_a \rangle_+\big) \ (D_a > 0)

	Args:
	    rate: :math:`D_a` (1/s).
	    hill_a: Curvature :math:`A` (-).
	    hill_b: Rate constant :math:`b` (1/s).
	    f_ecc: Eccentric saturation :math:`f_{ecc}` (-).

	Returns:
	    UFL scalar (-).
	"""
	b_ecc = (f_ecc - 1.0) * hill_b / (1.0 + hill_a)
	concentric = (1.0 + hill_a) * hill_b / (hill_b - macaulay_negative(rate)) - hill_a
	eccentric = f_ecc - (f_ecc - 1.0) * b_ecc / (b_ecc + macaulay(rate))
	return piecewise(rate, 0.0, concentric, eccentric)
