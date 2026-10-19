r"""Hill box: an active poro-elastic gel slab clamped on two opposite faces.

A hydrogel slab seeded with randomly oriented contractile cells (Hill chain) is clamped at both
ends of its long axis :math:`x` and drained on its four other faces. Under the activation
:math:`\alpha(t)` the cells contract: the gel shortens across :math:`x`, expels liquid, and the
cells may migrate, proliferate and reorient.

1. Geometry and boundary conditions
-----------------------------------
Box :math:`\Omega_0 = [0, L_x] \times [0, L_y] \times [0, L_z]` (``BoxGeometry``), all gel
(default :math:`875 \times 375 \times 50\ \mu\mathrm{m}`).

.. code-block:: text

                         TOP (z = L_z)
                 +---------------------------+
                /|      BACK (y = L_y)      /|
               / |                         / |
              +---------------------------+  |
     LEFT     |  |                        |  |     RIGHT
     x = 0    |  +------------------------|--+     x = L_x
     u = 0    | /                         | /      u = 0
              |/       FRONT (y = 0)      |/
              +---------------------------+
                       BOTTOM (z = 0)
       z  y
       | /        LEFT, RIGHT              : u = 0, impermeable
       |/___ x    FRONT, BACK, TOP, BOTTOM : p = 0 (drained), traction-free

* :math:`\mathbf{u} = \mathbf{0}` on ``LEFT`` (:math:`x=0`) and ``RIGHT`` (:math:`x=L_x`), impermeable there;
* :math:`p = 0` (drained) and traction-free on ``FRONT``, ``BACK``, ``BOTTOM``, ``TOP``;
* no cell flux anywhere (natural).

2. Unknowns (registration order = ``sol_tol`` order)
------------------------------------------------------
=========================  ======  =========================================  ==================
Unknown                    Space   Initial value                              Present if
=========================  ======  =========================================  ==================
:math:`\hat{\mathbf u}`    CG2     0                                          always
:math:`\theta,\varphi`     DG0     isotropic draw                             ``remodelling``
:math:`\hat p`             CG1     0                                          always
:math:`\Phi_c`             CG1     :math:`\Phi_{c,0}`                         ``cell_transport``
:math:`\tilde\phi_s`       CG1     :math:`\Phi_{s,0}`                         ``cell_transport``
:math:`\varepsilon_a`      DG0     0 (:math:`\lambda_a = 1`)                  always
=========================  ======  =========================================  ==================

Switched off, :math:`\theta,\varphi` stay at their random draw and :math:`\Phi_c = \Phi_{c,0}`
(uniform data): the remaining problem is :math:`(R_u, R_p, R_a)`, the poro-elastic gel with the
Hill chain.

3. Residuals
------------
.. math::
	R_u &= \int \big(\mathbf{P}_S + \mathbf{P}_h - J p \mathbf{F}^{-T}\big) : \nabla\delta\mathbf{u}\,\mathrm{d}V \\
	R_p &= \int \frac{J - J^n}{\Delta t}\delta p + \mathbf{K}_L \nabla p \cdot \nabla\delta p
	       - \mathbf{Q}_c \cdot \nabla\delta p \,\mathrm{d}V, \\
	\mathbf{K}_L &= \frac{k(\phi_\ell)}{\eta} J \mathbf{C}^{-1} \\
	R_c &= \int \frac{\Phi_c - \Phi_c^n}{\Delta t}\delta c - \mathbf{Q}_c \cdot \nabla\delta c
	       - r\,\mathrm{sat}\,\Phi_c\,\delta c \,\mathrm{d}V \\
	R_{\theta} &= \int \Big(\frac{\theta - \theta^n}{\Delta t}
	       - \frac{\alpha}{\tau_{eff}}\partial_\theta Q\Big)\delta\theta\,\mathrm{d}V \quad (\text{idem } \varphi) \\
	R_a &= \int \big[f_{max}\alpha f_{FL}(\lambda_a) h(D_a) + \eta_0 D_a + T_p - T_e\big]
	       \delta\varepsilon\,\mathrm{d}V \\
	R_s &= \int \Big(\tilde\phi_s - \frac{\Phi_{s,0}}{J}\Big)\delta\phi_s\,\mathrm{d}V

with :math:`\mathbf{P}_S` the Neo-Hookean + volumetric gel (section 5),
:math:`\mathbf{P}_h = \mathbf{F}\,\Phi_c T_e/I_4\,\mathbf{f}_0\otimes\mathbf{f}_0` (Hill chain),
:math:`I_4 = \mathbf f_0\cdot\mathbf C\mathbf f_0`, :math:`\mathbf{Q}_c` the diffusive + haptotactic cell
flux, :math:`\mathrm{sat} = \langle 1 - \Phi_c/\Phi_{c,max}\rangle_+` and :math:`Q` the orientation potential. In
:math:`R_a`, the cell is an elastic element (SEE, tension :math:`T_e`) in series with a contractile
element (CE, stretch :math:`\lambda_a = e^{\varepsilon_a}`, rate :math:`D_a = \dot\varepsilon_a`) itself in
parallel with a passive element (PEE, tension :math:`T_p`); :math:`f_{FL}` is the Gaussian force--length law
and :math:`h` the Hill force--velocity law (``hill_playground.materials.cell.active.hill``).

4. Nondimensionalisation
------------------------
Inputs are SI. Scales :math:`L_0 = L_z` (drainage length), :math:`\mu_0 = \mu_{gel}`,
:math:`t_0 = L_0^2\eta/(k\,\kappa_{gel})` (``CharacteristicScales``). ``nondimensionalization``:

* ``"none"``: SI mesh, unknowns and coefficients, every row factor 1;
* ``"operator"`` (default): mesh :math:`\hat{\mathbf X} = \mathbf X/L_0` (hence
  :math:`\hat{\mathbf u} = \mathbf u/L_0`), :math:`\hat p = p/\mu_0`, SI coefficients and SI time,
  each operator multiplied by ``scale_factor`` :math:`= s\,v_k/\mathcal N_{row}` (table below);
* ``"parameters"``: same mesh and unknowns, every coefficient converted with
  ``CharacteristicScales.nondimensionalize_*`` (:math:`\hat\mu, \hat\kappa, \hat f_{max}, \hat K,
  \hat D, \hat r, \hat b, \hat\tau, \hat t \dots`), every row factor 1.

Rows are named after the subsol of their test function (``ResidualRows``): for ``"operator"``,
``scale_factor`` :math:`= s\,v/\mathcal N_{row}` with :math:`v` the term's value-preserving factor
(``TermValues``: SI integral = :math:`v` times the form on the scaled mesh and unknowns, FE test function),
:math:`s` the scale of the row's test function (:math:`\delta\mathbf u = L_0\delta\hat{\mathbf u}`,
:math:`\delta p = \mu_0\delta\hat p`) and :math:`\mathcal N_{row}` the row normalization
(:math:`\mu_0 L_0^3` for ``displacement`` and ``eps_a``, :math:`\mu_0 L_0^3/t_0` for
``liquid_pressure``, :math:`L_0^3/t_0` for ``cell_fraction``, ``theta``, ``phi``, :math:`L_0^3` for
``substrate_smoothed``):

=================================  ======================  ===================  =======================
Term (``"operator"``)              Row (subsol)            :math:`s\,v`         ``scale_factor``
=================================  ======================  ===================  =======================
gel stresses, Hill SEE             ``displacement``        :math:`L_0^3`        :math:`1/\mu_0`
:math:`-Jp\mathbf F^{-T}`          ``displacement``        :math:`\mu_0 L_0^3`  1
:math:`(J-J^n)/\Delta t`           ``liquid_pressure``     :math:`\mu_0 L_0^3`  :math:`t_0`
Darcy                              ``liquid_pressure``     :math:`\mu_0^2 L_0`  :math:`\mu_0 t_0/L_0^2`
cell flux, :math:`\delta p`        ``liquid_pressure``     :math:`\mu_0 L_0`    :math:`t_0/L_0^2`
:math:`\dot\Phi_c`, proliferation  ``cell_fraction``       :math:`L_0^3`        :math:`t_0`
cell flux, :math:`\delta c`        ``cell_fraction``       :math:`L_0`          :math:`t_0/L_0^2`
substrate projection               ``substrate_smoothed``  :math:`L_0^3`        1
orientation                        ``theta``, ``phi``      :math:`L_0^3`        :math:`t_0`
contractile element                ``eps_a``               :math:`L_0^3`        :math:`1/\mu_0`
=================================  ======================  ===================  =======================

The resulting system is the one of ``"parameters"`` (same Jacobian, checked by
``tests/integration/demos/test_hill_box_scaling.py``).

``quadrature_degrees`` (same row names) sets a quadrature degree per row (``ResidualRows.measure``).

Both scaled modes are the SI system times a constant per row: Newton's iterates are identical to SI
(``tests/integration/analytical_consolidation_1D/test_terzaghi_nondim.py``). ``"parameters"``
integrates in :math:`\hat t` (QOI ``time_si`` gives :math:`t` in s). After the run the XDMF geometry and
time stamps are brought back to SI (``hill_playground.io.xdmf_units``): no ``ScaledKinematics`` is
needed since the mesh itself is scaled, so every gradient and measure is in :math:`\hat{\mathbf X}`.
With ``xdmf_si_only`` (default) the XDMF holds only SI fields: the dimensionless unknowns
(:math:`\theta, \varphi, \Phi_c, \tilde\phi_s, \varepsilon_a`) and the FOIs, no :math:`\hat{\mathbf u}`,
:math:`\hat p` nor ``*_old`` copies.
Exported fields and QOIs are SI in every mode, with the same names, order and spaces
(:math:`\mathbf u`, :math:`p` projected on the spaces of the unknowns: point data in ParaView).

5. Skeleton options
-------------------
:math:`J_c = \Phi_{s,0} + \Phi_c` (compaction point, :math:`\phi_\ell = 0`).

* ``permeability_law``: ``"constant"`` :math:`k_0`; ``"kozeny_carman"``; ``"holmes_mow"``;
  ``"compaction_power"`` :math:`k = k_0((J-J_c)/(1-J_{c,0}))^n`, :math:`n\ge1` (kinetic barrier:
  :math:`J > J_c` for all :math:`t`), see ``hill_playground.materials.permeability``;
* ``volumetric_law``: ``"quadratic"`` :math:`\frac{\kappa}{2}(J-1)^2`; ``"compaction"``
  :math:`\kappa(1-J_c)^2[x - \ln x - 1]`, :math:`x = (J-J_c)/(1-J_c)` (energetic barrier,
  :math:`J_\infty \in (J_c, 1)`), see ``hill_playground.materials.volumetric.compaction``.

6. Orientation outputs (``hill_playground.materials.orientation.display``)
---------------------------------------------------------------------------
Reference :math:`\mathbf f_0(\theta,\varphi)` and current
:math:`\mathbf f = \mathbf F\mathbf f_0/\|\mathbf F\mathbf f_0\|`:
``theta_deg``/``phi_deg`` (canonical, upper hemisphere), ``theta_wrapped_deg`` (in-plane axis,
period 180), ``tilt_xy_deg`` (0: fibre in the :math:`xy` plane), ``angle_x_deg`` (0: along the
held axis), with ``_current`` variants; QOIs ``order_x``/``order_z``
(:math:`\langle\frac32 f_i^2-\frac12\rangle`: 1 aligned, 0 isotropic, :math:`-\frac12` orthogonal;
``order_z`` :math:`\to -\frac12` = planar orientation) and ``mean_tilt_xy_deg``.

7. Loading
------------
Activation :math:`\alpha`: ramp 0 -> 1, then hold. No external load.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple, Union, get_args

import dolfin
import dolfin_mech
import fenicsoft
import fenicsoft.materials.orientation.spherical as spherical_mod
import fenicsoft.operators.cell as cell_ops
import hill_chain_helpers
from fenicsoft import io
from fenicsoft.mesh_geometries.box import BoxGeometry, BoxMarker, BoxMeshGenerator
from fenicsoft.mesh_geometries.mesh_geometries import GmshKeys, MeshOptions
from fenicsoft.operators.cell.rotation.orientation_change import OrientationChange

from hill_playground import integration
from hill_playground.io import xdmf_units
from hill_playground.materials import permeability as perm
from hill_playground.materials.cell.active.activation import ActivePowerConfig
from hill_playground.materials.cell.active.hill import law as hill_law
from hill_playground.materials.cell.active.hill.functions import SeeVariant
from hill_playground.materials.orientation import display
from hill_playground.materials.volumetric import compaction
from hill_playground.operators import ResidualRows
from hill_playground.operators.cell import ContractileEvolution, SaturatedProliferation
from hill_playground.scaling import CharacteristicScales

comm = dolfin.MPI.comm_world
rank = dolfin.MPI.rank(comm)

_HELD_MARKERS: Tuple[BoxMarker, ...] = (BoxMarker.LEFT, BoxMarker.RIGHT)
_DRAINED_MARKERS: Tuple[BoxMarker, ...] = (BoxMarker.FRONT, BoxMarker.BACK, BoxMarker.BOTTOM, BoxMarker.TOP)
_E_X = dolfin.Constant((1.0, 0.0, 0.0))
_E_Z = dolfin.Constant((0.0, 0.0, 1.0))

Nondimensionalization = Literal["none", "operator", "parameters"]
PermeabilityLaw = Literal["constant", "kozeny_carman", "holmes_mow", "compaction_power"]
VolumetricLaw = Literal["quadratic", "compaction"]
_CHOICES: Dict[str, Any] = {
	"nondimensionalization": Nondimensionalization,
	"permeability_law": PermeabilityLaw,
	"volumetric_law": VolumetricLaw,
	"see_variant": SeeVariant,
}
"""``UserInputs`` options restricted to a ``Literal``: checked at construction (``Literal`` is static only)."""


@dataclass(slots=True, frozen=True)
class UserInputs:
	r"""Configuration, SI units (module docstring for the meaning of each group).

	Attributes:
	    length_x: :math:`L_x`, held direction (m).
	    width_y: :math:`L_y` (m).
	    thickness_z: :math:`L_z`, also :math:`L_0` (m).
	    mesh_min_size: Gmsh minimum element size (m).
	    mesh_max_size: Gmsh maximum element size (m).
	    regenerate_mesh: Re-run Gmsh even if the mesh exists.
	    quadrature_degree: Default quadrature degree (every term without its own).
	    quadrature_degrees: Degree per residual row, overriding the default for that row's terms; rows
	        are named after the subsol of their test function (:data:`UNKNOWNS`).
	    foi_degree: Degree of the DG spaces of the FOIs.
	    nondimensionalization: ``"none"``, ``"operator"`` or ``"parameters"`` (module docstring, section 4).
	    cell_transport: Solve :math:`\Phi_c, \tilde\phi_s` (:math:`R_c, R_s`); else :math:`\Phi_c = \Phi_{c,0}`.
	    remodelling: Solve :math:`\theta, \varphi` (:math:`R_{\theta,\varphi}`); else frozen at the draw.
	    gel_shear_modulus: :math:`\mu_{gel}`, also :math:`\mu_0` (Pa).
	    gel_kappa: :math:`\kappa_{gel}` (Pa).
	    volumetric_law: ``"quadratic"`` or ``"compaction"`` (module docstring, section 5).
	    f_max: :math:`f_{max}` (Pa).
	    lmbda_opt: :math:`\lambda_{opt}`.
	    width_fl: :math:`W`.
	    use_force_length: Gaussian :math:`f_{FL}(\lambda_a)`, else 1.
	    k_e: SEE stiffness :math:`k_e` (Pa).
	    k_p: PEE stiffness :math:`k_p` (Pa).
	    hill_a: :math:`A`.
	    hill_b: :math:`b` (1/s).
	    f_ecc: :math:`f_{ecc}`.
	    eta_0: :math:`\eta_0` (Pa.s).
	    see_variant: SEE law (``hill.functions.SeeVariant``).
	    k_2e: Exponent of the ``"exponential"`` SEE.
	    orientation_random_seed: Seed of the isotropic :math:`(\theta, \varphi)` draw.
	    remodel_tau: Remodelling time :math:`\tau` (s).
	    permeability: :math:`k_0` (m^2).
	    viscosity_pa_s: :math:`\eta` (Pa.s).
	    permeability_law: :math:`k(\phi_\ell)` law (module docstring, section 5).
	    holmes_mow_exponent: :math:`\kappa_{HM}`.
	    holmes_mow_stiffening: :math:`M`.
	    compaction_exponent: :math:`n \ge 1` of ``"compaction_power"``.
	    rho_s: :math:`\rho_s` (kg/m^3), haptotaxis weight and :math:`\rho_0`.
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c_init: :math:`\Phi_{c,0}`.
	    phi_c_max: :math:`\Phi_{c,max}`.
	    cellular_diffusion: :math:`D_\phi` (m^2/s).
	    hapt_coeff: :math:`h_\phi` (m^5/(kg.s)).
	    prol_coeff: :math:`r` (1/s).
	    displacement_degree: Degree :math:`k` of :math:`\mathbf u` (:math:`p, \Phi_c, \tilde\phi_s` use :math:`k-1`).
	    steps: Ramp then hold (s).
	    activation_configs: :math:`\alpha(t)` per step.
	    sol_tol: Newton tolerance, the same for every (dimensionless) unknown.
	    solver_parameters: ``NonlinearSolver`` parameters (``sol_tol`` added at run time).
	    integrator_parameters: ``TimeIntegrator`` adaptive-``dt`` parameters.
	    xdmf_write_period: Output decimation (``DecimatedTimeIntegrator``), ``TimeInterval`` in s.
	    xdmf_si_only: Write only SI fields in the XDMF (no raw :math:`\hat{\mathbf u}, \hat p`, no ``*_old``).
	    stagnation_plateau_relative_error: Plateau-exit threshold.
	    project_prefix: Output folder.
	    stem_name: Output file stem.
	    stem_mesh_name: Mesh file stem.
	"""

	# --- Geometry ---
	length_x: float = 875e-6
	width_y: float = 375e-6
	thickness_z: float = 50e-6

	# --- Mesh ---
	mesh_min_size: float = thickness_z / 4
	mesh_max_size: float = thickness_z / 3
	regenerate_mesh: bool = True
	quadrature_degree: int = 4
	quadrature_degrees: Dict[str, int] = field(
		default_factory=dict
	)  # {"liquid_pressure": 2, "cell_fraction": 2, "substrate_smoothed": 2, "theta": 2, "phi": 2, "eps_a": 2}
	foi_degree: int = 0

	# --- Switches ---
	nondimensionalization: Nondimensionalization = "operator"
	cell_transport: bool = True
	remodelling: bool = True

	# --- Gel skeleton ---
	gel_shear_modulus: float = 1700.0
	gel_kappa: float = 1417.0
	volumetric_law: VolumetricLaw = "compaction"

	# --- Hill chain ---
	f_max: float = 20e3
	lmbda_opt: float = 1.05
	width_fl: float = 0.3
	use_force_length: bool = True
	k_e: float = 200e3
	k_p: float = 5e3
	hill_a: float = 0.25
	hill_b: float = 0.05
	f_ecc: float = 1.8
	eta_0: float = 0.0
	see_variant: SeeVariant = "mooney"
	k_2e: float = 0.0

	# --- Orientation ---
	orientation_random_seed: int = 0
	remodel_tau: float = 3600.0

	# --- Poromechanics ---
	permeability: float = 1.8e-15
	viscosity_pa_s: float = 1.0e-3
	permeability_law: PermeabilityLaw = "holmes_mow"
	holmes_mow_exponent: float = 2.0
	holmes_mow_stiffening: float = 0.0
	compaction_exponent: float = 2.0
	rho_s: float = 1060.0

	# --- Volume fractions ---
	phi_s0: float = 0.25
	phi_c_init: float = 0.05
	phi_c_max: float = 0.1

	# --- Cell transport ---
	cellular_diffusion: float = 4e-3 * 1e-6 / 86400.0  # 4e-3 mm^2/day
	hapt_coeff: float = 10 * (4e-3 * 1e-6 / 86400.0) / 1060.0
	prol_coeff: float = 0.1 / 86400.0  # 0.1 /day

	# --- Spaces ---
	displacement_degree: int = 2

	# --- Time ---
	steps: Sequence[io.Step] = field(
		default_factory=lambda: (
			io.Step(duration=60.0, init_dt=10.0, min_dt=0.25),  # ramp, dt ~ tau_a
			# io.Step(duration=600.0, init_dt=60.0, min_dt=0.25, max_dt=60.0),  # hold
			# io.Step(duration=600.0, init_dt=30.0, min_dt=5.0),  # step 0: 0% -> 100% ramp, 10 min
			io.Step(duration=42000.0, init_dt=30.0, min_dt=1.0, max_dt=100.0, min_simu_seconds=3 * 3600),
		)
	)
	activation_configs: Sequence[ActivePowerConfig] = field(
		default_factory=lambda: [
			ActivePowerConfig.ramp(k_step=0, ini=0.0, fin=1.0),
			ActivePowerConfig.constant(k_step=1, value=1.0),
		]
	)

	# --- Solver ---
	sol_tol: float = 1e-7
	solver_parameters: Dict[str, Any] = field(default_factory=lambda: {"n_iter_max": 12})
	integrator_parameters: Dict[str, Any] = field(
		default_factory=lambda: {
			"n_iter_for_accel": 4,
			"n_iter_for_decel": 10,
			"accel_coeff": 1.25,
			"decel_coeff": 1.25,
		}
	)
	xdmf_write_period: Union[int, integration.TimeInterval, Sequence[Union[int, integration.TimeInterval]]] = (
		1,
		# integration.TimeInterval(30.0),
		integration.TimeInterval(840.0),
	)
	stagnation_plateau_relative_error: float = 5e-7
	xdmf_si_only: bool = True

	# --- Output ---
	project_prefix: Path = Path("./outputs_hill_box_uniaxial")
	stem_name: str = "hill_box_uniaxial"
	stem_mesh_name: str = "hill_box_uniaxial_mesh"

	def __post_init__(self) -> None:
		r"""Check the saturation and the uniqueness of the CE equilibrium.

		Raises:
		    ValueError: :math:`\Phi_{s,0} + \Phi_{c,0} \ge 1`, uniqueness condition
		        :math:`k_e + k_p > f_{max} \max_\lambda |f_{FL}'(\lambda)\,\lambda|` violated, or a value outside its
		        ``Literal`` (``nondimensionalization``, ``permeability_law``, ``volumetric_law``, ``see_variant``).
		"""
		for name, choices in _CHOICES.items():
			value = getattr(self, name)
			if value not in get_args(choices):
				raise ValueError(f"Unknown {name} '{value}', expected one of {get_args(choices)}.")
		if self.phi_s0 + self.phi_c_init >= 1.0:
			raise ValueError(f"phi_s0 + phi_c_init = {self.phi_s0 + self.phi_c_init} must be < 1.")
		if self.use_force_length:
			hill_chain_helpers.check_hill_uniqueness(self.k_e, self.k_p, self.f_max, self.lmbda_opt, self.width_fl)

	@property
	def hydraulic_conductivity(self) -> float:
		r""":math:`k_0/\eta` (m^2/(Pa.s))."""
		return self.permeability / self.viscosity_pa_s

	@property
	def scales(self) -> CharacteristicScales:
		r"""Base scales :math:`L_0 = L_z`, :math:`\mu_0 = \mu_{gel}`, :math:`\rho_0 = \rho_s`.

		:math:`t_0 = L_0^2\eta/(k\kappa_{gel})` (consolidation time over the thickness).
		"""
		return CharacteristicScales.from_consolidation_scale(
			length=self.thickness_z,
			stress=self.gel_shear_modulus,
			density=self.rho_s,
			permeability=self.permeability,
			viscosity=self.viscosity_pa_s,
			consolidation_modulus=self.gel_kappa,
		)

	@property
	def probes(self) -> Dict[str, Tuple[float, float, float]]:
		"""Physical probe points (m): box centre, top-face centre, front-face centre."""
		cx, cy, cz = self.length_x / 2.0, self.width_y / 2.0, self.thickness_z / 2.0
		return {"center": (cx, cy, cz), "top": (cx, cy, self.thickness_z), "front": (cx, 0.0, cz)}


@dataclass(frozen=True)
class SolveUnits:
	r"""How SI quantities map to the solved system (module docstring, section 4).

	Attributes:
	    unknowns: Scales of mesh, :math:`\mathbf u`, :math:`p` (identity for ``"none"``).
	    coefficients: Scales of the coefficients and of the time (identity unless ``"parameters"``).
	"""

	unknowns: CharacteristicScales
	coefficients: CharacteristicScales

	@classmethod
	def from_inputs(cls, user_inputs: UserInputs) -> "SolveUnits":
		"""Scales actually applied for ``user_inputs.nondimensionalization``.

		Args:
		    user_inputs: Configuration.

		Returns:
		    The solve units.
		"""
		mode, identity = user_inputs.nondimensionalization, CharacteristicScales.identity()
		if mode == "none":
			return cls(identity, identity)
		return cls(user_inputs.scales, user_inputs.scales if mode == "parameters" else identity)


@dataclass(frozen=True)
class Coefficients:
	r"""Coefficients entering the forms, in solve units (:meth:`CharacteristicScales.nondimensionalize_*`).

	Stresses :math:`\mu, \kappa, f_{max}, k_e, k_p`; viscosity :math:`\eta_0`; rates :math:`b, r`;
	times :math:`\tau` and the steps; :math:`k_0/\eta`, :math:`D_\phi`, :math:`h_\phi\rho_s`.
	SI when ``coefficients`` is the identity.
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
	def from_inputs(cls, user_inputs: UserInputs, scales: CharacteristicScales) -> "Coefficients":
		r"""Convert the SI inputs with ``scales`` (identity: unchanged).

		Args:
		    user_inputs: Configuration.
		    scales: ``SolveUnits.coefficients``.

		Returns:
		    The coefficients.
		"""
		s, ui = scales, user_inputs

		def as_time(step: io.Step) -> io.Step:
			times = {"duration", "init_dt", "min_dt", "max_dt", "min_simu_seconds"}
			values = {k: getattr(step, k) for k in times if getattr(step, k, None) is not None}
			return dataclasses.replace(step, **{k: s.nondimensionalize_time(v) for k, v in values.items()})

		def as_period(period: Any) -> Any:
			if isinstance(period, integration.TimeInterval):
				return integration.TimeInterval(s.nondimensionalize_time(period.seconds))
			if isinstance(period, (list, tuple)):
				return type(period)(as_period(p) for p in period)
			return period

		return cls(
			mu=s.nondimensionalize_stress(ui.gel_shear_modulus),
			kappa=s.nondimensionalize_stress(ui.gel_kappa),
			f_max=s.nondimensionalize_stress(ui.f_max),
			k_e=s.nondimensionalize_stress(ui.k_e),
			k_p=s.nondimensionalize_stress(ui.k_p),
			eta_0=s.nondimensionalize_time(s.nondimensionalize_stress(ui.eta_0)),  # Pa.s / (mu0 t0)
			hill_b=s.nondimensionalize_rate(ui.hill_b),
			remodel_tau=s.nondimensionalize_time(ui.remodel_tau),
			conductivity=s.nondimensionalize_permeability_over_viscosity(ui.hydraulic_conductivity),
			cellular_diffusion=s.nondimensionalize_diffusivity(ui.cellular_diffusion),
			# h_phi * rho_s is a diffusivity: h_hat = h rho0 t0 / L0^2, rho_hat = rho_s / rho0.
			hapt_coeff=s.nondimensionalize_diffusivity(ui.hapt_coeff * s.density),
			rho_s=s.nondimensionalize_density(ui.rho_s),
			prol_coeff=s.nondimensionalize_rate(ui.prol_coeff),
			steps=tuple(as_time(step) for step in ui.steps),
			xdmf_write_period=as_period(ui.xdmf_write_period),
		)


UNKNOWNS: Tuple[str, ...] = (
	"displacement",
	"theta",
	"phi",
	"liquid_pressure",
	"cell_fraction",
	"substrate_smoothed",
	"eps_a",
)
"""Every possible subsol (= residual row) of the demo, whatever the switches."""


@dataclass(frozen=True)
class TermValues:
	r"""Value-preserving factor :math:`v` of each term (module docstring, section 4); all 1 unless ``"operator"``.

	SI integral = :math:`v` times the form written on the scaled mesh and unknowns, with the FE (dimensionless)
	test function; the test-function scale :math:`s` is applied per row by ``ResidualRows``.

	Attributes:
	    gel: :math:`\int \mathbf P:\nabla\delta\mathbf u`: :math:`L_0^2`.
	    fluid: :math:`-\int Jp\mathbf F^{-T}:\nabla\delta\mathbf u` with :math:`p = \mu_0\hat p`: :math:`\mu_0 L_0^2`.
	    volume: :math:`\int f\,\delta x` with :math:`f` in SI (rates, sources, :math:`R_a`, :math:`R_s`): :math:`L_0^3`.
	    darcy: :math:`\int \mathbf K_L\nabla p\cdot\nabla\delta p`: :math:`\mu_0 L_0`.
	    cell_flux: :math:`\int \mathbf Q_c\cdot\nabla\delta x`: :math:`L_0`.
	"""

	gel: float = 1.0
	fluid: float = 1.0
	volume: float = 1.0
	darcy: float = 1.0
	cell_flux: float = 1.0

	@classmethod
	def from_inputs(cls, user_inputs: UserInputs) -> "TermValues":
		"""Factors for ``"operator"``, ones otherwise.

		Args:
		    user_inputs: Configuration.

		Returns:
		    The term values.
		"""
		if user_inputs.nondimensionalization != "operator":
			return cls()
		s = user_inputs.scales
		return cls(
			gel=s.length**2,
			fluid=s.stress * s.length**2,
			volume=s.length**3,
			darcy=s.stress * s.length,
			cell_flux=s.length,
		)


def row_normalizations(user_inputs: UserInputs) -> Dict[str, float]:
	r"""Row normalization :math:`\mathcal N_{row}` per subsol (module docstring, section 4); none unless ``"operator"``.

	Args:
	    user_inputs: Configuration.

	Returns:
	    :math:`\mathcal N_{row}` for every name of :data:`UNKNOWNS`, or an empty dict.
	"""
	if user_inputs.nondimensionalization != "operator":
		return {}
	s = user_inputs.scales
	volume = s.length**3
	rate = volume / s.time
	return {
		"displacement": s.stress * volume,
		"theta": rate,
		"phi": rate,
		"liquid_pressure": s.stress * rate,
		"cell_fraction": rate,
		"substrate_smoothed": volume,
		"eps_a": s.stress * volume,
	}


def row_test_scales(user_inputs: UserInputs) -> Dict[str, float]:
	r"""Test-function scale :math:`s` per subsol (:math:`\delta x = s\,\delta\hat x`); none unless ``"operator"``.

	Args:
	    user_inputs: Configuration.

	Returns:
	    :math:`s_{\mathbf u} = L_0`, :math:`s_p = \mu_0` (others 1), or an empty dict.
	"""
	if user_inputs.nondimensionalization != "operator":
		return {}
	s = user_inputs.scales
	return {"displacement": s.length, "liquid_pressure": s.stress}


def build_residual_rows(problem: dolfin_mech.problems.Problem, user_inputs: UserInputs) -> ResidualRows:
	"""Quadrature degrees, test scales and normalizations of the rows present in ``problem`` (switches applied).

	Args:
	    problem: Problem with its subsols.
	    user_inputs: Configuration.

	Returns:
	    The row settings.

	Raises:
	    ValueError: A ``quadrature_degrees`` key is not one of :data:`UNKNOWNS` (typo).
	"""
	unknown = sorted(set(user_inputs.quadrature_degrees) - set(UNKNOWNS))
	if unknown:
		raise ValueError(f"quadrature_degrees: unknown rows {unknown}, expected names among {UNKNOWNS}.")
	present = {subsol.name for subsol in problem.subsols}
	return ResidualRows.from_problem(
		problem,
		quadrature_degrees={k: v for k, v in user_inputs.quadrature_degrees.items() if k in present},
		test_scales={k: v for k, v in row_test_scales(user_inputs).items() if k in present},
		normalizations={k: v for k, v in row_normalizations(user_inputs).items() if k in present},
	)


def build_mesh(user_inputs: UserInputs) -> io.SimulationPaths:
	"""Generate the box mesh with Gmsh on rank 0 (unless it exists) and convert it to XDMF.

	Args:
	    user_inputs: Configuration.

	Returns:
	    Output-folder layout.
	"""
	simulation_paths = io.configure_environment(user_inputs.project_prefix, user_inputs.stem_name)
	mesh_file_path = simulation_paths.mesh_folder / f"{user_inputs.stem_mesh_name}.msh"

	if not mesh_file_path.exists() or user_inputs.regenerate_mesh:
		if rank == 0:
			begin_t = time.time()
			geometry = BoxGeometry(
				length=user_inputs.length_x, width=user_inputs.width_y, height=user_inputs.thickness_z
			)
			mesh_options = MeshOptions(
				min_size=user_inputs.mesh_min_size,
				max_size=user_inputs.mesh_max_size,
				extra_opts={GmshKeys.TERMINAL: 0, GmshKeys.OPTIMIZE_NETGEN: 1},
			)
			with BoxMeshGenerator(geometry, mesh_options, simulation_paths.mesh_folder) as generator:
				msh_file = generator.generate(mesh_stem_name=user_inputs.stem_mesh_name)
			io.convert_msh_to_xdmf(msh_file, prune_z=False)
			logging.getLogger(__name__).info(f"Mesh creation completed in {time.time() - begin_t:.2f}s.")
		comm.Barrier()

	return simulation_paths


def initialise_problem(
	mesh: dolfin.Mesh,
	domains_mf: dolfin.MeshFunction,
	boundaries_mf: dolfin.MeshFunction,
	points_mf: dolfin.MeshFunction,
	user_inputs: UserInputs,
	coefficients: Coefficients,
) -> dolfin_mech.problems.Problem:
	r"""Register the unknowns (module docstring, section 2) and build the data fields.

	Sets on ``problem``: the subsols, ``theta_func``/``phi_func`` (subsol or frozen draw),
	``theta_initial``/``phi_initial``, ``phic_func`` (subsol or :math:`\Phi_{c,0}`), ``phi_s0``,
	``kinematics``, ``permeability`` (law of section 5), ``k_l_tensor``
	(:math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`) and ``sol_tol`` (one entry per subsol, registration order).

	Args:
	    mesh: Mesh (rescaled when nondimensionalised).
	    domains_mf: Cell tags.
	    boundaries_mf: Facet tags.
	    points_mf: Point tags.
	    user_inputs: Configuration.
	    coefficients: Coefficients in solve units (steps, :math:`k_0/\eta`).

	Returns:
	    Problem with its function spaces, kinematics and steps.
	"""
	problem = dolfin_mech.problems.Problem()
	problem.set_mesh(mesh=mesh, define_facet_normals=True)
	problem.set_measures(domains=domains_mf, boundaries=boundaries_mf, points=points_mf)
	sol_tol: List[float] = []
	lower_degree = user_inputs.displacement_degree - 1

	problem.displacement_subsol = problem.add_vector_subsol(
		name="displacement", family="CG", degree=user_inputs.displacement_degree
	)
	sol_tol.append(user_inputs.sol_tol)

	theta_init, phi_init = spherical_mod.sample_isotropic_theta_phi_dg0(
		mesh, seed=user_inputs.orientation_random_seed, family="DG", degree=0
	)
	problem.theta_initial = theta_init.copy(deepcopy=True)
	problem.theta_initial.rename("theta_initial", "theta_initial")
	problem.phi_initial = phi_init.copy(deepcopy=True)
	problem.phi_initial.rename("phi_initial", "phi_initial")
	if user_inputs.remodelling:
		problem.theta_subsol = problem.add_scalar_subsol(name="theta", family="DG", degree=0, init_fun=theta_init)
		problem.phi_subsol = problem.add_scalar_subsol(name="phi", family="DG", degree=0, init_fun=phi_init)
		sol_tol += [user_inputs.sol_tol, user_inputs.sol_tol]

	problem.pl_subsol = problem.add_scalar_subsol(
		name="liquid_pressure", family="CG", degree=lower_degree, init_val=0.0
	)
	sol_tol.append(user_inputs.sol_tol)

	problem.phi_s0 = dolfin.Constant(user_inputs.phi_s0)
	if user_inputs.cell_transport:
		problem.phic_subsol = problem.add_scalar_subsol(
			name="cell_fraction", family="CG", degree=lower_degree, init_val=user_inputs.phi_c_init
		)
		problem.phis_subsol = problem.add_scalar_subsol(
			name="substrate_smoothed", family="CG", degree=lower_degree, init_val=user_inputs.phi_s0
		)
		sol_tol += [user_inputs.sol_tol, user_inputs.sol_tol]

	problem.eps_a_subsol = problem.add_scalar_subsol(name="eps_a", family="DG", degree=0, init_val=0.0)
	sol_tol.append(user_inputs.sol_tol)
	problem.sol_tol = sol_tol

	problem.set_solution_finite_element()
	problem.set_solution_function_space()
	problem.set_solution_functions()
	problem.set_quadrature_degree(user_inputs.quadrature_degree)
	problem.set_foi_finite_elements_DG(user_inputs.foi_degree)
	problem.set_foi_function_spaces()

	problem.theta_func = problem.theta_subsol.subfunc if user_inputs.remodelling else theta_init
	problem.phi_func = problem.phi_subsol.subfunc if user_inputs.remodelling else phi_init
	problem.phic_func = (
		problem.phic_subsol.subfunc if user_inputs.cell_transport else dolfin.Constant(user_inputs.phi_c_init)
	)

	# Mesh in X_hat = X/L0 and U = u/L0: grad_Xhat(U) = grad_X(u), so plain Kinematics is exact.
	problem.kinematics = dolfin_mech.kinematics.Kinematics(
		U=problem.displacement_subsol.subfunc, U_old=problem.displacement_subsol.func_old, Q_expr=problem.Q_expr
	)

	for step_idx, step in enumerate(coefficients.steps):
		problem.add_step(Deltat=step.duration, dt_ini=step.init_dt, dt_min=step.min_dt, dt_max=step.max_dt)
		if step.min_simu_seconds is not None:
			problem.steps[step_idx].min_simu_seconds = step.min_simu_seconds

	problem.permeability = build_permeability(problem, user_inputs, coefficients)
	problem.k_l_tensor = problem.permeability.tensor
	return problem


def build_permeability(
	problem: dolfin_mech.problems.Problem, user_inputs: UserInputs, coefficients: Coefficients
) -> Any:
	r""":math:`k(\phi_\ell)/\eta` law of ``user_inputs.permeability_law`` (module docstring, section 5).

	Args:
	    problem: Problem with ``kinematics``, ``phi_s0``, ``phic_func``.
	    user_inputs: Configuration.
	    coefficients: :math:`k_0/\eta` in solve units.

	Returns:
	    Law exposing ``ratio``, ``hydraulic_conductivity`` and ``tensor``.
	"""
	kin, law = problem.kinematics, user_inputs.permeability_law
	if law == "constant":
		return perm.constant.Permeability(kin, perm.constant.Parameters(conductivity_0=coefficients.conductivity))
	common: Dict[str, Any] = dict(
		conductivity_0=coefficients.conductivity,
		phi_s0=problem.phi_s0,
		phi_c=problem.phic_func,
		phi_c0=user_inputs.phi_c_init,
	)
	if law == "kozeny_carman":
		return perm.kozeny_carman.Permeability(kin, perm.kozeny_carman.Parameters(**common))
	if law == "holmes_mow":
		params = perm.holmes_mow.Parameters(
			**common, exponent=user_inputs.holmes_mow_exponent, stiffening=user_inputs.holmes_mow_stiffening
		)
		return perm.holmes_mow.Permeability(kin, params)
	if law == "compaction_power":
		params = perm.compaction_power.Parameters(**common, exponent=user_inputs.compaction_exponent)
		return perm.compaction_power.Permeability(kin, params)
	raise ValueError(f"Unknown permeability_law '{law}'.")


def build_materials(
	problem: dolfin_mech.problems.Problem, user_inputs: UserInputs, coefficients: Coefficients
) -> Tuple[Any, hill_law.Constitutive]:
	r"""Gel law :math:`\mathbf{P}_S + \mathbf{P}_h` (Neo-Hooke + volumetric law of section 5 + Hill chain).

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.
	    coefficients: Moduli in solve units.

	Returns:
	    ``(gel_material, hill_material)``; the second owns ``p_act`` (:math:`\alpha`).
	"""
	kin = problem.kinematics
	isochoric = fenicsoft.materials.iso.neo_hooke.Constitutive(
		kinematics=kin,
		parameters=fenicsoft.materials.iso.neo_hooke.Parameters.from_shear_modulus(mu=coefficients.mu),
		name="gel_isochoric",
	)
	if user_inputs.volumetric_law == "compaction":
		volumetric: Any = compaction.CompactionVolumetricPenalty(
			kin,
			compaction.Parameters(
				kappa=coefficients.kappa,
				solid_jacobian=perm.porosity.compaction_jacobian(problem.phi_s0, problem.phic_func),
			),
			name="gel_volumetric",
		)
	elif user_inputs.volumetric_law == "quadratic":
		volumetric = fenicsoft.materials.volumetric.QuadraticVolumetricPenalty(
			kinematics=kin,
			parameters=fenicsoft.materials.volumetric.VolumetricPenaltyParameters(kappa=coefficients.kappa),
			name="gel_volumetric",
		)
	else:
		raise ValueError(f"Unknown volumetric_law '{user_inputs.volumetric_law}'.")
	hill_params = hill_law.Parameters(
		phi_c=problem.phic_func,
		f0=spherical_mod.f0_from_theta_phi(problem.theta_func, problem.phi_func),
		activation_configs=user_inputs.activation_configs,
		f_max=coefficients.f_max,
		simulation_steps=problem.steps,
		use_force_length=user_inputs.use_force_length,
		lmbda_opt=user_inputs.lmbda_opt,
		W=user_inputs.width_fl,
		eps_a=problem.eps_a_subsol.subfunc,
		eps_a_old=problem.eps_a_subsol.func_old,
		k_e=coefficients.k_e,
		k_p=coefficients.k_p,
		hill_a=user_inputs.hill_a,
		hill_b=coefficients.hill_b,
		f_ecc=user_inputs.f_ecc,
		eta_0=coefficients.eta_0,
		see_variant=user_inputs.see_variant,
		k_2e=user_inputs.k_2e,
	)
	hill_material = hill_law.Constitutive(kin, hill_params, name="cell_hill_chain")
	return isochoric + volumetric + hill_material, hill_material


def build_step_operators(
	problem: dolfin_mech.problems.Problem,
	user_inputs: UserInputs,
	hill_material: hill_law.Constitutive,
	coefficients: Coefficients,
	rows: ResidualRows,
	values: TermValues,
) -> Tuple[List[Any], ContractileEvolution]:
	r"""Operators shared by every step, with their measure and ``scale_factor`` (module docstring, section 4).

	Each operator takes ``rows.measure(dV, row)`` and ``rows.scale_factor(row, v)``, ``row`` being the
	subsol of its test function.

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.
	    hill_material: Hill law (owner of ``p_act``).
	    coefficients: Coefficients in solve units.
	    rows: Quadrature degree and normalization per row.
	    values: Value-preserving factor per term.

	Returns:
	    ``(operators, contractile_evolution)``.
	"""
	kin = problem.kinematics
	p_hat, dp = problem.pl_subsol.subfunc, problem.pl_subsol.dsubtest
	c = coefficients

	def dv(row: str) -> dolfin.Measure:
		return rows.measure(problem.dV, row)

	contractile = ContractileEvolution(
		law=hill_material,
		test=problem.eps_a_subsol.dsubtest,
		dx=dv("eps_a"),
		scale_factor=rows.scale_factor("eps_a", values.volume),
	)
	ops: List[Any] = [
		fenicsoft.operators.fluid_stress.biphasic.pk1.FluidStress(
			kinematics=kin,
			function=p_hat,
			displacement=problem.displacement_subsol.subfunc,
			test=problem.displacement_subsol.dsubtest,
			measure=dv("displacement"),
			scale_factor=rows.scale_factor("displacement", values.fluid),
		),
		fenicsoft.operators.DarcyGradient(
			function=p_hat,
			test=dp,
			conductivity=problem.k_l_tensor,
			dx=dv("liquid_pressure"),
			scale_factor=rows.scale_factor("liquid_pressure", values.darcy),
		),
		fenicsoft.operators.ChangeRate(
			function=kin.J,
			function_old=kin.J_old,
			test=dp,
			dx=dv("liquid_pressure"),
			scale_factor=rows.scale_factor("liquid_pressure", values.volume),
		),
		contractile,
	]

	if user_inputs.cell_transport:
		phi_c, dc = problem.phic_subsol.subfunc, problem.phic_subsol.dsubtest
		phi_s = problem.phis_subsol.subfunc

		def hapt(test: Any, row: str) -> Any:
			return cell_ops.SaturatedHaptotaxisFluxSmooth(
				phi_c,
				phi_s,
				c.rho_s,
				problem,
				c.hapt_coeff,
				user_inputs.phi_c_max,
				test,
				dv(row),
				rows.scale_factor(row, values.cell_flux),
			)

		def diff(test: Any, row: str) -> Any:
			return cell_ops.IsotropicDiffusion(
				phi_c, problem, c.cellular_diffusion, test, dv(row), rows.scale_factor(row, values.cell_flux)
			)

		ops += [
			hapt(dp, "liquid_pressure"),
			diff(dp, "liquid_pressure"),
			fenicsoft.operators.ChangeRate(
				function=phi_c,
				function_old=problem.phic_subsol.func_old,
				test=dc,
				dx=dv("cell_fraction"),
				scale_factor=rows.scale_factor("cell_fraction", values.volume),
			),
			SaturatedProliferation(
				phi_c,
				c.prol_coeff,
				user_inputs.phi_c_max,
				dc,
				dv("cell_fraction"),
				scale_factor=rows.scale_factor("cell_fraction", values.volume),
			),
			hapt(dc, "cell_fraction"),
			diff(dc, "cell_fraction"),
			cell_ops.SubstrateProjection(
				phi_s_func=phi_s,
				test_phi_s=problem.phis_subsol.dsubtest,
				kin_j=kin.J,
				phi_s0=problem.phi_s0,
				measure=dv("substrate_smoothed"),
				scale_factor=rows.scale_factor("substrate_smoothed", values.volume),
			),
		]

	if user_inputs.remodelling:
		theta, phi = problem.theta_subsol, problem.phi_subsol
		builders: Tuple[Tuple[Any, Callable[[Any], Any]], ...] = (
			(theta, lambda th: spherical_mod.f0_from_theta_phi(th, phi.subfunc)),
			(phi, lambda ph: spherical_mod.f0_from_theta_phi(theta.subfunc, ph)),
		)
		ops += [
			OrientationChange(
				angle=subsol.subfunc,
				angle_old=subsol.func_old,
				test=subsol.dsubtest,
				kinematics=kin,
				tau=c.remodel_tau,
				activation=hill_material.p_act,
				dx=dv(subsol.name),
				f0_builder=builder,
				scale_factor=rows.scale_factor(subsol.name, values.volume),
			)
			for subsol, builder in builders
		]
	return ops, contractile


def set_weak_form_and_bcs(
	problem: dolfin_mech.problems.Problem,
	boundaries_mf: dolfin.MeshFunction,
	user_inputs: UserInputs,
	coefficients: Coefficients,
) -> dolfin_mech.problems.Problem:
	r"""Register the residuals, the BCs (module docstring, section 1), the FOIs and QOIs.

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    boundaries_mf: Facet tags.
	    user_inputs: Configuration.
	    coefficients: Coefficients in solve units.

	Returns:
	    The assembled problem.
	"""
	rows = build_residual_rows(problem, user_inputs)
	values = TermValues.from_inputs(user_inputs)
	gel_material, hill_material = build_materials(problem, user_inputs, coefficients)
	shared_ops, contractile = build_step_operators(problem, user_inputs, hill_material, coefficients, rows, values)

	for step_idx in range(len(problem.steps)):
		problem.add_operator(
			fenicsoft.operators.hyper_elasticity.pk1.HyperElasticity(
				disp=problem.displacement_subsol.subfunc,
				disp_test=problem.displacement_subsol.dsubtest,
				material=gel_material,
				measure=rows.measure(problem.dV, "displacement"),
				k_step=step_idx,
				scale_factor=rows.scale_factor("displacement", values.gel),
			),
			k_step=step_idx,
		)
		for operator in shared_ops:
			problem.add_operator(operator=operator, k_step=step_idx)
		for marker in _HELD_MARKERS:
			problem.add_constraint(
				V=problem.displacement_subsol.fs,
				sub_domains=boundaries_mf,
				sub_domain_id=int(marker),
				val=[0.0, 0.0, 0.0],
				k_step=step_idx,
			)
		for marker in _DRAINED_MARKERS:
			problem.add_constraint(
				V=problem.pl_subsol.fs, sub_domains=boundaries_mf, sub_domain_id=int(marker), val=0.0, k_step=step_idx
			)

	add_outputs(problem, user_inputs, gel_material, hill_material, contractile, SolveUnits.from_inputs(user_inputs))
	return problem


def add_outputs(
	problem: dolfin_mech.problems.Problem,
	user_inputs: UserInputs,
	gel_material: Any,
	hill_material: hill_law.Constitutive,
	contractile: ContractileEvolution,
	units: SolveUnits,
) -> None:
	r"""FOIs and QOIs, all SI, same names, order and spaces in every mode (module docstring, sections 4, 6).

	:math:`\mathbf u = L_0\hat{\mathbf u}` and :math:`p = \mu_0\hat p` are scaled copies of the unknowns,
	on their CG spaces (point data, no projection); the other fields are cell-wise
	(``local_solver``) projections on the DG FOI spaces (cell data).
	``units.unknowns`` redimensionalizes mesh, :math:`\mathbf u`, :math:`p`; ``units.coefficients``
	the stresses built from the moduli and the time.

	Args:
	    problem: Problem.
	    user_inputs: Configuration.
	    gel_material: Composite gel law.
	    hill_material: Hill law.
	    contractile: :math:`R_a` operator (provides :math:`T_0`).
	    units: Applied scales.
	"""
	kin = problem.kinematics
	su, sc = units.unknowns, units.coefficients
	pressure = su.redimensionalize_stress(problem.pl_subsol.subfunc)
	f0 = spherical_mod.f0_from_theta_phi(problem.theta_func, problem.phi_func)
	f0_initial = spherical_mod.f0_from_theta_phi(problem.theta_initial, problem.phi_initial)
	fibre = display.current_direction(kin, f0)
	theta_deg, phi_deg = spherical_mod.canonical_theta_phi_degrees(problem.theta_func, problem.phi_func)

	def foi(name: str, expr: Any, fs: Any) -> None:
		# DG FOI spaces: cell-by-cell projection (cached local factorizations), not a global solve.
		problem.add_foi(name=name, expr=expr, fs=fs, update_type="local_solver")

	def scaled_copy(name: str, source: dolfin.Function, factor: float) -> None:
		# factor * source on the source's own space: exact, a vector scaling instead of a global L2 projection.
		copy = problem.add_foi(name=name, expr=source, fs=source.function_space(), update_type="interpolate")

		def update() -> None:
			copy.func.vector()[:] = factor * source.vector().get_local()

		copy.update = update

	# 1. Unknowns in SI, on their own (CG) spaces.
	scaled_copy("displacement_si", problem.displacement_subsol.func, su.length)
	scaled_copy("pressure_si", problem.pl_subsol.func, su.stress)

	# 2. Scalars, cell data.
	scalar_fois: Dict[str, Any] = {
		"jacobian": kin.J,
		"cell_fraction": problem.phic_func,
		"fluid_fraction": perm.porosity.eulerian_porosity(kin.J, problem.phi_s0, problem.phic_func),
		"permeability_ratio": problem.permeability.ratio,
		"theta": problem.theta_func,
		"phi": problem.phi_func,
		"theta_deg": theta_deg,
		"phi_deg": phi_deg,
		"theta_wrapped_deg": spherical_mod.wrapped_theta_degrees(problem.theta_func),
		"tilt_xy_deg": display.tilt_degrees(f0, _E_Z),
		"tilt_xy_current_deg": display.tilt_degrees(fibre, _E_Z),
		"angle_x_deg": display.axis_angle_degrees(f0, _E_X),
		"angle_x_current_deg": display.axis_angle_degrees(fibre, _E_X),
		"lambda_fibre": dolfin.sqrt(hill_material.inv4),
		"lambda_a": hill_material.lmbda_a,
		"lambda_e": dolfin.sqrt(hill_material.inv4e),
		"tension_see": sc.redimensionalize_stress(hill_material.tension_see),
		"tension_pee": sc.redimensionalize_stress(hill_material.tension_pee),
		"tension_ce": sc.redimensionalize_stress(hill_material.tension_ce),
		"tension_isometric": sc.redimensionalize_stress(contractile.isometric_tension),
	}
	subsol_names = {subsol.name for subsol in problem.subsols}
	for name, expr in scalar_fois.items():
		if name not in subsol_names:  # theta, phi, cell_fraction: the subsol itself is written
			foi(name, expr, problem.sfoi_fs)

	# 3. Vectors and tensors, cell data.
	foi("fibre", f0, problem.vfoi_fs)
	foi("fibre_initial", f0_initial, problem.vfoi_fs)
	foi("fibre_current", fibre, problem.vfoi_fs)
	foi("cell_cauchy", sc.redimensionalize_stress(hill_material.cauchy), problem.mfoi_fs)
	foi("total_cauchy", sc.redimensionalize_stress(gel_material.cauchy) - pressure * kin.I, problem.mfoi_fs)

	# QOIs. Point probes in mesh coordinates.
	probes = {k: [su.nondimensionalize_length(c) for c in v] for k, v in user_inputs.probes.items()}
	problem.displacement_subsol.func.set_allow_extrapolation(True)
	problem.pl_subsol.func.set_allow_extrapolation(True)

	def _time_si(_: Any) -> float:
		return sc.redimensionalize_time(hill_material.current_time)

	def _activation(_: Any) -> float:
		return hill_material.current_activation

	problem.add_qoi(name="time_si", expr=_time_si, point=probes["center"], update_type="direct")
	problem.add_qoi(name="cellular_activation", expr=_activation, point=probes["center"], update_type="direct")
	problem.add_qoi(
		name="top_center_disp_z",
		expr=su.redimensionalize_length(problem.displacement_subsol.func[2]),
		point=probes["top"],
		update_type="direct",
	)
	problem.add_qoi(
		name="front_center_disp_y",
		expr=su.redimensionalize_length(problem.displacement_subsol.func[1]),
		point=probes["front"],
		update_type="direct",
	)
	problem.add_qoi(
		name="center_pressure",
		expr=su.redimensionalize_stress(problem.pl_subsol.func),
		point=probes["center"],
		update_type="direct",
	)

	# Volume averages: dolfin_mech assembles expr and divides by norm (mesh volume, scale-free ratio).
	volume = dolfin.assemble(dolfin.Constant(1.0) * problem.dV)
	averages: Dict[str, Any] = {
		"mean_lambda_a": hill_material.lmbda_a,
		"mean_tension_see": sc.redimensionalize_stress(hill_material.tension_see),
		"mean_tension_isometric": sc.redimensionalize_stress(contractile.isometric_tension),
		"mean_jacobian": kin.J,
		"mean_permeability_ratio": problem.permeability.ratio,
		"order_x": display.order_parameter(fibre, _E_X),
		"order_z": display.order_parameter(fibre, _E_Z),
		"mean_tilt_xy_deg": display.tilt_degrees(fibre, _E_Z),
	}
	for name, expr in averages.items():
		problem.add_qoi(name=name, expr=expr * problem.dV, norm=volume, update_type="assembly")


def apply_initial_bcs(problem: dolfin_mech.problems.Problem) -> None:
	"""Apply every Dirichlet BC (global, then step 0) to the initial solution vector.

	Args:
	    problem: Assembled problem.
	"""
	for constraint in problem.constraints:
		constraint.bc.apply(problem.sol_func.vector())
	if len(problem.steps) > 0:
		for constraint in problem.steps[0].constraints:
			constraint.bc.apply(problem.sol_func.vector())


def solve_problem(
	problem: dolfin_mech.problems.Problem,
	simulation_paths: io.SimulationPaths,
	user_inputs: UserInputs,
	coefficients: Coefficients,
) -> bool:
	"""Newton + adaptive time stepping (playground ``DecimatedTimeIntegrator``: selected fields, rank-0 logs).

	Args:
	    problem: Assembled problem.
	    simulation_paths: Output layout.
	    user_inputs: Configuration.
	    coefficients: Write period in solve units.

	Returns:
	    True if every step converged.
	"""
	written = None
	if user_inputs.xdmf_si_only:
		# Dimensionless unknowns as solved (theta, phi, Phi_c, phi_s, eps_a) + SI FOIs; no u_hat, p_hat, *_old.
		raw = {"displacement", "liquid_pressure"}
		written = [s.func for s in problem.subsols if s.name not in raw] + problem.get_fois_func_lst()
	integrator = integration.DecimatedTimeIntegrator(
		problem=problem,
		solver=dolfin_mech.core.NonlinearSolver(
			problem=problem, parameters={**user_inputs.solver_parameters, "sol_tol": problem.sol_tol}
		),
		parameters=user_inputs.integrator_parameters,
		print_out=str(simulation_paths.outputs_base_name),  # type: ignore
		print_sta=str(simulation_paths.outputs_base_name),  # type: ignore
		write_qois=str(simulation_paths.results_base_name),  # type: ignore
		write_sol=str(simulation_paths.results_base_name),  # type: ignore
		write_period=coefficients.xdmf_write_period,
		stagnation_plateau_relative_error=user_inputs.stagnation_plateau_relative_error,
		functions_to_write=written,
	)
	begin_t = time.time()
	success = integrator.integrate()
	comm.Barrier()
	if rank == 0:
		logging.getLogger(__name__).info(f"Solve: {time.time() - begin_t:.1f} s, success={success}.")
	return success


def main(user_inputs: UserInputs = UserInputs()) -> bool:
	"""Mesh, rescale (if nondimensionalised), assemble, solve.

	Args:
	    user_inputs: Configuration.

	Returns:
	    True if the simulation completed.
	"""
	simulation_paths = build_mesh(user_inputs)
	mesh, domains_mf, boundaries_mf, points_mf = io.load_mesh_data(
		simulation_paths.mesh_folder / user_inputs.stem_mesh_name
	)
	units = SolveUnits.from_inputs(user_inputs)
	# In memory only: X_hat = X / L0 (identity for "none").
	mesh.coordinates()[:] = units.unknowns.nondimensionalize_length(mesh.coordinates())
	mesh.bounding_box_tree().build(mesh)

	coefficients = Coefficients.from_inputs(user_inputs, units.coefficients)
	problem = initialise_problem(mesh, domains_mf, boundaries_mf, points_mf, user_inputs, coefficients)
	problem = set_weak_form_and_bcs(problem, boundaries_mf, user_inputs, coefficients)
	apply_initial_bcs(problem)
	success = solve_problem(problem, simulation_paths, user_inputs, coefficients)
	# ParaView geometry and time stamps back to SI (the fields are already SI).
	if rank == 0:
		xdmf_units.redimensionalize_results(
			Path(f"{simulation_paths.results_base_name}.xdmf"), units.unknowns.length, units.coefficients.time
		)
	comm.Barrier()
	return success


if __name__ == "__main__":
	io.configure_logging(rank)
	if not main():
		comm.Abort(errorcode=1)
