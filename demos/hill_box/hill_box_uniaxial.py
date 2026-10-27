r"""Hill box: an active poro-elastic gel slab clamped on two opposite faces.

A hydrogel slab seeded with randomly oriented contractile cells (Hill chain) is clamped at both
ends of its long axis :math:`x` and drained on its four other faces. Under the activation
:math:`\alpha(t)` the cells contract: the gel narrows in :math:`y` and :math:`z` between the clamped
:math:`x`-faces, expels liquid, and the cells may migrate, proliferate and reorient.

Files: this script (model and solve), ``hill_box_scaling.py`` (SI inputs to dimensionless
coefficients), ``hill_box_outputs.py`` (SI fields and QOIs), ``post_processing.py`` (orientation figures).

Geometry and boundary conditions
--------------------------------
Box :math:`\Omega_0 = [0, L_x] \times [0, L_y] \times [0, L_z]`, all gel (default
:math:`875 \times 375 \times 50\ \mu\mathrm{m}`).

.. image:: /_static/resources/hill_box_bcs.svg
	:alt: Gel slab clamped and impermeable on LEFT and RIGHT, drained and traction-free elsewhere
	:width: 100%
	:align: center

Unknowns
--------
=========================  ======  ==============================  ==================
Unknown                    Space   Initial value                   Present if
=========================  ======  ==============================  ==================
:math:`\hat{\mathbf u}`    CG2     0                               always
:math:`\theta,\varphi`     DG0     isotropic draw                  ``remodelling``
:math:`\hat p`             CG1     0                               always
:math:`\Phi_c`             CG1     :math:`\Phi_{c,0}`              ``cell_transport``
:math:`\tilde\phi_s`       CG1     :math:`\Phi_{s,0}`              ``cell_transport``
:math:`\varepsilon_a`      DG0     0 (:math:`\lambda_a = 1`)       always
=========================  ======  ==============================  ==================

Switched off, :math:`\theta,\varphi` stay at their random draw and :math:`\Phi_c = \Phi_{c,0}`.

Residuals
---------
.. math::
	R_u &= \int \big(\mathbf{P}_S + \mathbf{P}_h - J p \mathbf{F}^{-T}\big) : \nabla\delta\mathbf{u}\,\mathrm{d}V \\
	R_p &= \int \frac{J - J^n}{\Delta t}\delta p + \mathbf{K}_L \nabla p \cdot \nabla\delta p
	       - \mathbf{Q}_c \cdot \nabla\delta p \,\mathrm{d}V,
	       \qquad \mathbf{K}_L = \frac{k(\phi_\ell)}{\eta} J \mathbf{C}^{-1} \\
	R_c &= \int \frac{\Phi_c - \Phi_c^n}{\Delta t}\delta c - \mathbf{Q}_c \cdot \nabla\delta c
	       - r\,\mathrm{sat}\,\Phi_c\,\delta c \,\mathrm{d}V \\
	R_{\theta} &= \int \Big(\frac{\theta - \theta^n}{\Delta t}
	       - \frac{\alpha}{\tau_{eff}}\partial_\theta Q\Big)\delta\theta\,\mathrm{d}V \quad (\text{idem } \varphi) \\
	R_a &= \int \big[f_{max}\alpha f_{FL}(\lambda_a) h(D_a) + \eta_0 D_a + T_p - T_e\big]
	       \delta\varepsilon\,\mathrm{d}V \\
	R_s &= \int \Big(\tilde\phi_s - \frac{\Phi_{s,0}}{J}\Big)\delta\phi_s\,\mathrm{d}V

with :math:`\mathbf{P}_S` the Neo-Hookean + volumetric gel (``volumetric_law``),
:math:`\mathbf{P}_h = \mathbf{F}\,\Phi_c T_e/I_4\,\mathbf{f}_0\otimes\mathbf{f}_0` (Hill chain),
:math:`I_4 = \mathbf f_0\cdot\mathbf C\mathbf f_0`, :math:`\mathbf{Q}_c` the diffusive + haptotactic cell
flux, :math:`\mathrm{sat} = \langle 1 - \Phi_c/\Phi_{c,max}\rangle_+` and :math:`Q` the orientation potential.
In :math:`R_a`, the cell is an elastic element (SEE, tension :math:`T_e`) in series with a contractile
element (CE, stretch :math:`\lambda_a = e^{\varepsilon_a}`, rate :math:`D_a = \dot\varepsilon_a`) itself in
parallel with a passive element (PEE, tension :math:`T_p`); :math:`f_{FL}` is the Gaussian force--length law
and :math:`h` the Hill force--velocity law (``fenicsoft.materials.cell.active.hill``).

Nondimensionalisation (by the parameters)
-----------------------------------------
Inputs are SI. Scales :math:`L_0 = L_z`, :math:`\mu_0 = \mu_{gel}`, :math:`t_0 = L_0^2\eta/(k\,\kappa_{gel})`
(``UserInputs.scales``). The mesh is divided by :math:`L_0` after loading (:math:`\hat{\mathbf u} = \mathbf u/L_0`
follows), :math:`\hat p = p/\mu_0`, and every coefficient is converted by ``hill_box_scaling.Coefficients``;
the solver time is :math:`\hat t = t/t_0`. The weak form is written as in SI. Outputs are SI
(``hill_box_outputs``); the XDMF geometry and time stamps are put back in m and s after the run.

Options
-------
* ``permeability_law``: ``"constant"``, ``"kozeny_carman"``, ``"holmes_mow"``, ``"compaction_power"``
  (``fenicsoft.materials.permeability``);
* ``volumetric_law``: ``"quadratic"`` :math:`\frac{\kappa}{2}(J-1)^2` or ``"compaction"``
  (barrier at :math:`J_{\min} = \Phi_{s,0} + \Phi_c`, ``fenicsoft.materials.volumetric.compaction``);
* ``cell_transport``, ``remodelling``: solve :math:`(\Phi_c, \tilde\phi_s)` and :math:`(\theta, \varphi)`;
* ``quadrature_degrees``: quadrature degree per residual row (row = subsol of the test function).

Loading: activation :math:`\alpha` ramped 0 -> 1, then held. No external load.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence, Tuple, Union, get_args

import dolfin
import dolfin_mech
import fenicsoft
import fenicsoft.materials.orientation.spherical as spherical_mod
import fenicsoft.operators.cell as cell_ops
from fenicsoft import integration, io
from fenicsoft.integration import Step
from fenicsoft.io import xdmf_units
from fenicsoft.materials import permeability as perm
from fenicsoft.materials import saturated_mixture
from fenicsoft.materials.cell.active.activation import ActivePowerConfig
from fenicsoft.materials.cell.active.hill import law as hill_law
from fenicsoft.materials.cell.active.hill.functions import SeeVariant
from fenicsoft.mesh_geometries.box import BoxGeometry, BoxMarker, BoxMeshGenerator
from fenicsoft.mesh_geometries.mesh_geometries import GmshKeys, MeshOptions
from fenicsoft.operators import ResidualRows
from fenicsoft.operators.cell import ContractileEvolution
from fenicsoft.operators.cell.rotation.orientation_change import OrientationChange
from fenicsoft.scaling import CharacteristicScales
from hill_box_outputs import add_outputs
from hill_box_scaling import Coefficients

comm = dolfin.MPI.comm_world
rank = dolfin.MPI.rank(comm)

_HELD_MARKERS: Tuple[BoxMarker, ...] = (BoxMarker.LEFT, BoxMarker.RIGHT)
_DRAINED_MARKERS: Tuple[BoxMarker, ...] = (BoxMarker.FRONT, BoxMarker.BACK, BoxMarker.BOTTOM, BoxMarker.TOP)

PermeabilityLaw = Literal["constant", "kozeny_carman", "holmes_mow", "compaction_power"]
VolumetricLaw = Literal["quadratic", "compaction"]
_CHOICES: Dict[str, Any] = {
	"permeability_law": PermeabilityLaw,
	"volumetric_law": VolumetricLaw,
	"see_variant": SeeVariant,
}
"""``UserInputs`` options restricted to a ``Literal``: checked at construction (``Literal`` is static only)."""


@dataclass(slots=True, frozen=True)
class UserInputs:
	r"""Configuration, SI units.

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
	    cell_transport: Solve :math:`\Phi_c, \tilde\phi_s` (:math:`R_c, R_s`); else :math:`\Phi_c = \Phi_{c,0}`.
	    remodelling: Solve :math:`\theta, \varphi` (:math:`R_{\theta,\varphi}`); else frozen at the draw.
	    gel_shear_modulus: :math:`\mu_{gel}`, also :math:`\mu_0` (Pa).
	    gel_kappa: :math:`\kappa_{gel}` (Pa).
	    volumetric_law: ``"quadratic"`` or ``"compaction"`` (module docstring, "Options").
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
	    see_variant: SEE law (``fenicsoft.materials.cell.active.hill.functions.SeeVariant``).
	    k_2e: Exponent of the ``"exponential"`` SEE.
	    orientation_random_seed: Seed of the isotropic :math:`(\theta, \varphi)` draw.
	    remodel_tau: Remodelling time :math:`\tau` (s).
	    permeability: :math:`k_0` (m^2).
	    viscosity_pa_s: :math:`\eta` (Pa.s).
	    permeability_law: :math:`k(\phi_\ell)` law (module docstring, "Options").
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
	    xdmf_write_period: Output decimation (``DecimatedTimeIntegrator``): ``int`` (substeps) or ``TimeInterval``
	        (s), one value or one per step (default: every substep of the ramp, then every 840 s).
	    xdmf_si_only: XDMF holds the SI FOIs and the unknowns already dimensionless
	        (:math:`\theta, \varphi, \Phi_c, \tilde\phi_s, \varepsilon_a`);
	        no :math:`\hat{\mathbf u}, \hat p` nor ``*_old``.
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
	steps: Sequence[Step] = field(
		default_factory=lambda: (
			Step(duration=10.0, init_dt=1.0, min_dt=0.25),  # ramp, dt ~ tau_a (was 60 and 10)
			# Step(duration=600.0, init_dt=60.0, min_dt=0.25, max_dt=60.0),  # hold
			# Step(duration=600.0, init_dt=30.0, min_dt=5.0),  # step 0: 0% -> 100% ramp, 10 min
			Step(duration=180000.0, init_dt=1000.0, min_dt=100.0, max_dt=3000.0, min_simu_seconds=3 * 3600),
		)
	)
	activation_configs: Sequence[ActivePowerConfig] = field(
		default_factory=lambda: [
			ActivePowerConfig.ramp(k_step=0, ini=0.0, fin=1.0),
			ActivePowerConfig.constant(k_step=1, value=1.0),
		]
	)

	# --- Solver ---
	sol_tol: float = 1e-6
	solver_parameters: Dict[str, Any] = field(default_factory=lambda: {"n_iter_max": 12})
	integrator_parameters: Dict[str, Any] = field(
		default_factory=lambda: {
			"n_iter_for_accel": 3,
			"n_iter_for_decel": 6,
			"accel_coeff": 1.25,
			"decel_coeff": 1.25,
		}
	)
	xdmf_write_period: Union[int, integration.TimeInterval, Sequence[Union[int, integration.TimeInterval]]] = (
		1,
		# integration.TimeInterval(30.0),
		integration.TimeInterval(840.0),
	)
	stagnation_plateau_relative_error: float = 1e-5
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
		        ``Literal`` (``permeability_law``, ``volumetric_law``, ``see_variant``).
		"""
		for name, choices in _CHOICES.items():
			value = getattr(self, name)
			if value not in get_args(choices):
				raise ValueError(f"Unknown {name} '{value}', expected one of {get_args(choices)}.")
		solid_jacobian = saturated_mixture.compaction_jacobian(self.phi_s0, self.phi_c_init)
		if solid_jacobian >= 1.0:
			raise ValueError(f"J_min = phi_s0 + phi_c_init = {solid_jacobian} must be < 1.")
		if self.use_force_length:
			hill_law.check_uniqueness(self.k_e, self.k_p, self.f_max, self.lmbda_opt, self.width_fl)

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


def build_residual_rows(problem: dolfin_mech.problems.Problem, user_inputs: UserInputs) -> ResidualRows:
	"""Quadrature degree per row (subsol of the test function), for the rows present in ``problem``.

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
	r"""Register the unknowns (module docstring) and build the data fields.

	Sets on ``problem``: the subsols, ``theta_func``/``phi_func`` (subsol or frozen draw),
	``theta_initial``/``phi_initial``, ``phic_func`` (subsol or :math:`\Phi_{c,0}`), ``phi_s0``,
	``kinematics``, ``permeability`` (``permeability_law``), ``k_l_tensor``
	(:math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`) and ``sol_tol`` (one entry per subsol, registration order).

	Args:
	    mesh: Mesh (rescaled when nondimensionalised).
	    domains_mf: Cell tags.
	    boundaries_mf: Facet tags.
	    points_mf: Point tags.
	    user_inputs: Configuration.
	    coefficients: Dimensionless coefficients (steps, :math:`k_0/\eta`).

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
	r""":math:`k(\phi_\ell)/\eta` law of ``user_inputs.permeability_law`` (module docstring, "Options").

	Args:
	    problem: Problem with ``kinematics``, ``phi_s0``, ``phic_func``.
	    user_inputs: Configuration.
	    coefficients: :math:`k_0/\eta` dimensionless.

	Returns:
	    Law exposing ``ratio``, ``hydraulic_conductivity`` and ``tensor``.

	Raises:
	    ValueError: Unknown ``permeability_law``.
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
	r"""Gel law :math:`\mathbf{P}_S + \mathbf{P}_h` (Neo-Hooke + ``volumetric_law`` + Hill chain).

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.
	    coefficients: Moduli dimensionless.

	Returns:
	    ``(gel_material, hill_material)``; the second owns ``p_act`` (:math:`\alpha`).

	Raises:
	    ValueError: Unknown ``volumetric_law``.
	"""
	kin = problem.kinematics
	isochoric = fenicsoft.materials.iso.neo_hooke.Constitutive(
		kinematics=kin,
		parameters=fenicsoft.materials.iso.neo_hooke.Parameters.from_shear_modulus(mu=coefficients.mu),
		name="gel_isochoric",
	)
	if user_inputs.volumetric_law == "compaction":
		volumetric: Any = fenicsoft.materials.volumetric.CompactionVolumetricPenalty(
			kin,
			fenicsoft.materials.volumetric.CompactionVolumetricPenaltyParameters(
				kappa=coefficients.kappa,
				solid_jacobian=saturated_mixture.compaction_jacobian(problem.phi_s0, problem.phic_func),
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
) -> Tuple[List[Any], ContractileEvolution]:
	r"""Operators shared by every step; each integrates with the quadrature degree of its row.

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.
	    hill_material: Hill law (owner of ``p_act``).
	    coefficients: Dimensionless coefficients.
	    rows: Quadrature degree per row.

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
	)
	ops: List[Any] = [
		fenicsoft.operators.fluid_stress.biphasic.pk1.FluidStress(
			kinematics=kin,
			function=p_hat,
			displacement=problem.displacement_subsol.subfunc,
			test=problem.displacement_subsol.dsubtest,
			measure=dv("displacement"),
		),
		fenicsoft.operators.DarcyGradient(
			function=p_hat,
			test=dp,
			conductivity=problem.k_l_tensor,
			dx=dv("liquid_pressure"),
		),
		fenicsoft.operators.ChangeRate(
			function=kin.J,
			function_old=kin.J_old,
			test=dp,
			dx=dv("liquid_pressure"),
		),
		contractile,
	]

	if user_inputs.cell_transport:
		phi_c, dc = problem.phic_subsol.subfunc, problem.phic_subsol.dsubtest
		phi_s = problem.phis_subsol.subfunc

		def hapt(test: Any, row: str) -> Any:
			return cell_ops.SaturatedHaptotaxisFluxSmooth(
				phi_c, phi_s, c.rho_s, problem, c.hapt_coeff, user_inputs.phi_c_max, test, dv(row)
			)

		def diff(test: Any, row: str) -> Any:
			return cell_ops.IsotropicDiffusion(phi_c, problem, c.cellular_diffusion, test, dv(row))

		ops += [
			hapt(dp, "liquid_pressure"),
			diff(dp, "liquid_pressure"),
			fenicsoft.operators.ChangeRate(
				function=phi_c,
				function_old=problem.phic_subsol.func_old,
				test=dc,
				dx=dv("cell_fraction"),
			),
			cell_ops.SaturatedProliferation(
				phi_c,
				c.prol_coeff,
				user_inputs.phi_c_max,
				dc,
				dv("cell_fraction"),
			),
			hapt(dc, "cell_fraction"),
			diff(dc, "cell_fraction"),
			cell_ops.SubstrateProjection(
				phi_s_func=phi_s,
				test_phi_s=problem.phis_subsol.dsubtest,
				kin_j=kin.J,
				phi_s0=problem.phi_s0,
				measure=dv("substrate_smoothed"),
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
	r"""Register the residuals, the BCs (module docstring), the FOIs and QOIs (``hill_box_outputs``).

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    boundaries_mf: Facet tags.
	    user_inputs: Configuration.
	    coefficients: Dimensionless coefficients.

	Returns:
	    The assembled problem.
	"""
	rows = build_residual_rows(problem, user_inputs)
	gel_material, hill_material = build_materials(problem, user_inputs, coefficients)
	shared_ops, contractile = build_step_operators(problem, user_inputs, hill_material, coefficients, rows)

	for step_idx in range(len(problem.steps)):
		problem.add_operator(
			fenicsoft.operators.hyper_elasticity.pk1.HyperElasticity(
				disp=problem.displacement_subsol.subfunc,
				disp_test=problem.displacement_subsol.dsubtest,
				material=gel_material,
				measure=rows.measure(problem.dV, "displacement"),
				k_step=step_idx,
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

	add_outputs(problem, user_inputs, gel_material, hill_material, contractile, user_inputs.scales)
	return problem


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
	"""Newton + adaptive time stepping (``integration.DecimatedTimeIntegrator``: selected fields, rank-0 logs).

	Args:
	    problem: Assembled problem.
	    simulation_paths: Output layout.
	    user_inputs: Configuration.
	    coefficients: Write period dimensionless.

	Returns:
	    True if every step completed (at :math:`t_{fin}` or by plateau exit).
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
	integrator.close()  # release the results XDMF/HDF5 before xdmf_units rewrites them
	comm.Barrier()
	if rank == 0:
		logging.getLogger(__name__).info(f"Solve: {timedelta(seconds=int(time.time() - begin_t))}, success={success}.")
	return success


def main(user_inputs: UserInputs = UserInputs()) -> bool:
	"""Mesh, scale, assemble, solve.

	Args:
	    user_inputs: Configuration.

	Returns:
	    True if the simulation completed.
	"""
	simulation_paths = build_mesh(user_inputs)
	mesh, domains_mf, boundaries_mf, points_mf = io.load_mesh_data(
		simulation_paths.mesh_folder / user_inputs.stem_mesh_name
	)
	scales = user_inputs.scales
	mesh.coordinates()[:] = scales.nondimensionalize_length(mesh.coordinates())  # X_hat = X / L0
	mesh.bounding_box_tree().build(mesh)

	coefficients = Coefficients.from_inputs(user_inputs)
	problem = initialise_problem(mesh, domains_mf, boundaries_mf, points_mf, user_inputs, coefficients)
	problem = set_weak_form_and_bcs(problem, boundaries_mf, user_inputs, coefficients)
	apply_initial_bcs(problem)
	success = solve_problem(problem, simulation_paths, user_inputs, coefficients)
	# ParaView geometry and time stamps back to SI (the fields are already SI).
	if rank == 0:
		xdmf_units.redimensionalize_results(
			Path(f"{simulation_paths.results_base_name}.xdmf"), scales.length, scales.time
		)
	comm.Barrier()
	return success


if __name__ == "__main__":
	io.configure_logging(rank)
	if not main():
		comm.Abort(errorcode=1)
