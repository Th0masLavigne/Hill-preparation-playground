r"""Hill box: the model of equations.tex on a gel slab held on two faces (simplified stage B).

A first check of the implementation before the microtissue-pillar demos: the CORE gel of
stage B, without pillars, held by its two x-faces.

1. Geometry and boundary conditions
-------------------------------------
Box :math:`\Omega_0 = [0, L_x] \times [0, L_y] \times [0, L_z]` (``BoxGeometry``), all gel.

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

3. Residuals (equations.tex, Sec. "Weak forms and residuals")
---------------------------------------------------------------
.. math::
	R_u &= \int \big(\mathbf{P}_S + \mathbf{P}_h - J p \mathbf{F}^{-T}\big) : \nabla\delta\mathbf{u}\,\mathrm{d}V \\
	R_p &= \int \frac{J - J^n}{\Delta t}\delta p + \mathbf{K}_L \nabla p \cdot \nabla\delta p
	       - \mathbf{Q}_c \cdot \nabla\delta p \,\mathrm{d}V, \qquad \mathbf{K}_L = \frac{k}{\eta} J \mathbf{C}^{-1} \\
	R_c &= \int \frac{\Phi_c - \Phi_c^n}{\Delta t}\delta c - \mathbf{Q}_c \cdot \nabla\delta c
	       - r\,\mathrm{sat}\,\Phi_c\,\delta c \,\mathrm{d}V \\
	R_{\theta} &= \int \Big(\frac{\theta - \theta^n}{\Delta t}
	       - \frac{\alpha}{\tau_{eff}}\partial_\theta Q\Big)\delta\theta\,\mathrm{d}V \quad (\text{idem } \varphi) \\
	R_a &= \int \big[f_{max}\alpha f_{FL}(\lambda_a) h(D_a) + \eta_0 D_a + T_p - T_e\big]
	       \delta\varepsilon\,\mathrm{d}V \\
	R_s &= \int \Big(\tilde\phi_s - \frac{\Phi_{s,0}}{J}\Big)\delta\phi_s\,\mathrm{d}V

with :math:`\mathbf{P}_S` the Neo-Hookean + :math:`\frac{\kappa}{2}(J-1)^2` gel,
:math:`\mathbf{P}_h = \mathbf{F}\,\Phi_c T_e/I_4\,\mathbf{f}_0\otimes\mathbf{f}_0` (Hill chain) and
:math:`\mathbf{Q}_c` the diffusive + haptotactic cell flux.

4. Nondimensionalisation (equations.tex, Part II)
---------------------------------------------------
Inputs are SI. With ``nondimensionalize=True`` (default) the mesh is rescaled by :math:`L_0`
(:math:`\hat{\mathbf X} = \mathbf X/L_0`, hence :math:`\hat{\mathbf u} = \mathbf u/L_0`), the
pressure unknown is :math:`\hat p = p/\mu_0`, and every operator is multiplied by
``scale_factor`` :math:`= v_k/\mathcal N_{row}` (Table tab:sf) so that the assembled system is
:math:`\tilde R_k = R_k/\mathcal N_k`:

=====================================  ====  ==========================
Term                                   Row   ``scale_factor``
=====================================  ====  ==========================
gel stresses (incl. Hill SEE)          u     :math:`1/(\mu_0 L_0)`
:math:`-Jp\mathbf F^{-T}`              u     :math:`1/L_0`
:math:`(J-J^n)/\Delta t`               p     :math:`t_0/\mu_0`
Darcy                                  p     :math:`t_0/L_0^2`
cell flux tested by :math:`\delta p`   p     :math:`t_0/(\mu_0 L_0^2)`
rate, proliferation; orientation       c     :math:`t_0`
cell flux tested by :math:`\delta c`   c     :math:`t_0/L_0^2`
contractile element                    a     :math:`1/\mu_0`
substrate projection                   s     1
=====================================  ====  ==========================

:math:`L_0 = L_z` (drainage length), :math:`\mu_0 = \mu_{gel}`,
:math:`t_0 = L_0^2\eta/(k\,\kappa_{gel})`. Exported fields and QOIs are SI (:math:`\mathbf u`
in m, :math:`p` in Pa; tensions are built from SI moduli and are already in Pa).

5. Loading
------------
Activation :math:`\alpha`: ramp 0 -> 1, then hold. No external load.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import dolfin
import dolfin_mech
import fenicsoft
import fenicsoft.materials.orientation.spherical as spherical_mod
import fenicsoft.operators.cell as cell_ops
import hill_chain_helpers
from fenicsoft import integration, io
from fenicsoft.materials.cell.active.aniso.law import ActivePowerConfig
from fenicsoft.mesh_geometries.box import BoxGeometry, BoxMarker, BoxMeshGenerator
from fenicsoft.mesh_geometries.mesh_geometries import GmshKeys, MeshOptions
from fenicsoft.operators.cell.rotation.orientation_change import OrientationChange

from hill_playground.materials.cell.active.hill import law as hill_law
from hill_playground.operators.cell import ContractileEvolution, SaturatedProliferation
from hill_playground.scaling import CharacteristicScales

comm = dolfin.MPI.comm_world
rank = dolfin.MPI.rank(comm)

_HELD_MARKERS: Tuple[BoxMarker, ...] = (BoxMarker.LEFT, BoxMarker.RIGHT)
_DRAINED_MARKERS: Tuple[BoxMarker, ...] = (BoxMarker.FRONT, BoxMarker.BACK, BoxMarker.BOTTOM, BoxMarker.TOP)


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
	    quadrature_degree: Quadrature degree.
	    foi_degree: Degree of the DG spaces of the FOIs.
	    nondimensionalize: Solve the dimensionless system (module docstring, section 4).
	    cell_transport: Solve :math:`\Phi_c, \tilde\phi_s` (:math:`R_c, R_s`); else :math:`\Phi_c = \Phi_{c,0}`.
	    remodelling: Solve :math:`\theta, \varphi` (:math:`R_{\theta,\varphi}`); else frozen at the draw.
	    gel_shear_modulus: :math:`\mu_{gel}`, also :math:`\mu_0` (Pa).
	    gel_kappa: :math:`\kappa_{gel}` (Pa).
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
	    see_variant: SEE law (``hill.functions.SEE_VARIANTS``).
	    k_2e: Exponent of the ``"exponential"`` SEE.
	    orientation_random_seed: Seed of the isotropic :math:`(\theta, \varphi)` draw.
	    remodel_tau: Remodelling time :math:`\tau` (s).
	    permeability: :math:`k` (m^2).
	    viscosity_pa_s: :math:`\eta` (Pa.s).
	    rho_s: :math:`\rho_s` (kg/m^3), haptotaxis weight and :math:`\rho_0`.
	    phi_s0: :math:`\Phi_{s,0}`.
	    phi_c_init: :math:`\Phi_{c,0}`.
	    phi_c_max: :math:`\Phi_{c,max}`.
	    cellular_diffusion: :math:`D_\phi` (m^2/s).
	    hapt_coeff: :math:`h_\phi`.
	    prol_coeff: :math:`r` (1/s).
	    displacement_degree: Degree :math:`k` of :math:`\mathbf u` (:math:`p, \Phi_c, \tilde\phi_s` use :math:`k-1`).
	    steps: Ramp then hold (s).
	    activation_configs: :math:`\alpha(t)` per step.
	    sol_tol: Newton tolerance, the same for every (dimensionless) unknown.
	    solver_parameters: ``NonlinearSolver`` parameters (``sol_tol`` added at run time).
	    integrator_parameters: ``TimeIntegrator`` adaptive-``dt`` parameters.
	    xdmf_write_period: Output decimation (``DecimatedTimeIntegrator``).
	    stagnation_plateau_relative_error: Plateau-exit threshold.
	    project_prefix: Output folder.
	    stem_name: Output file stem.
	    stem_mesh_name: Mesh file stem.
	"""

	# --- Geometry: the CORE gel patch of the pillar demos ---
	length_x: float = 375e-6
	width_y: float = 875e-6
	thickness_z: float = 50e-6

	# --- Mesh ---
	mesh_min_size: float = 50e-6 / 4
	mesh_max_size: float = 50e-6 / 2
	regenerate_mesh: bool = True
	quadrature_degree: int = 4
	foi_degree: int = 0

	# --- Switches ---
	nondimensionalize: bool = True
	cell_transport: bool = True
	remodelling: bool = True

	# --- Gel skeleton ---
	gel_shear_modulus: float = 1700.0
	gel_kappa: float = 1417.0

	# --- Hill chain (equations.tex, Table tab:groups) ---
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
	see_variant: str = "mooney"
	k_2e: float = 0.0

	# --- Orientation ---
	orientation_random_seed: int = 0
	remodel_tau: float = 3600.0

	# --- Poromechanics ---
	permeability: float = 1.8e-15
	viscosity_pa_s: float = 1.0e-3
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
			io.Step(duration=60.0, init_dt=2.0, min_dt=0.25),  # ramp, dt ~ tau_a
			io.Step(duration=600.0, init_dt=2.0, min_dt=0.25, max_dt=60.0),  # hold
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
			"n_iter_for_accel": 4,
			"n_iter_for_decel": 10,
			"accel_coeff": 1.25,
			"decel_coeff": 1.25,
		}
	)
	xdmf_write_period: Union[int, integration.TimeInterval, Sequence[Union[int, integration.TimeInterval]]] = (
		1,
		integration.TimeInterval(30.0),
	)
	stagnation_plateau_relative_error: float = 5e-3

	# --- Output ---
	project_prefix: Path = Path("./outputs_hill_box_uniaxial")
	stem_name: str = "hill_box_uniaxial"
	stem_mesh_name: str = "hill_box_uniaxial_mesh"

	def __post_init__(self) -> None:
		r"""Check the saturation and the uniqueness of the CE equilibrium.

		Raises:
		    ValueError: :math:`\Phi_{s,0} + \Phi_{c,0} \ge 1`, or Eq. uniqueness violated.
		"""
		if self.phi_s0 + self.phi_c_init >= 1.0:
			raise ValueError(f"phi_s0 + phi_c_init = {self.phi_s0 + self.phi_c_init} must be < 1.")
		if self.use_force_length:
			hill_chain_helpers.check_hill_uniqueness(self.k_e, self.k_p, self.f_max, self.lmbda_opt, self.width_fl)

	@property
	def hydraulic_conductivity(self) -> float:
		r""":math:`k/\eta` (m^2/(Pa.s))."""
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
class Scales:
	r"""Scales actually applied: :math:`(L_0, \mu_0, t_0)`, or ones when not nondimensionalised.

	Attributes:
	    length: :math:`L_0` (m).
	    stress: :math:`\mu_0` (Pa).
	    time: :math:`t_0` (s).
	"""

	length: float
	stress: float
	time: float

	@classmethod
	def from_inputs(cls, user_inputs: UserInputs) -> "Scales":
		"""Scales of ``user_inputs``, or (1, 1, 1) if ``nondimensionalize`` is False.

		Args:
		    user_inputs: Configuration.

		Returns:
		    The applied scales.
		"""
		if not user_inputs.nondimensionalize:
			return cls(1.0, 1.0, 1.0)
		s = user_inputs.scales
		return cls(s.length, s.stress, s.time)


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
) -> dolfin_mech.problems.Problem:
	r"""Register the unknowns (module docstring, section 2) and build the data fields.

	Sets on ``problem``: the subsols, ``theta_func``/``phi_func`` (subsol or frozen draw),
	``theta_initial``/``phi_initial``, ``phic_func`` (subsol or :math:`\Phi_{c,0}`), ``phi_s0``,
	``kinematics``, ``k_l_tensor`` (:math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`) and
	``sol_tol`` (one entry per subsol, registration order).

	Args:
	    mesh: Mesh (rescaled when nondimensionalised).
	    domains_mf: Cell tags.
	    boundaries_mf: Facet tags.
	    points_mf: Point tags.
	    user_inputs: Configuration.

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

	for step_idx, step in enumerate(user_inputs.steps):
		problem.add_step(Deltat=step.duration, dt_ini=step.init_dt, dt_min=step.min_dt, dt_max=step.max_dt)
		if step.min_simu_seconds is not None:
			problem.steps[step_idx].min_simu_seconds = step.min_simu_seconds

	problem.k_l_tensor = hill_chain_helpers.pulled_back_conductivity(
		dolfin.Constant(user_inputs.hydraulic_conductivity), problem.kinematics
	)
	return problem


def build_materials(
	problem: dolfin_mech.problems.Problem, user_inputs: UserInputs
) -> Tuple[Any, hill_law.Constitutive]:
	r"""Gel law :math:`\mathbf{P}_S + \mathbf{P}_h` (Neo-Hooke + volumetric penalty + Hill chain).

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.

	Returns:
	    ``(gel_material, hill_material)``; the second owns ``p_act`` (:math:`\alpha`).
	"""
	passive = fenicsoft.materials.iso.neo_hooke.Constitutive(
		kinematics=problem.kinematics,
		parameters=fenicsoft.materials.iso.neo_hooke.Parameters.from_shear_modulus(mu=user_inputs.gel_shear_modulus),
		name="gel_isochoric",
	) + fenicsoft.materials.volumetric.QuadraticVolumetricPenalty(
		kinematics=problem.kinematics,
		parameters=fenicsoft.materials.volumetric.VolumetricPenaltyParameters(kappa=user_inputs.gel_kappa),
		name="gel_volumetric",
	)
	hill_params = hill_law.Parameters(
		phi_c=problem.phic_func,
		f0=spherical_mod.f0_from_theta_phi(problem.theta_func, problem.phi_func),
		activation_configs=user_inputs.activation_configs,
		f_max=user_inputs.f_max,
		simulation_steps=problem.steps,
		use_force_length=user_inputs.use_force_length,
		lmbda_opt=user_inputs.lmbda_opt,
		W=user_inputs.width_fl,
		eps_a=problem.eps_a_subsol.subfunc,
		eps_a_old=problem.eps_a_subsol.func_old,
		k_e=user_inputs.k_e,
		k_p=user_inputs.k_p,
		hill_a=user_inputs.hill_a,
		hill_b=user_inputs.hill_b,
		f_ecc=user_inputs.f_ecc,
		eta_0=user_inputs.eta_0,
		see_variant=user_inputs.see_variant,
		k_2e=user_inputs.k_2e,
	)
	hill_material = hill_law.Constitutive(problem.kinematics, hill_params, name="cell_hill_chain")
	return passive + hill_material, hill_material


def build_step_operators(
	problem: dolfin_mech.problems.Problem,
	user_inputs: UserInputs,
	hill_material: hill_law.Constitutive,
	scales: Scales,
) -> Tuple[List[Any], ContractileEvolution]:
	r"""Operators shared by every step, with their ``scale_factor`` (module docstring, section 4).

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.
	    hill_material: Hill law (owner of ``p_act``).
	    scales: Applied scales.

	Returns:
	    ``(operators, contractile_evolution)``.
	"""
	length, stress, t0 = scales.length, scales.stress, scales.time
	dv = problem.dV
	kin = problem.kinematics
	p_hat, dp = problem.pl_subsol.subfunc, problem.pl_subsol.dsubtest

	contractile = ContractileEvolution(
		law=hill_material, test=problem.eps_a_subsol.dsubtest, dx=dv, scale_factor=1.0 / stress
	)
	ops: List[Any] = [
		fenicsoft.operators.fluid_stress.biphasic.pk1.FluidStress(
			kinematics=kin,
			function=p_hat,
			displacement=problem.displacement_subsol.subfunc,
			test=problem.displacement_subsol.dsubtest,
			measure=dv,
			scale_factor=1.0 / length,
		),
		fenicsoft.operators.DarcyGradient(
			function=p_hat, test=dp, conductivity=problem.k_l_tensor, dx=dv, scale_factor=t0 / length**2
		),
		fenicsoft.operators.ChangeRate(
			function=kin.J, function_old=kin.J_old, test=dp, dx=dv, scale_factor=t0 / stress
		),
		contractile,
	]

	if user_inputs.cell_transport:
		phi_c, dc = problem.phic_subsol.subfunc, problem.phic_subsol.dsubtest
		phi_s = problem.phis_subsol.subfunc

		def hapt(test: Any, factor: float) -> Any:
			return cell_ops.SaturatedHaptotaxisFluxSmooth(
				phi_c,
				phi_s,
				user_inputs.rho_s,
				problem,
				user_inputs.hapt_coeff,
				user_inputs.phi_c_max,
				test,
				dv,
				factor,
			)

		def diff(test: Any, factor: float) -> Any:
			return cell_ops.IsotropicDiffusion(phi_c, problem, user_inputs.cellular_diffusion, test, dv, factor)

		ops += [
			hapt(dp, t0 / (stress * length**2)),
			diff(dp, t0 / (stress * length**2)),
			fenicsoft.operators.ChangeRate(
				function=phi_c, function_old=problem.phic_subsol.func_old, test=dc, dx=dv, scale_factor=t0
			),
			SaturatedProliferation(phi_c, user_inputs.prol_coeff, user_inputs.phi_c_max, dc, dv, scale_factor=t0),
			hapt(dc, t0 / length**2),
			diff(dc, t0 / length**2),
			cell_ops.SubstrateProjection(
				phi_s_func=phi_s,
				test_phi_s=problem.phis_subsol.dsubtest,
				kin_j=kin.J,
				phi_s0=problem.phi_s0,
				measure=dv,
				scale_factor=1.0,
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
				tau=user_inputs.remodel_tau,
				activation=hill_material.p_act,
				dx=dv,
				f0_builder=builder,
				scale_factor=t0,
			)
			for subsol, builder in builders
		]
	return ops, contractile


def set_weak_form_and_bcs(
	problem: dolfin_mech.problems.Problem, boundaries_mf: dolfin.MeshFunction, user_inputs: UserInputs
) -> dolfin_mech.problems.Problem:
	r"""Register the residuals, the BCs (module docstring, section 1), the FOIs and QOIs.

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    boundaries_mf: Facet tags.
	    user_inputs: Configuration.

	Returns:
	    The assembled problem.
	"""
	scales = Scales.from_inputs(user_inputs)
	gel_material, hill_material = build_materials(problem, user_inputs)
	shared_ops, contractile = build_step_operators(problem, user_inputs, hill_material, scales)

	for step_idx in range(len(problem.steps)):
		problem.add_operator(
			fenicsoft.operators.hyper_elasticity.pk1.HyperElasticity(
				disp=problem.displacement_subsol.subfunc,
				disp_test=problem.displacement_subsol.dsubtest,
				material=gel_material,
				measure=problem.dV,
				k_step=step_idx,
				scale_factor=1.0 / (scales.stress * scales.length),
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

	add_outputs(problem, user_inputs, gel_material, hill_material, contractile, scales)
	return problem


def add_outputs(
	problem: dolfin_mech.problems.Problem,
	user_inputs: UserInputs,
	gel_material: Any,
	hill_material: hill_law.Constitutive,
	contractile: ContractileEvolution,
	scales: Scales,
) -> None:
	r"""FOIs and QOIs, all in SI (:math:`\mathbf u = L_0\hat{\mathbf u}`, :math:`p = \mu_0\hat p`).

	Args:
	    problem: Problem.
	    user_inputs: Configuration.
	    gel_material: Composite gel law.
	    hill_material: Hill law.
	    contractile: :math:`R_a` operator (provides :math:`T_0`).
	    scales: Applied scales.
	"""
	kin = problem.kinematics
	pressure = scales.stress * problem.pl_subsol.subfunc

	def foi(name: str, expr: Any, fs: Any) -> None:
		problem.add_foi(name=name, expr=expr, fs=fs, update_type="project")

	scalar_fois: Dict[str, Any] = {
		"jacobian": kin.J,
		"pressure": pressure,
		"cell_fraction": problem.phic_func,
		"fluid_fraction": 1.0 - (problem.phic_func + problem.phi_s0) / kin.J,
		"theta": problem.theta_func,
		"phi": problem.phi_func,
		"lambda_fibre": dolfin.sqrt(hill_material.inv4),
		"lambda_a": hill_material.lmbda_a,
		"lambda_e": dolfin.sqrt(hill_material.inv4e),
		"tension_see": hill_material.tension_see,
		"tension_pee": hill_material.tension_pee,
		"tension_ce": hill_material.tension_ce,
		"tension_isometric": contractile.isometric_tension,
	}
	for name, expr in scalar_fois.items():
		foi(name, expr, problem.sfoi_fs)
	foi("displacement_physical", scales.length * problem.displacement_subsol.subfunc, problem.vfoi_fs)
	foi("fibre", spherical_mod.f0_from_theta_phi(problem.theta_func, problem.phi_func), problem.vfoi_fs)
	foi("fibre_initial", spherical_mod.f0_from_theta_phi(problem.theta_initial, problem.phi_initial), problem.vfoi_fs)
	foi("cell_cauchy", hill_material.cauchy, problem.mfoi_fs)
	foi("total_cauchy", gel_material.cauchy - pressure * kin.I, problem.mfoi_fs)

	# Point probes, in mesh (hat) coordinates.
	probes = {k: [c / scales.length for c in v] for k, v in user_inputs.probes.items()}
	problem.displacement_subsol.func.set_allow_extrapolation(True)
	problem.pl_subsol.func.set_allow_extrapolation(True)
	problem.add_qoi(
		name="top_center_disp_z",
		expr=scales.length * problem.displacement_subsol.func[2],
		point=probes["top"],
		update_type="direct",
	)
	problem.add_qoi(
		name="front_center_disp_y",
		expr=scales.length * problem.displacement_subsol.func[1],
		point=probes["front"],
		update_type="direct",
	)
	problem.add_qoi(
		name="center_pressure",
		expr=scales.stress * problem.pl_subsol.func,
		point=probes["center"],
		update_type="direct",
	)

	def _activation(_: Any) -> float:
		return hill_material.current_activation

	problem.add_qoi(name="cellular_activation", expr=_activation, point=probes["center"], update_type="direct")

	# Gel averages: dolfin_mech assembles expr and divides by norm (hat volume, so the ratio is scale-free).
	volume = dolfin.assemble(dolfin.Constant(1.0) * problem.dV)
	averages: Dict[str, Any] = {
		"mean_lambda_a": hill_material.lmbda_a,
		"mean_tension_see": hill_material.tension_see,
		"mean_tension_isometric": contractile.isometric_tension,
		"mean_jacobian": kin.J,
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
	problem: dolfin_mech.problems.Problem, simulation_paths: io.SimulationPaths, user_inputs: UserInputs
) -> bool:
	"""Newton + adaptive time stepping (``DecimatedTimeIntegrator``), writing XDMF and QOIs.

	Args:
	    problem: Assembled problem.
	    simulation_paths: Output layout.
	    user_inputs: Configuration.

	Returns:
	    True if every step converged.
	"""
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
		write_period=user_inputs.xdmf_write_period,
		stagnation_plateau_relative_error=user_inputs.stagnation_plateau_relative_error,
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
	if user_inputs.nondimensionalize:
		# In memory only: X_hat = X / L0.
		mesh.coordinates()[:] /= user_inputs.scales.length
		mesh.bounding_box_tree().build(mesh)

	problem = initialise_problem(mesh, domains_mf, boundaries_mf, points_mf, user_inputs)
	problem = set_weak_form_and_bcs(problem, boundaries_mf, user_inputs)
	apply_initial_bcs(problem)
	return solve_problem(problem, simulation_paths, user_inputs)


if __name__ == "__main__":
	io.configure_logging(rank)
	if not main():
		comm.Abort(errorcode=1)
