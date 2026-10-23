r"""Terzaghi consolidation, nondimensionalised by the parameters (one step).

The 1D consolidation of a poro-elastic column, solved in dimensionless variables: every coefficient is
divided by its characteristic scale *before* the weak form is written, so the operators, the weak form and
the solver are exactly those of an SI simulation. Every line that differs from the SI demo is tagged
``# NONDIM``.

Problem
-------
Rectangle :math:`[0, L] \times [0, W]` (plane strain), saturated with an incompressible fluid, loaded at
:math:`t = 0` by :math:`p_0` on its left face, which is drained:

.. image:: /_static/resources/terzaghi_plane_strain.svg
	:alt: Plane-strain gel rectangle loaded by p0 at x = 0 (drained), u_x = 0 at x = L, rollers on y = 0 and y = W
	:width: 100%
	:align: center

Initial state: :math:`\mathbf u = \mathbf 0` and :math:`p = p_0` (undrained response to the load), except on the
drained face where :math:`p = 0` from :math:`t = 0` (step-0 BC applied to the initial vector).

Equations (reference configuration, St Venant--Kirchhoff skeleton)
-------------------------------------------------------------------
.. math::
	R_u &= \int_{\Omega_0} \big(\mathbf S - J p\, \mathbf C^{-1}\big) : \delta\mathbf E \,\mathrm dV
	       + \int_{x=0} p_0\, J\mathbf F^{-T}\mathbf N\cdot\delta\mathbf u \,\mathrm dS = 0 \\
	R_p &= \int_{\Omega_0} \frac{J - J^n}{\Delta t}\,\delta p
	       + \mathbf K_L \nabla p \cdot \nabla \delta p \,\mathrm dV = 0

with :math:`\mathbf K_L = k\, J\mathbf C^{-1}` the Darcy conductivity pulled back to the reference configuration
(``materials.permeability.constant``) and :math:`k` the hydraulic conductivity :math:`k_{int}/\eta`.
The exact solution (small strains) is Terzaghi's,
see ``terzaghi_analytics.py``.

Nondimensionalisation by the parameters: the recipe
-----------------------------------------------------
1. **Scales.** :math:`L_0 = L`, :math:`\mu_0 = M` (oedometric modulus), :math:`t_0 = L^2/c_v` with
   :math:`c_v = k M`: the dimensionless time is Terzaghi's time factor, :math:`\hat t = T`
   (``UserInputs.scales``, a :class:`.CharacteristicScales`).
2. **Mesh.** :math:`\hat{\mathbf X} = \mathbf X/L_0`: the mesh coordinates are divided by :math:`L_0` once,
   after loading. Then :math:`\hat{\mathbf u} = \mathbf u/L_0` and :math:`\hat\nabla = L_0\nabla`, so
   :math:`\mathbf F`, :math:`J`, :math:`\mathbf E` are unchanged: the kinematics needs no change.
3. **Unknowns.** :math:`\hat p = p/\mu_0` (initial value :math:`p_0/\mu_0`).
4. **Coefficients.** :math:`\hat E = E/\mu_0` (:math:`\nu` unchanged), :math:`\hat p_0 = p_0/\mu_0`,
   :math:`\hat k = k\,\mu_0 t_0/L_0^2 = 1`, durations :math:`\hat t = t/t_0`
   (:class:`DimensionlessCoefficients`, built with ``CharacteristicScales.nondimensionalize_*``).
5. **Weak form.** Unchanged: dividing :math:`R_u` by :math:`\mu_0 L_0^{d-1}` and :math:`R_p` by
   :math:`L_0^d/t_0` gives the same integrals written with hats. Every ``scale_factor`` stays 1.
6. **Outputs.** QOIs are redimensionalised in their expression (:math:`u = L_0\hat u`, :math:`p = \mu_0\hat p`);
   the solver time is :math:`\hat t` (``t0`` converts it); the XDMF geometry and time stamps are put back in
   m and s after the run (``io.xdmf_units``). The XDMF fields stay dimensionless.

Why: in SI, the rows of the Jacobian differ by many orders of magnitude (:math:`E \sim 10^4` Pa,
:math:`k \sim 10^{-13}`, :math:`L \sim 10^{-4}`); in hats every coefficient is :math:`\mathcal O(1)`, so the
linear solves are well conditioned and one relative tolerance fits every unknown.

Run
---
``python3 main.py`` writes ``outputs/`` next to this file: the mesh, the results (XDMF, QOIs), and in
``images/`` the figure ``terzaghi.png`` (profiles, settlement, errors) and ``errors.csv``.
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import dolfin
import dolfin_mech
import numpy as np
from fenicsoft import io, materials, operators
from fenicsoft.mesh_geometries import GmshKeys, MeshOptions, RectangleGeometry, RectangleMarker, RectangleMeshGenerator
from numpy.typing import NDArray

from hill_playground.io import xdmf_units
from hill_playground.materials import permeability
from hill_playground.scaling import CharacteristicScales, nondimensionalize_step

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared analytics and post-processing
import post_process  # noqa: E402
from terzaghi_analytics import TerzaghiCase  # noqa: E402

comm = dolfin.MPI.comm_world
rank = dolfin.MPI.rank(comm)


@dataclass(frozen=True)
class UserInputs:
	r"""Configuration, SI units.

	Attributes:
	    project_prefix: Output folder.
	    stem_name: Results file stem.
	    stem_mesh_name: Mesh file stem.
	    regenerate_mesh: Re-run Gmsh even if the mesh exists.
	    length: :math:`L` (m), drainage length.
	    width: :math:`W` (m).
	    min_char_length: Gmsh minimum element size (m).
	    max_char_length: Gmsh maximum element size (m).
	    load_magnitude: :math:`p_0` (Pa), small against :math:`M` so that Terzaghi (linear) applies.
	    young: :math:`E` (Pa).
	    poisson: :math:`\nu`.
	    hydraulic_conductivity: :math:`k` (m^2/(Pa.s)).
	    quadrature_degree: Quadrature degree.
	    function_degree: Degree of :math:`\mathbf u` (:math:`p` uses one less).
	    sol_tol: Newton relative tolerances for :math:`(\hat{\mathbf u}, \hat p)`.
	    n_iter_max: Newton iterations per time step.
	    steps: Time steps, SI (s): 1 ms first (boundary layer at the drained face), accelerated up to 0.05 s.
	    time_integrator_param: Adaptive time stepping.
	    n_probes: Pressure probes along the axis :math:`y = W/2`.
	"""

	project_prefix: Path = Path(__file__).resolve().parent / "outputs"
	stem_name: str = "terzaghi_nondim_parameters"
	stem_mesh_name: str = "rectangle_mesh"
	regenerate_mesh: bool = False
	length: float = 1e-4
	width: float = 1e-5
	min_char_length: float = 1e-4 / 50
	max_char_length: float = 1e-4 / 50
	load_magnitude: float = 1.0  # p0 / M ~ 1e-4: small strains, as Terzaghi
	young: float = 5e3
	poisson: float = 0.4
	hydraulic_conductivity: float = 1.8e-15 / 1e-2
	quadrature_degree: int = 4
	function_degree: int = 2
	sol_tol: Tuple[float, float] = (1e-10, 1e-10)
	n_iter_max: int = 10
	steps: Sequence[io.Step] = (io.Step(duration=15.0, init_dt=1e-3, min_dt=1e-4, max_dt=5e-2),)
	time_integrator_param: Dict[str, Any] = field(
		default_factory=lambda: {"n_iter_for_accel": 4, "n_iter_for_decel": 16, "accel_coeff": 1.1, "decel_coeff": 2}
	)
	n_probes: int = 21

	@property
	def oedometric_modulus(self) -> float:
		r""":math:`M = E(1-\nu)/((1+\nu)(1-2\nu))` (Pa)."""
		return float(materials.get_longitudinal_modulus_young_poisson(self.young, self.poisson))

	@property
	def scales(self) -> CharacteristicScales:  # NONDIM
		r""":math:`L_0 = L`, :math:`\mu_0 = M`, :math:`t_0 = L^2/(kM)` (the density scale is unused).

		:math:`k` is already :math:`k_{\text{gel}}/\eta`, hence ``viscosity=1``.
		"""
		return CharacteristicScales.from_consolidation_scale(
			length=self.length,
			stress=self.oedometric_modulus,
			density=1.0,
			permeability=self.hydraulic_conductivity,
			viscosity=1.0,
			consolidation_modulus=self.oedometric_modulus,
		)

	@property
	def analytical(self) -> TerzaghiCase:
		"""Terzaghi's solution for these inputs (drainage from :math:`t = 0`)."""
		return TerzaghiCase(self.length, self.load_magnitude, self.oedometric_modulus, self.hydraulic_conductivity)

	@property
	def probe_x(self) -> NDArray[np.float64]:
		"""Probe abscissae (m), from the drained face to the impermeable one."""
		return np.linspace(0.0, self.length, self.n_probes)


@dataclass(frozen=True)
class DimensionlessCoefficients:  # NONDIM
	r"""Every coefficient entering the weak form, divided by its scale.

	Attributes:
	    young: :math:`\hat E = E/\mu_0`.
	    load: :math:`\hat p_0 = p_0/\mu_0`.
	    conductivity: :math:`\hat k = k\,\mu_0 t_0/L_0^2` (here 1).
	    steps: Durations and time steps in :math:`\hat t = t/t_0`.
	"""

	young: float
	load: float
	conductivity: float
	steps: Tuple[io.Step, ...]

	@classmethod
	def from_inputs(cls, user_inputs: UserInputs) -> "DimensionlessCoefficients":
		"""Convert the SI inputs with ``user_inputs.scales``.

		Args:
		    user_inputs: Configuration.

		Returns:
		    The dimensionless coefficients.
		"""
		s = user_inputs.scales
		return cls(
			young=s.nondimensionalize_stress(user_inputs.young),
			load=s.nondimensionalize_stress(user_inputs.load_magnitude),
			conductivity=s.nondimensionalize_permeability_over_viscosity(user_inputs.hydraulic_conductivity),
			steps=tuple(nondimensionalize_step(step, s) for step in user_inputs.steps),
		)


def main(user_inputs: UserInputs = UserInputs()) -> post_process.Errors:
	"""Mesh, scale, assemble, solve, compare with Terzaghi.

	Args:
	    user_inputs: Configuration.

	Returns:
	    Errors against the analytical solution.
	"""
	paths = build_mesh(user_inputs)
	mesh, domains_mf, boundaries_mf, points_mf = io.load_mesh_data(paths.mesh_folder / user_inputs.stem_mesh_name)
	scales = user_inputs.scales
	mesh.coordinates()[:] = scales.nondimensionalize_length(mesh.coordinates())  # NONDIM: X_hat = X / L0
	mesh.bounding_box_tree().build(mesh)
	coefficients = DimensionlessCoefficients.from_inputs(user_inputs)  # NONDIM

	problem = initialise_problem(mesh, domains_mf, boundaries_mf, points_mf, user_inputs, coefficients)
	material = materials.iso.kirchhoff.Constitutive(
		kinematics=problem.kinematics,
		parameters=materials.iso.kirchhoff.Parameters.from_young_poisson(
			young=coefficients.young,  # NONDIM: E_hat
			poisson=user_inputs.poisson,
		),
		name="kirchhoff",
	)
	set_boundary_dirichlet(problem, boundaries_mf)
	set_boundary_neumann(problem, coefficients.load)
	set_weak_form(problem, material, coefficients.conductivity)
	set_qois(problem, user_inputs)
	solve_problem(problem, paths, user_inputs)

	if rank == 0:  # NONDIM: ParaView geometry in m and time stamps in s
		xdmf_units.redimensionalize_results(Path(f"{paths.results_base_name}.xdmf"), scales.length, scales.time)
	comm.Barrier()
	return post_processing(paths, user_inputs)


def build_mesh(user_inputs: UserInputs) -> io.SimulationPaths:
	"""Gmsh rectangle on rank 0 (unless it exists), converted to XDMF.

	Args:
	    user_inputs: Configuration.

	Returns:
	    Output-folder layout.
	"""
	paths = io.configure_environment(user_inputs.project_prefix, user_inputs.stem_name)
	if user_inputs.regenerate_mesh or not (paths.mesh_folder / f"{user_inputs.stem_mesh_name}.msh").exists():
		if rank == 0:
			geometry = RectangleGeometry(length=user_inputs.length, width=user_inputs.width)
			options = MeshOptions(
				min_size=user_inputs.min_char_length,
				max_size=user_inputs.max_char_length,
				generate_quads=False,
				extra_opts={GmshKeys.TERMINAL: 0, GmshKeys.OPTIMIZE_NETGEN: 1},
			)
			with RectangleMeshGenerator(geometry, options, paths.mesh_folder) as generator:
				io.convert_msh_to_xdmf(generator.generate(mesh_stem_name=user_inputs.stem_mesh_name), prune_z=True)
		comm.Barrier()
	return paths


def initialise_problem(
	mesh: dolfin.Mesh,
	domains_mf: dolfin.MeshFunction,
	boundaries_mf: dolfin.MeshFunction,
	points_mf: dolfin.MeshFunction,
	user_inputs: UserInputs,
	coefficients: DimensionlessCoefficients,
) -> dolfin_mech.problems.Problem:
	r"""Unknowns :math:`(\hat{\mathbf u}, \hat p)` (Taylor--Hood :math:`P_k/P_{k-1}`), kinematics, time steps.

	Args:
	    mesh: Mesh, already in :math:`\hat{\mathbf X}`.
	    domains_mf: Cell tags.
	    boundaries_mf: Facet tags.
	    points_mf: Point tags.
	    user_inputs: Configuration.
	    coefficients: Initial pressure and steps in hats.

	Returns:
	    The problem.
	"""
	problem = dolfin_mech.problems.Problem()
	problem.set_mesh(mesh, define_facet_normals=True)
	problem.set_measures(domains=domains_mf, boundaries=boundaries_mf, points=points_mf)
	problem.set_quadrature_degree(quadrature_degree=user_inputs.quadrature_degree)
	problem.displacement_subsol = problem.add_vector_subsol(
		name="displacement", family="CG", degree=user_inputs.function_degree
	)
	problem.pl_subsol = problem.add_scalar_subsol(
		name="liquid_pressure",
		family="CG",
		degree=user_inputs.function_degree - 1,
		init_val=coefficients.load,  # NONDIM: p_hat(t=0) = p0 / mu0
	)
	problem.set_solution_finite_element()
	problem.set_solution_function_space()
	problem.set_solution_functions()
	problem.kinematics = dolfin_mech.kinematics.Kinematics(
		U=problem.displacement_subsol.subfunc, U_old=problem.displacement_subsol.func_old
	)
	for step in coefficients.steps:  # NONDIM: durations in t_hat
		problem.add_step(Deltat=step.duration, dt_ini=step.init_dt, dt_min=step.min_dt, dt_max=step.max_dt)
	return problem


def set_boundary_dirichlet(problem: dolfin_mech.problems.Problem, boundaries_mf: dolfin.MeshFunction) -> None:
	r"""Drained left face, rollers on the lateral faces, clamped right face; applied to the initial state too.

	Args:
	    problem: Problem.
	    boundaries_mf: Facet tags (``RectangleMarker``).
	"""
	u_fs, p_fs = problem.displacement_subsol.fs, problem.pl_subsol.fs
	for k_step in range(len(problem.steps)):
		for fs, marker in (
			(p_fs, RectangleMarker.LEFT),  # p = 0
			(u_fs.sub(1), RectangleMarker.FRONT),  # u_y = 0
			(u_fs.sub(1), RectangleMarker.BACK),  # u_y = 0
			(u_fs.sub(0), RectangleMarker.RIGHT),  # u_x = 0
		):
			problem.add_constraint(V=fs, sub_domains=boundaries_mf, sub_domain_id=marker, val=0.0, k_step=k_step)
	for constraint in problem.steps[0].constraints:  # the drained face starts at p = 0
		constraint.bc.apply(problem.sol_func.vector())


def set_boundary_neumann(problem: dolfin_mech.problems.Problem, load: float) -> None:
	r"""Follower pressure :math:`\hat p_0` on the left face, every step.

	Args:
	    problem: Problem.
	    load: :math:`\hat p_0`.
	"""
	for k_step in range(len(problem.steps)):
		problem.add_operator(
			operator=dolfin_mech.operators.loading.SurfacePressure(
				U_test=problem.displacement_subsol.dsubtest,
				kinematics=problem.kinematics,
				N=problem.mesh_normals,
				measure=problem.dS(int(RectangleMarker.LEFT)),
				P_val=load,  # NONDIM: p0_hat
			),
			k_step=k_step,
		)


def set_weak_form(problem: dolfin_mech.problems.Problem, material: Any, conductivity: float) -> None:
	r"""Skeleton stress, fluid stress, :math:`\dot J` and Darcy: :math:`R_u` and :math:`R_p`.

	Args:
	    problem: Problem.
	    material: Skeleton law (built with :math:`\hat E`).
	    conductivity: :math:`\hat k`.
	"""
	u, du = problem.displacement_subsol.subfunc, problem.displacement_subsol.dsubtest
	p, dp = problem.pl_subsol.subfunc, problem.pl_subsol.dsubtest
	kin = problem.kinematics
	darcy_conductivity = permeability.constant.Permeability(  # K_L = k J C^-1
		kin, permeability.constant.Parameters(conductivity_0=conductivity)
	).tensor
	residual_terms = [
		operators.hyper_elasticity.pk2.HyperElasticity(disp=u, disp_test=du, material=material, measure=problem.dV),
		operators.fluid_stress.biphasic.pk2.FluidStress(
			kinematics=kin, function=p, displacement=u, test=du, measure=problem.dV
		),
		operators.ChangeRate(function=kin.J, function_old=kin.J_old, test=dp, dx=problem.dV),
		operators.DarcyGradient(
			function=p,
			test=dp,
			conductivity=darcy_conductivity,
			dx=problem.dV,  # NONDIM: k_hat
		),
	]
	for k_step in range(len(problem.steps)):
		for term in residual_terms:
			problem.add_operator(operator=term, k_step=k_step)


def set_qois(problem: dolfin_mech.problems.Problem, user_inputs: UserInputs) -> None:
	r"""SI QOIs: settlement ``u0`` at :math:`(0, W/2)` and pressures ``p_i`` along :math:`y = W/2`.

	Args:
	    problem: Problem.
	    user_inputs: Configuration (probe positions in m).
	"""
	s = user_inputs.scales
	u_func, p_func = problem.displacement_subsol.func, problem.pl_subsol.func
	# No set_allow_extrapolation: under MPI every rank would then return a value and the QOI averages them.
	mid = s.nondimensionalize_length(user_inputs.width / 2.0)  # NONDIM: probes in mesh coordinates
	problem.add_qoi(name="u0", expr=s.redimensionalize_length(u_func[0]), point=[0.0, mid], update_type="direct")
	for i, x in enumerate(user_inputs.probe_x):
		problem.add_qoi(
			name=f"p_{i}",
			expr=s.redimensionalize_stress(p_func),  # NONDIM: p = mu0 p_hat
			point=[s.nondimensionalize_length(x), mid],
			update_type="direct",
		)


def solve_problem(problem: dolfin_mech.problems.Problem, paths: io.SimulationPaths, user_inputs: UserInputs) -> None:
	"""Newton + adaptive time stepping; writes the XDMF fields and the QOIs.

	Args:
	    problem: Assembled problem.
	    paths: Output layout.
	    user_inputs: Solver settings.

	Raises:
	    RuntimeError: The time integration failed.
	"""
	solver = dolfin_mech.core.NonlinearSolver(
		problem=problem,
		parameters={"sol_tol": list(user_inputs.sol_tol), "n_iter_max": user_inputs.n_iter_max},
		relax_type="constant",
		write_iter=False,
	)
	integrator = dolfin_mech.core.TimeIntegrator(
		problem=problem,
		solver=solver,
		parameters=user_inputs.time_integrator_param,
		print_out=str(paths.outputs_base_name),  # type: ignore
		print_sta=str(paths.outputs_base_name),  # type: ignore
		write_qois=str(paths.results_base_name) + "-qois",  # type: ignore
		write_sol=str(paths.results_base_name),  # type: ignore
	)
	begin = time.time()
	success = integrator.integrate()
	integrator.close()
	comm.Barrier()
	if not success:
		raise RuntimeError("Time integration failed.")
	if rank == 0:
		logging.getLogger(__name__).info(f"Solve: {timedelta(seconds=int(time.time() - begin))}, success={success}.")


def post_processing(paths: io.SimulationPaths, user_inputs: UserInputs) -> post_process.Errors:
	r"""Errors against Terzaghi, ``images/terzaghi.png`` and ``images/errors.csv`` (rank 0).

	Args:
	    paths: Output layout.
	    user_inputs: Configuration.

	Returns:
	    The errors (on every rank).
	"""
	errors = None
	if rank == 0:
		table = post_process.read_qois(
			Path(f"{paths.results_base_name}-qois.dat"),
			user_inputs.n_probes,
			time_scale=user_inputs.scales.time,  # NONDIM: solver time is t_hat
		)
		errors = post_process.compute_errors(user_inputs.analytical, table, user_inputs.probe_x)
		errors.history.to_csv(paths.images_folder / "errors.csv", index=False)
		figure = post_process.plot_results(
			user_inputs.analytical,
			table,
			user_inputs.probe_x,
			errors,
			paths.images_folder / "terzaghi.png",
			"Terzaghi, nondimensionalised by the parameters (one step)",
		)
		logging.getLogger(__name__).info(f"{errors}\nFigure: {figure}")
	return comm.bcast(errors, root=0)


if __name__ == "__main__":
	io.configure_logging(rank)
	errors = main()
	if rank == 0:
		print(errors)
