r"""Terzaghi consolidation, nondimensionalised by the operators (initialising step + consolidation).

The 1D consolidation of a poro-elastic column. The mesh and the unknowns are scaled, but the coefficients and
the time stay in SI: each residual term is multiplied by a constant ``scale_factor`` so that every row of the
system becomes dimensionless. Every line that differs from the SI demo is tagged ``# NONDIM``.

Problem
-------
Rectangle :math:`[0, L] \times [0, W]` (plane strain), saturated with an incompressible fluid, loaded by
:math:`p_0` on its left face, which is drained:

.. image:: /_static/resources/terzaghi_plane_strain.svg
	:alt: Plane-strain gel rectangle loaded by p0 at x = 0 (drained), u_x = 0 at x = L, rollers on y = 0 and y = W
	:width: 100%
	:align: center

Initial state: :math:`\mathbf u = \mathbf 0` and :math:`p = p_0` everywhere (undrained response to the load).
Two steps: an **initialising step** of :math:`10^{-6}` s brings the pressure of the left face from :math:`p_0`
to 0 (consistent initial and boundary values), then the **consolidation** step keeps it at 0.

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

Nondimensionalisation by the operators: the recipe
----------------------------------------------------
1. **Scales.** :math:`L_0 = L`, :math:`\mu_0 = M` (oedometric modulus), :math:`t_0 = L^2/c_v` with
   :math:`c_v = k M` (``UserInputs.scales``, a :class:`.CharacteristicScales`).
2. **Mesh.** :math:`\hat{\mathbf X} = \mathbf X/L_0` (coordinates divided once, after loading). Then
   :math:`\hat{\mathbf u} = \mathbf u/L_0` and :math:`\hat\nabla = L_0\nabla`: :math:`\mathbf F`, :math:`J`,
   :math:`\mathbf E` are unchanged, the kinematics needs no change.
3. **Unknowns.** :math:`\hat p = p/\mu_0` (initial value :math:`p_0/\mu_0`). Coefficients and time stay in SI.
4. **Rows.** A row gathers the terms tested by the same test function, i.e. by the same unknown. Each SI row
   is divided by a normalisation that makes it :math:`\mathcal O(1)`: :math:`\mathcal N_u = \mu_0 L_0^d`
   (a stress times a volume) and :math:`\mathcal N_p = \mu_0 L_0^d/t_0` (a rate times a volume, times
   :math:`\mu_0` because :math:`\delta p = \mu_0\,\delta\hat p`); :math:`d` is the dimension of the mesh.
5. **Terms.** Written on the scaled mesh with :math:`\hat p` and the dimensionless FE test function, a term
   equals its SI value divided by a constant :math:`v`. With the scale :math:`s` of the test function
   (:math:`\delta\mathbf u = L_0\,\delta\hat{\mathbf u}`, :math:`\delta p = \mu_0\,\delta\hat p`), its
   ``scale_factor`` is :math:`s\,v/\mathcal N` (``ResidualRows.scale_factor``, class :class:`RowScaling`):

   * skeleton stress, row ``displacement``: :math:`v = L_0^{d-1}`, ``scale_factor`` :math:`1/\mu_0`;
   * fluid stress :math:`Jp\,\mathbf C^{-1}`, row ``displacement``: :math:`v = \mu_0 L_0^{d-1}`, ``scale_factor`` 1;
   * load :math:`p_0`, row ``displacement``: :math:`v = L_0^{d-1}`, ``scale_factor`` :math:`1/\mu_0`;
   * :math:`(J - J^n)/\Delta t`, row ``liquid_pressure``: :math:`v = L_0^d`, ``scale_factor`` :math:`t_0`;
   * Darcy, row ``liquid_pressure``: :math:`v = \mu_0 L_0^{d-2}`, ``scale_factor`` :math:`\mu_0 t_0/L_0^2`.

   The resulting system is the one of ``nondim_parameters`` (dimensionless coefficients), but the
   coefficients keep their SI values and the solver time stays in s.
6. **Outputs.** QOIs are redimensionalised in their expression (:math:`u = L_0\hat u`, :math:`p = \mu_0\hat p`);
   the XDMF geometry is put back in m after the run (``io.xdmf_units``); the XDMF fields stay
   dimensionless.

Why: in SI, the rows of the Jacobian differ by many orders of magnitude (:math:`E \sim 10^4` Pa,
:math:`k \sim 10^{-13}`, :math:`L \sim 10^{-4}`); scaled, every row is :math:`\mathcal O(1)`, so the linear
solves are well conditioned and one relative tolerance fits every unknown. Choose this method when the
coefficients must stay in SI (shared material laws, SI time for loads or activations).

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
from fenicsoft import integration, io, materials, operators
from fenicsoft.mesh_geometries import GmshKeys, MeshOptions, RectangleGeometry, RectangleMarker, RectangleMeshGenerator
from numpy.typing import NDArray

from hill_playground.io import xdmf_units
from hill_playground.materials import permeability
from hill_playground.operators import ResidualRows
from hill_playground.scaling import CharacteristicScales

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared analytics and post-processing
import post_process  # noqa: E402
from terzaghi_analytics import TerzaghiCase  # noqa: E402

comm = dolfin.MPI.comm_world
rank = dolfin.MPI.rank(comm)

INITIALISING_DURATION = 1e-6
"""Duration (s) of the initialising step, where the drained face goes from :math:`p_0` to 0."""


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
	    steps: Initialising step, then consolidation (s): 1 ms first, accelerated up to 0.05 s.
	    time_integrator_param: Adaptive time stepping (``n_iter_for_decel`` < ``n_iter_max``).
	    n_probes: Pressure probes along the axis :math:`y = W/2`.
	"""

	project_prefix: Path = Path(__file__).resolve().parent / "outputs"
	stem_name: str = "terzaghi_nondim_operator"
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
	steps: Sequence[integration.Step] = (
		integration.Step(duration=INITIALISING_DURATION, init_dt=INITIALISING_DURATION, min_dt=INITIALISING_DURATION),
		integration.Step(duration=15.0, init_dt=1e-3, min_dt=1e-4, max_dt=5e-2),
	)
	time_integrator_param: Dict[str, Any] = field(
		default_factory=lambda: {"n_iter_for_accel": 4, "n_iter_for_decel": 8, "accel_coeff": 1.1, "decel_coeff": 2}
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
		"""Terzaghi's solution for these inputs, drainage from the end of the initialising step."""
		return TerzaghiCase(
			self.length,
			self.load_magnitude,
			self.oedometric_modulus,
			self.hydraulic_conductivity,
			t_start=INITIALISING_DURATION,
		)

	@property
	def probe_x(self) -> NDArray[np.float64]:
		"""Probe abscissae (m), from the drained face to the impermeable one."""
		return np.linspace(0.0, self.length, self.n_probes)


@dataclass(frozen=True)
class RowScaling:  # NONDIM
	r"""Row settings and value-preserving factor :math:`v` of each term (recipe, steps 4--5).

	Attributes:
	    rows: Test scales :math:`s` and normalisations :math:`\mathcal N` per row.
	    values: :math:`v` per term (``skeleton``, ``fluid``, ``load``, ``rate``, ``darcy``).
	"""

	rows: ResidualRows
	values: Dict[str, float]

	@classmethod
	def build(cls, problem: dolfin_mech.problems.Problem, scales: CharacteristicScales) -> "RowScaling":
		r"""Rows named after the subsols of ``problem``.

		Args:
		    problem: Problem with its subsols.
		    scales: :math:`L_0, \mu_0, t_0`.

		Returns:
		    The row scaling.
		"""
		length, stress, t0 = scales.length, scales.stress, scales.time
		d = problem.mesh.geometry().dim()
		rows = ResidualRows.from_problem(
			problem,
			test_scales={"displacement": length, "liquid_pressure": stress},
			normalizations={"displacement": stress * length**d, "liquid_pressure": stress * length**d / t0},
		)
		values = {
			"skeleton": length ** (d - 1),
			"fluid": stress * length ** (d - 1),
			"load": length ** (d - 1),
			"rate": length**d,
			"darcy": stress * length ** (d - 2),
		}
		return cls(rows=rows, values=values)

	def factor(self, row: str, term: str) -> float:
		r"""``scale_factor`` :math:`= s\,v/\mathcal N` of ``term`` in ``row``.

		Args:
		    row: ``"displacement"`` or ``"liquid_pressure"``.
		    term: Key of :attr:`values`.

		Returns:
		    The factor.
		"""
		return self.rows.scale_factor(row, self.values[term])


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

	problem = initialise_problem(mesh, domains_mf, boundaries_mf, points_mf, user_inputs)
	scaling = RowScaling.build(problem, scales)  # NONDIM
	material = materials.iso.kirchhoff.Constitutive(
		kinematics=problem.kinematics,
		parameters=materials.iso.kirchhoff.Parameters.from_young_poisson(
			young=user_inputs.young, poisson=user_inputs.poisson
		),
		name="kirchhoff",
	)
	p0_hat = scales.nondimensionalize_stress(user_inputs.load_magnitude)  # NONDIM: p0 / mu0
	set_boundary_dirichlet(problem, boundaries_mf, p0_hat)
	set_boundary_neumann(problem, user_inputs.load_magnitude * scaling.factor("displacement", "load"))  # NONDIM
	set_weak_form(problem, material, user_inputs.hydraulic_conductivity, scaling)  # NONDIM: row scaling
	set_qois(problem, user_inputs)
	solve_problem(problem, paths, user_inputs)

	if rank == 0:  # NONDIM: ParaView geometry in m (the time is already in s)
		xdmf_units.redimensionalize_results(Path(f"{paths.results_base_name}.xdmf"), scales.length)
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
) -> dolfin_mech.problems.Problem:
	r"""Unknowns :math:`(\hat{\mathbf u}, \hat p)` (Taylor--Hood :math:`P_k/P_{k-1}`), kinematics, time steps.

	Args:
	    mesh: Mesh, already in :math:`\hat{\mathbf X}`.
	    domains_mf: Cell tags.
	    boundaries_mf: Facet tags.
	    points_mf: Point tags.
	    user_inputs: Configuration.

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
		init_val=user_inputs.scales.nondimensionalize_stress(user_inputs.load_magnitude),  # NONDIM: p0 / mu0
	)
	problem.set_solution_finite_element()
	problem.set_solution_function_space()
	problem.set_solution_functions()
	problem.kinematics = dolfin_mech.kinematics.Kinematics(
		U=problem.displacement_subsol.subfunc, U_old=problem.displacement_subsol.func_old
	)
	for step in user_inputs.steps:  # SI durations: the time is not scaled
		problem.add_step(Deltat=step.duration, dt_ini=step.init_dt, dt_min=step.min_dt, dt_max=step.max_dt)
	return problem


def set_boundary_dirichlet(
	problem: dolfin_mech.problems.Problem, boundaries_mf: dolfin.MeshFunction, initial_pressure: float
) -> None:
	r"""Drained left face (:math:`\hat p_0 \to 0` in the initialising step, then 0), rollers on the lateral faces.

	Roller :math:`u_x = 0` on the right face.

	Args:
	    problem: Problem.
	    boundaries_mf: Facet tags (``RectangleMarker``).
	    initial_pressure: :math:`\hat p_0`.
	"""
	u_fs, p_fs = problem.displacement_subsol.fs, problem.pl_subsol.fs
	problem.add_constraint(
		V=p_fs,
		sub_domains=boundaries_mf,
		sub_domain_id=RectangleMarker.LEFT,
		val_ini=initial_pressure,  # NONDIM: p0_hat
		val_fin=0.0,
		k_step=0,
	)
	problem.add_constraint(V=p_fs, sub_domains=boundaries_mf, sub_domain_id=RectangleMarker.LEFT, val=0.0, k_step=1)
	for k_step in range(len(problem.steps)):
		for fs, marker in (
			(u_fs.sub(1), RectangleMarker.FRONT),  # u_y = 0
			(u_fs.sub(1), RectangleMarker.BACK),  # u_y = 0
			(u_fs.sub(0), RectangleMarker.RIGHT),  # u_x = 0
		):
			problem.add_constraint(V=fs, sub_domains=boundaries_mf, sub_domain_id=marker, val=0.0, k_step=k_step)


def set_boundary_neumann(problem: dolfin_mech.problems.Problem, load: float) -> None:
	r"""Follower pressure on the left face, every step.

	``SurfacePressure`` has no ``scale_factor``: the term is linear in the pressure, so the factor multiplies
	the pressure itself.

	Args:
	    problem: Problem.
	    load: :math:`p_0` times its ``scale_factor``.
	"""
	for k_step in range(len(problem.steps)):
		problem.add_operator(
			operator=dolfin_mech.operators.loading.SurfacePressure(
				U_test=problem.displacement_subsol.dsubtest,
				kinematics=problem.kinematics,
				N=problem.mesh_normals,
				measure=problem.dS(int(RectangleMarker.LEFT)),
				P_val=load,
			),
			k_step=k_step,
		)


def set_weak_form(
	problem: dolfin_mech.problems.Problem, material: Any, conductivity: float, scaling: RowScaling
) -> None:
	r"""Skeleton stress, fluid stress, :math:`\dot J` and Darcy, each with its ``scale_factor``.

	Args:
	    problem: Problem.
	    material: Skeleton law (SI moduli).
	    conductivity: :math:`k` (SI).
	    scaling: ``scale_factor`` of each term.
	"""
	u, du = problem.displacement_subsol.subfunc, problem.displacement_subsol.dsubtest
	p, dp = problem.pl_subsol.subfunc, problem.pl_subsol.dsubtest
	kin = problem.kinematics
	darcy_conductivity = permeability.constant.Permeability(  # K_L = k J C^-1
		kin, permeability.constant.Parameters(conductivity_0=conductivity)
	).tensor
	residual_terms = [
		operators.hyper_elasticity.pk2.HyperElasticity(
			disp=u,
			disp_test=du,
			material=material,
			measure=problem.dV,
			scale_factor=scaling.factor("displacement", "skeleton"),  # NONDIM: 1/mu0
		),
		operators.fluid_stress.biphasic.pk2.FluidStress(
			kinematics=kin,
			function=p,
			displacement=u,
			test=du,
			measure=problem.dV,
			scale_factor=scaling.factor("displacement", "fluid"),  # NONDIM: 1
		),
		operators.ChangeRate(
			function=kin.J,
			function_old=kin.J_old,
			test=dp,
			dx=problem.dV,
			scale_factor=scaling.factor("liquid_pressure", "rate"),  # NONDIM: t0
		),
		operators.DarcyGradient(
			function=p,
			test=dp,
			conductivity=darcy_conductivity,
			dx=problem.dV,
			scale_factor=scaling.factor("liquid_pressure", "darcy"),  # NONDIM: mu0 t0 / L0^2
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
		table = post_process.read_qois(Path(f"{paths.results_base_name}-qois.dat"), user_inputs.n_probes)
		errors = post_process.compute_errors(user_inputs.analytical, table, user_inputs.probe_x)
		errors.history.to_csv(paths.images_folder / "errors.csv", index=False)
		figure = post_process.plot_results(
			user_inputs.analytical,
			table,
			user_inputs.probe_x,
			errors,
			paths.images_folder / "terzaghi.png",
			"Terzaghi, nondimensionalised by the operators (initialising step + consolidation)",
		)
		logging.getLogger(__name__).info(f"{errors}\nFigure: {figure}")
	return comm.bcast(errors, root=0)


if __name__ == "__main__":
	io.configure_logging(rank)
	errors = main()
	if rank == 0:
		print(errors)
