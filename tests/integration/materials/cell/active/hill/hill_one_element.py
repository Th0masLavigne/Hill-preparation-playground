r"""One-element FE runs of the gel + Hill chain (no fluid), for comparison with :mod:`hill_0d_reference`.

Unit cube ``UnitCubeMesh(1, 1, 1)`` (6 tetrahedra, every node on the boundary), CG1 :math:`\mathbf u`,
DG0 :math:`\varepsilon_a`, :math:`\mathbf f_0 = \mathbf e_x`, uniform :math:`\Phi_c`:

.. math::
	R_u = \int (\mathbf P_S + \mathbf P_h) : \nabla\delta\mathbf u \,\mathrm dV, \qquad
	R_a = \int \big[f_{max}\alpha f_{FL} h(D_a) + \eta_0 D_a + T_p - T_e\big]\delta\varepsilon \,\mathrm dV

Two kinematic set-ups, both homogeneous:

* ``"isometric"``: :math:`u_x = (\lambda-1)x`, :math:`u_y = u_z = 0` on every face (:math:`\mathbf F` imposed);
* ``"free"``: symmetry planes :math:`u_x(0,\cdot)=u_y(\cdot,0,\cdot)=u_z(\cdot,\cdot,0)=0`, other faces traction-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dolfin
import dolfin_mech
import fenicsoft
import numpy as np
import pandas as pd
from hill_0d_reference import HillChain0D

from hill_playground.materials.cell.active.activation import ActivePowerConfig
from hill_playground.materials.cell.active.hill import law as hill_law
from hill_playground.operators.cell import ContractileEvolution

QOI_NAMES = ("alpha", "eps_a", "lambda", "lambda_t", "pk1_xx", "pk1_yy", "tension_see")


@dataclass(frozen=True)
class OneElementRun:
	r"""Run settings.

	Attributes:
	    model: Shared parameters with the 0D reference.
	    kinematics: ``"isometric"`` or ``"free"``.
	    lmbda: Imposed stretch (``"isometric"`` only).
	    ramp_duration: :math:`\alpha: 0 \to 1` duration (s).
	    hold_duration: :math:`\alpha = 1` duration (s).
	    dt: Fixed time step (s).
	"""

	model: HillChain0D = HillChain0D()
	kinematics: Literal["isometric", "free"] = "isometric"
	lmbda: float = 1.0
	ramp_duration: float = 20.0
	hold_duration: float = 200.0
	dt: float = 2.0


def _face(axis: int, value: float) -> dolfin.SubDomain:
	return dolfin.CompiledSubDomain(f"on_boundary && near(x[{axis}], {value})")


def run(settings: OneElementRun, out_dir: Path) -> pd.DataFrame:
	"""Solve and return the QOI history (one row per time step, column ``t`` first).

	Args:
	    settings: Run settings.
	    out_dir: Output folder.

	Returns:
	    Columns ``t`` + :data:`QOI_NAMES`.

	Raises:
	    RuntimeError: Time integration failed.
	"""
	m = settings.model
	mesh = dolfin.UnitCubeMesh(1, 1, 1)
	problem = dolfin_mech.problems.Problem()
	problem.set_mesh(mesh=mesh, define_facet_normals=True)
	problem.set_measures(domains=None, boundaries=None, points=None)
	disp = problem.add_vector_subsol(name="displacement", family="CG", degree=1)
	eps = problem.add_scalar_subsol(name="eps_a", family="DG", degree=0, init_val=0.0)
	problem.set_solution_finite_element()
	problem.set_solution_function_space()
	problem.set_solution_functions()
	problem.set_quadrature_degree(2)
	kin = dolfin_mech.kinematics.Kinematics(U=disp.subfunc, U_old=disp.func_old)

	for duration in (settings.ramp_duration, settings.hold_duration):
		problem.add_step(Deltat=duration, dt_ini=settings.dt, dt_min=settings.dt, dt_max=settings.dt)

	gel = fenicsoft.materials.iso.neo_hooke.Constitutive(
		kinematics=kin, parameters=fenicsoft.materials.iso.neo_hooke.Parameters.from_shear_modulus(mu=m.mu)
	) + fenicsoft.materials.volumetric.QuadraticVolumetricPenalty(
		kinematics=kin, parameters=fenicsoft.materials.volumetric.VolumetricPenaltyParameters(kappa=m.kappa)
	)
	hill = hill_law.Constitutive(
		kin,
		hill_law.Parameters(
			phi_c=dolfin.Constant(m.phi_c),
			f0=dolfin.Constant((1.0, 0.0, 0.0)),
			activation_configs=[
				ActivePowerConfig.ramp(k_step=0, ini=0.0, fin=1.0),
				ActivePowerConfig.constant(k_step=1, value=1.0),
			],
			f_max=m.f_max,
			simulation_steps=problem.steps,
			use_force_length=True,
			lmbda_opt=m.lmbda_opt,
			W=m.width,
			eps_a=eps.subfunc,
			eps_a_old=eps.func_old,
			k_e=m.k_e,
			k_p=m.k_p,
			hill_a=m.hill_a,
			hill_b=m.hill_b,
			f_ecc=m.f_ecc,
			eta_0=m.eta_0,
		),
	)
	material = gel + hill
	contractile = ContractileEvolution(law=hill, test=eps.dsubtest, dx=problem.dV)

	for k_step in range(len(problem.steps)):
		problem.add_operator(
			fenicsoft.operators.hyper_elasticity.pk1.HyperElasticity(
				disp=disp.subfunc, disp_test=disp.dsubtest, material=material, measure=problem.dV, k_step=k_step
			),
			k_step=k_step,
		)
		problem.add_operator(contractile, k_step=k_step)
		if settings.kinematics == "isometric":
			for comp, axis_side in ((1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)):
				problem.add_constraint(V=disp.fs.sub(comp), sub_domain=_face(comp, axis_side), val=0.0, k_step=k_step)
			problem.add_constraint(V=disp.fs.sub(0), sub_domain=_face(0, 0.0), val=0.0, k_step=k_step)
			problem.add_constraint(V=disp.fs.sub(0), sub_domain=_face(0, 1.0), val=settings.lmbda - 1.0, k_step=k_step)
		else:
			for comp in range(3):
				problem.add_constraint(V=disp.fs.sub(comp), sub_domain=_face(comp, 0.0), val=0.0, k_step=k_step)

	volume = dolfin.assemble(dolfin.Constant(1.0) * problem.dV)
	pk1 = material.pk1
	averages = {
		"eps_a": eps.subfunc,
		"lambda": kin.F[0, 0],
		"lambda_t": kin.F[1, 1],
		"pk1_xx": pk1[0, 0],
		"pk1_yy": pk1[1, 1],
		"tension_see": hill.tension_see,
	}
	problem.add_qoi(name="alpha", expr=lambda _: hill.current_activation, point=[0.5, 0.5, 0.5], update_type="direct")
	for name, expr in averages.items():
		problem.add_qoi(name=name, expr=expr * problem.dV, norm=volume, update_type="assembly")

	for constraint in problem.steps[0].constraints:
		constraint.bc.apply(problem.sol_func.vector())

	out_dir.mkdir(parents=True, exist_ok=True)
	base = str(out_dir / "one_element")
	integrator = dolfin_mech.core.TimeIntegrator(
		problem=problem,
		solver=dolfin_mech.core.NonlinearSolver(
			problem=problem, parameters={"sol_tol": [1e-10, 1e-10], "n_iter_max": 20}
		),
		parameters={"n_iter_for_accel": 0, "n_iter_for_decel": 100},
		print_out=base,
		print_sta=base,
		write_qois=base + "-qois",
		write_sol=False,
	)
	if not integrator.integrate():
		raise RuntimeError("One-element Hill run failed.")
	integrator.close()
	table = pd.read_csv(base + "-qois.dat", sep=r"\s+", skiprows=1, names=["t", *QOI_NAMES])
	return table[table["t"] > 0.0].reset_index(drop=True)


def stress_check(model: HillChain0D, table: pd.DataFrame) -> np.ndarray:
	r"""0D :math:`P_{xx}` evaluated on the FE state, row by row (Pa).

	Args:
	    model: Parameters.
	    table: Output of :func:`run`.

	Returns:
	    :math:`P_{xx}(\lambda, \lambda_t, \varepsilon_a)`.
	"""
	rows = zip(table["lambda"], table["lambda_t"], table["eps_a"])
	return np.array([model.total_pk1_xx(lmbda, lmbda_t, eps_a) for lmbda, lmbda_t, eps_a in rows])
