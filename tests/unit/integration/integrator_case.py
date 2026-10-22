r"""Tiny transient diffusion :math:`\dot c - \Delta c = 0`, no-flux, :math:`c_0 = 1 + \cos\pi x`.

One CG1 subsol and one FOI.

:math:`c \to 1`: the relative change per step decays as :math:`e^{-\pi^2 t}`; with ``dt = 1/40`` and the threshold
``1e-3`` the plateau exit triggers around :math:`t \approx 0.6`.

Run as a script (also under ``mpirun``): ``python integrator_case.py <out_stem> <root_only_logs 0|1>``.
"""

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import dolfin
import dolfin_mech
import fenicsoft

from hill_playground.integration import DecimatedTimeIntegrator


def run(
	stem: str,
	functions: str = "default",
	root_only_logs: bool = True,
	n_steps: int = 40,
	min_simu_seconds: Optional[float] = None,
) -> Any:
	"""Build and integrate; ``functions``: ``"default"`` (base list) or ``"foi"`` (FOI only).

	Args:
	    stem: Output stem (``.out``, ``.sta``, ``-qois.dat``, ``.xdmf``).
	    functions: Written-field selection.
	    root_only_logs: Rank-0 logs.
	    n_steps: Number of time steps.
	    min_simu_seconds: Enables the plateau exit after this time.

	Returns:
	    ``(success, integrator)``.
	"""
	problem = dolfin_mech.problems.Problem()
	mesh = dolfin.UnitSquareMesh(8, 8)
	problem.set_mesh(mesh=mesh)
	problem.set_measures()
	c0 = dolfin.interpolate(dolfin.Expression("1.0 + cos(pi*x[0])", degree=2), dolfin.FunctionSpace(mesh, "CG", 1))
	c = problem.add_scalar_subsol(name="c", family="CG", degree=1, init_fun=c0)
	problem.set_solution_finite_element()
	problem.set_solution_function_space()
	problem.set_solution_functions()
	problem.set_foi_finite_elements_DG(0)
	problem.set_foi_function_spaces()
	dt = 1.0 / n_steps
	problem.add_step(Deltat=1.0, dt_ini=dt, dt_min=dt, dt_max=dt)
	problem.steps[0].min_simu_seconds = min_simu_seconds
	for operator in (
		fenicsoft.operators.ChangeRate(function=c.subfunc, function_old=c.func_old, test=c.dsubtest, dx=problem.dV),
		fenicsoft.operators.DarcyGradient(
			function=c.subfunc, test=c.dsubtest, conductivity=dolfin.Identity(2), dx=problem.dV
		),
	):
		problem.add_operator(operator, k_step=0)
	double = problem.add_foi(name="double_c", expr=2.0 * c.subfunc, fs=problem.sfoi_fs, update_type="project")
	problem.add_qoi(name="mean_c", expr=c.subfunc * problem.dV)
	written: Optional[Sequence[Any]] = [double.func] if functions == "foi" else None
	integrator = DecimatedTimeIntegrator(
		problem=problem,
		solver=dolfin_mech.core.NonlinearSolver(problem=problem, parameters={"sol_tol": [1e-8]}),
		parameters={},
		print_out=stem,
		print_sta=stem,
		write_qois=stem + "-qois",
		write_sol=stem,
		write_period=1,
		functions_to_write=written,
		root_only_logs=root_only_logs,
		stagnation_plateau_relative_error=1e-3,
	)
	success = integrator.integrate()
	integrator.close()
	return success, integrator


if __name__ == "__main__":
	Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
	sys.exit(0 if run(sys.argv[1], root_only_logs=bool(int(sys.argv[2])))[0] else 1)
