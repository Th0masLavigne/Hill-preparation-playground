"""Unit tests of the R_a operator.

R_a = T_CE + T_p - T_e, T_CE = f_max alpha f_FL(lambda_a) h(D_a) + eta_0 D_a, on one cell, u = 0 (lambda = 1),
f0 = e_x, alpha = 1, nominal parameters. Reference states (scalar solves of R_a = 0, D_a = 0):
isometric cell (lambda_a = 0.92218, T_e = 16.27 kPa) and free cell without SEE (lambda_a = 0.62564).
"""

import math
from typing import Any

import dolfin
import pytest

from hill_playground.operators.cell.hill import ContractileEvolution

F_MAX = 20e3
LAMBDA_A_ISO, T_E_ISO = 0.9221836, 16273.16  # isometric: f_max f_FL + T_p = T_e at lambda = 1
LAMBDA_A_FREE = 0.6256378  # free cell (T_e = 0): f_max f_FL + T_p = 0


def _operator(hill_setup: Any, dt: float) -> ContractileEvolution:
	"""R_a on the cube, with dt set.

	Args:
	    hill_setup: One-cell setup.
	    dt: Time step (s).

	Returns:
	    The operator.
	"""
	op = ContractileEvolution(
		law=hill_setup.law, test=dolfin.TestFunction(hill_setup.dg0), dx=dolfin.dx(domain=hill_setup.mesh)
	)
	op.set_dt(dt)
	return op


def _solve(hill_setup: Any, op: ContractileEvolution) -> float:
	"""Newton solve of R_a = 0 for eps_a (u fixed).

	Args:
	    hill_setup: One-cell setup.
	    op: The operator.

	Returns:
	    lambda_a = exp(eps_a).
	"""
	jac = dolfin.derivative(op.res_form, hill_setup.eps_a, dolfin.TrialFunction(hill_setup.dg0))
	dolfin.solve(
		op.res_form == 0, hill_setup.eps_a, J=jac, solver_parameters={"newton_solver": {"relative_tolerance": 1e-10}}
	)
	return math.exp(hill_setup.eps_a.vector().get_local()[0])


def test_isometric_equilibrium_zeroes_residual(hill_setup: Any) -> None:
	"""D_a = 0 at lambda_a = 0.92218: R_a = 0 and T_e = 16.27 kPa."""
	hill_setup.eps_a.vector()[:] = math.log(LAMBDA_A_ISO)
	hill_setup.eps_a_old.vector()[:] = math.log(LAMBDA_A_ISO)
	residual = dolfin.assemble(_operator(hill_setup, 1.0).res_form).get_local()
	assert abs(residual[0]) < 1e-6 * F_MAX
	assert math.isclose(hill_setup.mean(hill_setup.law.tension_see), T_E_ISO, rel_tol=1e-5)


def test_newton_finds_isometric_state(hill_setup: Any) -> None:
	"""From rest, dt >> tau_a: one implicit step lands on the isometric state."""
	assert math.isclose(_solve(hill_setup, _operator(hill_setup, 1e6)), LAMBDA_A_ISO, rel_tol=1e-5)


def test_free_cell(make_hill_setup: Any) -> None:
	"""No SEE (k_e = 0): T_0 + T_p = 0 gives lambda_a = 0.6256."""
	setup = make_hill_setup(k_e=0.0)
	assert math.isclose(_solve(setup, _operator(setup, 1e6)), LAMBDA_A_FREE, rel_tol=1e-5)


def test_passive_cell_is_series_springs(make_hill_setup: Any) -> None:
	"""Alpha = 0 (no motor): T_e = T_p, i.e. lambda_a = 1 at lambda = 1 (rest state), Newton from lambda_a = 0.9."""
	setup = make_hill_setup()
	setup.law.p_act.assign(0.0)
	setup.eps_a.vector()[:] = math.log(0.9)
	assert math.isclose(_solve(setup, _operator(setup, 1.0)), 1.0, abs_tol=1e-10)


def test_flat_force_length(make_hill_setup: Any) -> None:
	"""use_force_length=False: T_0 = f_max alpha (flat force--length), whatever lambda_a."""
	setup = make_hill_setup(use_force_length=False)
	setup.eps_a.vector()[:] = math.log(0.7)
	op = _operator(setup, 1.0)
	assert math.isclose(setup.mean(op.isometric_tension), F_MAX, rel_tol=1e-12)


@pytest.mark.parametrize("dt", [1e-3, 1.0, 1e3])
def test_jacobian_diagonal_positive(hill_setup: Any, dt: float) -> None:
	"""dR_a/deps_a > 0 at rest, for any dt (unique implicit update)."""
	op = _operator(hill_setup, dt)
	jac = dolfin.assemble(dolfin.derivative(op.res_form, hill_setup.eps_a, dolfin.TrialFunction(hill_setup.dg0)))
	assert jac.array()[0, 0] > 0.0


def test_scale_factor_multiplies_residual(hill_setup: Any) -> None:
	"""scale_factor = 1/mu0 divides the assembled residual by mu0."""
	hill_setup.eps_a.vector()[:] = math.log(0.95)
	ref = dolfin.assemble(_operator(hill_setup, 1.0).res_form).get_local()[0]
	op = ContractileEvolution(
		law=hill_setup.law,
		test=dolfin.TestFunction(hill_setup.dg0),
		dx=dolfin.dx(domain=hill_setup.mesh),
		scale_factor=1.0 / 1700.0,
	)
	op.set_dt(1.0)
	assert math.isclose(dolfin.assemble(op.res_form).get_local()[0], ref / 1700.0, rel_tol=1e-12)


def test_ce_tension_is_motor_plus_damper(make_hill_setup: Any) -> None:
	"""T_CE = T_0 h(D_a) + eta_0 D_a, with D_a = ln(0.95)/dt."""
	eta_0, dt = 500.0, 2.0
	setup = make_hill_setup(eta_0=eta_0)
	setup.eps_a.vector()[:] = math.log(0.95)
	op = _operator(setup, dt)
	expected = setup.mean(op.tension_motor) + eta_0 * math.log(0.95) / dt
	assert math.isclose(setup.mean(op.tension_ce), expected, rel_tol=1e-12)


def test_ce_tension_balances_springs_at_convergence(make_hill_setup: Any) -> None:
	"""R_a = 0: T_CE = T_e - T_p (series equilibrium), damper active."""
	setup = make_hill_setup(eta_0=500.0)
	op = _operator(setup, 1.0)
	_solve(setup, op)
	balance = setup.mean(setup.law.tension_see) - setup.mean(setup.law.tension_pee)
	assert math.isclose(setup.mean(op.tension_ce), balance, rel_tol=1e-8)
