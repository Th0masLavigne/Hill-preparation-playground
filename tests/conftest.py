"""Shared fixtures: a one-cell Hill-chain problem (unit cube, f0 = e_x, alpha held at 1)."""

from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import MagicMock

import dolfin
import pytest
from dolfin_mech.kinematics import Kinematics

from hill_playground.materials.cell.active.activation import ActivePowerConfig
from hill_playground.materials.cell.active.hill import law as hill_law

K_E, K_P, F_MAX, PHI_C = 200e3, 5e3, 20e3, 0.05


@dataclass
class HillSetup:
	"""Objects of a one-cell Hill-chain problem.

	Attributes:
	    mesh: Unit cube (6 tetrahedra).
	    u: CG1 displacement.
	    dg0: DG0 space of eps_a.
	    eps_a: Current eps_a.
	    eps_a_old: Previous eps_a.
	    law: Hill law, alpha = 1.
	"""

	mesh: Any
	u: Any
	dg0: Any
	eps_a: Any
	eps_a_old: Any
	law: hill_law.Constitutive

	def mean(self, expr: Any) -> float:
		"""Volume average of a UFL scalar.

		Args:
		    expr: UFL scalar.

		Returns:
		    Its average over the cube.
		"""
		return dolfin.assemble(expr * dolfin.dx(domain=self.mesh))


def make_setup(**overrides: Any) -> HillSetup:
	"""Build a :class:`HillSetup`; ``overrides`` replace ``hill_law.Parameters`` fields.

	Args:
	    **overrides: Parameter overrides.

	Returns:
	    The setup, with ``update_time`` already called (alpha = 1).
	"""
	mesh = dolfin.UnitCubeMesh(1, 1, 1)
	u = dolfin.Function(dolfin.VectorFunctionSpace(mesh, "CG", 1))
	dg0 = dolfin.FunctionSpace(mesh, "DG", 0)
	eps_a, eps_a_old = dolfin.Function(dg0), dolfin.Function(dg0)
	step = MagicMock(t_ini=0.0, t_fin=1.0)
	kwargs: dict[str, Any] = dict(
		phi_c=dolfin.Constant(PHI_C),
		f0=dolfin.Constant((1.0, 0.0, 0.0)),
		activation_configs=[ActivePowerConfig.constant(k_step=0, value=1.0)],
		f_max=F_MAX,
		simulation_steps=[step],
		eps_a=eps_a,
		eps_a_old=eps_a_old,
		k_e=K_E,
		k_p=K_P,
	)
	kwargs.update(overrides)
	law = hill_law.Constitutive(Kinematics(U=u), hill_law.Parameters(**kwargs))
	law.update_time(0.5)
	return HillSetup(mesh, u, dg0, eps_a, eps_a_old, law)


@pytest.fixture
def hill_setup() -> HillSetup:
	"""Default one-cell setup (nominal parameters, alpha = 1).

	Returns:
	    The setup.
	"""
	return make_setup()


@pytest.fixture
def make_hill_setup() -> Callable[..., HillSetup]:
	"""Factory of one-cell setups with parameter overrides.

	Returns:
	    :func:`make_setup`.
	"""
	return make_setup
