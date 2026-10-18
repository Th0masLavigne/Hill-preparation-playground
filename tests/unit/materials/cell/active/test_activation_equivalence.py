"""Equivalence with fenicsoft: ``ActivationSchedule`` gives the same alpha as the iso and aniso laws.

Random config sets (constant, ramp, function of t; missing, duplicated and out-of-range ``k_step``)
and random ``set_k_step``/``update_time`` sequences. Guards the transfer (TRANSFER.md, issues 4 and 14).
"""

import random
from types import SimpleNamespace
from typing import Any, Callable

import dolfin
from dolfin_mech.kinematics import Kinematics
from fenicsoft.materials.cell.active.aniso import law as aniso
from fenicsoft.materials.cell.active.iso import law as iso
from fenicsoft.materials.cell.active.iso.law import ActivePowerConfig as FenicsoftConfig

from hill_playground.materials.cell.active.activation import ActivationSchedule, ActivePowerConfig


def _configs(cls: Any, specs: list[tuple[str, int, float, float]]) -> list[Any]:
	def func(a: float, b: float) -> Callable[[float], float]:
		return lambda t: a + b * t

	builders = {
		"constant": lambda k, a, b: cls.constant(k_step=k, value=a),
		"ramp": lambda k, a, b: cls.ramp(k_step=k, ini=a, fin=b),
		"function": lambda k, a, b: cls.function(k_step=k, func=func(a, b)),
	}
	return [builders[kind](k, a, b) for kind, k, a, b in specs]


def _alpha(obj: Any, t_step: float) -> Any:
	try:
		obj.update_time(t_step)
	except IndexError:  # function config on a step that does not exist: same failure everywhere
		return "IndexError"
	return float(obj.p_cell if hasattr(obj, "p_cell") else obj.p_act)


def test_same_alpha_as_iso_and_aniso() -> None:
	"""100 random cases x 20 calls: identical alpha (or identical failure) for the three implementations."""
	rng = random.Random(0)
	mesh = dolfin.UnitCubeMesh(1, 1, 1)
	kin = Kinematics(U=dolfin.Function(dolfin.VectorFunctionSpace(mesh, "CG", 1)))
	for _ in range(100):
		n_steps, steps, t = rng.randint(1, 4), [], 0.0
		for _ in range(n_steps):
			duration = rng.uniform(0.5, 3.0)
			steps.append(SimpleNamespace(t_ini=t, t_fin=t + duration))
			t += duration
		specs = [
			(
				rng.choice(["constant", "ramp", "function"]),
				rng.randint(0, n_steps + 1),
				rng.uniform(-1, 2),
				rng.uniform(-1, 2),
			)
			for _ in range(rng.randint(0, 5))
		]
		laws = [
			aniso.Constitutive(
				kin,
				aniso.Parameters(
					phi_c=dolfin.Constant(0.1),
					f0=dolfin.Constant((1.0, 0.0, 0.0)),
					activation_configs=_configs(FenicsoftConfig, specs),
					f_max=1.0,
					simulation_steps=steps,
				),
			),
			iso.Constitutive(
				kin,
				iso.Parameters(
					phi_c=dolfin.Constant(0.1), power_configs=_configs(FenicsoftConfig, specs), simulation_steps=steps
				),
			),
			ActivationSchedule(_configs(ActivePowerConfig, specs), steps),
		]
		for _ in range(20):
			if rng.random() < 0.3:
				k_step = rng.randint(0, n_steps + 1)
				for obj in laws:
					obj.set_k_step(k_step)
			t_step = rng.random()
			values = [_alpha(obj, t_step) for obj in laws]
			assert values[0] == values[1] == values[2], (specs, values)
