r"""Free-floating cylinder: fields (FOIs) and quantities (QOIs) of interest, all in SI.

The solve is dimensionless (``hill_cylinder_scaling``); every output is multiplied back by its scale, so the
XDMF fields and the QOI file are in SI.

Fields (XDMF)
-------------
* point data: ``displacement_si``, ``pressure_si`` (scaled copies of the unknowns); with ``cell_transport``, the
  CG subsols ``cell_fraction`` and ``substrate_smoothed``;
* cell data: ``jacobian``, ``fluid_fraction``, ``permeability_ratio``, ``theta``, ``phi`` (DG0 subsols with
  ``remodelling``, else FOIs), ``cell_fraction`` (FOI without ``cell_transport``), ``eps_a`` (subsol),
  ``theta_deg``/``phi_deg`` (canonical, upper hemisphere), ``theta_wrapped_deg`` (in-plane axis, period 180),
  ``tilt_xy_deg`` (0: fibre in the :math:`xy` plane, normal to the axis :math:`z`), ``angle_x_deg``
  (0: along :math:`x`), with ``_current`` variants for
  :math:`\mathbf f = \mathbf F\mathbf f_0/\|\mathbf F\mathbf f_0\|`; stretches ``lambda_*``, tensions ``tension_*``,
  ``fibre``, ``fibre_initial``, ``fibre_current``, ``cell_cauchy``, ``total_cauchy``.

QOIs (``.dat``)
---------------
``time_si``, ``cellular_activation``, probes (``top_center_disp_z``: height change; ``wall_disp_y`` at
:math:`(0, -R, H/2)`: :math:`-\Delta R`; ``center_pressure``), volume averages ``mean_*`` (``mean_jacobian``:
:math:`V/V_0`), order parameters ``order_x``, ``order_z`` (:math:`\langle\frac32 f_i^2-\frac12\rangle`: 1 aligned,
0 isotropic, :math:`-\frac12` orthogonal; :math:`z` is the axis) and ``mean_tilt_xy_deg``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import dolfin
import dolfin_mech
import fenicsoft.materials.orientation.spherical as spherical_mod
from fenicsoft.materials import saturated_mixture
from fenicsoft.materials.cell.active.hill import law as hill_law
from fenicsoft.materials.orientation import display, generic
from fenicsoft.operators.cell import ContractileEvolution
from fenicsoft.scaling import CharacteristicScales

if TYPE_CHECKING:
	from hill_cylinder_free import UserInputs

_E_X = dolfin.Constant((1.0, 0.0, 0.0))
_E_Z = dolfin.Constant((0.0, 0.0, 1.0))


def add_outputs(
	problem: dolfin_mech.problems.Problem,
	user_inputs: UserInputs,
	gel_material: Any,
	hill_material: hill_law.Constitutive,
	contractile: ContractileEvolution,
	scales: CharacteristicScales,
) -> None:
	r"""Register the FOIs and QOIs of the module docstring, all in SI.

	:math:`\mathbf u = L_0\hat{\mathbf u}` and :math:`p = \mu_0\hat p` are scaled copies of the unknowns,
	on their CG spaces (point data, no projection); the other fields are cell-wise
	(``local_solver``) projections on the DG FOI spaces (cell data).

	Args:
	    problem: Problem.
	    user_inputs: Configuration.
	    gel_material: Composite gel law.
	    hill_material: Hill law.
	    contractile: :math:`R_a` operator (provides :math:`T_0` and :math:`T_{CE}`).
	    scales: :math:`L_0, \mu_0, t_0` (``UserInputs.scales``).
	"""
	kin = problem.kinematics
	s = scales
	pressure = s.redimensionalize_stress(problem.pl_subsol.subfunc)
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
	scaled_copy("displacement_si", problem.displacement_subsol.func, s.length)
	scaled_copy("pressure_si", problem.pl_subsol.func, s.stress)

	# 2. Scalars, cell data.
	scalar_fois: Dict[str, Any] = {
		"jacobian": kin.J,
		"cell_fraction": problem.phic_func,
		"fluid_fraction": saturated_mixture.eulerian_porosity(kin.J, problem.phi_s0, problem.phic_func),
		"permeability_ratio": problem.permeability.ratio,
		"theta": problem.theta_func,
		"phi": problem.phi_func,
		"theta_deg": theta_deg,
		"phi_deg": phi_deg,
		"theta_wrapped_deg": spherical_mod.wrapped_theta_degrees(problem.theta_func),
		"tilt_xy_deg": display.tilt_degrees(f0, _E_Z),
		"tilt_xy_current_deg": display.tilt_degrees(fibre, _E_Z),
		"angle_x_deg": generic.fiber_reorientation_angle(f0, _E_X, in_degrees=True),
		"angle_x_current_deg": generic.fiber_reorientation_angle(fibre, _E_X, in_degrees=True),
		"lambda_fibre": dolfin.sqrt(hill_material.inv4),
		"lambda_a": hill_material.lmbda_a,
		"lambda_e": dolfin.sqrt(hill_material.inv4e),
		"tension_see": s.redimensionalize_stress(hill_material.tension_see),
		"tension_pee": s.redimensionalize_stress(hill_material.tension_pee),
		"tension_ce": s.redimensionalize_stress(contractile.tension_ce),
		"tension_isometric": s.redimensionalize_stress(contractile.isometric_tension),
	}
	subsol_names = {subsol.name for subsol in problem.subsols}
	for name, expr in scalar_fois.items():
		if name not in subsol_names:  # theta, phi, cell_fraction: the subsol itself is written
			foi(name, expr, problem.sfoi_fs)

	# 3. Vectors and tensors, cell data.
	foi("fibre", f0, problem.vfoi_fs)
	foi("fibre_initial", f0_initial, problem.vfoi_fs)
	foi("fibre_current", fibre, problem.vfoi_fs)
	foi("cell_cauchy", s.redimensionalize_stress(hill_material.cauchy), problem.mfoi_fs)
	foi("total_cauchy", s.redimensionalize_stress(gel_material.cauchy) - pressure * kin.I, problem.mfoi_fs)

	# QOIs. Point probes in mesh coordinates.
	probes = {k: [s.nondimensionalize_length(c) for c in v] for k, v in user_inputs.probes.items()}
	# No set_allow_extrapolation: under MPI every rank would then return a value and the QOI averages them.

	def _time_si(_: Any) -> float:
		return s.redimensionalize_time(hill_material.current_time)

	def _activation(_: Any) -> float:
		return hill_material.current_activation

	problem.add_qoi(name="time_si", expr=_time_si, point=probes["center"], update_type="direct")
	problem.add_qoi(name="cellular_activation", expr=_activation, point=probes["center"], update_type="direct")
	problem.add_qoi(
		name="top_center_disp_z",
		expr=s.redimensionalize_length(problem.displacement_subsol.func[2]),
		point=probes["top"],
		update_type="direct",
	)
	problem.add_qoi(
		name="wall_disp_y",
		expr=s.redimensionalize_length(problem.displacement_subsol.func[1]),
		point=probes["wall"],
		update_type="direct",
	)
	problem.add_qoi(
		name="center_pressure",
		expr=s.redimensionalize_stress(problem.pl_subsol.func),
		point=probes["center"],
		update_type="direct",
	)

	# Volume averages: dolfin_mech assembles expr and divides by norm (mesh volume, scale-free ratio).
	volume = dolfin.assemble(dolfin.Constant(1.0) * problem.dV)
	averages: Dict[str, Any] = {
		"mean_lambda_a": hill_material.lmbda_a,
		"mean_tension_see": s.redimensionalize_stress(hill_material.tension_see),
		"mean_tension_isometric": s.redimensionalize_stress(contractile.isometric_tension),
		"mean_jacobian": kin.J,
		"mean_permeability_ratio": problem.permeability.ratio,
		"order_x": display.order_parameter(fibre, _E_X),
		"order_z": display.order_parameter(fibre, _E_Z),
		"mean_tilt_xy_deg": display.tilt_degrees(fibre, _E_Z),
	}
	for name, expr in averages.items():
		problem.add_qoi(name=name, expr=expr * problem.dV, norm=volume, update_type="assembly")
