r"""One-element FE runs of the gel + Hill chain (no fluid), for comparison with :mod:`hill_0d_reference`.

Unit cube ``UnitCubeMesh(1, 1, 1)`` (6 tetrahedra, every node on the boundary), CG1 :math:`\mathbf u`,
DG0 :math:`\varepsilon_a`, :math:`\mathbf f_0 = \mathbf e_x`, uniform :math:`\Phi_c`:

.. math::
	R_u = \int (\mathbf P_S + \mathbf P_h) : \nabla\delta\mathbf u \,\mathrm dV, \qquad
	R_a = \int \big[f_{max}\alpha f_{FL} h(D_a) + \eta_0 D_a + T_p - T_e\big]\delta\varepsilon \,\mathrm dV

Two kinematic set-ups, both homogeneous:

* ``"isometric"``: :math:`u_x = (\lambda-1)x` on the :math:`x` faces, :math:`u_y = 0` on the :math:`y` faces,
  :math:`u_z = 0` on the :math:`z` faces; every node lies on these faces, so :math:`\mathbf F` is imposed;
* ``"free"``: symmetry planes :math:`u_x(0,\cdot)=u_y(\cdot,0,\cdot)=u_z(\cdot,\cdot,0)=0`, other faces traction-free.

.. image:: /_static/resources/hill_one_element_bcs.svg
	:alt: Unit cube, isometric (u imposed on every face) and free (symmetry planes x, y, z = 0)
	:width: 90%
	:align: center
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import dolfin
import dolfin_mech
import fenicsoft
import numpy as np
import pandas as pd
from fenicsoft.materials.cell.active.activation import ActivePowerConfig
from hill_0d_reference import HillChain0D
from numpy.typing import NDArray

from hill_playground.materials.cell.active.hill import law as hill_law
from hill_playground.operators import ResidualRows
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
	    quadrature_degrees: Degree per row (``"displacement"``, ``"eps_a"``), default 2 (``ResidualRows``).
	    normalizations: Row normalization :math:`\mathcal N_{row}` (``ResidualRows``); does not change the solution.
	"""

	model: HillChain0D = HillChain0D()
	kinematics: Literal["isometric", "free"] = "isometric"
	lmbda: float = 1.0
	ramp_duration: float = 20.0
	hold_duration: float = 200.0
	dt: float = 2.0
	quadrature_degrees: Mapping[str, int] = field(default_factory=dict)
	normalizations: Mapping[str, float] = field(default_factory=dict)


def _face(axis: int, value: float) -> dolfin.SubDomain:
	return dolfin.CompiledSubDomain(f"on_boundary && near(x[{axis}], {value})")


def run(settings: OneElementRun, out_dir: Path) -> pd.DataFrame:
	"""Solve and return the QOI history (one row per time step, column ``t`` first).

	Args:
	    settings: Run settings.
	    out_dir: Output folder.

	Returns:
	    Columns ``t`` + :data:`QOI_NAMES`; ``attrs["wiring"]``: ``(scale_factor, quadrature degree or None)`` of
	    the operator of each row (``"displacement"``, ``"eps_a"``).

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
	# Rows = subsols of the test functions: displacement (R_u) and eps_a (R_a).
	rows = ResidualRows.from_problem(
		problem, quadrature_degrees=settings.quadrature_degrees, normalizations=settings.normalizations
	)
	contractile = ContractileEvolution(
		law=hill,
		test=eps.dsubtest,
		dx=rows.measure(problem.dV, "eps_a"),
		scale_factor=rows.scale_factor("eps_a"),
	)

	elasticity = []
	for k_step in range(len(problem.steps)):
		elasticity.append(
			fenicsoft.operators.hyper_elasticity.pk1.HyperElasticity(
				disp=disp.subfunc,
				disp_test=disp.dsubtest,
				material=material,
				measure=rows.measure(problem.dV, "displacement"),
				k_step=k_step,
				scale_factor=rows.scale_factor("displacement"),
			)
		)
		problem.add_operator(elasticity[-1], k_step=k_step)
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
		"lambda": kin.F[0, 0],  # type: ignore  # ufl_legacy: __getitem__ added at runtime
		"lambda_t": kin.F[1, 1],  # type: ignore
		"pk1_xx": pk1[0, 0],
		"pk1_yy": pk1[1, 1],
		"tension_see": hill.tension_see,
	}

	def activation(_: Any) -> float:
		return hill.current_activation

	problem.add_qoi(name="alpha", expr=activation, point=[0.5, 0.5, 0.5], update_type="direct")
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
		print_out=base,  # type: ignore  # dolfin_mech: str = file basename (untyped, default True)
		print_sta=base,  # type: ignore
		write_qois=base + "-qois",  # type: ignore
		write_sol=False,
	)
	if not integrator.integrate():
		raise RuntimeError("One-element Hill run failed.")
	integrator.close()
	table = pd.read_csv(base + "-qois.dat", sep=r"\s+", skiprows=1, names=["t", *QOI_NAMES])
	table = table.query("t > 0.0").reset_index(drop=True)
	table.attrs["wiring"] = {
		row: (op.scale_factor, op.measure.metadata().get("quadrature_degree"))
		for row, op in (("displacement", elasticity[0]), ("eps_a", contractile))
	}
	return table


def stress_check(model: HillChain0D, table: pd.DataFrame) -> NDArray[np.float64]:
	r"""0D :math:`P_{xx}` evaluated on the FE state, row by row (Pa).

	Args:
	    model: Parameters.
	    table: Output of :func:`run`.

	Returns:
	    :math:`P_{xx}(\lambda, \lambda_t, \varepsilon_a)`.
	"""
	rows = zip(table["lambda"], table["lambda_t"], table["eps_a"])
	return np.array([model.total_pk1_xx(lmbda, lmbda_t, eps_a) for lmbda, lmbda_t, eps_a in rows])


def reference(model: HillChain0D, settings: OneElementRun, table: pd.DataFrame) -> pd.DataFrame:
	r"""0D reference on the FE times: :math:`\lambda, \lambda_t, \varepsilon_a, T_e`.

	Args:
	    model: Parameters.
	    settings: Run settings (``kinematics``, ``lmbda``).
	    table: Output of :func:`run` (``t``, ``alpha``).

	Returns:
	    Columns ``t``, ``lambda``, ``lambda_t``, ``eps_a``, ``tension_see``.
	"""
	import hill_0d_reference as ref

	if settings.kinematics == "isometric":
		eps = ref.isometric_history(model, settings.lmbda, table["t"], table["alpha"])
		state = np.column_stack([np.full_like(eps, settings.lmbda), np.ones_like(eps), eps])
	else:
		state = ref.free_history(model, table["t"], table["alpha"])
	out = pd.DataFrame(state, columns=["lambda", "lambda_t", "eps_a"])
	out.insert(0, "t", table["t"].to_numpy())
	out["tension_see"] = [model.tension_see(lm**2 * np.exp(-2.0 * e)) for lm, e in zip(out["lambda"], out["eps_a"])]
	return out


def errors(model: HillChain0D, settings: OneElementRun, table: pd.DataFrame) -> dict[str, float]:
	r"""Max differences FE vs 0D over the history.

	Args:
	    model: Parameters.
	    settings: Run settings.
	    table: Output of :func:`run`.

	Returns:
	    ``eps_a`` (abs), ``stretch`` (abs, :math:`\lambda` and :math:`\lambda_t`), ``tension_see`` and
	    ``pk1_xx`` (relative to :math:`\max|T_e|`).
	"""
	ref_table = reference(model, settings, table)
	t_scale = np.abs(ref_table["tension_see"]).max()
	return {
		"eps_a": float(np.abs(table["eps_a"] - ref_table["eps_a"]).max()),
		"stretch": float(
			np.abs(table[["lambda", "lambda_t"]].to_numpy() - ref_table[["lambda", "lambda_t"]].to_numpy()).max()
		),
		"tension_see": float(np.abs(table["tension_see"] - ref_table["tension_see"]).max() / t_scale),
		"pk1_xx": float(np.abs(table["pk1_xx"] - stress_check(model, table)).max() / t_scale),
	}


def error_history(model: HillChain0D, settings: OneElementRun, table: pd.DataFrame) -> NDArray[np.float64]:
	r"""Per-step max over the checked fields of the FE vs 0D difference (same norms as :func:`errors`).

	Args:
	    model: Parameters.
	    settings: Run settings.
	    table: Output of :func:`run`.

	Returns:
	    :math:`\max(|\Delta\varepsilon_a|, |\Delta\lambda|, |\Delta\lambda_t|, |\Delta T_e|/T, |\Delta P_{xx}|/T)`
	    per step, :math:`T = \max|T_e|`.
	"""
	ref_table = reference(model, settings, table)
	t_scale = np.abs(ref_table["tension_see"]).max()
	columns = [np.abs(table[name] - ref_table[name]).to_numpy() for name in ("eps_a", "lambda", "lambda_t")] + [
		np.abs(table["tension_see"] - ref_table["tension_see"]).to_numpy() / t_scale,
		np.abs(table["pk1_xx"] - stress_check(model, table)).to_numpy() / t_scale,
	]
	return np.max(columns, axis=0)


def plot(model: HillChain0D, runs: dict[str, tuple[OneElementRun, pd.DataFrame]], path: Path) -> Path:
	r"""FE (markers) vs 0D (lines): :math:`\lambda_a`, :math:`T_e`, gel stretches, error history.

	Labels containing ``"rows"`` are drawn dashed with ``x`` markers (they coincide with the default case).

	Args:
	    model: Parameters.
	    runs: ``label -> (settings, table)``.
	    path: PNG file.

	Returns:
	    ``path``.
	"""
	import matplotlib

	matplotlib.use("Agg")
	import hill_0d_reference as ref
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D

	floor = 1e-18
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))
	ax_la, ax_te, ax_st, ax_err = axes.ravel()
	for label, (settings, table) in runs.items():
		ref_table = reference(model, settings, table)
		rows_case = "rows" in label
		ls, marker = ("--", "x") if rows_case else ("-", "o")
		fe = dict(ls="none", marker=marker, ms=4, mfc="none")
		every = slice(2, None, 4) if rows_case else slice(0, None, 4)  # shifted: both marker sets visible
		(line,) = ax_la.plot(ref_table["t"], np.exp(ref_table["eps_a"]), ls=ls, lw=1.2, label=label)
		color = line.get_color()
		ax_la.plot(table["t"][every], np.exp(table["eps_a"][every]), color=color, **fe)
		if settings.kinematics == "isometric":
			eq = np.exp(ref.isometric_equilibrium(model, settings.lmbda, 1.0))
			ax_la.axhline(eq, ls=":", lw=0.8, color=color)
		ax_te.plot(ref_table["t"], ref_table["tension_see"] / 1e3, ls=ls, lw=1.2, color=color, label=label)
		ax_te.plot(table["t"][every], table["tension_see"][every] / 1e3, color=color, **fe)
		if settings.kinematics == "free":
			for col in ("lambda", "lambda_t"):
				ax_st.plot(
					ref_table["t"], ref_table[col], ls=ls, lw=1.2, color=color, label=label if col == "lambda" else None
				)
				ax_st.plot(table["t"][every], table[col][every], color=color, **fe)
			if not rows_case:
				t_end = ref_table["t"].iloc[-1]
				for col, name in (("lambda", r"$\lambda$"), ("lambda_t", r"$\lambda_t$")):
					ax_st.annotate(name, (t_end, ref_table[col].iloc[-1]), xytext=(0, 6), textcoords="offset points")
		history = error_history(model, settings, table)
		ax_err.semilogy(
			table["t"],
			np.maximum(history, floor),
			ls=ls,
			lw=1.0,
			color=color,
			label=f"{label}: max {history.max():.1e}",
		)
	ramp = next(iter(runs.values()))[0].ramp_duration
	for ax in axes.ravel():
		ax.axvline(ramp, color="0.7", lw=0.8)
		ax.set_xlabel("t (s)")
	ax_la.set(
		ylabel=r"$\lambda_a = e^{\varepsilon_a}$", title=r"CE stretch $\lambda_a$ (dotted: isometric equilibrium)"
	)
	ax_te.set(ylabel=r"$T_e$ (kPa)", title="SEE tension")
	ax_st.set(ylabel="stretch", title=r"Free faces: gel stretches $\lambda$ (along $f_0$), $\lambda_t$")
	ax_err.set(
		ylabel="max FE - 0D difference",
		title=rf"Error: max $|\Delta\varepsilon_a|, |\Delta\lambda|, |\Delta\lambda_t|, "
		rf"|\Delta T_e|/T, |\Delta P_{{xx}}|/T$ "
		f"(floor {floor:.0e})",
	)
	ax_err.title.set_fontsize(10)
	key = [
		Line2D([], [], color="k", lw=1.2, label="0D reference (line)"),
		Line2D([], [], color="k", ls="none", marker="o", mfc="none", label="FE (markers)"),
		Line2D([], [], color="k", ls="--", marker="x", label="rows case (dashed, x)"),
		Line2D([], [], color="0.7", lw=0.8, label="end of activation ramp"),
	]
	for ax, where in ((ax_la, "upper right"), (ax_te, "lower right"), (ax_st, "center right")):
		handles, _ = ax.get_legend_handles_labels()
		ax.legend(handles=handles + key, fontsize=8, loc=where)
	bottom, top = ax_la.get_ylim()
	ax_la.set_ylim(bottom, top + 0.7 * (top - bottom))  # room for the legend above the curves
	ax_err.legend(fontsize=8)
	fig.tight_layout()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=130)
	plt.close(fig)
	return path
