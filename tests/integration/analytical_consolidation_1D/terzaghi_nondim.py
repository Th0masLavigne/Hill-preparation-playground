r"""Terzaghi consolidation solved three ways: SI, operator ``scale_factor``, dimensionless parameters.

Poro-elastic gel: isochoric Neo-Hooke :math:`\frac{\mu}{2}(\bar I_1 - 3)` + :math:`\frac{\kappa}{2}(J-1)^2`
(hence :math:`M = \kappa + \frac43\mu` below),
biphasic fluid stress :math:`-Jp\mathbf F^{-T}`, Darcy with :math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`.
Column :math:`[0, L] \times [0, h]^2` (3D, one cell across), lateral rollers, :math:`u_x = 0` at
:math:`x = L` (impermeable), load :math:`p_0` and :math:`p = 0` at :math:`x = 0`, :math:`p(t=0) = p_0`.

.. image:: /_static/resources/terzaghi_column.svg
	:alt: Gel column loaded by p0 at x = 0 (drained), clamped and impermeable at x = L, lateral rollers
	:width: 90%
	:align: center

Small-strain analytical solution (:math:`M = \kappa + \frac43\mu`, :math:`c_v = (k/\eta) M`,
:math:`T = c_v t / L^2`, :math:`m_j = (2j-1)\pi/2`):

.. math::
	p(x, t) = \sum_j \frac{2 p_0}{m_j} \sin\Big(m_j \frac{x}{L}\Big)\, e^{-m_j^2 T}, \qquad
	u_x(0, t) = \frac{p_0 L}{M}\Big[1 - \sum_j \frac{2}{m_j^2} e^{-m_j^2 T}\Big]

Modes (scales :math:`L_0 = L`, :math:`\mu_0 = \mu`, :math:`t_0 = L_0^2\eta/(k\kappa)`):

* ``"none"``: mesh :math:`X`, unknowns :math:`(u, p)`, SI coefficients, row factors 1;
* ``"operator"``: mesh :math:`\hat X`, unknowns :math:`(\hat u, \hat p)`, SI coefficients and time,
  ``scale_factor`` :math:`= s_k v/\mathcal N_k` with the test scales :math:`s_{\mathbf u} = L_0`,
  :math:`s_p = \mu_0`;
* ``"parameters"``: mesh :math:`\hat X`, unknowns :math:`(\hat u, \hat p)`, coefficients
  :math:`(\hat\mu, \hat\kappa, \hat K, \hat p_0)` and time :math:`\hat t`, row factors 1.

Each operator takes ``ResidualRows.measure``/``scale_factor`` of its row (subsol of its test function:
``displacement`` or ``liquid_pressure``); ``quadrature_degrees`` sets a degree per row (``"<mode>+rows"``).
Coefficients of ``"parameters"`` come from ``CharacteristicScales.nondimensionalize_*``; outputs are brought
back with ``redimensionalize_*``. Row scaling leaves Newton's iterates unchanged: the three modes agree to
round-off. Run as a script to get the figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import dolfin
import dolfin_mech
import fenicsoft
import numpy as np
import pandas as pd
from fenicsoft.scaling import CharacteristicScales
from numpy.typing import ArrayLike, NDArray

from hill_playground.operators import ResidualRows

Mode = Literal["none", "operator", "parameters"]
MODES: tuple[Mode, ...] = ("none", "operator", "parameters")
_LOAD = 1


@dataclass(frozen=True)
class TerzaghiInputs:
	r"""SI inputs.

	Attributes:
	    length: :math:`L` (m), drainage length.
	    n_cells: Elements along :math:`x`.
	    shear_modulus: :math:`\mu` (Pa).
	    kappa: :math:`\kappa` (Pa).
	    hydraulic_conductivity: :math:`k/\eta` (m^2/(Pa.s)).
	    viscosity: :math:`\eta` (Pa.s), only for :math:`t_0`.
	    load: :math:`p_0` (Pa), small against :math:`\mu` (linear regime).
	    t_end_factor: Final time in units of :math:`L^2/c_v`.
	    n_steps: Number of (fixed) time steps.
	    n_probes: Pressure probes along :math:`x`.
	    sol_tol: Newton relative tolerance.
	    probe_x: Probe abscissae :math:`x_i = i L/(n-1)` (m), set from ``length`` and ``n_probes``.
	"""

	length: float = 50e-6
	n_cells: int = 40
	shear_modulus: float = 1700.0
	kappa: float = 1417.0
	hydraulic_conductivity: float = 1.8e-15 / 1e-3
	viscosity: float = 1e-3
	load: float = 1.0
	t_end_factor: float = 1.0
	n_steps: int = 200
	n_probes: int = 21
	sol_tol: float = 1e-10
	probe_x: tuple[float, ...] = field(init=False)

	def __post_init__(self) -> None:
		"""Probe abscissae :math:`x_i = i L/(n-1)`."""
		object.__setattr__(self, "probe_x", tuple(np.linspace(0.0, self.length, self.n_probes)))

	@property
	def oedometric_modulus(self) -> float:
		r""":math:`M = \kappa + \frac43\mu`."""
		return self.kappa + 4.0 * self.shear_modulus / 3.0

	@property
	def consolidation_coefficient(self) -> float:
		r""":math:`c_v = (k/\eta) M`."""
		return self.hydraulic_conductivity * self.oedometric_modulus

	@property
	def t_end(self) -> float:
		r""":math:`t_{end}` (s)."""
		return self.t_end_factor * self.length**2 / self.consolidation_coefficient

	@property
	def scales(self) -> CharacteristicScales:
		r""":math:`L_0 = L`, :math:`\mu_0 = \mu`, :math:`t_0 = L^2\eta/(k\kappa)`."""
		return CharacteristicScales.from_consolidation_scale(
			length=self.length,
			stress=self.shear_modulus,
			density=1000.0,
			permeability=self.hydraulic_conductivity * self.viscosity,
			viscosity=self.viscosity,
			consolidation_modulus=self.kappa,
		)


def analytical_pressure(inputs: TerzaghiInputs, x: ArrayLike, t: float, n_terms: int = 400) -> NDArray[np.float64]:
	r""":math:`p(x, t)` (Pa), Terzaghi series.

	Args:
	    inputs: Inputs.
	    x: Abscissae (m), drained at 0.
	    t: Time (s), > 0.
	    n_terms: Series terms.

	Returns:
	    Pressure at ``x``.
	"""
	time_factor = inputs.consolidation_coefficient * t / inputs.length**2
	m = (2.0 * np.arange(1, n_terms + 1) - 1.0) * math.pi / 2.0
	xi = np.asarray(x)[:, None] / inputs.length
	return (2.0 * inputs.load / m * np.sin(m * xi) * np.exp(-(m**2) * time_factor)).sum(axis=1)


def analytical_settlement(inputs: TerzaghiInputs, t: ArrayLike, n_terms: int = 400) -> NDArray[np.float64]:
	r""":math:`u_x(0, t)` (m).

	Args:
	    inputs: Inputs.
	    t: Times (s).
	    n_terms: Series terms.

	Returns:
	    Settlement of the loaded face.
	"""
	time_factor = inputs.consolidation_coefficient * np.asarray(t)[:, None] / inputs.length**2
	m = (2.0 * np.arange(1, n_terms + 1) - 1.0) * math.pi / 2.0
	series = (2.0 / m**2 * np.exp(-(m**2) * time_factor)).sum(axis=1)
	return inputs.load * inputs.length / inputs.oedometric_modulus * (1.0 - series)


@dataclass(frozen=True)
class _Solve:
	"""Coefficients, value-preserving factors and row normalizations of one mode (solve units)."""

	scales: CharacteristicScales
	mu: float
	kappa: float
	conductivity: float
	load: float
	duration: float
	values: dict[str, float]
	test_scales: dict[str, float]
	normalizations: dict[str, float]

	@classmethod
	def build(cls, inputs: TerzaghiInputs, mode: Mode) -> "_Solve":
		s = inputs.scales if mode != "none" else CharacteristicScales.identity()
		if mode == "parameters":
			return cls(
				scales=s,
				mu=s.nondimensionalize_stress(inputs.shear_modulus),
				kappa=s.nondimensionalize_stress(inputs.kappa),
				conductivity=s.nondimensionalize_permeability_over_viscosity(inputs.hydraulic_conductivity),
				load=s.nondimensionalize_stress(inputs.load),
				duration=s.nondimensionalize_time(inputs.t_end),
				values={},
				test_scales={},
				normalizations={},
			)
		# "none" (s = identity: every factor is 1) and "operator" (SI coefficients, scaled rows):
		# scale_factor = s_row v / N_row, v = SI integral / form on the scaled mesh and unknowns (FE test
		# function), s_row = scale of the row's test function (du = L0 du_hat, dp = mu0 dp_hat).
		volume = s.length**3
		return cls(
			scales=s,
			mu=inputs.shear_modulus,
			kappa=inputs.kappa,
			conductivity=inputs.hydraulic_conductivity,
			load=inputs.load,
			duration=inputs.t_end,
			values={
				"gel": s.length**2,  # int P : grad du dV
				"load": s.length**2,  # int t . du dS
				"fluid": s.stress * s.length**2,  # int J p F^-T : grad du dV, p = mu0 p_hat
				"rate": volume,  # int (J - J^n)/dt dp dV
				"darcy": s.stress * s.length,  # int K grad p . grad dp dV
			},
			test_scales={"displacement": s.length, "liquid_pressure": s.stress},
			normalizations={"displacement": s.stress * volume, "liquid_pressure": s.stress * volume / s.time},
		)

	def value(self, term: str) -> float:
		return self.values.get(term, 1.0)


def solve(
	inputs: TerzaghiInputs, mode: Mode, out_dir: Path, quadrature_degrees: Optional[dict[str, int]] = None
) -> pd.DataFrame:
	r"""Run one mode; return SI time series (``t``, ``u0``, ``p_0`` ... ``p_{n-1}``).

	Args:
	    inputs: Inputs.
	    mode: ``"none"``, ``"operator"`` or ``"parameters"``.
	    out_dir: Output folder.
	    quadrature_degrees: Degree per row (``"displacement"``, ``"liquid_pressure"``), default 4.

	Returns:
	    One row per step: time (s), :math:`u_x(0)` (m), pressure at the probes (Pa).

	Raises:
	    RuntimeError: Time integration failed.
	"""
	c = _Solve.build(inputs, mode)
	s = c.scales
	h = inputs.length / 10.0
	mesh = dolfin.BoxMesh(dolfin.Point(0.0, 0.0, 0.0), dolfin.Point(inputs.length, h, h), inputs.n_cells, 1, 1)
	mesh.coordinates()[:] = s.nondimensionalize_length(mesh.coordinates())
	mesh.bounding_box_tree().build(mesh)
	length_mesh, h_mesh = s.nondimensionalize_length(inputs.length), s.nondimensionalize_length(h)

	boundaries = dolfin.MeshFunction("size_t", mesh, 2, 0)
	dolfin.CompiledSubDomain("on_boundary && near(x[0], 0.0)").mark(boundaries, _LOAD)

	problem = dolfin_mech.problems.Problem()
	problem.set_mesh(mesh=mesh, define_facet_normals=True)
	problem.set_measures(domains=None, boundaries=boundaries, points=None)
	problem.displacement_subsol = problem.add_vector_subsol(name="displacement", family="CG", degree=2)
	p_init = s.nondimensionalize_stress(inputs.load)
	problem.pl_subsol = problem.add_scalar_subsol(name="liquid_pressure", family="CG", degree=1, init_val=p_init)
	problem.set_solution_finite_element()
	problem.set_solution_function_space()
	problem.set_solution_functions()
	problem.set_quadrature_degree(4)
	kin = dolfin_mech.kinematics.Kinematics(
		U=problem.displacement_subsol.subfunc, U_old=problem.displacement_subsol.func_old
	)
	problem.kinematics = kin
	dt = c.duration / inputs.n_steps
	problem.add_step(Deltat=c.duration, dt_ini=dt, dt_min=dt, dt_max=dt)

	gel = fenicsoft.materials.iso.neo_hooke.Constitutive(
		kinematics=kin, parameters=fenicsoft.materials.iso.neo_hooke.Parameters.from_shear_modulus(mu=c.mu)
	) + fenicsoft.materials.volumetric.QuadraticVolumetricPenalty(
		kinematics=kin, parameters=fenicsoft.materials.volumetric.VolumetricPenaltyParameters(kappa=c.kappa)
	)
	u, du = problem.displacement_subsol.subfunc, problem.displacement_subsol.dsubtest
	p, dp = problem.pl_subsol.subfunc, problem.pl_subsol.dsubtest
	# Rows = subsols of the test functions: displacement (du) and liquid_pressure (dp).
	rows = ResidualRows.from_problem(
		problem, quadrature_degrees=quadrature_degrees, test_scales=c.test_scales, normalizations=c.normalizations
	)
	dv_u, dv_p = rows.measure(problem.dV, "displacement"), rows.measure(problem.dV, "liquid_pressure")
	operators = [
		fenicsoft.operators.hyper_elasticity.pk1.HyperElasticity(
			disp=u,
			disp_test=du,
			material=gel,
			measure=dv_u,
			k_step=0,
			scale_factor=rows.scale_factor("displacement", c.value("gel")),
		),
		fenicsoft.operators.fluid_stress.biphasic.pk1.FluidStress(
			kinematics=kin,
			function=p,
			displacement=u,
			test=du,
			measure=dv_u,
			scale_factor=rows.scale_factor("displacement", c.value("fluid")),
		),
		fenicsoft.operators.ChangeRate(
			function=kin.J,
			function_old=kin.J_old,
			test=dp,
			dx=dv_p,
			scale_factor=rows.scale_factor("liquid_pressure", c.value("rate")),
		),
		fenicsoft.operators.DarcyGradient(
			function=p,
			test=dp,
			conductivity=c.conductivity * kin.J * kin.C_inv,
			dx=dv_p,
			scale_factor=rows.scale_factor("liquid_pressure", c.value("darcy")),
		),
		# Follower load, linear in P (no scale_factor argument): the factor goes into P_val.
		dolfin_mech.operators.loading.SurfacePressure(
			U_test=du,
			kinematics=kin,
			N=problem.mesh_normals,
			measure=rows.measure(problem.dS(_LOAD), "displacement"),
			P_val=c.load * rows.scale_factor("displacement", c.value("load")),
		),
	]
	for operator in operators:
		problem.add_operator(operator=operator, k_step=0)

	def face(axis: int, value: float) -> dolfin.SubDomain:
		return dolfin.CompiledSubDomain(f"on_boundary && near(x[{axis}], {value})")

	problem.add_constraint(V=problem.pl_subsol.fs, sub_domain=face(0, 0.0), val=0.0, k_step=0)
	problem.add_constraint(V=problem.displacement_subsol.fs.sub(0), sub_domain=face(0, length_mesh), val=0.0, k_step=0)
	for axis in (1, 2):
		for side in (0.0, h_mesh):
			problem.add_constraint(
				V=problem.displacement_subsol.fs.sub(axis), sub_domain=face(axis, side), val=0.0, k_step=0
			)
	for constraint in problem.steps[0].constraints:
		constraint.bc.apply(problem.sol_func.vector())

	# No set_allow_extrapolation: under MPI every rank would then return a value and the QOI averages them.
	mid = h_mesh / 2.0
	problem.add_qoi(
		name="u0",
		expr=s.redimensionalize_length(problem.displacement_subsol.func[0]),
		point=[0.0, mid, mid],
		update_type="direct",
	)
	for i, x in enumerate(inputs.probe_x):
		problem.add_qoi(
			name=f"p_{i}",
			expr=s.redimensionalize_stress(problem.pl_subsol.func),
			point=[s.nondimensionalize_length(x), mid, mid],
			update_type="direct",
		)

	out_dir.mkdir(parents=True, exist_ok=True)
	base = str(out_dir / f"terzaghi_{mode}{'_rows' if quadrature_degrees else ''}")
	integrator = dolfin_mech.core.TimeIntegrator(
		problem=problem,
		solver=dolfin_mech.core.NonlinearSolver(
			problem=problem, parameters={"sol_tol": [inputs.sol_tol] * 2, "n_iter_max": 10}
		),
		parameters={"n_iter_for_accel": 0, "n_iter_for_decel": 100},
		print_out=base,  # type: ignore
		print_sta=base,  # type: ignore
		write_qois=base + "-qois",  # type: ignore
		write_sol=False,
	)
	if not integrator.integrate():
		raise RuntimeError(f"Terzaghi ({mode}) failed.")
	integrator.close()
	names = ["t", "u0", *[f"p_{i}" for i in range(inputs.n_probes)]]
	table = pd.read_csv(base + "-qois.dat", sep=r"\s+", skiprows=1, names=names)
	# Only the "parameters" mode integrates in t_hat = t / t0; "operator" keeps the SI clock.
	clock = s if mode == "parameters" else CharacteristicScales.identity()
	table["t"] = table["t"].map(clock.redimensionalize_time)
	return table.query("t > 0.0").reset_index(drop=True)


@dataclass(frozen=True)
class Errors:
	r"""Errors against the analytical solution.

	Attributes:
	    pressure: :math:`\max_t \|p_h - p\|_{\ell^2(\text{probes})} / (p_0\sqrt{n})`, for :math:`T \ge T_{min}`.
	    settlement: :math:`\max_t |u_h(0) - u(0)| / u_\infty`, for :math:`T \ge T_{min}`.
	"""

	pressure: float
	settlement: float


def errors(inputs: TerzaghiInputs, table: pd.DataFrame, t_factor_min: float = 0.05) -> Errors:
	r"""Relative errors; early times (:math:`T < T_{min}`, sharp boundary layer) are skipped.

	Args:
	    inputs: Inputs.
	    table: Output of :func:`solve`.
	    t_factor_min: :math:`T_{min}`.

	Returns:
	    The errors.
	"""
	x = np.array(inputs.probe_x)
	p_cols = [f"p_{i}" for i in range(inputs.n_probes)]
	t_min = t_factor_min * inputs.length**2 / inputs.consolidation_coefficient
	kept = table[table["t"] >= t_min]
	err_p = max(
		float(np.linalg.norm(row[p_cols].to_numpy(dtype=float) - analytical_pressure(inputs, x, float(row["t"]))))
		for _, row in kept.iterrows()
	) / (inputs.load * math.sqrt(inputs.n_probes))
	u_inf = inputs.load * inputs.length / inputs.oedometric_modulus
	err_u = np.abs(kept["u0"].to_numpy() - analytical_settlement(inputs, kept["t"].to_numpy())).max() / u_inf
	return Errors(pressure=float(err_p), settlement=float(err_u))


def plot(inputs: TerzaghiInputs, tables: dict[str, pd.DataFrame], path: Path) -> Path:
	"""Pressure profiles and settlement, simulations (markers) vs analytical (lines).

	Args:
	    inputs: Inputs.
	    tables: Output of :func:`solve` per mode.
	    path: PNG file.

	Returns:
	    ``path``.
	"""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	x = np.array(inputs.probe_x)
	t_c = inputs.length**2 / inputs.consolidation_coefficient
	fig, (ax_p, ax_u) = plt.subplots(1, 2, figsize=(12, 4.5))
	reference = next(iter(tables.values()))
	x_fine = np.linspace(0.0, inputs.length, 200)
	markers = {"none": "o", "operator": "s", "parameters": "^"}  # "<mode>+rows": "x"
	for t_factor in (0.02, 0.1, 0.3, 0.6, 1.0):
		row = reference.iloc[(reference["t"] - t_factor * t_c).abs().argmin()]
		(line,) = ax_p.plot(x_fine * 1e6, analytical_pressure(inputs, x_fine, row["t"]), lw=1.2)
		for mode, table in tables.items():
			r = table.iloc[(table["t"] - row["t"]).abs().argmin()]
			values = r[[f"p_{i}" for i in range(inputs.n_probes)]].to_numpy(dtype=float)
			label = f"T = {t_factor:g}" if mode == "none" else None
			ax_p.plot(x * 1e6, values, markers.get(mode, "x"), ms=4, mfc="none", color=line.get_color(), label=label)
	ax_p.set(xlabel="x (µm), drained at 0", ylabel="p (Pa)", title="Pressure: lines analytical, markers FE")
	ax_p.legend(fontsize=8)
	t_fine = np.linspace(0.0, reference["t"].max(), 300)
	ax_u.plot(t_fine, analytical_settlement(inputs, t_fine) * 1e9, "k-", lw=1.2, label="analytical")
	for mode, table in tables.items():
		e = errors(inputs, table)
		ax_u.plot(
			table["t"][::5],
			table["u0"][::5] * 1e9,
			markers.get(mode, "x"),
			ms=4,
			mfc="none",
			label=f"{mode}: err p {e.pressure:.1e}, u {e.settlement:.1e}",
		)
	ax_u.set(xlabel="t (s)", ylabel="u_x(0) (nm)", title="Settlement of the loaded face")
	ax_u.legend(fontsize=8)
	fig.tight_layout()
	path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(path, dpi=130)  # type: ignore
	plt.close(fig)
	return path


if __name__ == "__main__":
	out = Path("./outputs_terzaghi_nondim")
	cfg = TerzaghiInputs()
	results = {mode: solve(cfg, mode, out) for mode in MODES}
	for mode, table in results.items():
		print(mode, errors(cfg, table))
	print("figure:", plot(cfg, results, out / "terzaghi_nondim.png"))
