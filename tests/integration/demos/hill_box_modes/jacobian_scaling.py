r"""Diagnostic: Jacobian of the Hill box at the first step, per nondimensionalization mode.

Assembles :math:`\mathbf J = \partial\tilde R/\partial\hat x` (Dirichlet BCs applied) as the Newton
solver sees it, then reports:

* :math:`\kappa_2(\mathbf J) = \sigma_{max}/\sigma_{min}` (dense SVD: small meshes only);
* the median :math:`|J_{ii}|` per row (subsol), Dirichlet rows excluded: the order of magnitude of
  each block, which the row scaling should bring to :math:`\mathcal O(1)` up to the mesh size.

``"none"`` solves in SI; ``"operator"`` (each residual row multiplied by a constant ``scale_factor``) and
``"parameters"`` (dimensionless coefficients) solve the same dimensionless system, hence must give the same
matrix (see :mod:`hill_box_all_modes`, "Nondimensionalisation"). Run ``python jacobian_scaling.py`` for the
table.

Reading the diagonals (scaled mesh, :math:`\hat h = h/L_0`): a row whose leading term is a mass
term (:math:`\int \hat N_i^2\,\mathrm d\hat V \sim \hat h^3`: ``cell_fraction``, ``theta``, ``phi``,
``substrate_smoothed``) and a row led by a stiffness term (:math:`\int |\hat\nabla \hat N_i|^2 \sim
\hat h`: ``displacement``, ``liquid_pressure``) differ by :math:`\hat h^2` (here :math:`\sim 10^{-3}`
with the P1/P2 constants). This is the usual mass/stiffness ratio of the discretization, not a
scaling error: it does not depend on the units, and a row factor removing it would depend on the mesh.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io as std_io
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import dolfin
import hill_box_all_modes as demo
import numpy as np
import scipy.sparse as sp
from fenicsoft import io
from numpy.typing import NDArray


def small_box(mode: str, prefix: Path, **overrides: Any) -> demo.UserInputs:
	"""The default (small) box of :mod:`hill_box_all_modes` in ``mode``: about 2000 dofs.

	Args:
	    mode: ``nondimensionalization``.
	    prefix: Output folder.
	    **overrides: Other ``UserInputs`` fields.

	Returns:
	    The configuration.
	"""
	return dataclasses.replace(demo.UserInputs(), nondimensionalization=mode, project_prefix=prefix, **overrides)


def assemble_jacobian(
	user_inputs: demo.UserInputs, t_step: float = 0.25
) -> Tuple[sp.csr_matrix, Dict[str, NDArray[np.intp]]]:
	r"""Jacobian at the initial state, first step, step fraction ``t_step`` (BCs applied).

	Args:
	    user_inputs: Configuration.
	    t_step: Step fraction (sets :math:`\alpha` and the BC values).

	Returns:
	    ``(J, dofs)``: CSR matrix and the dof indices of each subsol.
	"""
	paths = demo.build_mesh(user_inputs)
	mesh, domains, boundaries, points = io.load_mesh_data(paths.mesh_folder / user_inputs.stem_mesh_name)
	units = demo.SolveUnits.from_inputs(user_inputs)
	mesh.coordinates()[:] = units.unknowns.nondimensionalize_length(mesh.coordinates())
	mesh.bounding_box_tree().build(mesh)
	coefficients = demo.Coefficients.from_inputs(user_inputs, units.coefficients)
	problem = demo.initialise_problem(mesh, domains, boundaries, points, user_inputs, coefficients)
	problem = demo.set_weak_form_and_bcs(problem, boundaries, user_inputs, coefficients)
	demo.apply_initial_bcs(problem)
	step = problem.steps[0]
	problem.set_variational_formulation(k_step=0)
	for operator in step.operators:
		operator.set_value_at_t_step(t_step)
		operator.set_dt(step.dt_ini)
	for constraint in step.constraints:
		constraint.set_value_at_t_step(t_step)
	matrix, _ = dolfin.assemble_system(
		problem.jac_form,
		-problem.res_form,
		bcs=[c.bc for c in problem.constraints + step.constraints],
		form_compiler_parameters=problem.form_compiler_parameters,
	)
	indptr, indices, values = dolfin.as_backend_type(matrix).mat().getValuesCSR()
	jacobian = sp.csr_matrix((values, indices, indptr), shape=(matrix.size(0), matrix.size(1)))
	dofs = {
		subsol.name: np.asarray(problem.sol_fs.sub(k).dofmap().dofs(), dtype=np.intp)
		for k, subsol in enumerate(problem.subsols)
	}
	return jacobian, dofs


def condition_number(jacobian: sp.csr_matrix) -> float:
	r""":math:`\kappa_2 = \sigma_{max}/\sigma_{min}` by dense SVD.

	Args:
	    jacobian: Matrix (a few thousand rows at most).

	Returns:
	    The 2-norm condition number.
	"""
	singular = np.linalg.svd(jacobian.toarray(), compute_uv=False)
	return float(singular[0] / singular[-1])


def block_diagonals(jacobian: sp.csr_matrix, dofs: Dict[str, NDArray[np.intp]]) -> Dict[str, float]:
	r"""Median :math:`|J_{ii}|` per subsol, Dirichlet rows (no off-diagonal entry) excluded.

	Args:
	    jacobian: Matrix.
	    dofs: Dof indices per subsol.

	Returns:
	    Median per subsol.
	"""
	diagonal = np.abs(jacobian.diagonal())
	off_diagonal = np.asarray(abs(jacobian).sum(axis=1)).ravel() - diagonal
	free = off_diagonal > 1e-14 * (off_diagonal + diagonal + 1e-300)
	return {name: float(np.median(diagonal[idx][free[idx]])) for name, idx in dofs.items()}


if __name__ == "__main__":
	io.configure_logging(0)
	logging.disable(logging.INFO)  # mesh generation logs
	dolfin.set_log_level(dolfin.LogLevel.WARNING)  # dolfin.info of io.load_mesh_data
	table = {}
	for mode in ("none", "operator", "parameters"):
		with contextlib.redirect_stdout(std_io.StringIO()):  # mesh reader prints
			matrix, blocks = assemble_jacobian(small_box(mode, Path(f"./outputs_jacobian_scaling_{mode}")))
		table[mode] = {"n": matrix.get_shape()[0], "cond2": condition_number(matrix), **block_diagonals(matrix, blocks)}  # type: ignore
	print("median |J_ii| per row (Dirichlet rows excluded) and 2-norm condition number")
	print(f"{'row':20s}" + "".join(f"{mode:>12s}" for mode in table))
	for key in next(iter(table.values())):
		cells = "".join(f"{row[key]:>12d}" if key == "n" else f"{row[key]:>12.2e}" for row in table.values())
		print(f"{key:20s}{cells}")
