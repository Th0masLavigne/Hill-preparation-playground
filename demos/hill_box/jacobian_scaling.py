r"""Diagnostic: Jacobian of the Hill box at the first step, per nondimensionalization mode.

Assembles :math:`\mathbf J = \partial\tilde R/\partial\hat x` (Dirichlet BCs applied) as the Newton
solver sees it, then reports:

* :math:`\kappa_2(\mathbf J) = \sigma_{max}/\sigma_{min}` (dense SVD: small meshes only);
* the median :math:`|J_{ii}|` per row (subsol), Dirichlet rows excluded: the order of magnitude of
  each block, which the row scaling should bring to :math:`\mathcal O(1)` up to the mesh size.

``"operator"`` and ``"parameters"`` must give the same matrix (module docstring of
:mod:`hill_box_uniaxial`, section 4). Run ``python jacobian_scaling.py`` for the table.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Tuple

import dolfin
import hill_box_uniaxial as demo
import numpy as np
import scipy.sparse as sp
from fenicsoft import io


def small_box(mode: str, prefix: Path, **overrides: Any) -> demo.UserInputs:
	"""A 100 x 50 x 50 um box, h = 25 um, one 20 s ramp (about 2000 dofs).

	Args:
	    mode: ``nondimensionalization``.
	    prefix: Output folder.
	    **overrides: Other ``UserInputs`` fields.

	Returns:
	    The configuration.
	"""
	settings: Dict[str, Any] = dict(
		length_x=100e-6,
		width_y=50e-6,
		mesh_max_size=25e-6,
		mesh_min_size=25e-6,
		nondimensionalization=mode,
		project_prefix=prefix,
		steps=(io.Step(duration=20.0, init_dt=5.0, min_dt=0.5),),
		activation_configs=[demo.ActivePowerConfig.ramp(k_step=0, ini=0.0, fin=1.0)],
	)
	settings.update(overrides)
	return dataclasses.replace(demo.UserInputs(), **settings)


def assemble_jacobian(
	user_inputs: demo.UserInputs, t_step: float = 0.25
) -> Tuple[sp.csr_matrix, Dict[str, np.ndarray]]:
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
	dofs = {subsol.name: np.array(problem.sol_fs.sub(k).dofmap().dofs()) for k, subsol in enumerate(problem.subsols)}
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


def block_diagonals(jacobian: sp.csr_matrix, dofs: Dict[str, np.ndarray]) -> Dict[str, float]:
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
	for mode in ("none", "operator", "parameters"):
		matrix, blocks = assemble_jacobian(small_box(mode, Path(f"./outputs_jacobian_scaling_{mode}")))
		medians = block_diagonals(matrix, blocks)
		print(f"{mode:10s} n = {matrix.shape[0]}  cond2 = {condition_number(matrix):.2e}")
		print("           " + "  ".join(f"{k}={v:.1e}" for k, v in medians.items()))
