r"""Microtissue between pillars, cut: stage A mesh and state restricted to the core patch. Serial only.

Keeps the cells ``GEL_CORE`` and ``PILLAR_CORE`` (``fenicsoft.mesh.extract_submesh_by_tags``), projects the
facet tags (``fenicsoft.mesh.project_facet_tags``), tags the two cut planes :math:`x = L_{side}` and
:math:`x = L_{side} + L_{core}` ``CUT_FACES`` (drained in stage B) and transfers the SI checkpoints of stage A
(``HANDOVER``):

* CG fields :math:`\mathbf u, p, \Phi_c`: interpolation (``fenicsoft.mesh.transfer_field_by_interpolation``),
  exact since the submesh nodes are parent nodes;
* DG0 fields :math:`\theta, \varphi, \varepsilon_a`: per-cell gather (``fenicsoft.mesh.build_dg0_gather_map``,
  ``fenicsoft.mesh.transfer_dg0_field``).

Writes the core mesh and the checkpoints into the mesh folder of stage B.
"""

from __future__ import annotations

import logging
from typing import Optional

import dolfin
import pillars_biaxial as stage_a
import pillars_uniaxial_release as stage_b
from fenicsoft import io
from fenicsoft.mesh import (
	build_dg0_gather_map,
	extract_submesh_by_tags,
	project_facet_tags,
	transfer_dg0_field,
	transfer_field_by_interpolation,
)

comm = dolfin.MPI.comm_world

_CORE = (int(stage_a.M.GEL_CORE), int(stage_a.M.PILLAR_CORE))
_PLANE_TOL = 1e-3
"""Tolerance of the cut planes, relative to the element size."""


def tag_cut_faces(submesh: dolfin.Mesh, boundaries_mf: dolfin.MeshFunction, user_inputs: stage_a.UserInputs) -> int:
	"""Tag ``CUT_FACES`` the exterior facets on the planes :math:`x = L_{side}` and :math:`x = L_{side} + L_{core}`.

	Args:
	    submesh: Core mesh (SI).
	    boundaries_mf: Its facet tags (modified).
	    user_inputs: Stage A configuration.

	Returns:
	    Number of tagged facets.
	"""
	x_left = user_inputs.side_length_x
	planes = dolfin.CompiledSubDomain(
		"on_boundary && (near(x[0], a, tol) || near(x[0], b, tol))",
		a=x_left,
		b=x_left + user_inputs.core_length_x,
		tol=_PLANE_TOL * submesh.hmin(),
	)
	before = int((boundaries_mf.array() == stage_a.CUT_FACES).sum())
	planes.mark(boundaries_mf, stage_a.CUT_FACES)
	return int((boundaries_mf.array() == stage_a.CUT_FACES).sum()) - before


def cut(
	stage_a_inputs: stage_a.UserInputs = stage_a.UserInputs(),
	stage_b_inputs: Optional[stage_b.UserInputs] = None,
) -> None:
	"""Extract the core, transfer the stage A checkpoints, write the stage B inputs.

	Args:
	    stage_a_inputs: Stage A configuration.
	    stage_b_inputs: Stage B configuration; default ``stage_b.UserInputs()``.

	Raises:
	    RuntimeError: Run with more than one MPI rank (``dolfin.SubMesh`` is serial).
	"""
	if dolfin.MPI.size(comm) != 1:
		raise RuntimeError("pillars_cut_to_core.py runs in serial (dolfin.SubMesh): no mpirun.")
	stage_b_inputs = stage_b_inputs or stage_b.UserInputs()
	prefix_a = stage_a_inputs.project_prefix  # not io.configure_environment: it empties the results folder
	mesh, domains_mf, boundaries_mf, _ = io.load_mesh_data(prefix_a / "mesh" / stage_a_inputs.stem_mesh_name)
	parent = stage_a.read_handover(mesh, prefix_a / "results", stage_a_inputs)

	submesh, sub_domains_mf = extract_submesh_by_tags(mesh, domains_mf, tags=_CORE)
	sub_boundaries_mf = project_facet_tags(submesh, boundaries_mf)
	n_cut = tag_cut_faces(submesh, sub_boundaries_mf, stage_a_inputs)

	transferred = {}
	for name, space in stage_a.handover_spaces(submesh, stage_a_inputs).items():
		transferred[name] = dolfin.Function(space, name=name)
		if space.ufl_element().family() == "Discontinuous Lagrange" and space.ufl_element().degree() == 0:
			gather = build_dg0_gather_map(submesh, parent[name].function_space(), space)
			transfer_dg0_field(parent[name], transferred[name], gather)
		else:
			transfer_field_by_interpolation(parent[name], transferred[name])

	paths_b = io.configure_environment(stage_b_inputs.project_prefix, stage_b_inputs.stem_name)
	io.write_submesh_to_xdmf(
		submesh, sub_domains_mf, sub_boundaries_mf, paths_b.mesh_folder / stage_b_inputs.stem_mesh_name
	)
	stage_a.write_handover(transferred, paths_b.mesh_folder, {name: 1.0 for name in transferred})
	logging.getLogger(__name__).info(
		f"Core mesh ({submesh.num_cells()} cells, {n_cut} cut facets) and {sorted(transferred)} written to "
		f"{paths_b.mesh_folder}."
	)


if __name__ == "__main__":
	io.configure_logging(0)
	cut()
