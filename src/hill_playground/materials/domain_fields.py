r"""Indicator fields of a set of tagged subdomains (cell tags of the mesh).

With :math:`\Omega_T` the union of the cells whose tag is in :math:`T`:

.. math::
	\chi_T(\mathbf x) = \mathbb 1_{\Omega_T}(\mathbf x) \quad (\mathrm{DG}_0), \qquad
	\chi_T^{int}(\mathbf x_v) = \begin{cases} 1 & \text{every cell around vertex } v \text{ is in } \Omega_T \\
	0 & \text{otherwise} \end{cases} \quad (\mathrm{CG}_1)

:math:`\chi_T^{int} = 0` on the interface between :math:`\Omega_T` and the other cells.
Piecewise data: :math:`a = a_{out} + (a_{in} - a_{out})\,\chi_T`.
"""

from typing import Sequence

import dolfin
import numpy as np
from fenicsoft.io import build_indicator_meshfunction


def cell_indicator(
	mesh: dolfin.Mesh, domains_mf: dolfin.MeshFunction, tags: Sequence[int], name: str = "cell_indicator"
) -> dolfin.Function:
	r""":math:`\chi_T`: 1 on the cells tagged in ``tags``, 0 elsewhere (DG0).

	Args:
	    mesh: Mesh.
	    domains_mf: Cell tags.
	    tags: Tags of :math:`\Omega_T`.
	    name: Name of the function.

	Returns:
	    DG0 function.
	"""
	space = dolfin.FunctionSpace(mesh, "DG", 0)
	indicator = dolfin.Function(space, name=name)
	inside = build_indicator_meshfunction(mesh, domains_mf, tags).array().astype(np.float64)
	cells = np.arange(mesh.num_cells(), dtype=np.uintp)
	values = indicator.vector().get_local()
	values[space.dofmap().entity_dofs(mesh, mesh.topology().dim(), cells)] = inside
	indicator.vector().set_local(values)
	indicator.vector().apply("insert")
	return indicator


def interior_vertex_indicator(
	mesh: dolfin.Mesh, domains_mf: dolfin.MeshFunction, tags: Sequence[int], name: str = "interior_indicator"
) -> dolfin.Function:
	r""":math:`\chi_T^{int}`: 1 at the vertices whose cells are all in :math:`\Omega_T`, 0 elsewhere (CG1).

	A vertex touches a cell outside :math:`\Omega_T` iff :math:`\int (1 - \chi_T)\,\varphi_v\,\mathrm dV > 0`
	(:math:`\varphi_v` its hat function): one assembly, exact across MPI partitions.

	Args:
	    mesh: Mesh.
	    domains_mf: Cell tags.
	    tags: Tags of :math:`\Omega_T`.
	    name: Name of the function.

	Returns:
	    CG1 function.
	"""
	space = dolfin.FunctionSpace(mesh, "CG", 1)
	outside = 1.0 - cell_indicator(mesh, domains_mf, tags)
	touch = dolfin.assemble(outside * dolfin.TestFunction(space) * dolfin.dx(domain=mesh))
	indicator = dolfin.Function(space, name=name)
	indicator.vector().set_local((touch.get_local() == 0.0).astype(np.float64))
	indicator.vector().apply("insert")
	return indicator
