r"""Microtissue between pillars, stage B: the side patches released, the core held by its four pillars.

Starts from the stage A state on the core patch (``pillars_cut_to_core.py``): :math:`\mathbf u, p, \Phi_c, \theta,
\varphi, \varepsilon_a`, reference configuration unchanged. Model, pillar pins, nondimensionalisation and outputs:
those of stage A (``pillars_biaxial.py``, reused). The time restarts at 0; ``fibre_initial`` is the stage A
final orientation.

Boundary conditions (:data:`STAGE_B`)
-------------------------------------
* :math:`\mathbf u = \mathbf 0` on ``PILLAR_BASE_CORE`` (four pillars);
* :math:`p = 0` on ``FRONT/BACK/TOP/BOTTOM_GEL`` and on the two cut faces ``CUT_FACES`` (in contact with the
  medium after the release); traction-free, no cell flux;
* :math:`\alpha = 1` held.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Union

import dolfin
import pillars_biaxial as stage_a
from fenicsoft import io
from fenicsoft.integration import Step

from hill_playground import integration
from hill_playground.materials.cell.active.activation import ActivePowerConfig

rank = dolfin.MPI.rank(dolfin.MPI.comm_world)

M = stage_a.M
STAGE_B = stage_a.PillarRegion(
	gel=(int(M.GEL_CORE),),
	pillars=(int(M.PILLAR_CORE),),
	clamped=(int(M.PILLAR_BASE_CORE),),
	drained=(*(int(m) for m in (M.FRONT_GEL, M.BACK_GEL, M.TOP_GEL, M.BOTTOM_GEL)), stage_a.CUT_FACES),
)
"""Stage B: core patch, its four pillar bases clamped, cut faces drained."""


@dataclass(slots=True, frozen=True)
class UserInputs(stage_a.UserInputs):
	r"""Stage B configuration, SI units: stage A inputs with one hold step and the stage B outputs.

	Attributes:
	    steps: Hold (s); first time step ``init_dt`` = 1 s ≈ :math:`t_0` (sudden release).
	    activation_configs: :math:`\alpha = 1`.
	    xdmf_write_period: One period for the single step.
	    project_prefix: Output folder (its ``mesh`` folder holds the cut).
	    stem_name: Output file stem.
	    stem_mesh_name: Core mesh file stem.
	"""

	steps: Sequence[Step] = field(
		default_factory=lambda: (
			Step(duration=42000.0, init_dt=1.0, min_dt=0.25, max_dt=3000.0, min_simu_seconds=3 * 3600),
		)
	)
	activation_configs: Sequence[ActivePowerConfig] = field(
		default_factory=lambda: [ActivePowerConfig.constant(k_step=0, value=1.0)]
	)
	xdmf_write_period: Union[int, integration.TimeInterval, Sequence[Any]] = integration.TimeInterval(840.0)
	project_prefix: Path = Path("./outputs_pillars_uniaxial_release")
	stem_name: str = "pillars_uniaxial_release"
	stem_mesh_name: str = "pillars_core_mesh"


def main(user_inputs: UserInputs = UserInputs()) -> bool:
	"""Stage B: load the cut, scale, assemble, solve, checkpoint.

	Args:
	    user_inputs: Configuration.

	Returns:
	    True if the simulation completed.

	Raises:
	    FileNotFoundError: ``pillars_cut_to_core.py`` not run.
	"""
	simulation_paths = io.configure_environment(user_inputs.project_prefix, user_inputs.stem_name)
	mesh, domains_mf, boundaries_mf, _ = io.load_mesh_data(simulation_paths.mesh_folder / user_inputs.stem_mesh_name)
	initial = stage_a.read_handover(mesh, simulation_paths.mesh_folder, user_inputs)
	return stage_a.run(user_inputs, simulation_paths, mesh, domains_mf, boundaries_mf, STAGE_B, initial)


if __name__ == "__main__":
	io.configure_logging(rank)
	if not main():
		dolfin.MPI.comm_world.Abort(errorcode=1)
