r"""Microtissue between pillars, stage A: an active poro-elastic gel held biaxially by twelve micropillars.

A hydrogel slab seeded with randomly oriented contractile cells (Hill chain) hangs on twelve mushroom-shaped
micropillars clamped at their bases (biaxial constraint, Mostert et al. 2026). Under the activation
:math:`\alpha(t)` the cells contract against the pillars, the gel compacts and expels liquid, and the cells may
migrate, proliferate and reorient. Stage B (``pillars_uniaxial_release.py``) then keeps the core patch only.

Files: this script (stage A: model, solve, checkpoints; shared with the cut and stage B),
``pillars_scaling.py`` (SI inputs to dimensionless coefficients), ``pillars_outputs.py`` (SI fields and QOIs),
``pillars_cut_to_core.py``, ``pillars_uniaxial_release.py``, ``pillars_post_processing.py``. Pipeline
(``batch_hill_microtissue_pillars.sh``): this script (``mpirun``) -> cut (serial) -> stage B (``mpirun``) ->
post-processing.

Geometry and boundary conditions
--------------------------------
Three gel patches ``GEL_SIDE_LEFT | GEL_CORE | GEL_SIDE_RIGHT``, :math:`z \in [0, T]`, four pillars each
(``fenicsoft.mesh_geometries.microtissue_pillars``): a shaft of radius :math:`r_p` through the gel, from its
clamped base :math:`z_b = T - (h_p - t_l)` below the gel to :math:`z = T`, capped by a lid of radius
:math:`r_l > r_p` **above the gel**, :math:`z \in [T, T + t_l]`. The gel/pillar interface is the shaft wall in the
gel and the annulus :math:`r_p < r < r_l` under the lid, on the top face of the gel (``SIDE_PILLARS_*``).

.. image:: /_static/resources/hill_microtissue_pillars_bcs.svg
	:alt: Top view of the three gel patches and twelve pillars, and section of one pillar with its lid on the gel
	:width: 100%
	:align: center

* :math:`\mathbf u = \mathbf 0` on the pillar bases (``region.clamped``: all twelve in stage A);
* :math:`p = 0` on the free gel faces ``*_GEL`` (``region.drained``; the annuli under the lids are not part of
  ``TOP_GEL``); pillars and the rest impermeable;
* traction-free elsewhere, no cell flux through the gel boundary.

Unknowns
--------
=======================  =====  =================================================  ==================
Unknown                  Space  Initial value                                      Present if
=======================  =====  =================================================  ==================
:math:`\hat{\mathbf u}`  CG2    0                                                  always
:math:`\theta,\varphi`   DG0    isotropic draw                                     ``remodelling``
:math:`\hat p`           CG1    0                                                  always
:math:`\Phi_c`           CG1    :math:`\Phi_{c,0}`, 0 strictly inside the pillars  ``cell_transport``
:math:`\tilde\phi_s`     CG1    :math:`\Phi_{s,0}` of the gel                      ``cell_transport``
:math:`\varepsilon_a`    DG0    0 (:math:`\lambda_a = 1`)                          always
=======================  =====  =================================================  ==================

Switched off, :math:`\theta,\varphi` stay at their random draw and :math:`\Phi_c` at its initial field.
:math:`\Phi_c = \Phi_{c,0}` at every vertex touching the gel, 0 at the vertices whose cells are all pillar
(``materials.domain_fields``); :math:`\Phi_c \in P_1`, hence ``displacement_degree = 2``.

Residuals
---------
``equations.tex``, :math:`\Omega_0 = \Omega_0^{gel} \cup \Omega_0^{pil}`:

.. math::
	R_u &= \int_{gel} \big(\mathbf{P}_S + \mathbf{P}_h\big) : \nabla\delta\mathbf{u}
	       + \int_{pil} \mathbf{P}_S^{pil} : \nabla\delta\mathbf{u}
	       - \int J p \mathbf{F}^{-T} : \nabla\delta\mathbf{u} \\
	R_p &= \int \frac{J - J^n}{\Delta t}\delta p + \int_{gel} \mathbf{K}_L \nabla p \cdot \nabla\delta p
	       + \int_{pil} \mathbf{K}_L^{pil} \nabla p \cdot \nabla\delta p
	       - \int_{gel} \mathbf{Q}_c \cdot \nabla\delta p \\
	R_c &= \int_{gel} \Big[\frac{\Phi_c - \Phi_c^n}{\Delta t}\delta c - \mathbf{Q}_c \cdot \nabla\delta c
	       - r\,\mathrm{sat}\,\Phi_c\,\delta c\Big] + \int_{pil} \frac{\Phi_c - \Phi_c^n}{\Delta t}\delta c \\
	R_{\theta} &= \int \Big(\frac{\theta - \theta^n}{\Delta t}
	       - \frac{\alpha}{\tau_{eff}}\partial_\theta Q\Big)\delta\theta \quad (\text{idem } \varphi) \\
	R_a &= \int_{gel} \big[f_{max}\alpha f_{FL}(\lambda_a) h(D_a) + \eta_0 D_a + T_p - T_e\big]\delta\varepsilon
	       + \int_{pil} \frac{\varepsilon_a - \varepsilon_a^n}{\Delta t}\delta\varepsilon \\
	R_s &= \int \Big(\tilde\phi_s - \frac{\Phi_{s,0}}{J}\Big)\delta\phi_s

(all :math:`\mathrm dV`) with, in the gel, :math:`\mathbf{P}_S` the Neo-Hookean + volumetric gel
(``volumetric_law``), :math:`\mathbf{P}_h = \mathbf{F}\,\Phi_c T_e/I_4\,\mathbf{f}_0\otimes\mathbf{f}_0` (Hill chain),
:math:`I_4 = \mathbf f_0\cdot\mathbf C\mathbf f_0`, :math:`\mathbf{K}_L = (k(\phi_\ell)/\eta) J \mathbf{C}^{-1}`,
:math:`\mathbf{Q}_c` the diffusive + haptotactic cell flux,
:math:`\mathrm{sat} = \langle 1 - \Phi_c/\Phi_{c,max}\rangle_+`,
:math:`Q` the orientation potential, :math:`D_a = (\varepsilon_a - \varepsilon_a^n)/\Delta t` (Hill chain:
SEE :math:`T_e` in series with the CE, itself in parallel with the PEE :math:`T_p`; ``materials.cell.active.hill``);
in the pillars, :math:`\mathbf P_S^{pil}` Neo-Hooke + quadratic volumetric (``pillar_shear_modulus``,
``pillar_kappa``), :math:`\mathbf K_L^{pil} = r_k\,(k_0/\eta) J\mathbf C^{-1}` (``pillar_permeability_ratio``
:math:`r_k`), :math:`\Phi_{s,0}` = ``pillar_phi_s0`` in :math:`R_s`. The pillar term of :math:`R_a` is
:math:`\mu_0 t_0 \int D_a\,\delta\varepsilon` in SI (``equations.tex``, Eq. ``eq:Ra``), written here dimensionless.

Cell-free pillars (``equations.tex``, Remark ``rem:pillar-pin``): :math:`\Phi_c` and :math:`\varepsilon_a` are
frozen there by their rate term alone (``ChangeRate`` on :math:`\Omega_0^{pil}`): they keep their initial values
(:math:`\Phi_c = 0`, :math:`\lambda_a = 1`), no cell flux crosses the interface (natural condition of the gel rows),
and no scale is involved. An identity :math:`\frac{1}{t_0}\int_{pil}\Phi_c\delta c` would also act on the
:math:`P_1` interface rows: a cell sink of relative weight :math:`\Delta t/t_0` (up to :math:`3\cdot 10^3` here).

Nondimensionalisation (by the parameters)
-----------------------------------------
Inputs are SI. Scales :math:`L_0 = T` (gel thickness, drainage length), :math:`\mu_0 = \mu_{gel}`,
:math:`t_0 = L_0^2\eta/(k\,\kappa_{gel})` (``UserInputs.scales``). The mesh is divided by :math:`L_0` after loading
(:math:`\hat{\mathbf u} = \mathbf u/L_0` follows), :math:`\hat p = p/\mu_0`, and every coefficient, the pillar moduli
included, is converted by ``pillars_scaling.Coefficients``; the solver time is :math:`\hat t = t/t_0`. Outputs and
checkpoints are SI (``pillars_outputs``, :data:`HANDOVER`); the XDMF geometry and time stamps are put back in m
and s after the run.

Options
-------
* ``permeability_law`` (gel): ``"constant"``, ``"kozeny_carman"``, ``"holmes_mow"``, ``"compaction_power"``
  (``materials.permeability``);
* ``volumetric_law`` (gel): ``"quadratic"`` :math:`\frac{\kappa}{2}(J-1)^2` or ``"compaction"``
  (barrier at :math:`J_{\min} = \Phi_{s,0} + \Phi_c`, ``materials.volumetric.compaction``);
* ``cell_transport``, ``remodelling``: solve :math:`(\Phi_c, \tilde\phi_s)` and :math:`(\theta, \varphi)`;
* ``quadrature_degrees``: quadrature degree per residual row (row = subsol of the test function).

Loading: activation :math:`\alpha` ramped 0 -> 1, then held. No external load.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, get_args

import dolfin
import dolfin_mech
import fenicsoft
import fenicsoft.materials.orientation.spherical as spherical_mod
import fenicsoft.operators.cell as cell_ops
from fenicsoft import io
from fenicsoft.mesh_geometries.mesh_geometries import GmshKeys, MeshOptions, RefinementZone
from fenicsoft.mesh_geometries.microtissue_pillars import (
	MicrotissuePillarsGeometry,
	MicrotissuePillarsMarker,
	MicrotissuePillarsMeshGenerator,
)
from fenicsoft.operators.cell.rotation.orientation_change import OrientationChange
from pillars_outputs import add_outputs
from pillars_scaling import Coefficients

from hill_playground import integration
from hill_playground.io import xdmf_units
from hill_playground.materials import domain_fields, saturated_mixture
from hill_playground.materials import permeability as perm
from hill_playground.materials.cell.active.activation import ActivePowerConfig
from hill_playground.materials.cell.active.hill import law as hill_law
from hill_playground.materials.cell.active.hill.functions import SeeVariant
from hill_playground.materials.volumetric import compaction
from hill_playground.operators import ResidualRows
from hill_playground.operators.cell import ContractileEvolution, SaturatedProliferation
from hill_playground.scaling import CharacteristicScales

comm = dolfin.MPI.comm_world
rank = dolfin.MPI.rank(comm)

M = MicrotissuePillarsMarker
GEL_ENVELOPE: Tuple[int, ...] = tuple(
	int(m) for m in (M.LEFT_GEL, M.RIGHT_GEL, M.FRONT_GEL, M.BACK_GEL, M.TOP_GEL, M.BOTTOM_GEL)
)
"""Free gel faces (drained)."""
CUT_FACES: int = max(int(m) for m in MicrotissuePillarsMarker) + 1
"""Facet tag of the faces cut between the core and the side patches (stage B mesh, ``pillars_cut_to_core``)."""
HANDOVER: Tuple[str, ...] = ("displacement", "liquid_pressure", "cell_fraction", "theta", "phi", "eps_a")
"""Fields checkpointed by stage A and read by stage B (subsol names; frozen fields when switched off)."""


@dataclass(frozen=True)
class PillarRegion:
	r"""Tags of a stage: gel and pillar cells, clamped and drained facets.

	Attributes:
	    gel: Gel cell tags.
	    pillars: Pillar cell tags.
	    clamped: Facet tags with :math:`\mathbf u = \mathbf 0` (pillar bases).
	    drained: Facet tags with :math:`p = 0`.
	"""

	gel: Tuple[int, ...]
	pillars: Tuple[int, ...]
	clamped: Tuple[int, ...]
	drained: Tuple[int, ...]


STAGE_A = PillarRegion(
	gel=tuple(int(m) for m in (M.GEL_SIDE_LEFT, M.GEL_CORE, M.GEL_SIDE_RIGHT)),
	pillars=tuple(int(m) for m in (M.PILLAR_SIDE_LEFT, M.PILLAR_CORE, M.PILLAR_SIDE_RIGHT)),
	clamped=tuple(int(m) for m in (M.PILLAR_BASE_SIDE_LEFT, M.PILLAR_BASE_CORE, M.PILLAR_BASE_SIDE_RIGHT)),
	drained=GEL_ENVELOPE,
)
"""Stage A: whole microtissue, every pillar base clamped."""

_LID_TOPS = (M.PILLAR_LID_TOP_SIDE_LEFT, M.PILLAR_LID_TOP_CORE, M.PILLAR_LID_TOP_SIDE_RIGHT)


PermeabilityLaw = Literal["constant", "kozeny_carman", "holmes_mow", "compaction_power"]
VolumetricLaw = Literal["quadratic", "compaction"]
_CHOICES: Dict[str, Any] = {
	"permeability_law": PermeabilityLaw,
	"volumetric_law": VolumetricLaw,
	"see_variant": SeeVariant,
}
"""``UserInputs`` options restricted to a ``Literal``: checked at construction (``Literal`` is static only)."""


@dataclass(slots=True, frozen=True)
class UserInputs:
	r"""Stage A configuration, SI units.

	Attributes:
	    side_length_x: :math:`L_{side}`, x-extent of each side patch (m).
	    core_length_x: :math:`L_{core}`, x-extent of the core patch (m).
	    width_y: :math:`W` (m).
	    gel_thickness_z: :math:`T`, also :math:`L_0` (m).
	    pillar_radius: Shaft radius :math:`r_p` (m).
	    pillar_lid_radius: Lid radius :math:`r_l` (m).
	    lid_thickness: Lid thickness :math:`t_l` (m), above the gel.
	    pillar_height: Pillar length :math:`h_p` (shaft + lid, m), :math:`h_p - t_l > T`.
	    pillar_edge_margin: Pillar centre to the outer edges :math:`m` (m).
	    epsilon: Geometric tolerance of ``MicrotissuePillarsGeometry``.
	    create_lids: Build the lids.
	    mesh_min_size: Gmsh minimum element size (m).
	    mesh_max_size: Gmsh maximum element size, bulk gel (m).
	    mesh_refine_min_size: Element size near the lid tops (m).
	    mesh_refine_dist_min: Distance to a lid top kept at ``mesh_refine_min_size`` (m).
	    mesh_refine_dist_max: Distance to a lid top back to ``mesh_max_size`` (m).
	    regenerate_mesh: Re-run Gmsh even if the mesh exists.
	    quadrature_degree: Default quadrature degree (every term without its own).
	    quadrature_degrees: Degree per residual row, overriding the default for that row's terms; rows
	        are named after the subsol of their test function (:data:`UNKNOWNS`).
	    foi_degree: Degree of the DG spaces of the FOIs.
	    cell_transport: Solve :math:`\Phi_c, \tilde\phi_s` (:math:`R_c, R_s`); else :math:`\Phi_c = \Phi_{c,0}`.
	    remodelling: Solve :math:`\theta, \varphi` (:math:`R_{\theta,\varphi}`); else frozen at the draw.
	    gel_shear_modulus: :math:`\mu_{gel}`, also :math:`\mu_0` (Pa).
	    gel_kappa: :math:`\kappa_{gel}` (Pa).
	    volumetric_law: ``"quadratic"`` or ``"compaction"`` (module docstring, "Options").
	    f_max: :math:`f_{max}` (Pa).
	    lmbda_opt: :math:`\lambda_{opt}`.
	    width_fl: :math:`W`.
	    use_force_length: Gaussian :math:`f_{FL}(\lambda_a)`, else 1.
	    k_e: SEE stiffness :math:`k_e` (Pa).
	    k_p: PEE stiffness :math:`k_p` (Pa).
	    hill_a: :math:`A`.
	    hill_b: :math:`b` (1/s).
	    f_ecc: :math:`f_{ecc}`.
	    eta_0: :math:`\eta_0` (Pa.s).
	    see_variant: SEE law (``hill.functions.SeeVariant``).
	    k_2e: Exponent of the ``"exponential"`` SEE.
	    orientation_random_seed: Seed of the isotropic :math:`(\theta, \varphi)` draw.
	    remodel_tau: Remodelling time :math:`\tau` (s).
	    permeability: :math:`k_0` (m^2).
	    viscosity_pa_s: :math:`\eta` (Pa.s).
	    permeability_law: :math:`k(\phi_\ell)` law (module docstring, "Options").
	    holmes_mow_exponent: :math:`\kappa_{HM}`.
	    holmes_mow_stiffening: :math:`M`.
	    compaction_exponent: :math:`n \ge 1` of ``"compaction_power"``.
	    rho_s: :math:`\rho_s` (kg/m^3), haptotaxis weight and :math:`\rho_0`.
	    pillar_shear_modulus: :math:`\mu_{pil}` (Pa).
	    pillar_kappa: :math:`\kappa_{pil}` (Pa).
	    pillar_permeability_ratio: :math:`r_k = k_{pil}/k_0`, in :math:`(0, 1]`.
	    pillar_phi_s0: :math:`\Phi_{s,0}` of the pillars in :math:`R_s` (1: full solid).
	    phi_s0: :math:`\Phi_{s,0}` of the gel.
	    phi_c_init: :math:`\Phi_{c,0}` of the gel.
	    phi_c_max: :math:`\Phi_{c,max}`.
	    cellular_diffusion: :math:`D_\phi` (m^2/s).
	    hapt_coeff: :math:`h_\phi` (m^5/(kg.s)).
	    prol_coeff: :math:`r` (1/s).
	    displacement_degree: Degree of :math:`\mathbf u`, 2 (:math:`p, \Phi_c, \tilde\phi_s` in :math:`P_1`).
	    steps: Ramp then hold (s).
	    activation_configs: :math:`\alpha(t)` per step.
	    sol_tol: Newton tolerance, the same for every (dimensionless) unknown.
	    solver_parameters: ``NonlinearSolver`` parameters (``sol_tol`` added at run time).
	    integrator_parameters: ``TimeIntegrator`` adaptive-``dt`` parameters.
	    xdmf_write_period: Output decimation (``DecimatedTimeIntegrator``): ``int`` (substeps) or ``TimeInterval``
	        (s), one value or one per step (default: every substep of the ramp, then every 840 s).
	    xdmf_si_only: XDMF holds the SI FOIs and the unknowns already dimensionless
	        (:math:`\theta, \varphi, \Phi_c, \tilde\phi_s, \varepsilon_a`);
	        no :math:`\hat{\mathbf u}, \hat p` nor ``*_old``.
	    stagnation_plateau_relative_error: Plateau-exit threshold.
	    project_prefix: Output folder.
	    stem_name: Output file stem.
	    stem_mesh_name: Mesh file stem.
	"""

	# --- Geometry (MicrotissuePillarsGeometry) ---
	side_length_x: float = 250e-6
	core_length_x: float = 375e-6
	width_y: float = 875e-6
	gel_thickness_z: float = 50e-6
	pillar_radius: float = 37.5e-6
	pillar_lid_radius: float = 62.5e-6
	lid_thickness: float = 10e-6
	pillar_height: float = 200e-6
	pillar_edge_margin: float = 110e-6
	epsilon: float = 1e-6
	create_lids: bool = True

	# --- Mesh ---
	mesh_min_size: float = lid_thickness / 2
	mesh_max_size: float = gel_thickness_z / 3
	mesh_refine_min_size: float = lid_thickness / 2
	mesh_refine_dist_min: float = 0.0
	mesh_refine_dist_max: float = pillar_lid_radius
	regenerate_mesh: bool = True
	quadrature_degree: int = 4
	quadrature_degrees: Dict[str, int] = field(
		default_factory=dict
	)  # {"liquid_pressure": 2, "cell_fraction": 2, "substrate_smoothed": 2, "theta": 2, "phi": 2, "eps_a": 2}
	foi_degree: int = 0

	# --- Switches ---
	cell_transport: bool = True
	remodelling: bool = True

	# --- Gel skeleton ---
	gel_shear_modulus: float = 1700.0
	gel_kappa: float = 1417.0
	volumetric_law: VolumetricLaw = "compaction"

	# --- Hill chain ---
	f_max: float = 20e3
	lmbda_opt: float = 1.05
	width_fl: float = 0.3
	use_force_length: bool = True
	k_e: float = 200e3
	k_p: float = 5e3
	hill_a: float = 0.25
	hill_b: float = 0.05
	f_ecc: float = 1.8
	eta_0: float = 0.0
	see_variant: SeeVariant = "mooney"
	k_2e: float = 0.0

	# --- Orientation ---
	orientation_random_seed: int = 0
	remodel_tau: float = 3600.0

	# --- Poromechanics ---
	permeability: float = 1.8e-15
	viscosity_pa_s: float = 1.0e-3
	permeability_law: PermeabilityLaw = "holmes_mow"
	holmes_mow_exponent: float = 2.0
	holmes_mow_stiffening: float = 0.0
	compaction_exponent: float = 2.0
	rho_s: float = 1060.0

	# --- Pillars ---
	pillar_shear_modulus: float = 385.0e3
	pillar_kappa: float = 833.0e3
	pillar_permeability_ratio: float = 1.0 / 100.0
	pillar_phi_s0: float = 1.0

	# --- Volume fractions ---
	phi_s0: float = 0.25
	phi_c_init: float = 0.05
	phi_c_max: float = 0.1

	# --- Cell transport ---
	cellular_diffusion: float = 4e-3 * 1e-6 / 86400.0  # 4e-3 mm^2/day
	hapt_coeff: float = 10 * (4e-3 * 1e-6 / 86400.0) / 1060.0
	prol_coeff: float = 0.1 / 86400.0  # 0.1 /day

	# --- Spaces ---
	displacement_degree: int = 2

	# --- Time ---
	steps: Sequence[io.Step] = field(
		default_factory=lambda: (
			io.Step(duration=10.0, init_dt=1.0, min_dt=0.25),  # ramp, dt ~ tau_a
			io.Step(duration=42000.0, init_dt=100.0, min_dt=1.0, max_dt=3000.0, min_simu_seconds=3 * 3600),
		)
	)
	activation_configs: Sequence[ActivePowerConfig] = field(
		default_factory=lambda: [
			ActivePowerConfig.ramp(k_step=0, ini=0.0, fin=1.0),
			ActivePowerConfig.constant(k_step=1, value=1.0),
		]
	)

	# --- Solver ---
	sol_tol: float = 1e-6
	solver_parameters: Dict[str, Any] = field(default_factory=lambda: {"n_iter_max": 12})
	integrator_parameters: Dict[str, Any] = field(
		default_factory=lambda: {
			"n_iter_for_accel": 3,
			"n_iter_for_decel": 6,
			"accel_coeff": 1.25,
			"decel_coeff": 1.25,
		}
	)
	xdmf_write_period: Union[int, integration.TimeInterval, Sequence[Union[int, integration.TimeInterval]]] = (
		1,
		integration.TimeInterval(840.0),
	)
	stagnation_plateau_relative_error: float = 1e-5
	xdmf_si_only: bool = True

	# --- Output ---
	project_prefix: Path = Path("./outputs_pillars_biaxial")
	stem_name: str = "pillars_biaxial"
	stem_mesh_name: str = "pillars_biaxial_mesh"

	def __post_init__(self) -> None:
		r"""Check the saturation, the uniqueness of the CE equilibrium and the pillar inputs.

		Raises:
		    ValueError: :math:`\Phi_{s,0} + \Phi_{c,0} \ge 1`, uniqueness condition
		        :math:`k_e + k_p > f_{max} \max_\lambda |f_{FL}'(\lambda)\,\lambda|` violated, a value outside its
		        ``Literal`` (``permeability_law``, ``volumetric_law``, ``see_variant``), ``displacement_degree != 2``
		        (initial :math:`\Phi_c` built on :math:`P_1`), :math:`r_k \notin (0, 1]` or
		        ``pillar_phi_s0`` :math:`\notin [0, 1]`.
		"""
		for name, choices in _CHOICES.items():
			value = getattr(self, name)
			if value not in get_args(choices):
				raise ValueError(f"Unknown {name} '{value}', expected one of {get_args(choices)}.")
		solid_jacobian = saturated_mixture.compaction_jacobian(self.phi_s0, self.phi_c_init)
		if solid_jacobian >= 1.0:
			raise ValueError(f"J_min = phi_s0 + phi_c_init = {solid_jacobian} must be < 1.")
		if self.use_force_length:
			hill_law.check_uniqueness(self.k_e, self.k_p, self.f_max, self.lmbda_opt, self.width_fl)
		if self.displacement_degree != 2:
			raise ValueError(f"displacement_degree must be 2 (Phi_c in P1), got {self.displacement_degree}.")
		if not 0.0 < self.pillar_permeability_ratio <= 1.0:
			raise ValueError(f"pillar_permeability_ratio must be in (0, 1], got {self.pillar_permeability_ratio}.")
		if not 0.0 <= self.pillar_phi_s0 <= 1.0:
			raise ValueError(f"pillar_phi_s0 must be in [0, 1], got {self.pillar_phi_s0}.")

	@property
	def hydraulic_conductivity(self) -> float:
		r""":math:`k_0/\eta` (m^2/(Pa.s))."""
		return self.permeability / self.viscosity_pa_s

	@property
	def scales(self) -> CharacteristicScales:
		r"""Base scales :math:`L_0 = T`, :math:`\mu_0 = \mu_{gel}`, :math:`\rho_0 = \rho_s`.

		:math:`t_0 = L_0^2\eta/(k\kappa_{gel})` (consolidation time over the gel thickness).
		"""
		return CharacteristicScales.from_consolidation_scale(
			length=self.gel_thickness_z,
			stress=self.gel_shear_modulus,
			density=self.rho_s,
			permeability=self.permeability,
			viscosity=self.viscosity_pa_s,
			consolidation_modulus=self.gel_kappa,
		)

	@property
	def core_center_xy(self) -> Tuple[float, float]:
		"""In-plane centre of the core patch (m)."""
		return self.side_length_x + self.core_length_x / 2.0, self.width_y / 2.0

	@property
	def probes(self) -> Dict[str, Tuple[float, float, float]]:
		"""Physical probe points (m), core patch: centre, top-face centre, front face (:math:`y = 0`)."""
		x, y = self.core_center_xy
		t = self.gel_thickness_z
		return {"center": (x, y, t / 2.0), "top": (x, y, t), "front": (x, 0.0, t / 2.0)}


@dataclass(frozen=True)
class PillarFields:
	r"""Data fields of the gel/pillar split.

	Attributes:
	    gel: :math:`\chi_{gel}` (DG0).
	    phi_s0: :math:`\Phi_{s,0}` (gel ``phi_s0``, pillars ``pillar_phi_s0``).
	    phi_c0: Initial :math:`\Phi_c` (CG1): :math:`\Phi_{c,0}(1 - \chi^{int}_{pil})`, :math:`\chi^{int}_{pil} = 1`
	        strictly inside the pillars.
	"""

	gel: dolfin.Function
	phi_s0: Any
	phi_c0: dolfin.Function


UNKNOWNS: Tuple[str, ...] = (
	"displacement",
	"theta",
	"phi",
	"liquid_pressure",
	"cell_fraction",
	"substrate_smoothed",
	"eps_a",
)
"""Every possible subsol (= residual row) of the demo, whatever the switches."""


def build_residual_rows(problem: dolfin_mech.problems.Problem, user_inputs: UserInputs) -> ResidualRows:
	"""Quadrature degree per row (subsol of the test function), for the rows present in ``problem``.

	Args:
	    problem: Problem with its subsols.
	    user_inputs: Configuration.

	Returns:
	    The row settings.

	Raises:
	    ValueError: A ``quadrature_degrees`` key is not one of :data:`UNKNOWNS` (typo).
	"""
	unknown = sorted(set(user_inputs.quadrature_degrees) - set(UNKNOWNS))
	if unknown:
		raise ValueError(f"quadrature_degrees: unknown rows {unknown}, expected names among {UNKNOWNS}.")
	present = {subsol.name for subsol in problem.subsols}
	return ResidualRows.from_problem(
		problem,
		quadrature_degrees={k: v for k, v in user_inputs.quadrature_degrees.items() if k in present},
	)


def build_geometry(user_inputs: UserInputs) -> MicrotissuePillarsGeometry:
	"""Three-patch geometry with twelve pillars.

	Args:
	    user_inputs: Configuration.

	Returns:
	    The geometry, SI.
	"""
	return MicrotissuePillarsGeometry(
		side_length_x=user_inputs.side_length_x,
		core_length_x=user_inputs.core_length_x,
		width_y=user_inputs.width_y,
		gel_thickness_z=user_inputs.gel_thickness_z,
		pillar_radius=user_inputs.pillar_radius,
		pillar_lid_radius=user_inputs.pillar_lid_radius,
		lid_thickness=user_inputs.lid_thickness,
		pillar_height=user_inputs.pillar_height,
		pillar_edge_margin=user_inputs.pillar_edge_margin,
		epsilon=user_inputs.epsilon,
		create_lids=user_inputs.create_lids,
	)


def build_mesh(user_inputs: UserInputs) -> io.SimulationPaths:
	"""Generate the mesh with Gmsh on rank 0 (unless it exists), refined at the lid tops, and convert it to XDMF.

	Gmsh options of the fenicsoft pillar demo: Netgen optimizer off (segfault on this geometry), geometry tolerance
	:math:`10^{-6}`, 2D algorithm MeshAdapt (periodic shaft surfaces).

	Args:
	    user_inputs: Configuration.

	Returns:
	    Output-folder layout.
	"""
	simulation_paths = io.configure_environment(user_inputs.project_prefix, user_inputs.stem_name)
	mesh_file_path = simulation_paths.mesh_folder / f"{user_inputs.stem_mesh_name}.msh"

	if not mesh_file_path.exists() or user_inputs.regenerate_mesh:
		if rank == 0:
			begin_t = time.time()
			refinements = [
				RefinementZone(
					marker=marker,
					min_size=user_inputs.mesh_refine_min_size,
					dist_min=user_inputs.mesh_refine_dist_min,
					dist_max=user_inputs.mesh_refine_dist_max,
				)
				for marker in _LID_TOPS
			]
			mesh_options = MeshOptions(
				min_size=user_inputs.mesh_min_size,
				max_size=user_inputs.mesh_max_size,
				refinements=refinements,
				extra_opts={
					GmshKeys.TERMINAL: 0,
					GmshKeys.OPTIMIZE: 1,
					GmshKeys.SMOOTHING: 10,
					GmshKeys.GEOMETRY_TOLERANCE: 1e-6,
					GmshKeys.ALGORITHM_2D: 1,
				},
			)
			geometry = build_geometry(user_inputs)
			with MicrotissuePillarsMeshGenerator(geometry, mesh_options, simulation_paths.mesh_folder) as generator:
				msh_file = generator.generate(mesh_stem_name=user_inputs.stem_mesh_name)
			io.convert_msh_to_xdmf(msh_file, prune_z=False)
			logging.getLogger(__name__).info(f"Mesh creation completed in {time.time() - begin_t:.2f}s.")
		comm.Barrier()

	return simulation_paths


def build_pillar_fields(
	mesh: dolfin.Mesh, domains_mf: dolfin.MeshFunction, region: PillarRegion, user_inputs: UserInputs
) -> PillarFields:
	r"""Indicators, :math:`\Phi_{s,0}` and initial :math:`\Phi_c` of the gel/pillar split (``domain_fields``).

	Args:
	    mesh: Mesh.
	    domains_mf: Cell tags.
	    region: Tags of the stage.
	    user_inputs: Configuration.

	Returns:
	    The fields.
	"""
	gel = domain_fields.cell_indicator(mesh, domains_mf, region.gel, name="gel_indicator")
	pillar_interior = domain_fields.interior_vertex_indicator(mesh, domains_mf, region.pillars)
	phi_c0 = dolfin.Function(pillar_interior.function_space(), name="cell_fraction_initial")
	phi_c0.vector()[:] = user_inputs.phi_c_init * (1.0 - pillar_interior.vector().get_local())
	phi_s0 = user_inputs.pillar_phi_s0 + (user_inputs.phi_s0 - user_inputs.pillar_phi_s0) * gel
	return PillarFields(gel=gel, phi_s0=phi_s0, phi_c0=phi_c0)


def initialise_problem(
	mesh: dolfin.Mesh,
	domains_mf: dolfin.MeshFunction,
	boundaries_mf: dolfin.MeshFunction,
	user_inputs: UserInputs,
	coefficients: Coefficients,
	region: PillarRegion,
	initial_fields: Optional[Mapping[str, dolfin.Function]] = None,
) -> Tuple[dolfin_mech.problems.Problem, PillarFields]:
	r"""Register the unknowns (module docstring) and build the data fields.

	Sets on ``problem``: the subsols, ``theta_func``/``phi_func`` (subsol or frozen draw),
	``theta_initial``/``phi_initial``, ``phic_func`` (subsol or frozen initial :math:`\Phi_c`),
	``phi_s0`` (gel/pillar field),
	``kinematics``, ``permeability`` (``permeability_law``), ``k_l_tensor``
	(:math:`\mathbf K_L = (k/\eta) J \mathbf C^{-1}`) and ``sol_tol`` (one entry per subsol, registration order).

	Args:
	    mesh: Mesh (rescaled when nondimensionalised).
	    domains_mf: Cell tags.
	    boundaries_mf: Facet tags.
	    user_inputs: Configuration.
	    coefficients: Dimensionless coefficients (steps, :math:`k_0/\eta`).
	    region: Tags of the stage.
	    initial_fields: Initial value per subsol name, on the subsol's own space (stage B); ``theta``/``phi``
	        also replace the draw and ``cell_fraction`` the frozen field. Default: module docstring.

	Returns:
	    ``(problem, fields)``: problem with its function spaces, kinematics and steps; gel/pillar fields.
	"""
	fields = build_pillar_fields(mesh, domains_mf, region, user_inputs)
	initial = {"cell_fraction": fields.phi_c0, **(initial_fields or {})}
	problem = dolfin_mech.problems.Problem()
	problem.set_mesh(mesh=mesh, define_facet_normals=True)
	problem.set_measures(domains=domains_mf, boundaries=boundaries_mf, points=None)
	sol_tol: List[float] = []
	lower_degree = user_inputs.displacement_degree - 1

	def scalar(name: str, family: str, degree: int, init_val: float) -> Any:
		if name in initial:
			return problem.add_scalar_subsol(name=name, family=family, degree=degree, init_fun=initial[name])
		return problem.add_scalar_subsol(name=name, family=family, degree=degree, init_val=init_val)

	displacement_fe = dolfin.VectorElement("CG", mesh.ufl_cell(), user_inputs.displacement_degree)
	problem.displacement_subsol = problem.add_subsol(
		name="displacement", fe=displacement_fe, init_fun=initial.get("displacement")
	)
	sol_tol.append(user_inputs.sol_tol)

	if "theta" in initial and "phi" in initial:
		theta_init, phi_init = initial["theta"], initial["phi"]
	else:
		theta_init, phi_init = spherical_mod.sample_isotropic_theta_phi_dg0(
			mesh, seed=user_inputs.orientation_random_seed, family="DG", degree=0
		)
	problem.theta_initial = theta_init.copy(deepcopy=True)
	problem.theta_initial.rename("theta_initial", "theta_initial")
	problem.phi_initial = phi_init.copy(deepcopy=True)
	problem.phi_initial.rename("phi_initial", "phi_initial")
	if user_inputs.remodelling:
		problem.theta_subsol = problem.add_scalar_subsol(name="theta", family="DG", degree=0, init_fun=theta_init)
		problem.phi_subsol = problem.add_scalar_subsol(name="phi", family="DG", degree=0, init_fun=phi_init)
		sol_tol += [user_inputs.sol_tol, user_inputs.sol_tol]

	problem.pl_subsol = scalar("liquid_pressure", "CG", lower_degree, 0.0)
	sol_tol.append(user_inputs.sol_tol)

	problem.phi_s0 = fields.phi_s0
	if user_inputs.cell_transport:
		problem.phic_subsol = scalar("cell_fraction", "CG", lower_degree, user_inputs.phi_c_init)
		problem.phis_subsol = scalar("substrate_smoothed", "CG", lower_degree, user_inputs.phi_s0)
		sol_tol += [user_inputs.sol_tol, user_inputs.sol_tol]

	problem.eps_a_subsol = scalar("eps_a", "DG", 0, 0.0)
	sol_tol.append(user_inputs.sol_tol)
	problem.sol_tol = sol_tol

	problem.set_solution_finite_element()
	problem.set_solution_function_space()
	problem.set_solution_functions()
	problem.set_quadrature_degree(user_inputs.quadrature_degree)
	problem.set_foi_finite_elements_DG(user_inputs.foi_degree)
	problem.set_foi_function_spaces()

	problem.theta_func = problem.theta_subsol.subfunc if user_inputs.remodelling else theta_init
	problem.phi_func = problem.phi_subsol.subfunc if user_inputs.remodelling else phi_init
	problem.phic_func = problem.phic_subsol.subfunc if user_inputs.cell_transport else initial["cell_fraction"]

	# Mesh in X_hat = X/L0 and U = u/L0: grad_Xhat(U) = grad_X(u), so plain Kinematics is exact.
	problem.kinematics = dolfin_mech.kinematics.Kinematics(
		U=problem.displacement_subsol.subfunc, U_old=problem.displacement_subsol.func_old, Q_expr=problem.Q_expr
	)

	for step_idx, step in enumerate(coefficients.steps):
		problem.add_step(Deltat=step.duration, dt_ini=step.init_dt, dt_min=step.min_dt, dt_max=step.max_dt)
		if step.min_simu_seconds is not None:
			problem.steps[step_idx].min_simu_seconds = step.min_simu_seconds

	problem.permeability = build_permeability(problem, user_inputs, coefficients)
	problem.k_l_tensor = problem.permeability.tensor
	return problem, fields


def build_permeability(
	problem: dolfin_mech.problems.Problem, user_inputs: UserInputs, coefficients: Coefficients
) -> Any:
	r""":math:`k(\phi_\ell)/\eta` law of the gel, ``user_inputs.permeability_law`` (module docstring, "Options").

	Args:
	    problem: Problem with ``kinematics``, ``phic_func``.
	    user_inputs: Configuration.
	    coefficients: :math:`k_0/\eta` dimensionless.

	Returns:
	    Law exposing ``ratio``, ``hydraulic_conductivity`` and ``tensor``.
	"""
	kin, law = problem.kinematics, user_inputs.permeability_law
	if law == "constant":
		return perm.constant.Permeability(kin, perm.constant.Parameters(conductivity_0=coefficients.conductivity))
	common: Dict[str, Any] = dict(
		conductivity_0=coefficients.conductivity,
		phi_s0=user_inputs.phi_s0,  # gel value: the law is integrated on the gel
		phi_c=problem.phic_func,
		phi_c0=user_inputs.phi_c_init,
	)
	if law == "kozeny_carman":
		return perm.kozeny_carman.Permeability(kin, perm.kozeny_carman.Parameters(**common))
	if law == "holmes_mow":
		params = perm.holmes_mow.Parameters(
			**common, exponent=user_inputs.holmes_mow_exponent, stiffening=user_inputs.holmes_mow_stiffening
		)
		return perm.holmes_mow.Permeability(kin, params)
	if law == "compaction_power":
		params = perm.compaction_power.Parameters(**common, exponent=user_inputs.compaction_exponent)
		return perm.compaction_power.Permeability(kin, params)
	raise ValueError(f"Unknown permeability_law '{law}'.")


def build_materials(
	problem: dolfin_mech.problems.Problem, user_inputs: UserInputs, coefficients: Coefficients
) -> Tuple[Any, hill_law.Constitutive]:
	r"""Gel law :math:`\mathbf{P}_S + \mathbf{P}_h` (Neo-Hooke + ``volumetric_law`` + Hill chain).

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.
	    coefficients: Moduli dimensionless.

	Returns:
	    ``(gel_material, hill_material)``; the second owns ``p_act`` (:math:`\alpha`).
	"""
	kin = problem.kinematics
	isochoric = fenicsoft.materials.iso.neo_hooke.Constitutive(
		kinematics=kin,
		parameters=fenicsoft.materials.iso.neo_hooke.Parameters.from_shear_modulus(mu=coefficients.mu),
		name="gel_isochoric",
	)
	if user_inputs.volumetric_law == "compaction":
		volumetric: Any = compaction.CompactionVolumetricPenalty(
			kin,
			compaction.Parameters(
				kappa=coefficients.kappa,
				solid_jacobian=saturated_mixture.compaction_jacobian(user_inputs.phi_s0, problem.phic_func),
			),
			name="gel_volumetric",
		)
	elif user_inputs.volumetric_law == "quadratic":
		volumetric = fenicsoft.materials.volumetric.QuadraticVolumetricPenalty(
			kinematics=kin,
			parameters=fenicsoft.materials.volumetric.VolumetricPenaltyParameters(kappa=coefficients.kappa),
			name="gel_volumetric",
		)
	else:
		raise ValueError(f"Unknown volumetric_law '{user_inputs.volumetric_law}'.")
	hill_params = hill_law.Parameters(
		phi_c=problem.phic_func,
		f0=spherical_mod.f0_from_theta_phi(problem.theta_func, problem.phi_func),
		activation_configs=user_inputs.activation_configs,
		f_max=coefficients.f_max,
		simulation_steps=problem.steps,
		use_force_length=user_inputs.use_force_length,
		lmbda_opt=user_inputs.lmbda_opt,
		W=user_inputs.width_fl,
		eps_a=problem.eps_a_subsol.subfunc,
		eps_a_old=problem.eps_a_subsol.func_old,
		k_e=coefficients.k_e,
		k_p=coefficients.k_p,
		hill_a=user_inputs.hill_a,
		hill_b=coefficients.hill_b,
		f_ecc=user_inputs.f_ecc,
		eta_0=coefficients.eta_0,
		see_variant=user_inputs.see_variant,
		k_2e=user_inputs.k_2e,
	)
	hill_material = hill_law.Constitutive(kin, hill_params, name="cell_hill_chain")
	return isochoric + volumetric + hill_material, hill_material


def build_step_operators(
	problem: dolfin_mech.problems.Problem,
	user_inputs: UserInputs,
	hill_material: hill_law.Constitutive,
	coefficients: Coefficients,
	rows: ResidualRows,
	gel_dx: dolfin.Measure,
) -> Tuple[List[Any], ContractileEvolution]:
	r"""Gel operators shared by every step; each integrates with the quadrature degree of its row.

	On ``gel_dx``: :math:`R_a`, the Darcy term with ``problem.k_l_tensor``, the cell flux, rate and proliferation.
	On the whole mesh: :math:`-Jp\mathbf F^{-T}`, :math:`(J - J^n)/\Delta t`, :math:`R_s`, :math:`R_{\theta,\varphi}`.

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.
	    hill_material: Hill law (owner of ``p_act``).
	    coefficients: Dimensionless coefficients.
	    rows: Quadrature degree per row.
	    gel_dx: Gel measure.

	Returns:
	    ``(operators, contractile_evolution)``.
	"""
	kin = problem.kinematics
	p_hat, dp = problem.pl_subsol.subfunc, problem.pl_subsol.dsubtest
	c = coefficients

	def dv(row: str) -> dolfin.Measure:
		return rows.measure(problem.dV, row)

	def dv_gel(row: str) -> dolfin.Measure:
		return rows.measure(gel_dx, row)

	contractile = ContractileEvolution(
		law=hill_material,
		test=problem.eps_a_subsol.dsubtest,
		dx=dv_gel("eps_a"),
	)
	ops: List[Any] = [
		fenicsoft.operators.fluid_stress.biphasic.pk1.FluidStress(
			kinematics=kin,
			function=p_hat,
			displacement=problem.displacement_subsol.subfunc,
			test=problem.displacement_subsol.dsubtest,
			measure=dv("displacement"),
		),
		fenicsoft.operators.DarcyGradient(
			function=p_hat,
			test=dp,
			conductivity=problem.k_l_tensor,
			dx=dv_gel("liquid_pressure"),
		),
		fenicsoft.operators.ChangeRate(
			function=kin.J,
			function_old=kin.J_old,
			test=dp,
			dx=dv("liquid_pressure"),
		),
		contractile,
	]

	if user_inputs.cell_transport:
		phi_c, dc = problem.phic_subsol.subfunc, problem.phic_subsol.dsubtest
		phi_s = problem.phis_subsol.subfunc

		def hapt(test: Any, row: str) -> Any:
			return cell_ops.SaturatedHaptotaxisFluxSmooth(
				phi_c, phi_s, c.rho_s, problem, c.hapt_coeff, user_inputs.phi_c_max, test, dv_gel(row)
			)

		def diff(test: Any, row: str) -> Any:
			return cell_ops.IsotropicDiffusion(phi_c, problem, c.cellular_diffusion, test, dv_gel(row))

		ops += [
			hapt(dp, "liquid_pressure"),
			diff(dp, "liquid_pressure"),
			fenicsoft.operators.ChangeRate(
				function=phi_c,
				function_old=problem.phic_subsol.func_old,
				test=dc,
				dx=dv_gel("cell_fraction"),
			),
			SaturatedProliferation(
				phi_c,
				c.prol_coeff,
				user_inputs.phi_c_max,
				dc,
				dv_gel("cell_fraction"),
			),
			hapt(dc, "cell_fraction"),
			diff(dc, "cell_fraction"),
			cell_ops.SubstrateProjection(
				phi_s_func=phi_s,
				test_phi_s=problem.phis_subsol.dsubtest,
				kin_j=kin.J,
				phi_s0=problem.phi_s0,
				measure=dv("substrate_smoothed"),
			),
		]

	if user_inputs.remodelling:
		theta, phi = problem.theta_subsol, problem.phi_subsol
		builders: Tuple[Tuple[Any, Callable[[Any], Any]], ...] = (
			(theta, lambda th: spherical_mod.f0_from_theta_phi(th, phi.subfunc)),
			(phi, lambda ph: spherical_mod.f0_from_theta_phi(theta.subfunc, ph)),
		)
		ops += [
			OrientationChange(
				angle=subsol.subfunc,
				angle_old=subsol.func_old,
				test=subsol.dsubtest,
				kinematics=kin,
				tau=c.remodel_tau,
				activation=hill_material.p_act,
				dx=dv(subsol.name),
				f0_builder=builder,
			)
			for subsol, builder in builders
		]
	return ops, contractile


def build_pillar_material(problem: dolfin_mech.problems.Problem, coefficients: Coefficients) -> Any:
	r""":math:`\mathbf P_S^{pil}`: Neo-Hooke + quadratic volumetric penalty (pillar moduli).

	Args:
	    problem: Problem with ``kinematics``.
	    coefficients: :math:`\hat\mu_{pil}, \hat\kappa_{pil}`.

	Returns:
	    Pillar law.
	"""
	kin = problem.kinematics
	isochoric = fenicsoft.materials.iso.neo_hooke.Constitutive(
		kinematics=kin,
		parameters=fenicsoft.materials.iso.neo_hooke.Parameters.from_shear_modulus(mu=coefficients.pillar_mu),
		name="pillar_isochoric",
	)
	volumetric = fenicsoft.materials.volumetric.QuadraticVolumetricPenalty(
		kinematics=kin,
		parameters=fenicsoft.materials.volumetric.VolumetricPenaltyParameters(kappa=coefficients.pillar_kappa),
		name="pillar_volumetric",
	)
	return isochoric + volumetric


def build_pillar_operators(
	problem: dolfin_mech.problems.Problem,
	user_inputs: UserInputs,
	coefficients: Coefficients,
	rows: Any,
	pillar_dx: dolfin.Measure,
) -> List[Any]:
	r"""Pillar terms: Darcy with :math:`\mathbf K_L^{pil}`, :math:`\Phi_c` and :math:`\varepsilon_a` frozen.

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    user_inputs: Configuration.
	    coefficients: Dimensionless coefficients.
	    rows: ``ResidualRows``.
	    pillar_dx: Pillar measure.

	Returns:
	    The operators.
	"""
	conductivity = user_inputs.pillar_permeability_ratio * coefficients.conductivity
	pillar_law = perm.constant.Permeability(problem.kinematics, perm.constant.Parameters(conductivity_0=conductivity))
	frozen = [problem.eps_a_subsol] + ([problem.phic_subsol] if user_inputs.cell_transport else [])
	return [
		fenicsoft.operators.DarcyGradient(
			function=problem.pl_subsol.subfunc,
			test=problem.pl_subsol.dsubtest,
			conductivity=pillar_law.tensor,
			dx=rows.measure(pillar_dx, "liquid_pressure"),
		),
	] + [
		fenicsoft.operators.ChangeRate(
			function=subsol.subfunc,
			function_old=subsol.func_old,
			test=subsol.dsubtest,
			dx=rows.measure(pillar_dx, subsol.name),
		)
		for subsol in frozen
	]


def set_weak_form_and_bcs(
	problem: dolfin_mech.problems.Problem,
	boundaries_mf: dolfin.MeshFunction,
	user_inputs: UserInputs,
	coefficients: Coefficients,
	region: PillarRegion,
	fields: PillarFields,
) -> dolfin_mech.problems.Problem:
	"""Register the residuals (module docstring), the BCs of ``region``, the FOIs and QOIs (``pillars_outputs``).

	Args:
	    problem: Problem from :func:`initialise_problem`.
	    boundaries_mf: Facet tags.
	    user_inputs: Configuration.
	    coefficients: Dimensionless coefficients.
	    region: Tags of the stage.
	    fields: Gel/pillar fields.

	Returns:
	    The assembled problem.
	"""
	gel_dx, pillar_dx = problem.dV(region.gel), problem.dV(region.pillars)
	rows = build_residual_rows(problem, user_inputs)
	gel_material, hill_material = build_materials(problem, user_inputs, coefficients)
	pillar_material = build_pillar_material(problem, coefficients)
	shared_ops, contractile = build_step_operators(problem, user_inputs, hill_material, coefficients, rows, gel_dx)
	shared_ops += build_pillar_operators(problem, user_inputs, coefficients, rows, pillar_dx)

	for step_idx in range(len(problem.steps)):
		for material, dx in ((gel_material, gel_dx), (pillar_material, pillar_dx)):
			problem.add_operator(
				fenicsoft.operators.hyper_elasticity.pk1.HyperElasticity(
					disp=problem.displacement_subsol.subfunc,
					disp_test=problem.displacement_subsol.dsubtest,
					material=material,
					measure=rows.measure(dx, "displacement"),
					k_step=step_idx,
				),
				k_step=step_idx,
			)
		for operator in shared_ops:
			problem.add_operator(operator=operator, k_step=step_idx)
		for marker in region.clamped:
			problem.add_constraint(
				V=problem.displacement_subsol.fs,
				sub_domains=boundaries_mf,
				sub_domain_id=marker,
				val=[0.0, 0.0, 0.0],
				k_step=step_idx,
			)
		for marker in region.drained:
			problem.add_constraint(
				V=problem.pl_subsol.fs, sub_domains=boundaries_mf, sub_domain_id=marker, val=0.0, k_step=step_idx
			)

	solid_cauchy = fields.gel * gel_material.cauchy + (1.0 - fields.gel) * pillar_material.cauchy
	add_outputs(problem, user_inputs, hill_material, contractile, gel_dx, solid_cauchy, fields.gel)
	return problem


def apply_initial_bcs(problem: dolfin_mech.problems.Problem) -> None:
	"""Apply every Dirichlet BC (global, then step 0) to the initial solution vector.

	Args:
	    problem: Assembled problem.
	"""
	for constraint in problem.constraints:
		constraint.bc.apply(problem.sol_func.vector())
	if len(problem.steps) > 0:
		for constraint in problem.steps[0].constraints:
			constraint.bc.apply(problem.sol_func.vector())


def solve_problem(
	problem: dolfin_mech.problems.Problem,
	simulation_paths: io.SimulationPaths,
	user_inputs: UserInputs,
	coefficients: Coefficients,
) -> bool:
	"""Newton + adaptive time stepping (``integration.DecimatedTimeIntegrator``: selected fields, rank-0 logs).

	Args:
	    problem: Assembled problem.
	    simulation_paths: Output layout.
	    user_inputs: Configuration.
	    coefficients: Write period dimensionless.

	Returns:
	    True if every step completed (at :math:`t_{fin}` or by plateau exit).
	"""
	written = None
	if user_inputs.xdmf_si_only:
		# Dimensionless unknowns as solved (theta, phi, Phi_c, phi_s, eps_a) + SI FOIs; no u_hat, p_hat, *_old.
		raw = {"displacement", "liquid_pressure"}
		written = [s.func for s in problem.subsols if s.name not in raw] + problem.get_fois_func_lst()
	integrator = integration.DecimatedTimeIntegrator(
		problem=problem,
		solver=dolfin_mech.core.NonlinearSolver(
			problem=problem, parameters={**user_inputs.solver_parameters, "sol_tol": problem.sol_tol}
		),
		parameters=user_inputs.integrator_parameters,
		print_out=str(simulation_paths.outputs_base_name),  # type: ignore
		print_sta=str(simulation_paths.outputs_base_name),  # type: ignore
		write_qois=str(simulation_paths.results_base_name),  # type: ignore
		write_sol=str(simulation_paths.results_base_name),  # type: ignore
		write_period=coefficients.xdmf_write_period,
		stagnation_plateau_relative_error=user_inputs.stagnation_plateau_relative_error,
		functions_to_write=written,
	)
	begin_t = time.time()
	success = integrator.integrate()
	comm.Barrier()
	if rank == 0:
		logging.getLogger(__name__).info(f"Solve: {timedelta(seconds=int(time.time() - begin_t))}, success={success}.")
	return success


def handover_functions(problem: dolfin_mech.problems.Problem) -> Dict[str, dolfin.Function]:
	"""Current value of each :data:`HANDOVER` field (subsol, or frozen field when switched off).

	Args:
	    problem: Problem.

	Returns:
	    Name -> function (dimensionless unknowns).
	"""
	subsols = {subsol.name: subsol.func for subsol in problem.subsols}
	frozen = {"theta": problem.theta_func, "phi": problem.phi_func, "cell_fraction": problem.phic_func}
	return {name: subsols.get(name, frozen.get(name)) for name in HANDOVER}  # type: ignore


def handover_scales(user_inputs: UserInputs) -> Dict[str, float]:
	r"""SI factor of each :data:`HANDOVER` field: :math:`L_0` (:math:`\mathbf u`), :math:`\mu_0` (:math:`p`), else 1.

	Args:
	    user_inputs: Configuration.

	Returns:
	    Name -> factor (SI value = factor times dimensionless value).
	"""
	s = user_inputs.scales
	return {name: {"displacement": s.length, "liquid_pressure": s.stress}.get(name, 1.0) for name in HANDOVER}


def handover_spaces(mesh: dolfin.Mesh, user_inputs: UserInputs) -> Dict[str, dolfin.FunctionSpace]:
	"""Function space of each :data:`HANDOVER` field (spaces of the unknowns).

	Args:
	    mesh: Mesh.
	    user_inputs: Configuration.

	Returns:
	    Name -> space.
	"""
	k = user_inputs.displacement_degree
	cg, dg0 = dolfin.FunctionSpace(mesh, "CG", k - 1), dolfin.FunctionSpace(mesh, "DG", 0)
	return {
		"displacement": dolfin.VectorFunctionSpace(mesh, "CG", k),
		"liquid_pressure": cg,
		"cell_fraction": cg,
		"theta": dg0,
		"phi": dg0,
		"eps_a": dg0,
	}


def write_handover(functions: Mapping[str, dolfin.Function], folder: Path, factors: Mapping[str, float]) -> None:
	"""Write ``factor * function`` per field to ``folder/<name>.xdmf`` (``io.write_function_checkpoint``).

	Args:
	    functions: Name -> function.
	    folder: Output folder.
	    factors: Name -> factor (SI conversion).
	"""
	for name, function in functions.items():
		copy = function.copy(deepcopy=True)
		copy.vector()[:] = factors[name] * function.vector().get_local()
		io.write_function_checkpoint(copy, folder / f"{name}.xdmf", name=name)


def read_handover(mesh: dolfin.Mesh, folder: Path, user_inputs: UserInputs) -> Dict[str, dolfin.Function]:
	"""Read every :data:`HANDOVER` field of ``folder`` on ``mesh`` (``io.read_function_checkpoint``).

	Args:
	    mesh: Mesh of the checkpoints.
	    folder: Checkpoint folder.
	    user_inputs: Configuration (spaces).

	Returns:
	    Name -> function, as written.

	Raises:
	    FileNotFoundError: A checkpoint is missing.
	"""
	functions = {}
	for name, space in handover_spaces(mesh, user_inputs).items():
		path = folder / f"{name}.xdmf"
		if not path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {path}.")
		functions[name] = dolfin.Function(space, name=name)
		io.read_function_checkpoint(functions[name], path, name=name)
	return functions


def run(
	user_inputs: UserInputs,
	simulation_paths: io.SimulationPaths,
	mesh: dolfin.Mesh,
	domains_mf: dolfin.MeshFunction,
	boundaries_mf: dolfin.MeshFunction,
	region: PillarRegion,
	initial_fields_si: Optional[Mapping[str, dolfin.Function]] = None,
) -> bool:
	"""Scale, assemble, solve, write the SI checkpoints if it completed (stages A and B).

	Args:
	    user_inputs: Configuration.
	    simulation_paths: Output layout.
	    mesh: SI mesh (scaled in place, restored at the end).
	    domains_mf: Cell tags.
	    boundaries_mf: Facet tags.
	    region: Tags of the stage.
	    initial_fields_si: SI initial value per subsol name (stage B); default: module docstring.

	Returns:
	    True if the simulation completed.
	"""
	scales = user_inputs.scales
	mesh.coordinates()[:] = scales.nondimensionalize_length(mesh.coordinates())  # X_hat = X / L0
	mesh.bounding_box_tree().build(mesh)
	initial = None
	if initial_fields_si is not None:
		factors = handover_scales(user_inputs)
		initial = {name: f.copy(deepcopy=True) for name, f in initial_fields_si.items()}
		for name, function in initial.items():
			function.vector()[:] = function.vector().get_local() / factors[name]

	coefficients = Coefficients.from_inputs(user_inputs)
	problem, fields = initialise_problem(
		mesh, domains_mf, boundaries_mf, user_inputs, coefficients, region, initial_fields=initial
	)
	problem = set_weak_form_and_bcs(problem, boundaries_mf, user_inputs, coefficients, region, fields)
	apply_initial_bcs(problem)
	success = solve_problem(problem, simulation_paths, user_inputs, coefficients)

	mesh.coordinates()[:] = scales.redimensionalize_length(mesh.coordinates())  # SI checkpoints
	mesh.bounding_box_tree().build(mesh)
	if success:
		write_handover(handover_functions(problem), simulation_paths.results_folder, handover_scales(user_inputs))
	if rank == 0:
		xdmf_units.redimensionalize_results(
			Path(f"{simulation_paths.results_base_name}.xdmf"), scales.length, scales.time
		)
	comm.Barrier()
	return success


def main(user_inputs: UserInputs = UserInputs()) -> bool:
	"""Stage A: mesh, scale, assemble, solve, checkpoint.

	Args:
	    user_inputs: Configuration.

	Returns:
	    True if the simulation completed.
	"""
	simulation_paths = build_mesh(user_inputs)
	mesh, domains_mf, boundaries_mf, _ = io.load_mesh_data(simulation_paths.mesh_folder / user_inputs.stem_mesh_name)
	return run(user_inputs, simulation_paths, mesh, domains_mf, boundaries_mf, STAGE_A)


if __name__ == "__main__":
	io.configure_logging(rank)
	if not main():
		comm.Abort(errorcode=1)
