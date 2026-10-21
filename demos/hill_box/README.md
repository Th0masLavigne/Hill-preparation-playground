# Hill box (uniaxial)

Active poro-elastic gel slab $875 \times 375 \times 50\ \mu\text{m}$ seeded with contractile cells
(Hill chain) of random isotropic orientation. The slab is clamped on its two $x$-faces, drained
($p = 0$) and traction-free on the four others. Under the activation $\alpha(t)$ (ramp 0 → 1, then hold)
the cells contract, the gel shortens across $x$ and expels liquid; the cells may also migrate,
proliferate and reorient.

```text
                     TOP (z = L_z)
             +---------------------------+
            /|      BACK (y = L_y)      /|
           / |                         / |
          +---------------------------+  |
 LEFT     |  |                        |  |     RIGHT
 x = 0    |  +------------------------|--+     x = L_x
 u = 0    | /                         | /      u = 0
          |/       FRONT (y = 0)      |/
          +---------------------------+
                   BOTTOM (z = 0)
   z  y
   | /        LEFT, RIGHT              : u = 0, impermeable
   |/___ x    FRONT, BACK, TOP, BOTTOM : p = 0 (drained), traction-free
              everywhere               : no cell flux
```

Unknowns: displacement $\mathbf u$, liquid pressure $p$, CE log-stretch $\varepsilon_a$, and optionally
the cell fraction $\Phi_c$ with the smoothed substrate fraction $\tilde\phi_s$ (`cell_transport`) and the
cell orientation angles $\theta, \varphi$ (`remodelling`). The full set of residuals is in the module
docstring of `hill_box_uniaxial.py`.

| Item | Where |
| --- | --- |
| Hill law ($R_u$), activation $\alpha(t)$ | `hill_playground.materials.cell.active.hill`, `...active.activation` |
| CE evolution ($R_a$) | `hill_playground.operators.cell.hill.ContractileEvolution` |
| Clipped proliferation | `hill_playground.operators.cell.SaturatedProliferation` |
| Scales ($L_0, \mu_0, t_0$), conversions | `hill_playground.scaling.CharacteristicScales` |
| $k(\phi_\ell)$, $\mathbf K_L = (k/\eta) J \mathbf C^{-1}$ | `hill_playground.materials.permeability` |
| Compaction volumetric energy | `hill_playground.materials.volumetric.compaction` |
| Orientation diagnostics | `hill_playground.materials.orientation.display` |
| Uniqueness check of the CE equilibrium | `hill_playground.materials.cell.active.hill.law.check_uniqueness` |

| Demo file | Content |
| --- | --- |
| `hill_box_uniaxial.py` | configuration (`UserInputs`), mesh, unknowns, laws, residuals, BCs, solve |
| `hill_box_scaling.py` | SI inputs → dimensionless coefficients (`Coefficients`) |
| `hill_box_outputs.py` | SI fields (XDMF) and QOIs (`.dat`) |
| `post_processing.py` | orientation figures from the results |

Inputs and outputs are SI. `UserInputs` options:

The solve is dimensionless, by the parameters: mesh $\hat{\mathbf X} = \mathbf X/L_0$ (so plain `Kinematics`
is exact), $\hat p = p/\mu_0$, coefficients converted by `hill_box_scaling.Coefficients`, time $\hat t = t/t_0$
(QOI `time_si` gives seconds). After the run the XDMF geometry and time stamps are put back in m and s
(`hill_playground.io.xdmf_units`). The two methods (parameters, operators) are explained step by step on
Terzaghi's consolidation in `demos/analytical_consolidation_1D/`.

- `permeability_law`: `"constant"` | `"kozeny_carman"` | `"holmes_mow"` | `"compaction_power"`.
- `volumetric_law`: `"quadratic"` | `"compaction"` ($J > J_c = \Phi_{s,0} + \Phi_c$).
- `cell_transport` ($\Phi_c$, $\tilde\phi_s$), `remodelling` ($\theta$, $\varphi$).

```bash
cd demos/hill_box
python3 hill_box_uniaxial.py              # or: mpirun -np 4 python3 hill_box_uniaxial.py
```

Outputs in `outputs_hill_box_uniaxial/results/`, all SI (`hill_box_outputs.py`):

- point data: `displacement_si`, `pressure_si` (SI copies of the unknowns, on their CG spaces);
- cell data: `jacobian`, `fluid_fraction`, `permeability_ratio`, `theta`, `phi`,
  `theta_deg`/`phi_deg` (canonical), `theta_wrapped_deg` (in-plane axis), `tilt_xy_deg` and
  `tilt_xy_current_deg` (0 = fibre in the $xy$ plane), `angle_x_deg` and `angle_x_current_deg`
  (0 = along the held axis), `lambda_*`, `tension_*`, `fibre`, `fibre_initial`, `fibre_current`,
  `cell_cauchy`, `total_cauchy`;
- QOIs: `time_si`, `cellular_activation`, probes, `mean_*`, `order_x`, `order_z`
  ($\langle \tfrac32 f_i^2 - \tfrac12\rangle$; `order_z` $\to -\tfrac12$ = planar orientation),
  `mean_tilt_xy_deg`.

With `xdmf_si_only=True` (default) the XDMF holds only SI fields: the dimensionless unknowns
(`theta`, `phi`, `cell_fraction`, `substrate_smoothed`, `eps_a`) and the FOIs above; the raw
`displacement`/`liquid_pressure` (dimensionless) and the `*_old` copies are not written.

Orientation post-processing (numpy, h5py, matplotlib; no dolfin, also on a run in progress):

```bash
python3 post_processing.py    # [--results <xdmf>] [--output-dir <dir>] [--bins 36] [--sphere-bins 30]
```

It writes to `outputs_hill_box_uniaxial/post_processing/`, for the reference fibre $\mathbf f_0$ (remodelling
only) and the current fibre $\mathbf f$ (remodelling + deformation), with volume-weighted averages:

- `orientation_histograms.png`: first vs last frame, in-plane angle to $x$, tilt out of $xy$, angle to the
  director, with the isotropic density for reference;
- `orientation_sphere.png`: equal-area projection of the fibre axes seen from $+z$ (centre: along $z$, rim:
  in the $xy$ plane), density relative to the isotropic one;
- `orientation_time.png`: order $S$, $Q_{xx}$, $Q_{zz}$ and mean tilt over the frames, QOIs `order_x`,
  `order_z`, `mean_tilt_xy_deg` overlaid as a check.

The console table gives the state of the order tensor $\mathbf Q = \langle \tfrac32 \mathbf f\otimes\mathbf f -
\tfrac12 \mathbf I\rangle$ (isotropic / prolate / oblate / biaxial), $S$, the biaxiality and the director.

ParaView: colour by `tilt_xy_current_deg` (threshold < 15° to see in-plane cells), glyphs of
`fibre_current` scaled by `tension_see`. The geometry is in m, so *Warp By Vector* uses
`displacement_si`.

Quadrature per row (row = subsol of the test function, `hill_playground.operators.ResidualRows`): `quadrature_degrees={"displacement": 3, "liquid_pressure": 2, "cell_fraction": 2,
"substrate_smoothed": 2, "theta": 2, "phi": 2, "eps_a": 2}` overrides `quadrature_degree` (4) row by row
(integral metadata `dV(degree=q)`). Measured on the size/2 box: assembly 1.83 → 1.26 s per Newton iteration
(−31 %), QOIs within 3e-4 of the degree-4 run; the `displacement` row (P2 gel, Hill and fluid stresses) carries
most of the assembly cost. Default: empty (degree 4 everywhere). Keys are checked (typo → error); rows of
switched-off unknowns are ignored.

Mesh refinement: the cost is the direct (MUMPS) solve. Measured here (2 cores): `mesh_max_size`
$L_z/2$ → 8 s per Newton iteration, $L_z/4$ → 210 s and ~5 GB, Newton still converging in 4
iterations at $\Delta t = 10$ s. If MUMPS runs out of memory, the solver reports
"Linear solver failed", the step is treated as diverged and $\Delta t$ is cut until `min_dt`:
check the `.out` file and the memory given to Docker.

Expected orders of magnitude (one cell held at $\lambda = 1$, $\alpha = 1$, default parameters):
$\lambda_a = 0.922$, $T_e = 16.3$ kPa, CE relaxation time $\tau_a \approx 1.7$ s.
In the box the gel shortens freely across $y, z$, so expect $\lambda_a$ between 0.92 and 0.63
(free cell) depending on the fibre direction.
