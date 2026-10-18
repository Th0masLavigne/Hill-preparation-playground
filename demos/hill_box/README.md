# Hill box (uniaxial)

First simulation of `equations.tex`: gel slab $875 \times 375 \times 50\ \mu\text{m}$ (the CORE patch of
the pillar demos), held on its two $x$-faces, drained ($p = 0$) and traction-free on the four
others, random isotropic cell orientation. Simplified stage B, without pillars, to check the
implementation.

| Item | Where |
| --- | --- |
| Hill law ($R_u$), activation $\alpha(t)$ | `hill_playground.materials.cell.active.hill`, `...active.activation` |
| CE evolution ($R_a$) | `hill_playground.operators.cell.hill.ContractileEvolution` |
| Clipped proliferation | `hill_playground.operators.cell.SaturatedProliferation` |
| Scales ($L_0, \mu_0, t_0$), conversions | `hill_playground.scaling.CharacteristicScales` |
| $k(\phi_\ell)$, $\mathbf K_L = (k/\eta) J \mathbf C^{-1}$ | `hill_playground.materials.permeability` |
| Compaction volumetric energy | `hill_playground.materials.volumetric.compaction` |
| Orientation diagnostics | `hill_playground.materials.orientation.display` |
| Uniqueness check | `hill_chain_helpers.py` (demo side) |

Inputs and outputs are SI. `UserInputs` options:

- `nondimensionalization`: `"none"` | `"operator"` (default, `scale_factor` per row, SI time) |
  `"parameters"` (coefficients converted with `CharacteristicScales.nondimensionalize_*`, time $\hat t$,
  QOI `time_si` gives seconds). The three give the same solution to round-off. The mesh itself is
  scaled ($\hat{\mathbf X} = \mathbf X/L_0$), so plain `Kinematics` is exact (no `ScaledKinematics`);
  after the run the XDMF geometry and time stamps are put back in m and s (`hill_playground.io.xdmf_units`).
- `permeability_law`: `"constant"` | `"kozeny_carman"` | `"holmes_mow"` | `"compaction_power"`.
- `volumetric_law`: `"quadratic"` | `"compaction"` ($J > J_c = \Phi_{s,0} + \Phi_c$).
- `cell_transport` ($\Phi_c$, $\tilde\phi_s$), `remodelling` ($\theta$, $\varphi$).

```bash
cd demos/hill_box
python3 hill_box_uniaxial.py              # or: mpirun -np 4 python3 hill_box_uniaxial.py
```

Outputs in `outputs_hill_box_uniaxial/results/`, same names, order and spaces in every mode:

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
`displacement`/`liquid_pressure` (solve units) and the `*_old` copies are not written.

ParaView: colour by `tilt_xy_current_deg` (threshold < 15° to see in-plane cells), glyphs of
`fibre_current` scaled by `tension_see`. The geometry is in m in every mode, so *Warp By Vector*
uses `displacement_si`.

Rows (`hill_playground.operators.ResidualRows`): a row is named after the subsol of its test function
(`displacement`, `liquid_pressure`, `cell_fraction`, `substrate_smoothed`, `theta`, `phi`, `eps_a`). Each
operator takes `rows.measure(dV, row)` and `rows.scale_factor(row, v)`, with `v` its value-preserving
factor (`TermValues`, written with the FE test function), the test scale from `row_test_scales`
($s_{\mathbf u} = L_0$, $s_p = \mu_0$) and the row normalization from `row_normalizations` (both
`"operator"` mode only): `scale_factor` $= s\,v/\mathcal N$. `python jacobian_scaling.py` prints
$\kappa_2(\mathbf J)$ and the diagonal block sizes per mode (none 1.7e20, operator = parameters 4.3e4).

Quadrature per row: `quadrature_degrees={"displacement": 3, "liquid_pressure": 2, "cell_fraction": 2,
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

Expected orders of magnitude (one cell, $\lambda = 1$, $\alpha = 1$, equations.tex):
$\lambda_a = 0.922$, $T_e = 16.3$ kPa, CE relaxation time $\tau_a \approx 1.7$ s.
In the box the gel shortens freely across $y, z$, so expect $\lambda_a$ between 0.92 and 0.63
(free cell) depending on the fibre direction.
