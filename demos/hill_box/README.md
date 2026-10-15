# Hill box (uniaxial)

First simulation of `equations.tex`: gel slab $375 \times 875 \times 50\ \mu\text{m}$ (the CORE patch of
the pillar demos), held on its two $x$-faces, drained ($p = 0$) and traction-free on the four
others, random isotropic cell orientation. Simplified stage B, without pillars, to check the
implementation.

| Item | Where |
| --- | --- |
| Hill law ($R_u$) | `hill_playground.materials.cell.active.hill` |
| CE evolution ($R_a$) | `hill_playground.operators.cell.hill.ContractileEvolution` |
| Clipped proliferation | `hill_playground.operators.cell.SaturatedProliferation` |
| Scales ($L_0, \mu_0, t_0$) | `hill_playground.scaling.CharacteristicScales` |
| $\mathbf K_L = (k/\eta) J \mathbf C^{-1}$, uniqueness check | `hill_chain_helpers.py` (demo side) |

Inputs and outputs are SI; the solved system is dimensionless (`nondimensionalize=True`).

Switches in `UserInputs`: `cell_transport` ($\Phi_c$, $\tilde\phi_s$), `remodelling` ($\theta$, $\varphi$).
Both `True` by default (full model); set both `False` to check $R_u, R_p, R_a$ alone first.

```bash
cd demos/hill_box
python3 hill_box_uniaxial.py              # or: mpirun -np 4 python3 hill_box_uniaxial.py
```

Outputs in `outputs_hill_box_uniaxial/`: XDMF fields (`lambda_a`, `tension_see`, `tension_ce`,
`pressure`, `displacement_physical`, `cell_cauchy`, ...) and QOIs (`mean_lambda_a`,
`mean_tension_see`, `top_center_disp_z`, `front_center_disp_y`, `center_pressure`, ...).

Expected orders of magnitude (one cell, $\lambda = 1$, $\alpha = 1$, equations.tex):
$\lambda_a = 0.922$, $T_e = 16.3$ kPa, CE relaxation time $\tau_a \approx 1.7$ s.
In the box the gel shortens freely across $y, z$, so expect $\lambda_a$ between 0.92 and 0.63
(free cell) depending on the fibre direction.
