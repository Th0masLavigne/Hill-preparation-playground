# Transfer to fenicsoft: elementary issues

Everything in `hill_playground/<path>` targets `fenicsoft/<path>` (same sub-package names).
One issue = one merge request, in the order below: each one only needs the issues above it.
Reference branch: `64-reproduce-experiment-from-mostert-et-al-2` (it has `scale_factor` and `OrientationChange`).

Legend: **new**: the file does not exist in fenicsoft. **change**: patch an existing fenicsoft file.
**copy**: a fenicsoft file copied here and modified; report the diff and do not create a second file.

| # | Issue | Kind | Depends on |
|---|---|---|---|
| 1 | Integration: select XDMF fields, write logs on rank 0 only | copy | – |
| 2 | Scaling: promote `CharacteristicScales` to `fenicsoft.scaling` | new | – |
| 3 | IO: write XDMF geometry and time back in SI | new | – |
| 4 | Active laws: shared activation schedule | new | – |
| 5 | Hill chain law (self-contained) | new | 4 |
| 6 | Contractile element operator $R_a$ | new | 5 |
| 7 | Saturated proliferation: clip the saturation | change | – |
| 8 | Permeability laws $k(\phi_\ell)$ | new | – |
| 9 | Volumetric energy with a compaction point | new | 8 (`porosity`, demo only) |
| 10 | Orientation diagnostics | new | – |
| 11 | Residual rows: quadrature degree, test scale and normalization per row | new | – |
| 12 | Hill one-element integration test | new | 5, 6, 11 |
| 13 | Terzaghi integration test, nondimensionalization methods | new | 2, 11 |
| 14 | Hill box demo | new | 1–11 |
| 15 | Follow-ups (iso/aniso activation, `ScaledKinematics`, tooling) | change | 4 |

---

## 1. Integration: select XDMF fields, write logs on rank 0 only

**Problem.**
- `dolfin_mech.core.TimeIntegrator` always writes every subsol, every `*_old` copy and every FOI. When the solve is nondimensionalized, the raw unknowns (û, p̂) are not SI, and the `*_old` copies duplicate every frame.
- Under `mpirun`, the `.out`, `.sta` and `.dat` files are opened by every rank, each with its own buffer and file offset. The ranks overwrite each other once the buffers flush. Reproduced with `mpirun -n 2`: the `.out` keeps 272 lines instead of 521.

**Changes** (`src/fenicsoft/integration/decimated_time_integrator.py`; diff = blocks tagged `CHANGE (playground)` in `hill_playground/integration/decimated_time_integrator.py`):
- `DecimatedTimeIntegrator.__init__(..., functions_to_write=None, root_only_logs=True)`.
- `_silence_logs()`: on ranks other than 0, the printers write to `os.devnull`. Every logged value is already MPI-reduced, so nothing is lost.
- `_reopen_xdmf()`: reopens the results XDMF on `functions_to_write` and rewrites the t = 0 frame, before the decimation proxy is built.
- Cleaner long-term alternative: add `functions_to_write` to `dolfin_mech.core.TimeIntegrator` itself, which removes the rewrite of t = 0.
- Module `Status` paragraph: remove it once merged.
- Permanent fix on the fenicsoft side: no dolfin_mech change is expected, and fenicsoft only uses `DecimatedTimeIntegrator`. `NonlinearSolver` needs no change: the base `TimeIntegrator.__init__` sets `solver.printer = self.printer`, so silencing `self.printer` also silences the Newton lines. Only a `NonlinearSolver` used on its own with `print_out=<file>` would still write from every rank.

**Tests.**
- `tests/unit/integration/test_decimated_time_integrator.py` and `integrator_case.py` → fenicsoft `tests/unit/io/`. They cover:
  - the selected fields and the base list;
  - the `mpirun -n 2` logs being identical to the serial ones (`.out` and `.sta` as text, `.dat` numerically to 1e-10);
  - a silenced rank, both after a run and at construction (rank faked to 1);
  - the plateau exit on a real problem, including the "before `min_simu_seconds`" and "not yet stagnant" branches.
- `integrator_case.py` is a no-flux diffusion with c₀ = 1 + cos πx, so c → 1 and the plateau is reached around t ≈ 0.6. Avoid a Dirichlet BC with a single subsol: it does not act on the solution and the case becomes trivial (c ≡ 0).
- The MPI test works with both MPICH and Open MPI. `--oversubscribe` and `--allow-run-as-root` are only added when `mpirun --version` says Open MPI, because MPICH's hydra rejects them. The parent's `OMPI_*`, `PMIX_*` and `PMI_*` variables are dropped for the child job, since dolfin has already initialised MPI in the parent.
- `test_decimated_time_integrator_unit.py` is the fenicsoft test with only its import changed. Do not copy it: keep fenicsoft's.

**Done when** the 59 integration unit tests pass with 100 % coverage of the module, and an `mpirun -n 2` run of a demo gives a `.out` with the same line count as the serial run.

## 2. Scaling: promote `CharacteristicScales` to `fenicsoft.scaling`

**Changes.**
- New `src/fenicsoft/scaling/{__init__.py, characteristic_scales.py}`, from `hill_playground/scaling`. It is the demo-side `microtissue_pillar_spherical_scale/src/characteristic_scales.py`, plus:
  - `identity()`;
  - conversions documented for floats and UFL expressions;
  - corrected docstring of `permeability_over_viscosity`: L₀²/(μ₀t₀) = (k/η)κ/μ₀, so K̂ = μ₀/κ.
- The scaled pillar demo (`microtissue_pillar_spherical_scale`, wrong row factors: see issue 11) is deleted from fenicsoft meanwhile. The pillar case will be recoded in the playground on top of issues 2 and 11, once the Hill box is validated.

**Tests.** `tests/unit/scaling/test_characteristic_scales.py` (100 % coverage).

## 3. IO: write XDMF geometry and time back in SI

**Changes.** New `src/fenicsoft/io/xdmf_units.py`. `redimensionalize_results(xdmf, length, time)` multiplies `Mesh/*/mesh/geometry` by L₀ and `<Time Value>` by t₀. It is idempotent through the HDF5 attribute `geometry_scale`. Call it on rank 0 after `integrate()` whenever the mesh was scaled.

**Tests.** `tests/unit/io/test_xdmf_units.py`.

## 4. Active laws: shared activation schedule

**Changes.** New `src/fenicsoft/materials/cell/active/activation.py`:
- `ActivePowerConfig`: same API as in `iso.law`;
- `ActivationSchedule`: owns `p_act` and the global `time`, with `set_k_step` and `update_time`.
- Difference from the machinery in `iso`/`aniso`: in `update_time`, `elif manager is not None` becomes `else`. A manager exists for every config without `func`, so the third case was unreachable (`# pragma: no cover` in aniso).

It is not a constitutive law. Active laws use it by composition, so no law depends on another one.

**Tests.**
- `tests/unit/materials/cell/active/test_activation.py`.
- `tests/unit/materials/cell/active/test_activation_equivalence.py`: 100 random config sets × 20 calls give the same α as `iso.Constitutive` and `aniso.Constitutive`, or the same `IndexError` for a function config on a missing step. The sets cover constant, ramp and function configs, and missing, duplicated or out-of-range `k_step`.
- fenicsoft demos: they only use `ActivePowerConfig.constant`, `.ramp` and `.function`, never the internals, so the change is transparent for them.

## 5. Hill chain law (self-contained)

**Changes.** New `src/fenicsoft/materials/cell/active/hill/{__init__.py, functions.py, law.py}`:
- `functions.py`: T_e (variants `mooney`, `tension_only`, `exponential`), Ψ_e, T_p, Ψ_p, f_FL, h(D_a);
- `law.py`:
  - `Parameters(MaterialParameters)` and `Constitutive(ConstitutiveLaw)`, with S_h = Φ_c T_e/I₄ f₀⊗f₀;
  - activation through `ActivationSchedule`, exposed as `current_activation` and `current_time`;
  - no import of `iso` or `aniso`.

**Tests.** `tests/unit/materials/cell/active/hill/test_{functions,law}_cell_hill.py` and the fixtures in `tests/conftest.py` (`HillSetup`, `make_setup`).

## 6. Contractile element operator $R_a$

**Changes.** New `src/fenicsoft/operators/cell/hill/{__init__.py, contractile_evolution.py}`: `ContractileEvolution`, with R_a = [f_max α f_FL h(D_a) + η₀D_a + T_p − T_e] δε and D_a = (ε_a − ε_aⁿ)/Δt, plus `scale_factor`.

**Tests.** `tests/unit/operators/cell/hill/test_contractile_evolution.py`.

## 7. Saturated proliferation: clip the saturation

**Changes.**
- In fenicsoft `operators/cell/proliferation/saturated.py`, use sat = max(0, 1 − Φ_c/Φ_c,max). Today it turns into a sink above Φ_c,max.
- Then delete `hill_playground/operators/cell/proliferation/saturated.py`.

**Tests.** Move `tests/unit/operators/cell/proliferation/test_saturated_prol.py` next to the existing fenicsoft tests.

## 8. Permeability laws $k(\phi_\ell)$

**Changes.** New package `src/fenicsoft/materials/permeability/`:
- `porosity.py`: J_c = Φ_s0 + Φ_c, Φ_ℓ = J − J_c, φ_ℓ = 1 − J_c/J, K_L = (k/η) J C⁻¹. Shared relations, not a law.
- One self-contained module per law, each with `Parameters`, `ratio`, `hydraulic_conductivity` and `tensor`: `constant`, `kozeny_carman`, `holmes_mow`, `compaction_power` (n ≥ 1 enforced).
- Afterwards, the pillar demos can replace their `permeability.py` (gel/pillar field) by these laws times a domain field.
- Add `materials.permeability` to fenicsoft's environment test if it has a layout check.

**Tests.** `tests/unit/materials/permeability/test_permeability_laws.py`.

## 9. Volumetric energy with a compaction point

**Changes.** New `src/fenicsoft/materials/volumetric/compaction.py`: `CompactionVolumetricPenalty` (Ehlers), W = κ(1−J_c)²[x − ln x − 1], with its own `Parameters(kappa, solid_jacobian, x_floor)`.

**Tests.** `tests/unit/materials/volumetric/test_compaction.py`.

## 10. Orientation diagnostics

**Changes.** New `src/fenicsoft/materials/orientation/display.py`: `current_direction`, `tilt_degrees`, `axis_angle_degrees`, `order_parameter` (head/tail symmetric, `atan2`).

**Tests.** `tests/unit/materials/orientation/test_display.py`.

## 11. Residual rows: quadrature degree, test scale and normalization per row

**Problem.** Every operator of a monolithic residual needs a `scale_factor` = s_row·v/𝒩_row. Today these are hand-written per demo (`RowFactors` in the pillar demos, magic factors in the tests).

equations.tex Table tab:sf omitted s_row, the scale of the row's test function: the FE test function is δx̂ while R_row of Table tab:N is written with δx = s δx̂ (s_u = L₀, s_p = μ₀, else 1). The solution is unchanged (row scaling), but with tab:sf as printed the u row was 1/L₀ too large and the p row 1/μ₀ too small: κ₂(J) = 9.3e8 instead of 7.5e4. Corrected factors (equations.tex fixed accordingly): gel 1/μ₀, fluid 1, J rate t₀, Darcy μ₀t₀/L₀², cell flux in R_p t₀/L₀²; the other rows are unchanged. Assembly could also be cheaper with a quadrature degree per row (measured: −31 % assembly on the Hill box).

**Changes.** New `src/fenicsoft/operators/residual_rows.py`: `ResidualRows(rows, quadrature_degrees, test_scales, normalizations)`.
- A row is named after the subsol of its test function (`displacement`, `liquid_pressure`, …). `ResidualRows.from_problem(problem, …)` takes the names from `problem.subsols`.
- `measure(dV, row)` returns `dV(degree=q_row)` (integral metadata), or `dV` unchanged when the row has no degree.
- `scale_factor(row, v)` returns s_row·v/𝒩_row, with s = 1 and 𝒩 = 1 when the row has none. v is the value-preserving factor written with the FE test function.
- Unknown row names, negative or non-integer degrees and non-positive scales or normalizations raise. The mappings are frozen.
- FFC check, on the devcontainer's legacy FFC:
  - mixed degrees are exact in residuals (rank 1) and Jacobians (rank 2);
  - they are **wrong in scalar functionals** (rank 0), where the low-degree integrand is counted twice. Never use per-row measures in an `"assembly"` QOI.
  - The first environment test (`x⁸` at degree 2 plus `x` at degree 8) did not detect it: `x` is integrated exactly at any degree. The test now checks a residual and its Jacobian.

**Tests.**
- `tests/unit/operators/test_residual_rows.py`.
- `tests/environment/test_env.py::test_quadrature_degree_per_integral` (rank 1 and rank 2) → fenicsoft environment tests.
- Wiring tests, in issues 12–14:
  - Hill one element: `ResidualRows` on R_u and R_a.
  - Terzaghi: all 3 modes through `ResidualRows`, plus the `+rows` variants.
  - Hill box demo.

**Follow-up.** The pillar case is recoded in the playground with `ResidualRows` (multi-domain: `rows.measure(problem.dV(PILLAR), row)` keeps the subdomain; the pillar identity Φ_c = 0 sits in the `cell_fraction` row with v = L₀³/t₀, i.e. factor 1).

## 12. Hill one-element integration test

**Changes.** `tests/integration/materials/cell/active/hill/` (4 files, including `conftest.py`): a one-element FE run of gel + Hill, compared with an independent 0D backward-Euler reference (scipy). Four cases: isometric λ = 1 and λ = 1.04, traction-free faces, and traction-free faces through `ResidualRows` with other per-row degrees (`displacement`: 1, `eps_a`: 0) and row normalizations (μ, f_max).
- The operators take their measure and `scale_factor` from `ResidualRows`. In the homogeneous one-element state every integrand is constant per cell, so the `rows` case must equal both the 0D reference and the default free run to round-off.
- The figure `hill_one_element.png` and `errors.csv` are written to `outputs_hill_one_element/`. The figure shows λ_a, T_e, the free-face gel stretches and the error history, FE (markers) against 0D (lines).
- The error table is printed in the pytest summary.

**Done when** ε_a, P_xx and (λ, λ_t) match the 0D reference to 1e-12–1e-10.

## 13. Terzaghi integration test, nondimensionalization methods

**Changes.**
- `tests/integration/analytical_consolidation_1D/{terzaghi_nondim.py, test_terzaghi_nondim.py, conftest.py}` → fenicsoft `tests/integration/analytical_consolidation_1D/nondimensional/`.
- Three modes: SI, operator `scale_factor`, and parameters converted with `nondimensionalize_*`.
- The operators take their measure and `scale_factor` from `ResidualRows` (rows `displacement` and `liquid_pressure`), with term value factors v (gel and load L₀², fluid μ₀L₀², rate L₀³, Darcy μ₀L₀), test scales (L₀, μ₀, operator mode) and row normalizations (μ₀L₀³, μ₀L₀³/t₀).
- `+rows` variants of the 3 modes: `{"displacement": 3, "liquid_pressure": 2}`. They stay within 1 % of the analytics, agree with each other to round-off, and differ from the default degree by 1e-12 < Δp/p₀ < 1e-3, which shows the degrees are applied. Runtime: about 3.8 min for the 6 runs. The figure and `errors.csv` go to `outputs_terzaghi_nondim/`; the error table is printed in the pytest summary.
- Merge the Terzaghi series with fenicsoft's `terzaghi_analytics.py` if wanted.

**Done when** the pressure and settlement errors are below 1 % for T ≥ 0.05 for all 6 cases (measured: 8.6e-3 and 3.9e-3), and the scaled modes are identical to SI with the same rows.

## 14. Hill box demo

**Changes.**
- `demos/hill_box/{hill_box_uniaxial.py, hill_chain_helpers.py, README.md}` → fenicsoft `demos/smooth_muscle_cells/porous_hydrogel/hill_box/`.
- Replace the `hill_playground.*` imports by `fenicsoft.*`.
- `UserInputs` options: `nondimensionalization` (`none`, `operator`, `parameters`), `permeability_law`, `volumetric_law`, `xdmf_si_only`, `quadrature_degrees`.
- Outputs:
  - `displacement_si` and `pressure_si` are scaled copies of the unknowns on their own CG spaces (vector scaling). A global `dolfin.project` on CG2 cost about 5 s per output, 15 % of the run.
  - The other FOIs use `update_type="local_solver"`, a cell-wise projection on the DG spaces, instead of `"project"`.
  - All output updates went from 17.1 s to 0.3 s on the size/2 box.
- Rows through `ResidualRows` (issue 11): `build_residual_rows()` gives each operator `rows.measure(dV, row)` and `rows.scale_factor(row, v)`.
  - `TermValues` holds the value factors (gel L₀², fluid μ₀L₀², volume L₀³, Darcy μ₀L₀, cell flux L₀).
  - `row_test_scales()` holds s_row (L₀ for `displacement`, μ₀ for `liquid_pressure`) and `row_normalizations()` holds 𝒩_row (`"operator"` mode only).
  - Row names are the subsols: `displacement`, `liquid_pressure`, `cell_fraction`, `substrate_smoothed`, `theta`, `phi`, `eps_a`. The cell flux Q_c is in both `liquid_pressure` and `cell_fraction`.
  - The `quadrature_degrees` keys are checked against `UNKNOWNS` (a typo raises an error); rows of switched-off unknowns are ignored. Default: empty, i.e. `quadrature_degree` everywhere.
  - Measured on the size/2 box: `{"displacement": 3}` and the other rows at 2 bring assembly from 1.83 to 1.26 s per iteration, with QOIs within 3e-4.
  - Checked after the refactor: same QOIs as before (difference 0.0), the 3 modes identical (3e-15), and the same holds with per-row degrees (`operator` vs `none` with rows: 2.5e-15).
- Conditioning: `demos/hill_box/jacobian_scaling.py` assembles J at the first step for the 3 modes (κ₂ by dense SVD, median |J_ii| per row). Test `tests/integration/demos/test_hill_box_scaling.py` (about 80 s): `operator` J = `parameters` J (≤ 1e-10), interior diagonal blocks O(1), κ₂: none 1.7e20, operator 4.3e4, parameters 4.3e4.
- Cost profile (size/2, warm JIT cache): MUMPS solve ≈ 78 %, assembly ≈ 21 %, outputs < 1 %. Turning off `cell_transport` and `remodelling` cuts the solve to 5.2 s per iteration (−37 %). size/4: about 210 s per solve and about 5 GB.

**Done when** the three `nondimensionalization` modes give identical QOIs (measured: 3e-15) and identical Jacobians (`operator` vs `parameters`), and an `mpirun` run gives a readable `.out`.

## 15. Follow-ups

- `iso.law` and `aniso.law` import `ActivePowerConfig` (and the `p_act` machinery) from `cell/active/activation.py` (issue 4), and remove their copies.
  - `aniso` keeps `p_act` and `iso` keeps `p_cell` as aliases of `ActivationSchedule.p_act`, plus `current_activation` / `current_power`.
  - Their unit tests read private attributes and must be adapted: `_current_manager` and `_step_managers` in `test_law_cell_iso.py`, `_current_config` in `test_law_cell_aniso.py`.
  - The equivalence test then becomes redundant.
- `ScaledKinematics` (pillar demo `src/`) goes with the deleted scaled pillar demo: the mesh is scaled, so `Kinematics` is exact.
- After each merge:
  - remove the transferred paths from `NEW_BRANCHES` in `tests/environment/test_env.py`;
  - delete the playground copies;
  - switch the playground imports to fenicsoft.
- Devcontainer: mount `~/.cache/dijitso` (the compiled forms) as a persistent volume. The Hill forms take about 90 s of JIT compilation on a fresh container.
- Tooling: autoapi documents the tests too, so docstrings there must be valid reST. Write `|x|` inside `:math:`, never bare (`make docs` fails on "Undefined substitution").
