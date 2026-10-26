# Transfer to fenicsoft: elementary issues

Everything in `hill_playground/<path>` targets `fenicsoft/<path>` (same sub-package names).
One issue = one merge request, in the order below: each one only needs the issues above it.
Reference branch: fenicsoft `main` (clean-up #93 merged; it carries `scale_factor` and `OrientationChange`).

**fenicsoft conventions since #93** (apply them in every transfer merge request):
- Layout: `Step` is in `fenicsoft.integration` (`integration.Step`), the mesh cut and field transfer in the new
  `fenicsoft.mesh` (`fenicsoft.mesh.transfer`); `fenicsoft.io` only reads and writes files. Tests mirror `src`
  (`tests/unit/integration/`, `tests/unit/mesh/`, ...).
- Demos: elementary cases on simple geometries in `demos/`, complex cases (real geometries, multi-stage pipelines,
  reproduced experiments) in `demos/applications/`. Each demo module docstring gives the problem, the BCs with a TikZ
  figure (`docs/src/_static/resources/<demo>_bcs.{tex,svg}`), the residuals and the validation.
- Archives: superseded code in `archives/{src,demos,tests}` (not installed, tested nor linted; documented under
  "Archives (not maintained)"). The anisotropic active law `materials/cell/active/aniso` and the demos using it
  (`hill_cube`, `hill_cylinder_*remodelling`, `microtissue_pillars_{planar,spherical}`, porous
  `microtissue_pillar_spherical`) are there; `microtissue_pillar_spherical_scale` was deleted.
- Docstrings: Google style, short (first line = objective, then the equation, `Args`/`Returns` without types, no
  history; tests: one line stating what is verified), as in the playground (fenicsoft README, "Contributing").
- `make linter` (ruff 0.3.0 of the pre-commit hook, D rules included: one-line summary, blank line after it),
  `make cov` (100 % of the transferred modules) and `make docs` (`sphinx -W`, docs include demos, applications,
  tests and archives) must pass.

**`__init__.py` files** are never copied: merge their imports and `__all__` into the existing fenicsoft ones (they export more). fenicsoft has no `__init__.py` in `operators/cell/*/` sub-folders: export the new classes from `operators/cell/__init__.py` instead. New top-level sub-package (`scaling`): add it to `fenicsoft/__init__.py` and its `__all__`.

Legend: **new**: the file does not exist in fenicsoft. **change**: patch an existing fenicsoft file.
**copy**: a fenicsoft file copied here and modified; report the diff and do not create a second file.

| # | Issue | Kind | Depends on |
|---|---|---|---|
| 0 | Shared UFL guards (package root) `piecewise`, `floored`, `capped`, `macaulay`, `macaulay_negative` | new | – |
| 1 | Integration: select XDMF fields, write logs on rank 0 only | copy | – |
| 2 | Scaling: promote `CharacteristicScales` to `fenicsoft.scaling` | new | 1 |
| 3 | IO: write XDMF geometry and time back in SI | new | – |
| 4 | Active laws: shared activation schedule | new | – |
| 5 | Hill chain law (self-contained) | new | 0, 4 |
| 6 | Contractile element operator $R_a$ | new | 5 |
| 7 | Saturated proliferation: clip the saturation | change | – |
| 8 | Permeability laws $k(\phi_\ell)$ | new | 0 |
| 9 | Volumetric energy with a compaction point | new | 0; 8 (`saturated_mixture`, demo only) |
| 10 | Orientation diagnostics | new | – |
| 11 | Residual rows: quadrature degree, test scale and normalization per row | new | – |
| 12 | Hill one-element integration test | new | 5, 6, 11 |
| 13 | Terzaghi integration test, nondimensionalization methods | new | 2, 11 |
| 14 | Hill box demo | new | 0–11 |
| 15 | Follow-ups (iso activation, `ScaledKinematics`, tooling) | change | 4 |
| 16 | Subdomain indicators (gel/pillar data fields) | new | – |
| 17 | Free-floating cylinder demo | new | 0–11 |
| 18 | Microtissue between pillars demo (stage A, cut, stage B) | new | 0–11, 16 |

---

## 0. Shared UFL guards

**Changes.** New `src/fenicsoft/ufl_functions.py`: `piecewise(x, x0, f_le, f_gt)` (the tie x = x0 and its UFL tangent go to the lower branch), and with it `floored(x, x_f)` = max(x, x_f), `capped(x, x_c)` = min(x, x_c), `macaulay(x)` = max(x, 0) (tangent 0 at 0) and `macaulay_negative(x)` = min(x, 0) (tangent 1 at 0), the Newton safeguards of the Hill, permeability and compaction laws and of `SaturatedProliferation` (written once; they were three copies). Package root, not `materials/`: operators use them too. Import it first in `fenicsoft/__init__.py`. Reuse candidates in fenicsoft: the clamped `sqrt` in `materials/orientation/generic.py` (`macaulay`), the saturation `conditional(phi_c > phi_c_max, 0, 1 - phi_c/phi_c_max)` in `operators/cell/haptotaxis/saturated.py` (`macaulay`).

**Tests.** `tests/unit/test_ufl_functions.py`.

## 1. Integration: select XDMF fields, write logs on rank 0 only

**Problem.**
- `dolfin_mech.core.TimeIntegrator` always writes every subsol, every `*_old` copy and every FOI. When the solve is nondimensionalized, the raw unknowns (û, p̂) are not SI, and the `*_old` copies duplicate every frame.
- Under `mpirun`, the `.out`, `.sta` and `.dat` files are opened by every rank, each with its own buffer and file offset. The ranks overwrite each other once the buffers flush. Reproduced with `mpirun -n 2`: the `.out` keeps 272 lines instead of 521.

**Changes** (`src/fenicsoft/integration/decimated_time_integrator.py`). The playground file replaces fenicsoft's: all docstrings were shortened (no tagged blocks), so diff the code without docstrings (e.g. `ast.unparse` of both files with docstrings stripped). Code changes:
- `DecimatedTimeIntegrator.__init__(..., functions_to_write=None, root_only_logs=True)`.
- `_silence_logs()`: on ranks other than 0, the printers write to `os.devnull`. Every logged value is already MPI-reduced, so nothing is lost.
- `_reopen_xdmf()`: reopens the results XDMF on `functions_to_write` and rewrites the t = 0 frame, before the decimation proxy is built.
- `_DecimatedXDMFFile`: the last written t is no longer reset at a step change (initially 0: the file already holds t = 0); only the period state is (call count, first frame of the step). Before, a solver failure on the first substep of a step forced a second frame at t_ini (= previous t_fin, or t = 0).
- Cleaner long-term alternative: add `functions_to_write` to `dolfin_mech.core.TimeIntegrator` itself, which removes the rewrite of t = 0.
- `_relative_error` removed: `is_subsol_stagnant` calls `dolfin_mech.core.compute_error` (same rule: scale within 1e-9 of 0 gives e = 0 if the change is too, else 1; checked in `dolfin_mech_devel_lavigne@master`). The default threshold is renamed `_DEFAULT_PLATEAU_RELATIVE_ERROR` (was `_RELATIVE_ERROR_ZERO_EPS`).
- `write_period` is validated even with `write_sol=False` (fenicsoft returned before the checks).
- A single integer `write_period < 1` (e.g. 0) now raises `ValueError`; fenicsoft (`isinstance(write_period, int) and write_period <= 1: return`) silently treated it as "no throttling".
- Plateau log message: no `s` after the elapsed time (solve units: s, or t₀ for a scaled solve).
- Permanent fix on the fenicsoft side: no dolfin_mech change is expected, and fenicsoft only uses `DecimatedTimeIntegrator`. `NonlinearSolver` needs no change: the base `TimeIntegrator.__init__` sets `solver.printer = self.printer`, so silencing `self.printer` also silences the Newton lines. Only a `NonlinearSolver` used on its own with `print_out=<file>` would still write from every rank.

**Tests.**
- `tests/unit/integration/test_decimated_time_integrator.py` and `integrator_case.py` → fenicsoft `tests/unit/integration/` (where fenicsoft's `test_decimated_time_integrator_unit.py` now is). They cover:
  - the selected fields and the base list;
  - the `mpirun -n 2` logs being identical to the serial ones (`.out` as text with every number masked, since timings differ; `.sta` as raw text; `.dat` numerically to 1e-10);
  - a silenced rank, both after a run and at construction (rank faked to 1);
  - the plateau exit on a real problem, including the "before `min_simu_seconds`" and "not yet stagnant" branches.
- `integrator_case.py` is a no-flux diffusion with c₀ = 1 + cos πx, so c → 1 and the plateau is reached around t ≈ 0.6. Avoid a Dirichlet BC with a single subsol: it does not act on the solution and the case becomes trivial (c ≡ 0).
- The MPI test works with both MPICH and Open MPI. `--oversubscribe` and `--allow-run-as-root` are only added when `mpirun --version` says Open MPI, because MPICH's hydra rejects them. The parent's `OMPI_*`, `PMIX_*` and `PMI_*` variables are dropped for the child job, since dolfin has already initialised MPI in the parent.
- `test_decimated_time_integrator_unit.py` is fenicsoft's test with the import changed, all docstrings shortened, the `_relative_error` tests removed (function gone), the constant renamed, and: `test_integrate_permanent_solver_failure_ends_the_step_unsuccessfully` now expects no write (t = 0 already in the file); new tests `..._force_at_previous_step_end_after_step_change_is_a_no_op`, `..._force_at_initial_time_is_a_no_op` (duplicate-frame fix) and `test_is_subsol_stagnant_zero_scale` (zero norm). Port these changes into fenicsoft's file, now `tests/unit/integration/test_decimated_time_integrator_unit.py` (its docstrings were already condensed in #93: diff the code, not the docstrings).

**Done when** the 61 integration unit test cases (59 functions, 2 of them parametrized) pass with 100 % coverage of the module, and an `mpirun -n 2` run of a demo gives a `.out` with the same line count as the serial run.

## 2. Scaling: promote `CharacteristicScales` to `fenicsoft.scaling`

**Changes.**
- New `src/fenicsoft/scaling/{__init__.py, characteristic_scales.py, time_steps.py}`, from `hill_playground/scaling`. It supersedes the demo-side copy of the deleted `microtissue_pillar_spherical_scale` (`src/characteristic_scales.py`), plus:
  - `identity()`;
  - conversions documented for floats and UFL expressions;
  - corrected docstring of `permeability_over_viscosity`: L₀²/(μ₀t₀) = (k/η)κ/μ₀, so K̂ = μ₀/κ;
  - raw-docstring math fixed (`1/\tau`, `k/\eta` were written with a double backslash).
- `time_steps.py`: `nondimensionalize_step` (`fenicsoft.integration.Step` time fields, `min_simu_seconds` included) and `nondimensionalize_write_period` (`TimeInterval` / t₀), shared by the scaled demos; both return Python `float` (see §15, `set_dt` with a NumPy scalar). Depends on `integration.TimeInterval`.
- Done in #93: the scaled pillar demo `microtissue_pillar_spherical_scale` (wrong row factors, see issue 11; its test expected (k/η)(μ₀/κ) for `permeability_over_viscosity`) is deleted from fenicsoft. The pillar case is recoded in the playground (issue 18).

**Tests.** `tests/unit/scaling/test_characteristic_scales.py` (100 % coverage), `tests/unit/scaling/test_time_steps.py`.

## 3. IO: write XDMF geometry and time back in SI

**Changes.** New `src/fenicsoft/io/xdmf_units.py`. `redimensionalize_results(xdmf, length, time)` multiplies `Mesh/*/mesh/geometry` by L₀ and `<Time Value>` by t₀. It is idempotent stage by stage through the HDF5 attributes `geometry_scale` and `time_scale`, each written right after its stage (the XDMF is replaced atomically): a second call skips the stamped stages and resumes an interrupted one; only an interruption inside a stage, before its stamp, makes the next call apply that stage again. Both attributes are recorded even when the scale is 1; `is_redimensionalized(h5)` is True once both are set. With L₀ = t₀ = 1 the call is a no-op and writes no stamp: it logs a warning (`logging`, as `fenicsoft.io.mesh`), since `is_redimensionalized` then stays False, as for a run in progress; the demo's main must handle that case. New `src/fenicsoft/io/xdmf_reading.py`: `read_time_stamps(xdmf)` returns the `<Time Value>` entries as floats (use it in tests rather than matching the formatted text), and `TIME_VALUE` is the pattern shared with `xdmf_units`. Call it on rank 0 after `integrate()` whenever the mesh was scaled. Export `redimensionalize_results`, `is_redimensionalized` and `read_time_stamps` from `fenicsoft.io`. New direct dependency of `src`: `h5py` (already in the image; add it to the `dependencies` of `pyproject.toml`).

**Tests.** `tests/unit/io/test_xdmf_units.py`, `tests/unit/io/test_xdmf_reading.py`.

## 4. Active laws: shared activation schedule

**Changes.** New `src/fenicsoft/materials/cell/active/activation.py`:
- `ActivePowerConfig`: same API as in `iso.law`;
- `ActivationSchedule`: owns `p_act` and the global `time`, with `set_k_step` and `update_time`.
- Difference from the machinery in `iso`: in `update_time`, `elif manager is not None` becomes `else`. A manager exists for every config without `func`, so the third case was unreachable (`# pragma: no cover` in iso; the archived aniso law had the same copy).

It is not a constitutive law. Active laws use it by composition, so no law depends on another one.

**Tests.**
- `tests/unit/materials/cell/active/test_activation.py`.
- `tests/unit/materials/cell/active/test_activation_equivalence.py`: 100 random config sets × 20 calls give the same α as `iso.Constitutive` (the aniso comparison was dropped with the archived law), or the same `IndexError` for a function config on a missing step. The sets cover constant, ramp and function configs, and missing, duplicated or out-of-range `k_step`.
- fenicsoft demos: they only use `ActivePowerConfig.constant`, `.ramp` and `.function`, never the internals, so the change is transparent for them.

## 5. Hill chain law (self-contained)

**Changes.** New `src/fenicsoft/materials/cell/active/hill/{__init__.py, functions.py, law.py}`:
- `functions.py`: T_e (variants `mooney`, `tension_only`, `exponential`), Ψ_e, T_p, Ψ_p, f_FL, h(D_a); guards from `fenicsoft.ufl_functions` (issue 0);
- `law.py`:
  - `Parameters(MaterialParameters)` and `Constitutive(ConstitutiveLaw)`, with S_h = Φ_c T_e/I₄ f₀⊗f₀;
  - activation through `ActivationSchedule`, exposed as `current_activation` and `current_time`;
  - no import of `iso` or `aniso`;
  - `check_uniqueness`: sufficient condition of a unique isometric CE equilibrium, k_e + k_p > f_max max|f_FL′λ| (Mooney SEE, or any SEE while taut). The max is sampled in u = (λ − λ_opt)/W ∈ [max(−4, −λ_opt/W), 4], so it holds for any W (a fixed window λ ∈ [0.05, 3] missed it for W ≳ 2).

**Tests.** `tests/unit/materials/cell/active/hill/test_{functions,law}_cell_hill.py` and the fixtures in `tests/conftest.py` (`HillSetup`, `make_setup`).

## 6. Contractile element operator $R_a$

**Changes.** New `src/fenicsoft/operators/cell/hill/contractile_evolution.py` (no `__init__.py`, export from `operators/cell/__init__.py`): `ContractileEvolution`, with R_a = [f_max α f_FL h(D_a) + η₀D_a + T_p − T_e] δε and D_a = (ε_a − ε_aⁿ)/Δt, plus `scale_factor`. Register it per step (`k_step`): `set_dt` is only called on step operators.

**Tests.** `tests/unit/operators/cell/hill/test_contractile_evolution.py`.

## 7. Saturated proliferation: clip the saturation

**Changes.**
- In fenicsoft `operators/cell/proliferation/saturated.py`, use sat = max(0, 1 − Φ_c/Φ_c,max). Today it turns into a sink above Φ_c,max.
- Then delete `hill_playground/operators/cell/proliferation/saturated.py`.

**Tests.** fenicsoft already has `tests/unit/operators/cell/proliferation/test_saturated_prol.py`: merge, do not overwrite.
- Add the playground cases (identical below Φ_c,max, zero source above), comparing with the analytical value r Φ_c (1 − Φ_c/Φ_c,max) instead of the old fenicsoft class (after the merge it is the same class).
- `test_saturated_proliferation_construction` builds the operator with a `MagicMock` `phi_c`: UFL `conditional` (`macaulay` → `floored` → `piecewise`, `dolfin.le`) rejects it once the clip is in. Replace the mock by a `dolfin.Function` or remove that test (the assembly tests cover it).

## 8. Permeability laws $k(\phi_\ell)$

**Changes.**
- New module `src/fenicsoft/materials/saturated_mixture.py`: J_min = Φ_s0 + Φ_c, Φ_ℓ = J − J_min, φ_ℓ = 1 − J_min/J, φ_s = Φ_s0/J. Mixture kinematics shared by the permeability laws and the compaction energy (issue 9), not a law. Φ_c = 0 by default (here and in `PorousPermeabilityParameters`): two-phase mixture, solid + liquid. Reuse candidate: φ_s = φ_s0/J re-coded in `operators/cell/haptotaxis/saturated.py`. Add it to `materials/__init__.py`.
- New package `src/fenicsoft/materials/permeability/`:
- `base.py`: `pulled_back_conductivity` K_L = (k/η) J C⁻¹; `PermeabilityParameters(MaterialParameters)` (k₀/η) and `PorousPermeabilityParameters` (+ Φ_s0 uniform, Φ_c, Φ_c0); `Permeability` (abstract `_ratio()` = k/k₀ read through the cached `ratio`, then `hydraulic_conductivity` and `tensor`) and `PorousPermeability` (+ `porosity` φ_ℓ, `porosity_0` φ_ℓ0, `fluid_ratio` Φ_ℓ/Φ_ℓ0).
- One module per law, with its `Parameters` (own fields only) and `Permeability` (only `_ratio`): `constant`, `kozeny_carman` (0 < Φ_s0 + Φ_c0 enforced: φ_ℓ0 = 1 divides by 0), `holmes_mow` (κ_HM ≥ 0 and M ≥ 0 enforced), `compaction_power` (n ≥ 1 enforced).

**Tests.** `tests/unit/materials/test_saturated_mixture.py`, `tests/unit/materials/permeability/test_permeability_laws.py`.

## 9. Volumetric energy with a compaction point

**Changes.** New `src/fenicsoft/materials/volumetric/compaction.py`: `CompactionVolumetricPenalty` (Ehlers), W = κ(1−J_min)²[x − ln x − 1], with its own `Parameters(kappa, solid_jacobian, x_floor)` (J_min < 1 checked when `solid_jacobian` is a number or a `dolfin.Constant`). Below x_floor, W is continued linearly in x, so S = 2∂W/∂C holds everywhere and S stays finite (checked by finite differences in the test).

**Tests.** `tests/unit/materials/volumetric/test_compaction.py`.

## 10. Orientation diagnostics

**Changes.** New `src/fenicsoft/materials/orientation/display.py`: `current_direction`, `tilt_degrees`, `axis_angle_degrees`, `order_parameter` (head/tail symmetric, `atan2`).

**Tests.** `tests/unit/materials/orientation/test_display.py`.

## 11. Residual rows: quadrature degree, test scale and normalization per row

**Problem.** Every operator of a monolithic residual needs a `scale_factor` = s_row·v/𝒩_row. Today these are hand-written per demo (inline `value_preserving_factor / N_row` in `set_weak_form_and_bcs` of the porous pillar demos, magic factors in the tests).

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

**Follow-up.** The pillar case is recoded in the playground with `ResidualRows` (issue 18; multi-domain: `rows.measure(problem.dV(tags), row)` keeps the subdomain). Φ_c and ε_a are frozen in the pillars by their rate term (`ChangeRate`), not by the identities of equations.tex (see issue 18).

## 12. Hill one-element integration test

**Changes.** `tests/integration/materials/cell/active/hill/{hill_0d_reference.py, hill_one_element.py, test_hill_one_element.py, conftest.py}`: a one-element FE run of gel + Hill, compared with an independent 0D backward-Euler reference (scipy). Four cases: isometric λ = 1 and λ = 1.04, traction-free faces, and traction-free faces through `ResidualRows` with other per-row degrees (`displacement`: 1, `eps_a`: 0) and row normalizations (μ, f_max).
- Fixed 2026-10-18: the normalizations were passed in the `test_scales` slot of `ResidualRows.from_problem` (rows multiplied instead of divided; the solution is the same, so the test still passed). They are now passed by keyword; re-run the test.
- The operators take their measure and `scale_factor` from `ResidualRows`. In the homogeneous one-element state every integrand is constant per cell, so the `rows` case must equal both the 0D reference and the default free run to round-off. Since the solution cannot reveal a mis-wiring, `run` also returns `attrs["wiring"]` (scale factor and quadrature degree of each operator), checked by `test_rows_are_wired_to_the_operators`.
- The 0D reference takes α(t_n) from the FE table; `test_activation_follows_the_ramp` checks that column against min(t/t_ramp, 1).
- The figure `hill_one_element.png` and `errors.csv` are written to `outputs_hill_one_element/`. The figure shows λ_a, T_e, the free-face gel stretches and the error history, FE (markers) against 0D (lines); the `rows` case is dashed with `x` markers (it coincides with the free case). The error history (`error_history`) is the per-step max over ε_a, λ, λ_t, T_e/T and P_xx/T, the norms of `errors`.
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
- `demos/hill_box/{hill_box_uniaxial.py, hill_box_scaling.py, hill_box_outputs.py, post_processing.py, README.md}` → fenicsoft `demos/smooth_muscle_cells/porous_hydrogel/hill_box/` (elementary: simple geometry), figure `docs/src/_static/resources/hill_box_bcs.{tex,svg}`.
- fenicsoft's `terzaghi_analytics.py` and the SI Terzaghi demos now carry condensed docstrings and the figure `terzaghi_plane_strain.{tex,svg}` (copied from the playground in #93).
- `demos/analytical_consolidation_1D/{nondim_parameters, nondim_operator, terzaghi_analytics.py, post_process.py, README.md}` → fenicsoft `demos/analytical_consolidation_1D/` (next to `single_step/`, `initialising_step/`).
  - **Name clash**: fenicsoft already has `terzaghi_analytics.py` and `post_process.py` there, with a different content and API (`single_step/` and `initialising_step/` import `post_process`). Either merge (keep fenicsoft's names used by the SI demos, add `TerzaghiCase`, `read_qois`, `compute_errors`, `plot_results`), or rename the playground ones (e.g. `terzaghi_case.py`, `post_process_nondim.py`) before the copy.
- The demo is nondimensionalised **by the parameters only** (`hill_box_scaling.Coefficients`, every `scale_factor` 1, solve in t̂). The three-mode version (`nondimensionalization` = `none`, `operator`, `parameters`) is the test reference `tests/integration/demos/hill_box_modes/hill_box_all_modes.py`, with `jacobian_scaling.py`.
- `tests/integration/demos/{test_hill_box_scaling.py, test_hill_box_inputs.py, test_hill_box_post_processing.py, hill_box_modes/}` → `demos/.../hill_box/test/` with a `sys.path` conftest (fenicsoft has no `tests/integration/demos/`). `testpaths = ["tests"]` does not collect it: run it explicitly (`pytest demos/.../hill_box/test`), or decide another location before the merge request.
- Replace the `hill_playground.*` imports by `fenicsoft.*`.
- `UserInputs` options (demo): `permeability_law`, `volumetric_law`, `see_variant`, `cell_transport`, `remodelling`, `xdmf_si_only`, `quadrature_degrees`; `hill_box_all_modes.UserInputs` adds `nondimensionalization`.
- Outputs:
  - `displacement_si` and `pressure_si` are scaled copies of the unknowns on their own CG spaces (vector scaling). A global `dolfin.project` on CG2 cost about 5 s per output, 15 % of the run.
  - The other FOIs use `update_type="local_solver"`, a cell-wise projection on the DG spaces, instead of `"project"`.
  - All output updates went from 17.1 s to 0.3 s on the size/2 box.
- Rows through `ResidualRows` (issue 11): in the demo, `build_residual_rows()` gives each operator `rows.measure(dV, row)` (quadrature degree per row only). In `hill_box_all_modes.py` it also gives `rows.scale_factor(row, v)`:
  - `TermValues` holds the value factors (gel L₀², fluid μ₀L₀², volume L₀³, Darcy μ₀L₀, cell flux L₀).
  - `row_test_scales()` holds s_row (L₀ for `displacement`, μ₀ for `liquid_pressure`) and `row_normalizations()` holds 𝒩_row (`"operator"` mode only).
  - Row names are the subsols: `displacement`, `liquid_pressure`, `cell_fraction`, `substrate_smoothed`, `theta`, `phi`, `eps_a`. The cell flux Q_c is in both `liquid_pressure` and `cell_fraction`.
  - The `quadrature_degrees` keys are checked against `UNKNOWNS` (a typo raises an error); rows of switched-off unknowns are ignored. Default: empty, i.e. `quadrature_degree` everywhere.
  - Measured on the size/2 box: `{"displacement": 3}` and the other rows at 2 bring assembly from 1.83 to 1.26 s per iteration, with QOIs within 3e-4.
  - Checked after the refactor: same QOIs as before (difference 0.0), the 3 modes identical (3e-15), and the same holds with per-row degrees (`operator` vs `none` with rows: 2.5e-15).
- Conditioning: `tests/integration/demos/hill_box_modes/jacobian_scaling.py` (with the 3-mode reference `hill_box_all_modes.py`) assembles J at the first step for the 3 modes (κ₂ by dense SVD, median |J_ii| per row, printed as a table). The mass rows (`cell_fraction`, `theta`, `phi`, ~1e-3; `substrate_smoothed`, ~|Ω̂_e| ≈ 0.02, no t₀/Δt factor) and the stiffness rows (`displacement`, `liquid_pressure`, ~2) differ by about (t₀/Δt) ĥ²/6 (ĥ = h/L₀ = 0.5, t₀ ≈ 1 s, Δt = 5 s): discretization and time step, not scaling. Test `tests/integration/demos/test_hill_box_scaling.py` (about 80 s): `operator` J = `parameters` J (≤ 1e-10), interior diagonal blocks O(1), κ₂: none 1.7e20, operator 4.3e4, parameters 4.3e4.
- Cost profile (size/2, warm JIT cache): MUMPS solve ≈ 78 %, assembly ≈ 21 %, outputs < 1 %. Turning off `cell_transport` and `remodelling` cuts the solve to 5.2 s per iteration (−37 %). size/4: about 210 s per solve and about 5 GB.

**Done when** the three `nondimensionalization` modes of `hill_box_all_modes.py` give identical QOIs (measured: 3e-15) and identical Jacobians (`operator` vs `parameters`), and an `mpirun` run gives a readable `.out`.

## 15. Follow-ups

- `iso.law` imports `ActivePowerConfig` (and the `p_act` machinery) from `cell/active/activation.py` (issue 4), and removes its copy (`aniso.law` is archived since #93).
  - `iso` keeps `p_cell` as an alias of `ActivationSchedule.p_act`, plus `current_power`.
  - Its unit test reads private attributes and must be adapted: `_current_manager` and `_step_managers` in `test_law_cell_iso.py`.
  - The equivalence test then becomes redundant.
  - The archived demos import `ActivePowerConfig` from `fenicsoft.materials.cell.active.iso.law`: point them to `activation.py` too (or leave them, they are not maintained).
- `ScaledKinematics` (pillar demo `src/`) went with the deleted scaled pillar demo (#93): the mesh is scaled, so `Kinematics` is exact.
- After each merge:
  - remove the transferred paths from `NEW_BRANCHES` in `tests/environment/test_env.py`;
  - delete the playground copies;
  - switch the playground imports to fenicsoft.
- Devcontainer: the playground mounts the JIT cache `/home/fenics_user/.cache/dolfin` (`DIJITSO_CACHE_DIR` of the image) as the named volume `hill-dijitso-cache`; do the same in fenicsoft. The Hill forms take about 90 s of JIT compilation on a fresh container.
- `set_dt` with a NumPy scalar: `dolfin_mech.core.TimeVaryingConstant.set_value_sca` only accepts `int`/`float`/`list`/`ndarray` (exact `type(val) in ...`), so a `numpy.float64` Δt is silently ignored and Δt stays 0 in every operator (`ChangeRate`, `OrientationChange`, `ContractileEvolution`, `Inertia`). The operators are left as in dolfin_mech/fenicsoft (`self.tv_dt.set_value(dt)`); the playground casts at the source instead (`scaling.time_steps` returns Python `float`, test `test_numpy_t0_gives_python_floats`). Root fix to raise in dolfin_mech: `isinstance(val, numbers.Real)` in `set_value_sca`.
- Docs: `skip_globals` in `docs/src/conf.py` now skips only undocumented module data (documented constants and their `:data:` cross-references are rendered); same change in fenicsoft.
- Tooling: autoapi documents the tests too, so docstrings there must be valid reST. Write `|x|` inside `:math:`, never bare (`make docs` fails on "Undefined substitution").

## 16. Subdomain indicators (gel/pillar data fields)

**Changes.** New `src/fenicsoft/materials/domain_fields.py`:
- `cell_indicator(mesh, domains, tags)`: χ_T, 1 on the tagged cells (DG0), from `fenicsoft.mesh.build_indicator_meshfunction`. Only owned dofs are written (ghost cells, `ghost_mode != "none"`, have dofs past the local vector). Piecewise data a = a_out + (a_in − a_out) χ_T (Φ_s0, conductivity, solid stress).
- `interior_vertex_indicator(mesh, domains, tags)`: χ_T^int, 1 at the vertices whose cells are all tagged (CG1), 0 on the interface; one assembly of ∫(1 − χ_T)φ_v, exact across MPI partitions (the vertex loop of the fenicsoft demo looked at the local cells only: a partition boundary could put a gel cell out of sight).
- Replaces the side files of the archived `archives/demos/smooth_muscle_cells/porous_hydrogel/microtissue_pillar_spherical` (the `_scale` copy is deleted): `domain_fields.build_domain_split_dg0_field` (→ `cell_indicator`), `permeability.py` (→ `materials.permeability.constant` on the pillar measure), `cell_fraction_porous.build_initial_phi_s0_field` / `build_initial_phi_c_field` (→ the two indicators), `validate_gel_saturation` (→ `GelInputs.__post_init__`, `saturated_mixture.compaction_jacobian`), `PillarInteriorIdentity` (→ frozen rate, issue 18).

**Tests.** `tests/unit/materials/test_domain_fields.py`.

## 17. Free-floating cylinder demo

**Changes.** `demos/hill_cylinder_free/{hill_cylinder_free.py, hill_cylinder_scaling.py, hill_cylinder_outputs.py, cylinder_post_processing.py, README.md}` → fenicsoft `demos/smooth_muscle_cells/porous_hydrogel/hill_cylinder_free/` (elementary: simple geometry), figure `docs/src/_static/resources/hill_cylinder_free_bcs.{tex,svg}`.
- Self-contained demo (imports no other demo): the Hill box model on a cylinder (`CylinderGeometry`), L₀ = H, drained on WALL, TOP, BOTTOM; nondimensionalised by the parameters.
- Rigid modes: isostatic 3-2-1 support (pointwise BCs, `dolfin_mech.core.PinpointSubDomain`) at the vertices nearest to (0,0,0), (R,0,0), (0,R,0), found over all ranks (`nearest_vertex`). Reuse candidate for `fenicsoft.constraints` if another free-body demo needs it.
- QOIs `top_center_disp_z` (height change), `wall_disp_y` (−ΔR at mid-height, probe inset by h_max²/(4R)), `mean_jacobian` (V/V₀).

**Done when** a local run reaches the plateau; checked here: one step (α 0 → 0.1, 1 s) on a 0.6 × 0.3 mm cylinder converges, serial = `mpirun -n 2` to 1e-15, and the post-processing reads the results.

## 18. Microtissue between pillars demo

**Changes.** `demos/hill_microtissue_pillars/{pillars_biaxial.py, pillars_scaling.py, pillars_outputs.py, pillars_cut_to_core.py, pillars_uniaxial_release.py, pillars_post_processing.py, batch_hill_microtissue_pillars.sh, README.md}` → fenicsoft `demos/applications/hill_microtissue_pillars/` (complex: multi-patch geometry, two stages and a cut), which supersedes the archived `microtissue_pillar_spherical` (aniso active law, SI); figure `hill_microtissue_pillars_bcs.{tex,svg}`. Being an application, its module docstrings may keep more context than an elementary demo, in the same style.
- Self-contained demo (imports no other demo; stage B and the cut reuse stage A of the same folder).
- Geometry: the lid sits **on top of the gel**, z ∈ [T, T + t_l] (`MicrotissuePillarsMeshGenerator` builds it so), base at z = T − (h_p − t_l); `SIDE_PILLARS_*` = pillar boundary minus base and lid top: the interface (shaft wall in the gel + annulus under the lid, not part of `TOP_GEL`, not drained) plus the shaft below the gel and the lid rim. The `fenicsoft.mesh_geometries.microtissue_pillars` docstrings (lid position, `pillar_base_z`, `PILLAR_LID_TOP_*`, `SIDE_PILLARS_*`) were corrected to match the code in #93.
- Gel: Hill box model on the gel measure; pillars: Neo-Hooke + quadratic volumetric (pillar moduli / μ₀), Darcy with r_k k₀ (constant law), R_θ,φ and R_s over the whole mesh, R_s with the gel Φ_s0 everywhere (a pillar value would make φ̃_s jump across the first gel element and pull the cells onto the pillars by haptotaxis); output `fluid_fraction` = χ_gel φ_ℓ (0 in the pillars, no pillar Φ_s0).
- Φ_c and ε_a frozen in the pillars by `ChangeRate` on the pillar measure (∫(x − xⁿ)/Δt δx): Φ_c = 0, λ_a = 1, no flux through the interface, no scale. equations.tex updated accordingly (2026-10-23, backup `equations.tex.bak-2026-10-23`): eq:Rc, eq:Ra, eq:Rc-nd, eq:Ra-nd, tab:sf, the strong form on Ω_pil and the new Remark rem:pillar-pin, which explains why the former identities (1/t₀)∫Φ_c δc and μ₀∫(λ_a − 1)δλ were dropped: with P1 Φ_c the identity also enters the interface rows (cell sink of relative weight Δt/t₀, up to 3e3 in the hold) and gives Φ_c < 0 inside (checked on a two-domain cube: interface value pulled from 1 to < 0.99).
- Initial Φ_c: Φ_c0 at every vertex touching the gel, 0 strictly inside the pillars (`materials.domain_fields`, issue 16; needs Φ_c ∈ P1: `displacement_degree` = 2, checked).
- Stage A → B: SI checkpoints of u, p, Φ_c, θ, φ, ε_a (`HANDOVER`; ε_a was not transferred in the fenicsoft demo, the aniso law had no internal variable); mesh restored to SI before writing. Cut (serial): `fenicsoft.mesh.extract_submesh_by_tags`, `project_facet_tags`, `transfer_field_by_interpolation` (CG) and `build_dg0_gather_map` / `transfer_dg0_field` (DG0); the cut planes are tagged `CUT_FACES` and drained in stage B (p = 0 there checked in the results).
- fenicsoft demo defects not carried over: `InterfaceResistance` on `dS(SIDE_PILLARS_*)` acts on exterior facets only (`problem.dS` is `exterior_facet`), i.e. on the pillar legs and lid surfaces, not on the gel/pillar interface (not used here: the interface is no-flux by construction); the vertex loop of the initial Φ_c sees the local cells only (MPI), replaced by `interior_vertex_indicator`; cut faces undrained; `cut_to_core` re-coded `transfer_field_by_interpolation`; `io.configure_environment` empties the results folder, so the cut reads stage A's folders directly.
- Batch for asimov kept (`batch_hill_microtissue_pillars.sh`): same docker pipeline, the whole playground mounted (`PYTHONPATH=/app/src`).

**Done when** stage A, cut, stage B and the post-processing run locally; checked here: one step of each stage on a half-size geometry (T = 25 µm), stage B starting exactly from the stage A state (same probe values), serial = `mpirun -n 2` (stage A) to 1e-15, post-processing on both stages.
