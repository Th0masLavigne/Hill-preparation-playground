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
| 11 | Hill one-element integration test | new | 5, 6 |
| 12 | Terzaghi integration test, nondimensionalization methods | new | 2 |
| 13 | Hill box demo | new | 1–10 |
| 14 | Follow-ups (iso/aniso activation, `ScaledKinematics`, tooling) | change | 4 |

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
- Make the pillar demos import `fenicsoft.scaling`, delete their `src/characteristic_scales.py`, and fix their test `test_from_consolidation_scale_scales_permeability_over_viscosity_by_stress_over_kappa` (it expects the inverted ratio).

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

## 11. Hill one-element integration test

**Changes.** `tests/integration/materials/cell/active/hill/` (3 files): a one-element FE run of gel + Hill, compared with an independent 0D backward-Euler reference (scipy). Three cases: isometric λ = 1 and λ = 1.04, and traction-free faces.

**Done when** ε_a, P_xx and (λ, λ_t) match the 0D reference to 1e-12–1e-10.

## 12. Terzaghi integration test, nondimensionalization methods

**Changes.**
- `tests/integration/analytical_consolidation_1D/{terzaghi_nondim.py, test_terzaghi_nondim.py, conftest.py}` → fenicsoft `tests/integration/analytical_consolidation_1D/nondimensional/`.
- Three modes: SI, operator `scale_factor`, and parameters converted with `nondimensionalize_*`. The figure and `errors.csv` go to `outputs_terzaghi_nondim/`; the error table is printed in the pytest summary.
- Merge the Terzaghi series with fenicsoft's `terzaghi_analytics.py` if wanted.

**Done when** the pressure and settlement errors are below 1 % for T ≥ 0.05 (measured: 8.6e-3 and 3.9e-3), and the three modes are identical to round-off.

## 13. Hill box demo

**Changes.**
- `demos/hill_box/{hill_box_uniaxial.py, hill_chain_helpers.py, README.md}` → fenicsoft `demos/smooth_muscle_cells/porous_hydrogel/hill_box/`.
- Replace the `hill_playground.*` imports by `fenicsoft.*`.

**Done when** the three `nondimensionalization` modes give identical QOIs (measured: 3e-15), and an `mpirun` run gives a readable `.out`.

## 14. Follow-ups

- `iso.law` and `aniso.law` import `ActivePowerConfig` (and the `p_act` machinery) from `cell/active/activation.py` (issue 4), and remove their copies.
  - `aniso` keeps `p_act` and `iso` keeps `p_cell` as aliases of `ActivationSchedule.p_act`, plus `current_activation` / `current_power`.
  - Their unit tests read private attributes and must be adapted: `_current_manager` and `_step_managers` in `test_law_cell_iso.py`, `_current_config` in `test_law_cell_aniso.py`.
  - The equivalence test then becomes redundant.
- `ScaledKinematics` (pillar demo `src/`) stays dormant: the mesh is scaled, so `Kinematics` is exact. Delete it, or keep it for a "physical mesh" variant (all gradients and measures would then need scaling).
- After each merge:
  - remove the transferred paths from `NEW_BRANCHES` in `tests/environment/test_env.py`;
  - delete the playground copies;
  - switch the playground imports to fenicsoft.
- Tooling: autoapi documents the tests too, so docstrings there must be valid reST. Write `|x|` inside `:math:`, never bare (`make docs` fails on "Undefined substitution").
