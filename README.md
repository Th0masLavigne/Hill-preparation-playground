# Hill-preparation-playground

> **Showcase.** This branch is a snapshot (27/10/2026) of the playground where the Hill-chain smooth muscle cell
> model was developed from its equations, before its transfer into
> [`fenicsoft`](https://src.koda.cnrs.fr/lms1/jean-marc-allain/fenicsoft), the FEniCS-based modelling package of the
> LMS (École polytechnique). That code now lives in fenicsoft (merge requests #95 to #106); this snapshot is kept
> as it was, with its fenicsoft and dolfin_mech versions pinned in `.devcontainer/fenicsoft.env` so it still
> installs and runs. New developments start from the `devel-base` branch.

Workspace to go **from equations to simulation** for the Hill-chain smooth muscle cell model,
before transfer into
[`fenicsoft`](https://src.koda.cnrs.fr/lms1/jean-marc-allain/fenicsoft).

The package `hill_playground` mirrors the `fenicsoft` tree one-to-one:
`src/hill_playground/<path>` → `src/fenicsoft/<path>` (same coding standards: ruff, tabs,
Google docstrings, Sphinx autoapi). It reuses fenicsoft's objects whenever possible and only
holds what is new.

```text
.
├── .devcontainer/          # VS Code devcontainer (same image/settings as fenicsoft)
│   ├── devcontainer.json
│   ├── fenicsoft.env       # <- choose the fenicsoft branch/tag/commit here
│   ├── install_fenicsoft.sh
│   ├── initialize.sh       # host-side: base image + bind-mount folders
│   └── secrets.env.example # token for the remote mode
├── src/hill_playground/    # -> fenicsoft/ (one issue per item: TRANSFER.md)
│   ├── ufl_functions.py              # piecewise, floored, capped, macaulay(_negative)       [new]
│   ├── integration/                  # DecimatedTimeIntegrator, field selection + rank-0 logs [copy]
│   ├── io/                           # XDMF geometry/time back to SI                         [new]
│   ├── materials/cell/active/        # activation.py: shared activation schedule             [new]
│   ├── materials/cell/active/hill/   # Hill law (R_u) + shared UFL functions                 [new]
│   ├── materials/permeability/       # k(phi_l) laws on a common base class                  [new]
│   ├── materials/saturated_mixture.py # J_min, Phi_l, phi_l, phi_s (mixture kinematics)      [new]
│   ├── materials/domain_fields.py    # subdomain indicators (gel/pillar data fields)         [new]
│   ├── materials/volumetric/         # volumetric energy with a compaction point             [new]
│   ├── materials/orientation/        # orientation diagnostics                               [new]
│   ├── operators/residual_rows.py    # quadrature degree, test scale, normalization per row   [new]
│   ├── operators/cell/hill/          # ContractileEvolution (R_a)                            [new]
│   ├── operators/cell/proliferation/ # clipped SaturatedProliferation                        [change]
│   └── scaling/                      # CharacteristicScales (promoted)                       [new]
├── tests/{unit,integration,environment,conventions}/
├── demos/analytical_consolidation_1D/  # Terzaghi, nondimensionalised: start here
├── demos/hill_box/                     # gel slab clamped on two faces
├── demos/hill_cylinder_free/           # free-floating cylinder
├── demos/hill_microtissue_pillars/     # microtissue between pillars (stage A, cut, stage B, batch)
├── docs/src/
└── TRANSFER.md             # transfer plan to fenicsoft, one merge request per issue
```

## Setup (VS Code + Dev Containers)

Expected layout on the host (same as today):

```text
~/Documents/git/
├── fenicsoft/                    # your clone (mounted in the container)
└── Hill-preparation-playground/  # this repo
~/Documents/local/build_docs_hill # created automatically, receives `make docs`
```

1. Choose the fenicsoft version in `.devcontainer/fenicsoft.env`:

   | `FENICSOFT_SOURCE` | Installs | `FENICSOFT_REF` |
   | --- | --- | --- |
   | `local` (default) | a throw-away copy of `../fenicsoft` at the ref; your checkout is untouched | local branch, `origin/<branch>`, tag or sha (pinned, see above) |
   | `editable` | `pip install -e ../fenicsoft` as currently checked out (edit both repos live) | ignored |
   | `remote` | from the GitLab, needs `FENICSOFT_TOKEN` (`secrets.env` or host env) | branch, tag or sha |

2. *Command palette* → **Dev Containers: Reopen in Container**.
   - `initialize.sh` reuses the `dolfin:v2019.2.0.dev0` image of fenicsoft's devcontainer, or builds it from
     `../fenicsoft/docker/dolfin/Dockerfile` if missing (Apple silicon: `export DOLFIN_PLATFORM=linux/amd64`
     to force x86_64).
   - `make install` then installs fenicsoft (+ restructured `dolfin_mech@master`), this package in editable
     mode and the pre-commit hooks.

3. Check: `make fenicsoft-info` and `pytest tests/environment -s`.

### Switching fenicsoft branch

Inside the container, no rebuild needed:

```bash
make fenicsoft FENICSOFT_REF=<branch|tag|sha>
make fenicsoft FENICSOFT_SOURCE=editable
make fenicsoft            # back to what fenicsoft.env says
```

A branch that only exists on the GitLab must be fetched on the host first (`git fetch` in `../fenicsoft`)
for the `local` mode, or use `FENICSOFT_SOURCE=remote`.

### Batch image (asimov)

`docker/hill_playground/build.sh` builds `hill_playground:<user>` from the same pieces (base `dolfin:v2019.2.0.dev0`
with your uid, `install_fenicsoft.sh` at the ref of `fenicsoft.env`, `FENICSOFT_REF=...` to override) plus this
package installed editable. The batches (`demos/*/batch_*.sh`) mount their own folder only, and the clone given by
`HILL_PLAYGROUND_DIR` over the editable install. Needs `../fenicsoft` (or `FENICSOFT_SOURCE=remote` + token) and
BuildKit (Docker ≥ 23).

## Daily commands

| | |
| --- | --- |
| `make test` / `make cov` | pytest (coverage ≥ 90 %) |
| `make linter` / `make format` | ruff check / format |
| `make docs` | Sphinx → `~/Documents/local/build_docs_hill` |
| `make clean` | remove caches and outputs |

## Demos

| Demo | Content |
| --- | --- |
| `demos/analytical_consolidation_1D/` | Terzaghi's consolidation solved in dimensionless variables, by the parameters (`nondim_parameters/`, one step) and by the operators (`nondim_operator/`, initialising step): error vs analytical and figures |
| `demos/hill_box/` | Active poro-elastic gel slab with Hill-chain cells, clamped on its two $x$-faces, drained and traction-free elsewhere |
| `demos/hill_cylinder_free/` | Free-floating cylinder, drained and traction-free on its whole boundary, isostatic 3-2-1 support |
| `demos/hill_microtissue_pillars/` | Microtissue between twelve pillars: stage A (biaxial), cut to the core, stage B (uniaxial release), ROI post-processing, batch script |
