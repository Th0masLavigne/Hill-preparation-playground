# Hill-preparation-playground

Workspace to go **from equations to simulation** before transfer into
[`fenicsoft`](https://src.koda.cnrs.fr/lms1/jean-marc-allain/fenicsoft).

Branch `devel-base`: empty template for new developments (structure, devcontainer, Docker image, tests and docs
tooling). The Hill-chain smooth muscle cell work developed here is now in fenicsoft (merge requests #95 to #106);
its pre-transfer state is the `main` branch.

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
├── docker/hill_playground/ # batch image (asimov): same stack + this package
├── src/hill_playground/    # -> src/fenicsoft/ (same sub-package names)
├── tests/{unit,integration,environment,conventions}/
├── demos/                  # one folder per demo -> fenicsoft demos/ or demos/applications/
├── docs/src/
└── TRANSFER.md             # transfer plan to fenicsoft, one merge request per issue
```

## Workflow

1. Write the equations (LaTeX), then the code in `src/hill_playground/<path>`, its unit tests in
   `tests/unit/<path>` and, if needed, a demo in `demos/<name>/`.
2. Before writing anything, check that fenicsoft does not already provide it (or can be extended).
3. List each transferable item as one issue in `TRANSFER.md` (target path, kind, dependencies).
4. One fenicsoft merge request per issue; once merged, delete the playground copy and import from fenicsoft.

`tests/environment/test_env.py` checks that every playground sub-package exists in fenicsoft, unless declared in
`NEW_BRANCHES`.

## Setup (VS Code + Dev Containers)

Expected layout on the host:

```text
~/Documents/git/
├── fenicsoft/                    # your clone (mounted in the container)
└── Hill-preparation-playground/  # this repo
~/Documents/local/build_docs_hill # created automatically, receives `make docs`
```

1. Choose the fenicsoft version in `.devcontainer/fenicsoft.env`:

   | `FENICSOFT_SOURCE` | Installs | `FENICSOFT_REF` |
   | --- | --- | --- |
   | `local` (default) | a throw-away copy of `../fenicsoft` at the ref; your checkout is untouched | local branch, `origin/<branch>`, tag or sha (default `main`) |
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
make fenicsoft FENICSOFT_REF=main
make fenicsoft FENICSOFT_SOURCE=editable
make fenicsoft            # back to what fenicsoft.env says
```

A branch that only exists on the GitLab must be fetched on the host first (`git fetch` in `../fenicsoft`)
for the `local` mode, or use `FENICSOFT_SOURCE=remote`.

### Batch image (asimov)

`docker/hill_playground/build.sh` builds `hill_playground:<user>` from the same pieces (base `dolfin:v2019.2.0.dev0`
with your uid, `install_fenicsoft.sh` at the ref of `fenicsoft.env`, `FENICSOFT_REF=...` to override) plus this
package installed editable. A batch script (`demos/<name>/batch_*.sh`) mounts its own demo folder, and the clone
given by `HILL_PLAYGROUND_DIR` over the editable install. Needs `../fenicsoft` (or `FENICSOFT_SOURCE=remote` +
token) and BuildKit (Docker ≥ 23).

## Daily commands

| | |
| --- | --- |
| `make test` / `make cov` | pytest (coverage ≥ 90 %) |
| `make linter` / `make format` | ruff check / format |
| `make docs` | Sphinx → `~/Documents/local/build_docs_hill` |
| `make clean` | remove caches and outputs |
