# Hill-preparation-playground

Workspace to go **from equations to simulation** for the Hill-chain smooth muscle cell model
(`lms_equations_poro/equations.tex`, implementation plan in `code.tex`), before transfer into
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
├── src/hill_playground/    # -> fenicsoft/
│   ├── materials/cell/active/hill/
│   └── operators/cell/hill/
├── tests/{unit,integration,environment}/
├── demos/
└── docs/src/
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
   | `local` (default) | a throw-away copy of `../fenicsoft` at the ref; your checkout is untouched | local branch, `origin/<branch>`, tag or sha |
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
make fenicsoft FENICSOFT_REF=77-add-micro-tissue-pillar-mesh-generator-for-mostert-et-al-demo
make fenicsoft FENICSOFT_SOURCE=editable
make fenicsoft            # back to what fenicsoft.env says
```

A branch that only exists on the GitLab must be fetched on the host first (`git fetch` in `../fenicsoft`)
for the `local` mode, or use `FENICSOFT_SOURCE=remote`.

## Daily commands

| | |
| --- | --- |
| `make test` / `make cov` | pytest (coverage ≥ 90 %) |
| `make linter` / `make format` | ruff check / format |
| `make docs` | Sphinx → `~/Documents/local/build_docs_hill` |
| `make clean` | remove caches and outputs |
