# Transfer to fenicsoft: elementary issues

Everything in `hill_playground/<path>` targets `fenicsoft/<path>` (same sub-package names).
One issue = one merge request, in the order of the table below: each one only needs the issues above it.
Reference branch: fenicsoft `main`.

**fenicsoft conventions** (apply them in every transfer merge request):
- Layout: `Step` is in `fenicsoft.integration`, the mesh cut and field transfer in `fenicsoft.mesh`;
  `fenicsoft.io` only reads and writes files. Tests mirror `src` (`tests/unit/<path>/`).
- Demos: elementary cases on simple geometries in `demos/`, complex cases (real geometries, multi-stage pipelines,
  reproduced experiments) in `demos/applications/`. Each demo module docstring gives the problem, the BCs with a TikZ
  figure (`docs/src/_static/resources/<demo>_bcs.{tex,svg}`), the residuals and the validation.
- Archives: superseded code in `archives/{src,demos,tests}` (not installed, tested nor linted).
- Docstrings: Google style, short (first line = objective, then the equation, `Args`/`Returns` without types, no
  history; tests: one line stating what is verified), see fenicsoft README, "Contributing".
- `make linter` (ruff of the pre-commit hook, D rules included), `make cov` (100 % of the transferred modules) and
  `make docs` (`sphinx -W`) must pass.
- Tests: every float comparison states its tolerance, guarded by `tests/conventions/test_explicit_tolerances.py`.
  Exact arithmetic: `rel=1e-12, abs=0.0`; zero target: `rel=0.0, abs=...`. Shared test helpers: fenicsoft
  `tests/conftest.py`.
- Floors, caps and positive/negative parts use `fenicsoft.ufl_functions`, never an inline `dolfin.conditional`.

**`__init__.py` files** are never copied: merge their imports and `__all__` into the existing fenicsoft ones.
fenicsoft has no `__init__.py` in `operators/cell/*/` sub-folders: export the new classes from
`operators/cell/__init__.py` instead. New top-level sub-package: add it to `fenicsoft/__init__.py` and its `__all__`.

**Status.** No open issue.

Legend: **new**: the file does not exist in fenicsoft. **change**: patch an existing fenicsoft file.
**copy**: a fenicsoft file copied here and modified; report the diff and do not create a second file.

| # | Issue | Kind | Depends on |
|---|---|---|---|
| 0 | *(title)* | new / change / copy | – |

<!-- Template of one issue section:

## 0. Title

**Objective.** One sentence, with the equation.

**Files.** `src/hill_playground/<path>.py` → `src/fenicsoft/<path>.py`; tests `tests/unit/<path>/test_<name>.py`.

**Changes.** What is added or patched in fenicsoft (classes, functions, exports).

**Tests.** What each test verifies.

**Done in #<MR>.** What was merged, what stays in the playground.
-->
