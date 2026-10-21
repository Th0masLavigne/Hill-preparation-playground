r"""Every float comparison in ``tests/`` states its tolerance.

.. math::
	|a - b| \le \max(\mathrm{rel}\,|b|,\ \mathrm{abs}) \quad \text{(pytest, math)}, \qquad
	|a - b| \le \mathrm{atol} + \mathrm{rtol}\,|b| \quad \text{(numpy)}

Required keywords (the defaults are hidden and scale dependent, e.g. ``numpy.isclose``'s ``atol=1e-8``):

* ``pytest.approx``: ``rel`` or ``abs``;
* ``math.isclose``: ``rel_tol`` or ``abs_tol``;
* ``numpy.isclose``, ``numpy.allclose``, ``numpy.testing.assert_allclose``: ``rtol`` and ``atol``.
"""

import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

TESTS_DIR = Path(__file__).resolve().parents[1]

#: call name -> (keyword groups; each group needs at least one of its keywords)
RULES: Dict[str, Tuple[Tuple[str, ...], ...]] = {
	"approx": (("rel", "abs"),),
	"isclose": (("rel_tol", "abs_tol"),),
	"np.isclose": (("rtol",), ("atol",)),
	"np.allclose": (("rtol",), ("atol",)),
	"assert_allclose": (("rtol",), ("atol",)),
}


def _rule_key(call: ast.Call) -> str:
	"""Name of the comparison called, ``""`` if not a tolerance comparison.

	Args:
		call: Call node.

	Returns:
		Key of :data:`RULES` or ``""``.
	"""
	func = call.func
	name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
	owner = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else ""
	if name in ("isclose", "allclose") and owner in ("np", "numpy"):
		return f"np.{name}"
	if name == "isclose" and owner == "math":
		return "isclose"
	return name if name in ("approx", "assert_allclose") else ""


def missing_tolerances(path: Path) -> List[str]:
	"""Comparisons of ``path`` lacking a required tolerance keyword.

	Args:
		path: Python test file.

	Returns:
		``"file:line name needs kw"`` entries (empty if compliant).
	"""
	issues = []
	for node in ast.walk(ast.parse(path.read_text())):
		key = _rule_key(node) if isinstance(node, ast.Call) else ""
		if not key:
			continue
		given = {kw.arg for kw in node.keywords}
		for group in RULES[key]:
			if not given.intersection(group):
				issues.append(f"{path.relative_to(TESTS_DIR)}:{node.lineno} {key} needs {' or '.join(group)}")
	return issues


@pytest.mark.parametrize("path", sorted(TESTS_DIR.rglob("*.py")), ids=lambda p: str(p.relative_to(TESTS_DIR)))
def test_tolerances_are_explicit(path: Path) -> None:
	"""No implicit default tolerance.

	Args:
		path: Python test file.
	"""
	issues = missing_tolerances(path)
	assert not issues, "\n".join(issues)
