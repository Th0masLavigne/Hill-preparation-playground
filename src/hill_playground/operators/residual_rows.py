r"""Per-row settings of a monolithic residual: quadrature degree, test-function scale, row normalization.

A row of the monolithic system gathers every term tested by the same test function, i.e. by
the same subsol: a row is named after its subsol (``"displacement"``, ``"liquid_pressure"``, ...).
With the scaled mesh and unknowns (:math:`\hat{\mathbf X} = \mathbf X/L_0`, :math:`\hat x_k = x_k/s_k`),
the FE test function is the dimensionless :math:`\delta\hat x_k` while the SI residual :math:`R_k` is written
with :math:`\delta x_k = s_k\,\delta\hat x_k` (:math:`s_{\mathbf u} = L_0`, :math:`s_p = \mu_0`, else 1).
For a term of value-preserving factor :math:`v` (SI integral = :math:`v` times the form written on the
scaled mesh and unknowns, with the FE test function):

.. math::
	\tilde R_k = \frac{R_k}{\mathcal N_k}, \qquad
	\texttt{scale\_factor} = \frac{s_k\, v}{\mathcal N_k}, \qquad
	\texttt{measure} = \mathrm{d}V(\text{degree} = q_k)

Missing settings default to :math:`s_k = 1`, :math:`\mathcal N_k = 1` and the problem's
quadrature degree. ``dV(degree=q)`` is integral metadata: FFC applies it to these integrals only,
also inside a summed residual (rank 1) and in the Jacobian derived from it (rank 2).

.. warning::
    Do not mix degrees inside a *scalar* functional (rank 0, e.g. an ``"assembly"`` QOI): the legacy
    FFC of the devcontainer then returns a wrong value (measured: the low-degree integrand counted
    twice). Residuals and Jacobians are exact (``tests/environment/test_env.py``).

Target: ``fenicsoft.operators.residual_rows``.
"""

import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import dolfin


@dataclass(frozen=True)
class ResidualRows:
	r"""Quadrature degree :math:`q_k`, test scale :math:`s_k` and normalization :math:`\mathcal N_k` per row.

	Attributes:
	    rows: Names of the rows (the subsol names of the problem).
	    quadrature_degrees: :math:`q_k` per row (integer :math:`\ge 0`); missing rows use the default.
	    test_scales: :math:`s_k` per row (:math:`> 0`), scale of the row's test function; missing rows use 1.
	    normalizations: :math:`\mathcal N_k` per row (:math:`> 0`); missing rows use 1.
	"""

	rows: Sequence[str]
	quadrature_degrees: Mapping[str, int] = field(default_factory=dict)
	test_scales: Mapping[str, float] = field(default_factory=dict)
	normalizations: Mapping[str, float] = field(default_factory=dict)

	def __post_init__(self) -> None:
		"""Freeze the mappings and check their keys and values.

		Raises:
		    ValueError: Unknown row name, negative or non-integer degree, or non-positive scale/normalization.
		"""
		object.__setattr__(self, "rows", tuple(self.rows))
		for name in ("quadrature_degrees", "test_scales", "normalizations"):
			setting = MappingProxyType(dict(getattr(self, name)))
			object.__setattr__(self, name, setting)
			unknown = sorted(set(setting) - set(self.rows))
			if unknown:
				raise ValueError(f"{name}: unknown rows {unknown}, expected names among {self.rows}.")
		for row, degree in self.quadrature_degrees.items():
			if isinstance(degree, bool) or not isinstance(degree, numbers.Integral) or degree < 0:
				raise ValueError(f"Quadrature degree of row '{row}' must be an integer >= 0, got {degree!r}.")
		for name in ("test_scales", "normalizations"):
			for row, value in getattr(self, name).items():
				if not value > 0.0:
					raise ValueError(f"{name} of row '{row}' must be > 0, got {value!r}.")

	@classmethod
	def from_problem(
		cls,
		problem: Any,
		quadrature_degrees: Optional[Mapping[str, int]] = None,
		test_scales: Optional[Mapping[str, float]] = None,
		normalizations: Optional[Mapping[str, float]] = None,
	) -> "ResidualRows":
		r"""Rows named after ``problem.subsols``.

		Args:
		    problem: ``dolfin_mech`` problem with its subsols registered.
		    quadrature_degrees: :math:`q_k` per subsol name.
		    test_scales: :math:`s_k` per subsol name.
		    normalizations: :math:`\mathcal N_k` per subsol name.

		Returns:
		    The row settings.
		"""
		return cls(
			rows=[subsol.name for subsol in problem.subsols],
			quadrature_degrees=quadrature_degrees or {},
			test_scales=test_scales or {},
			normalizations=normalizations or {},
		)

	def _check(self, row: str) -> None:
		if row not in self.rows:
			raise ValueError(f"Unknown row '{row}', expected one of {self.rows}.")

	def measure(self, measure: dolfin.Measure, row: str) -> dolfin.Measure:
		r"""``measure`` with the degree :math:`q_k` of ``row``, unchanged if none is set.

		Args:
		    measure: Base measure, e.g. ``problem.dV``, ``problem.dV(subdomain)`` or ``problem.dS``.
		    row: Row name (subsol of the term's test function).

		Returns:
		    ``measure(degree=q_k)`` (subdomain kept) or ``measure``.

		Raises:
		    ValueError: Unknown row.
		"""
		self._check(row)
		degree = self.quadrature_degrees.get(row)
		return measure if degree is None else measure(degree=degree)

	def scale_factor(self, row: str, value: float = 1.0) -> float:
		r"""``scale_factor`` of a term of ``row``: :math:`s_k\, v / \mathcal N_k`.

		Args:
		    row: Row name (subsol of the term's test function).
		    value: Value-preserving factor :math:`v` of the term (FE test function, no :math:`s_k`).

		Returns:
		    :math:`s_k\, v / \mathcal N_k`.

		Raises:
		    ValueError: Unknown row.
		"""
		self._check(row)
		return self.test_scales.get(row, 1.0) * value / self.normalizations.get(row, 1.0)
