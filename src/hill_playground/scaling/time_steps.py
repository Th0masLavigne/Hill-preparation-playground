r"""Time inputs of the integrator in solve time :math:`\hat t = t/t_0`.

Steps (duration, ``dt`` bounds, plateau floor) and write periods in s are divided by :math:`t_0`; write periods
given as a number of substeps are unchanged. Results are Python ``float``: ``dolfin_mech``'s
``TimeVaryingConstant.set_value`` ignores NumPy scalars, so a ``numpy.float64`` :math:`\Delta t` would leave every
``set_dt`` at 0.
"""

import dataclasses
from typing import Any, TypeVar

from ..integration import TimeInterval
from .characteristic_scales import CharacteristicScales

STEP_TIME_FIELDS: tuple[str, ...] = ("duration", "init_dt", "min_dt", "max_dt", "min_simu_seconds")
"""Time fields of a step (``fenicsoft.integration.Step``); absent or ``None`` fields are left as they are."""

T_Step = TypeVar("T_Step")


def nondimensionalize_step(step: T_Step, scales: CharacteristicScales) -> T_Step:
	r"""Every time field of ``step`` (:data:`STEP_TIME_FIELDS`) divided by :math:`t_0`, as ``float``.

	Args:
	    step: Frozen dataclass step, in s (``fenicsoft.integration.Step``).
	    scales: :math:`t_0`.

	Returns:
	    A copy of ``step`` in :math:`\hat t`.
	"""
	values = {k: getattr(step, k) for k in STEP_TIME_FIELDS if getattr(step, k, None) is not None}
	scaled = {k: float(scales.nondimensionalize_time(v)) for k, v in values.items()}
	return dataclasses.replace(step, **scaled)  # type: ignore[type-var]


def nondimensionalize_write_period(period: Any, scales: CharacteristicScales) -> Any:
	r"""Write period(s) of ``DecimatedTimeIntegrator`` in :math:`\hat t`.

	Args:
	    period: ``int`` (substeps, unchanged), :class:`TimeInterval` (s), or a list/tuple of them (one per step).
	    scales: :math:`t_0`.

	Returns:
	    Same structure, :class:`TimeInterval` divided by :math:`t_0`.
	"""
	if isinstance(period, TimeInterval):
		return TimeInterval(float(scales.nondimensionalize_time(period.seconds)))
	if isinstance(period, (list, tuple)):
		return type(period)(nondimensionalize_write_period(p, scales) for p in period)
	return period
