"""Unit tests of the step and write-period conversion to the solve time t_hat = t / t0."""

import pytest
from fenicsoft import io

from hill_playground.integration import TimeInterval
from hill_playground.scaling import CharacteristicScales, nondimensionalize_step, nondimensionalize_write_period

T0 = 4.0
SCALES = CharacteristicScales(length=1.0, stress=1.0, time=T0, density=1.0)


def test_every_time_field_of_a_step_is_divided_by_t0() -> None:
	"""duration, init_dt, min_dt, max_dt and min_simu_seconds: t / t0."""
	step = io.Step(duration=100.0, init_dt=8.0, min_dt=2.0, max_dt=20.0, min_simu_seconds=40.0)
	scaled = nondimensionalize_step(step, SCALES)
	for name in ("duration", "init_dt", "min_dt", "max_dt", "min_simu_seconds"):
		assert getattr(scaled, name) == pytest.approx(getattr(step, name) / T0, rel=1e-12), name


def test_optional_fields_left_at_none() -> None:
	"""max_dt and min_simu_seconds not given: still None."""
	scaled = nondimensionalize_step(io.Step(duration=100.0, init_dt=8.0, min_dt=2.0), SCALES)
	assert scaled.max_dt is None
	assert scaled.min_simu_seconds is None
	assert scaled.duration == pytest.approx(25.0, rel=1e-12)


def test_write_period() -> None:
	"""TimeInterval divided by t0, substep counts unchanged, sequence type kept."""
	assert nondimensionalize_write_period(3, SCALES) == 3
	assert nondimensionalize_write_period(TimeInterval(40.0), SCALES).seconds == pytest.approx(10.0, rel=1e-12)
	periods = nondimensionalize_write_period((1, TimeInterval(8.0)), SCALES)
	assert isinstance(periods, tuple)
	assert periods[0] == 1
	assert periods[1].seconds == pytest.approx(2.0, rel=1e-12)
	assert isinstance(nondimensionalize_write_period([TimeInterval(8.0)], SCALES), list)
