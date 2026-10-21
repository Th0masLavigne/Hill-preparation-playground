"""Unit tests of the activation schedule alpha(t) (constant, ramp, function of t)."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from hill_playground.materials.cell.active.activation import ActivationSchedule, ActivePowerConfig


@pytest.mark.parametrize(
	"kwargs",
	[
		{},
		{"val": 1.0, "func": abs},
		{"val": 1.0, "val_ini": 0.0},
		{"val_ini": 0.0},
	],
)
def test_config_rejects_ambiguous_definitions(kwargs: dict[str, Any]) -> None:
	"""Exactly one of ``val``, (``val_ini``, ``val_fin``) or ``func``."""
	with pytest.raises(ValueError):
		ActivePowerConfig(k_step=0, **kwargs)


def test_schedule_ramp_constant_function() -> None:
	"""Ramp: alpha_ini + t_step (alpha_fin - alpha_ini); func: g(t_ini + t_step (t_fin - t_ini))."""
	steps = [MagicMock(t_ini=0.0, t_fin=2.0), MagicMock(t_ini=2.0, t_fin=4.0), MagicMock(t_ini=4.0, t_fin=6.0)]
	schedule = ActivationSchedule(
		[
			ActivePowerConfig.ramp(k_step=0, ini=0.2, fin=1.0),
			ActivePowerConfig.constant(k_step=1, value=0.7),
			ActivePowerConfig.function(k_step=2, func=lambda t: t / 10.0),
		],
		steps,
	)
	assert schedule.value == 0.0
	schedule.update_time(0.5)
	assert schedule.value == pytest.approx(0.6, rel=1e-12)
	schedule.set_k_step(1)
	schedule.update_time(0.3)
	assert schedule.value == pytest.approx(0.7, rel=1e-12)
	schedule.set_k_step(2)
	schedule.update_time(0.5)
	assert schedule.value == pytest.approx(0.5, rel=1e-12)


def test_schedule_without_config_keeps_value() -> None:
	"""A step without config leaves alpha unchanged."""
	schedule = ActivationSchedule([ActivePowerConfig.constant(k_step=0, value=1.0)], [MagicMock(t_ini=0.0, t_fin=1.0)])
	schedule.update_time(0.1)
	schedule.set_k_step(3)
	schedule.update_time(0.9)
	assert schedule.value == 1.0


def test_schedule_tracks_global_time() -> None:
	"""Global time = t_ini + t_step (t_fin - t_ini) of the current step."""
	steps = [MagicMock(t_ini=0.0, t_fin=2.0), MagicMock(t_ini=2.0, t_fin=6.0)]
	schedule = ActivationSchedule([ActivePowerConfig.constant(k_step=0, value=1.0)], steps)
	schedule.update_time(0.5)
	assert schedule.time == pytest.approx(1.0, rel=1e-12)
	schedule.set_k_step(1)
	schedule.update_time(0.25)
	assert schedule.time == pytest.approx(3.0, rel=1e-12)


def test_function_on_missing_step_raises_index_error() -> None:
	"""A ``func`` config whose step does not exist fails at ``update_time``, as in fenicsoft iso/aniso."""
	steps = [MagicMock(t_ini=0.0, t_fin=1.0)]
	schedule = ActivationSchedule([ActivePowerConfig.function(k_step=3, func=lambda t: t)], steps)
	schedule.set_k_step(3)
	with pytest.raises(IndexError):
		schedule.update_time(0.5)
