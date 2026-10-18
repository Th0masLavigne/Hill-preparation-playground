r"""Activation schedule :math:`\alpha(t)` of active cell laws (loading, not a material).

Target: ``fenicsoft.materials.cell.active.activation``. Kept outside any constitutive law so that
an active law owns its schedule by composition and no law depends on another one: today
``ActivePowerConfig`` and the ``p_act`` machinery are written in ``iso.law`` and ``aniso.law``;
both can import them from here once merged.

Per step :math:`k`, :math:`\alpha` is either constant, a linear ramp in the step fraction
:math:`t_{step} \in [0, 1]`, or a function of the global time :math:`t`:

.. math::
	\alpha = \alpha_k, \qquad
	\alpha = \alpha_{ini} + t_{step}(\alpha_{fin} - \alpha_{ini}), \qquad
	\alpha = g_k(t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import dolfin
import dolfin_mech as dmech


@dataclass(frozen=True)
class ActivePowerConfig:
	r""":math:`\alpha` on one step: ``val`` | (``val_ini``, ``val_fin``) | ``func(t)``, exclusively.

	Same API as ``fenicsoft.materials.cell.active.iso.law.ActivePowerConfig``.

	Attributes:
	    k_step: Step index.
	    val: Constant :math:`\alpha_k`.
	    val_ini: Ramp start :math:`\alpha_{ini}`.
	    val_fin: Ramp end :math:`\alpha_{fin}`.
	    func: :math:`g_k(t)`, global time in s.
	"""

	k_step: int
	val: float | None = None
	val_ini: float | None = None
	val_fin: float | None = None
	func: Callable[[float], float] | None = None

	def __post_init__(self) -> None:
		"""Check that exactly one definition is given.

		Raises:
		    ValueError: Mixed or missing definitions.
		"""
		ramp = (self.val_ini, self.val_fin)
		if self.func is not None and any(v is not None for v in (self.val, *ramp)):
			raise ValueError("ActivePowerConfig cannot have both a 'func' and constant values defined.")
		if self.val is not None and any(v is not None for v in ramp):
			raise ValueError("ActivePowerConfig cannot define 'val' alongside 'val_ini' or 'val_fin'.")
		if self.func is None and self.val is None:
			if all(v is None for v in ramp):
				raise ValueError(
					"ActivePowerConfig must define either 'val', 'func', or a ramp ('val_ini' and 'val_fin')."
				)
			if any(v is None for v in ramp):
				raise ValueError("ActivePowerConfig ramp requires both 'val_ini' and 'val_fin' to be defined.")

	@classmethod
	def constant(cls, k_step: int, value: float) -> ActivePowerConfig:
		r""":math:`\alpha = \alpha_k` on step ``k_step``."""
		return cls(k_step=k_step, val=value)

	@classmethod
	def ramp(cls, k_step: int, ini: float, fin: float) -> ActivePowerConfig:
		r""":math:`\alpha: \alpha_{ini} \to \alpha_{fin}`, linear over step ``k_step``."""
		return cls(k_step=k_step, val_ini=ini, val_fin=fin)

	@classmethod
	def function(cls, k_step: int, func: Callable[[float], float]) -> ActivePowerConfig:
		r""":math:`\alpha = g_k(t)` on step ``k_step``."""
		return cls(k_step=k_step, func=func)

	def to_dict(self) -> dict[str, float | None]:
		"""Keyword arguments of ``dolfin_mech.core.TimeVaryingConstant``."""
		return {"val": self.val, "val_ini": self.val_ini, "val_fin": self.val_fin}


class ActivationSchedule:
	r"""Holds :math:`\alpha` as one ``dolfin.Constant`` (``p_act``) and updates it step by step.

	Args:
	    configs: One :class:`ActivePowerConfig` per step (steps without one keep the last value).
	    simulation_steps: ``problem.steps`` (``t_ini``, ``t_fin`` of each step).

	Attributes:
	    p_act: :math:`\alpha`, the single UFL constant shared by every form that uses it.
	    time: Global time :math:`t` of the last :meth:`update_time` (in the units of ``simulation_steps``).
	"""

	def __init__(self, configs: Sequence[ActivePowerConfig], simulation_steps: Sequence[dmech.core.Step]) -> None:
		"""Index the configs by step and create ``p_act = 0``."""
		self._steps = simulation_steps
		self._configs = {config.k_step: config for config in configs}
		self._managers = {
			config.k_step: dmech.core.TimeVaryingConstant(**config.to_dict())
			for config in configs
			if config.func is None
		}
		first = min(self._managers) if self._managers else 0
		self._config = self._configs.get(first)
		self._manager = self._managers.get(first)
		self.p_act = dolfin.Constant(0.0)
		self.time = 0.0
		self._k_step = first

	def set_k_step(self, k_step: int) -> None:
		"""Select the config of step ``k_step``.

		Args:
		    k_step: Step index.
		"""
		self._k_step = k_step
		self._config = self._configs.get(k_step)
		self._manager = self._managers.get(k_step)

	def update_time(self, t_step: float) -> None:
		r"""Set :math:`\alpha` at the step fraction :math:`t_{step}`.

		Args:
		    t_step: :math:`(t - t_{ini})/(t_{fin} - t_{ini}) \in [0, 1]`.
		"""
		if self._k_step < len(self._steps):
			step = self._steps[self._k_step]
			self.time = step.t_ini + t_step * (step.t_fin - step.t_ini)
		if self._config is None:
			return
		if self._config.func is not None:
			step = self._steps[self._config.k_step]  # IndexError on a missing step, as in iso/aniso
			self.p_act.assign(self._config.func(step.t_ini + t_step * (step.t_fin - step.t_ini)))
		else:  # constant or ramp: a TimeVaryingConstant exists for this step
			assert self._manager is not None  # built for every config without func
			self._manager.set_value_at_t_step(t_step)
			self.p_act.assign(float(self._manager.val))

	@property
	def value(self) -> float:
		r"""Current :math:`\alpha`."""
		return float(self.p_act)
