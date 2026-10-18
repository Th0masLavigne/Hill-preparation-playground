"""Time integration (target: ``fenicsoft.integration``).

:mod:`.decimated_time_integrator` is a copy of the fenicsoft module with two changes to report
(``functions_to_write``, rank-0 logs): see its ``Status``.
"""

from .decimated_time_integrator import DecimatedTimeIntegrator, TimeInterval, should_write_at_call

__all__ = ["DecimatedTimeIntegrator", "TimeInterval", "should_write_at_call"]
