"""Time integration.

:mod:`.decimated_time_integrator`: adaptive time integrator with throttled XDMF output, early exit on a
plateau, selected output fields and rank-0 logs.
"""

from .decimated_time_integrator import DecimatedTimeIntegrator, TimeInterval, should_write_at_call

__all__ = ["DecimatedTimeIntegrator", "TimeInterval", "should_write_at_call"]
