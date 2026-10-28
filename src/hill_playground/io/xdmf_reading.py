"""Read-only helpers on a results XDMF."""

import re
from pathlib import Path

TIME_VALUE = re.compile(r'(<Time Value=")([^"]+)(")')
"""``<Time Value="t">``: groups (prefix, value, closing quote)."""


def read_time_stamps(xdmf_path: Path) -> list[float]:
	"""Time stamps :math:`t_k` of an XDMF, in file order.

	Args:
	    xdmf_path: Results ``.xdmf``.

	Returns:
	    The ``<Time Value>`` entries as floats.
	"""
	return [float(m.group(2)) for m in TIME_VALUE.finditer(Path(xdmf_path).read_text())]
