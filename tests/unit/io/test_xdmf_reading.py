"""Unit tests of the read-only XDMF helpers."""

from pathlib import Path

import pytest

from hill_playground.io import xdmf_reading


def test_reads_time_stamps_in_file_order(tmp_path: Path) -> None:
	"""Any float spelling -> floats, in file order; no ``<Time>`` -> empty list."""
	xdmf = tmp_path / "res.xdmf"
	xdmf.write_text(
		'<Grid><Time Value="0"/></Grid><Grid><Time Value="5.000000000000000e-01"/></Grid><Time Value="3600.0"/>'
	)
	assert xdmf_reading.read_time_stamps(xdmf) == pytest.approx([0.0, 0.5, 3600.0], rel=1e-15)
	xdmf.write_text("<Grid/>")
	assert xdmf_reading.read_time_stamps(xdmf) == []
