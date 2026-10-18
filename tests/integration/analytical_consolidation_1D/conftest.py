"""Print the Terzaghi error table in the pytest terminal summary (no ``-s`` needed)."""

from pathlib import Path
from typing import Any

import pandas as pd

ERRORS_CSV = Path(__file__).resolve().parent / "outputs_terzaghi_nondim" / "errors.csv"


def pytest_sessionstart(session: Any) -> None:
	"""Forget the table of a previous session."""
	ERRORS_CSV.unlink(missing_ok=True)


def pytest_terminal_summary(terminalreporter: Any) -> None:
	"""Errors vs analytical, per nondimensionalization mode, and where the figure is."""
	if ERRORS_CSV.exists():
		terminalreporter.section("Terzaghi: relative errors vs analytical")
		terminalreporter.write_line(pd.read_csv(ERRORS_CSV).to_string(index=False, float_format="%.3e"))
		terminalreporter.write_line(f"figure: {ERRORS_CSV.parent / 'terzaghi_nondim.png'}")
