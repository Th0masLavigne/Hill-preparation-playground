"""Print the Hill one-element error table in the pytest terminal summary (no ``-s`` needed)."""

from pathlib import Path
from typing import Any

import pandas as pd

ERRORS_CSV = Path(__file__).resolve().parent / "outputs_hill_one_element" / "errors.csv"


ERRORS_CSV.unlink(missing_ok=True)  # forget the table of a previous session (this conftest loads once per session)


def pytest_terminal_summary(terminalreporter: Any) -> None:
	"""Max FE vs 0D differences per case, and where the figure is."""
	if ERRORS_CSV.exists():
		terminalreporter.section("Hill one element: FE vs 0D reference (max differences)")
		terminalreporter.write_line(pd.read_csv(ERRORS_CSV).to_string(index=False, float_format="%.2e"))
		terminalreporter.write_line(f"figure: {ERRORS_CSV.parent / 'hill_one_element.png'}")
