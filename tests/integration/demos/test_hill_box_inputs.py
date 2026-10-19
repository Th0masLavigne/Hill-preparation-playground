"""Integration: ``UserInputs`` of the Hill box demo rejects an unknown option at construction."""

import sys
from pathlib import Path
from typing import Any, get_args

import pytest

DEMO = Path(__file__).resolve().parents[3] / "demos" / "hill_box"
sys.path.insert(0, str(DEMO))

import hill_box_uniaxial as demo  # noqa: E402


@pytest.mark.parametrize("name", ["nondimensionalization", "permeability_law", "volumetric_law", "see_variant"])
def test_unknown_option_raises(name: str) -> None:
	"""A value outside the ``Literal`` stops before any mesh or solve."""
	with pytest.raises(ValueError, match=name):
		options: dict[str, Any] = {name: "typo"}
		demo.UserInputs(**options)


def test_defaults_are_valid() -> None:
	"""Every default lies in its ``Literal``."""
	inputs = demo.UserInputs()
	for name, choices in demo._CHOICES.items():  # pyright: ignore[reportPrivateUsage]
		assert getattr(inputs, name) in get_args(choices)
