"""Hill-chain smooth muscle cell model.

Sub-packages follow the layout of the destination package one-to-one: ``<path>`` keeps its name.

Author:
	Thomas Lavigne (LMS, Polytechnique)
Date:
	2026
"""

import importlib.metadata

try:
	__version__ = importlib.metadata.version("hill_playground")
except importlib.metadata.PackageNotFoundError:
	__version__ = "0.0.0"

from . import materials, operators, scaling

__all__ = ["materials", "operators", "scaling"]
