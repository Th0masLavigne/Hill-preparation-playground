"""Hill-chain smooth muscle cell model, staged for fenicsoft.

Sub-packages mirror ``fenicsoft`` one-to-one, so that
``hill_playground/<path>`` moves to ``fenicsoft/<path>`` without renaming.

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
