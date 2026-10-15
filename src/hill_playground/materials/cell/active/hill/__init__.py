r"""Hill chain SEE -- (CE :math:`\parallel` PEE).

Shared UFL functions of the chain and the active stress law (equations.tex, "The cell Hill chain").
Target: ``fenicsoft.materials.cell.active.hill``.
"""

from . import functions
from .law import Constitutive, Parameters

__all__ = ["Constitutive", "functions", "Parameters"]
