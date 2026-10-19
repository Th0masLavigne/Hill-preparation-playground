r"""Hill chain SEE -- (CE :math:`\parallel` PEE).

Series elastic element (SEE) in series with a contractile element (CE) itself in parallel with a passive
element (PEE). Shared UFL functions of the chain and the active stress law.
Target: ``fenicsoft.materials.cell.active.hill``.
"""

from . import functions
from .law import Constitutive, Parameters

__all__ = ["Constitutive", "functions", "Parameters"]
