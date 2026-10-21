r"""Hill chain SEE -- (CE :math:`\parallel` PEE): UFL functions and active stress law.

Series elastic element (SEE) in series with a contractile element (CE), the CE in parallel with a passive
element (PEE). Target: ``fenicsoft.materials.cell.active.hill``.
"""

from . import functions
from .law import Constitutive, Parameters, check_uniqueness

__all__ = ["check_uniqueness", "Constitutive", "functions", "Parameters"]
