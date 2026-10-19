r"""Evolution of the contractile strain :math:`\varepsilon_a = \ln \lambda_a`.

Residual :math:`R_a` of the Hill chain: force balance between the contractile element (CE), the
passive element (PEE) and the series elastic element (SEE).
Target: ``fenicsoft.operators.cell.hill``.
"""

from .contractile_evolution import ContractileEvolution

__all__ = ["ContractileEvolution"]
