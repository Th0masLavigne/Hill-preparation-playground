r"""Evolution of the contractile strain :math:`\varepsilon_a = \ln \lambda_a`.

Residual :math:`R_a` of the Hill chain (equations.tex, "The cell Hill chain").
Target: ``fenicsoft.operators.cell.hill``.
"""

from .contractile_evolution import ContractileEvolution

__all__ = ["ContractileEvolution"]
