r"""Hydraulic conductivity laws :math:`k(\phi_\ell)/\eta` and their pull-back :math:`\mathbf K_L`.

Target: ``fenicsoft.materials.permeability``.

equations.tex, Eq. darcy-lag and Remark rem:kphi. Each law module has its own ``Parameters`` and
``Permeability`` and only writes :math:`k/k_0` (``_ratio``, read as ``ratio``); :mod:`base` gives
:math:`k/\eta`, :math:`\mathbf K_L` and the porosity quantities, :mod:`porosity` holds the kinematic relations.

======================  =====================================================================================
Module                  :math:`k/k_0`
======================  =====================================================================================
``constant``            1
``kozeny_carman``       :math:`\dfrac{\phi_\ell^3/(1-\phi_\ell)^2}{\phi_{\ell,0}^3/(1-\phi_{\ell,0})^2}`
``holmes_mow``          :math:`\Big(\dfrac{J-J_c}{1-J_{c,0}}\Big)^{\kappa_{HM}} e^{\frac{M}{2}(J^2-1)}`
``compaction_power``    :math:`\Big(\dfrac{J-J_c}{1-J_{c,0}}\Big)^{n}`, :math:`n \ge 1`
======================  =====================================================================================

with :math:`J_c = \Phi_{s,0} + \Phi_c` and :math:`\phi_\ell = 1 - J_c/J`.
"""

from . import base, compaction_power, constant, holmes_mow, kozeny_carman, porosity
from .base import Permeability, PermeabilityParameters, PorousPermeability, PorousPermeabilityParameters

__all__ = [
	"base",
	"compaction_power",
	"constant",
	"holmes_mow",
	"kozeny_carman",
	"Permeability",
	"PermeabilityParameters",
	"porosity",
	"PorousPermeability",
	"PorousPermeabilityParameters",
]
