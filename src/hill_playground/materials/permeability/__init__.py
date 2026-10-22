r"""Hydraulic conductivity laws :math:`k(\phi_\ell)/\eta` and their pull-back :math:`\mathbf K_L`.

:math:`\mathbf K_L = (k/\eta) J\mathbf C^{-1}`. Each law module defines ``Parameters`` and ``Permeability`` and
writes only :math:`k/k_0` (``_ratio``); :mod:`.base` gives :math:`k/\eta` and :math:`\mathbf K_L`,
:mod:`.saturated_mixture` the kinematic relations.

======================  =====================================================================================
Module                  :math:`k/k_0`
======================  =====================================================================================
``constant``            1
``kozeny_carman``       :math:`\dfrac{\phi_\ell^3/(1-\phi_\ell)^2}{\phi_{\ell,0}^3/(1-\phi_{\ell,0})^2}`
``holmes_mow``          :math:`\Big(\dfrac{J-J_{\min}}{1-J_{\min,0}}\Big)^{\kappa_{HM}} e^{\frac{M}{2}(J^2-1)}`
``compaction_power``    :math:`\Big(\dfrac{J-J_{\min}}{1-J_{\min,0}}\Big)^{n}`, :math:`n \ge 1`
======================  =====================================================================================

with :math:`J_{\min} = \Phi_{s,0} + \Phi_c` and :math:`\phi_\ell = 1 - J_{\min}/J`. Two-phase mixture (solid + liquid):
leave ``phi_c`` and ``phi_c0`` at 0 (default), then :math:`J_{\min} = \Phi_{s,0}`.
"""

from . import base, compaction_power, constant, holmes_mow, kozeny_carman
from .base import Permeability, PermeabilityParameters, PorousPermeability, PorousPermeabilityParameters

__all__ = [
	"base",
	"compaction_power",
	"constant",
	"holmes_mow",
	"kozeny_carman",
	"Permeability",
	"PermeabilityParameters",
	"PorousPermeability",
	"PorousPermeabilityParameters",
]
