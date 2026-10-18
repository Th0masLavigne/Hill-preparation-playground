r"""Hydraulic conductivity laws :math:`k(\phi_\ell)/\eta` and their pull-back :math:`\mathbf K_L`.

Target: ``fenicsoft.materials.permeability``.

equations.tex, Eq. darcy-lag and Remark rem:kphi. Each law is self-contained (own ``Parameters``,
own class); :mod:`porosity` only holds the kinematic relations shared by all of them.

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

from . import compaction_power, constant, holmes_mow, kozeny_carman, porosity

__all__ = ["compaction_power", "constant", "holmes_mow", "kozeny_carman", "porosity"]
