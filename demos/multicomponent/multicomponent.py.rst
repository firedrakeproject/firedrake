Multicomponent flow -- microfluidic mixing of hydrocarbons
===================================================

.. rst-class:: emphasis

    In this tutorial we demonstrate how Firedrake can be used to
    simulate multicomponent flow; specifically the microfluidic 
    mixing of benzene and cyclohexane.

    The demo was contributed by `Aaron Baier-Reinio
    <mailto:baierreinio@maths.ox.ac.uk>`__ and `Kars Knook
    <mailto:knook@maths.ox.ac.uk>`__.

We consider a steady, isothermal, nonreacting mixture of benzene and cyclohexane in
a microfluidic container :math:`\Omega \subset \mathbb{R}^d`.

We model the mixture using the Stokes--Onsager--Stefan--Maxwell equations.
Momentum transport is modelled using the compressible 
Stokes momentum equation for a Newtonian fluid,

.. math::

    -\nabla \cdot \big\{
        2 \eta \epsilon(v) + (\zeta - 2\eta / d) (\nabla \cdot v) \mathbb{I}
    \big\} + \nabla p = 0 \quad \textrm{in}\ \Omega.

Here, the unknowns are the :math:`\mathbb{R}^2`-valued velocity :math:`v`,
scalar pressure :math:`p` and density :math:`rho`.
Moreover :math:`\epsilon (v)` denotes the symmetric gradient of :math:`v`,
:math:`\eta, \zeta > 0` are the shear and bulk viscosities,
:math:`d` is the spatial dimension (:math:`d=2` in this demo),
:math:`\mathbb{I}` the :math:`d \times d` identity matrix.

Let :math:`n` denote the number of chemical species in the mixture; 
in this example :math:`n=2` (benzene and cyclohexane).
The continuity equation for molar concentration :math:`c_i`
of species :math:`i \in 1:n` in the abscence of chemical reactions is

.. math::

    \partial_t c_i + \frac{1}{M_i} \nabla \cdot J_i = 0

where :math:`M_i > 0` is the molar mass of species :math:`i` and
:math:`J_i` its mass flux.

The mass fluxes must be modelled using a constitutive relation.
For example, in a Fickian model we might use
:math:`J_i = M_i c_i v - D_i \nabla c_i`
where the first term represents advection and the second term Fickian diffusion
(here :math:`D_i > 0` are Fickian diffusion coefficients).
The Fickian approach is appropriate for dilute mixtures (i.e. mixtures where all but
one of the species are present in trace amounts), but it typically is not
thermodynamically consistent in the non-dilute regime and fails to account for
cross-diffusional effects. We therefore model the mass fluxes using the 
Onsager--Stefan--Maxwell equations, which capture cross-diffusion in non-dilute
mixtures in a thermodynamically consistent fashion. Sometimes in the literature
these equations are referred to as the generalized Maxwell--Stefan equations [CITE BIRD].





+---------------------------+---------------------------+
| .. image:: benzene_0.png  | .. image:: benzene_4.png  |
|    :width: 100%           |    :width: 100%           |
+---------------------------+---------------------------+
