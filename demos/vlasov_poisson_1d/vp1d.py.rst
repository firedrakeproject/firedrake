1D Vlasov-Poisson Equation
===========================

This tutorial was contributed by `Colin Cotter
<mailto:colin.cotter@imperial.ac.uk>`__ and Werner Bauer.

A plasma is a continuum of moving particles with nonunique velocity
at each point in space. In :math:`d` dimensions, the plasma is
described by a density :math:`f(x,v,t)` where :math:`x\in \mathbb{R}^d`
are the physical coordinates and :math:`v \in \mathbb{R}^d` are velocity
coordinates. Hence, in :math:`d` dimensions, a :math:`2d`
dimensional mesh is required. To deal with this curse of
dimensionality, particle-in-cell methods are usually used. However,
in 1 dimension, it is tractable to simulate the plasma on a 2
dimensional mesh.

The Vlasov equation models the (collisionless) conservation of plasma
particles, according to 

.. math::
   f_t + \nabla_{\vec{x}} \cdot (\vec{v}f) + \nabla_{\vec{v}} \cdot (\vec{a}f) = 0,

where

.. math::
   \nabla_{\vec{x}} = (\partial_{x_1},\ldots, \partial_{x_d}), \quad
   \nabla_{\vec{v}} = (\partial_{v_1},\ldots, \partial_{v_d}).

To close the system, we need a formula for the acceleration :math:`\vec{a}`.
In the (single species) Vlasov-Poisson model, the acceleration is
determined by the electrostatic force,

.. math::
   \vec{a} = -\frac{1}{m}\nabla\phi,

where :math:`m`
is the mass per plasma particle, and :math:`\phi` is the electrostatic
potential determined by the Poisson equation,

.. math::
   -\nabla^2\phi = q\int_{\mathbb{R}^d} f(\vec{x},\vec{v},t)\mathrm{d} v,

where :math:`q` is the electric charge per plasma particle.

In this demo we specialise to :math:`d=1`, and the equations become

.. math::
   f_t + (fv)_x + (-f\phi_x/m)_v = 0, \quad
   -\phi_{xx} = q\int f(x,v,t)\mathrm{d} v,

with coordinates :math:`(x,v)\in \mathbb{R}^2`. From now on we will
relabel these coordinates :math:`(x,v)\mapsto (x_1,x_2)`, obtaining
the equivalent form,

.. math::
   f_t + \nabla\cdot(\vec{u}f) = 0, \quad \vec{u} = (v,-\phi_x/m), \quad
   -\phi_{x_1x_1} = q\int f(x_1,x_2,t)\mathrm{d} x_2,

where :math:`\nabla=(\partial_{x_1},\partial{x_2})`. From now we will
choose units such that :math:`q,m` are absorbed into the definition of
:math:`f`.

To proceed, we need to develop variational formulations of these
equations. For the density we will use a discontinuous Galerkin formulation,
and the continuity equation becomes 

.. math::

   \int_\Omega \! q \frac{\partial f}{\partial t} \, \mathrm{d} x
   &= \int_\Omega \! f \nabla \cdot (q \vec{u}) \, \mathrm{d} x\\
   &\quad- \int_{\Gamma_\mathrm{int}} \! \widetilde{f}(q_+ \vec{u} \cdot \vec{n}_+
     + q_- \vec{u} \cdot \vec{n}_-) \, \mathrm{d} S\\
   &\quad- \int_{\Gamma_{\mathrlap{\mathrm{ext, inflow}}}} q f_\mathrm{in} \vec{u} \cdot
   \vec{n} \, \mathrm{d} s\\
   &\quad- \int_{\Gamma_{\mathrlap{\mathrm{ext, outflow}}}} q f \vec{u} \cdot
   \vec{n} \, \mathrm{d} s
   \qquad \forall q \in V,

where :math:`\Gamma_\mathrm{int}` is the 
   
As usual, to implement this problem, we start by importing the
Firedrake namespace. ::

  from firedrake import *


A Python script version of this demo can be found :demo:`here <vp1d.py>`.
