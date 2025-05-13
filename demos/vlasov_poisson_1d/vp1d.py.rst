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
   f_t + \nabla_x \cdot (vf) + \nabla_v \cdot (af) = 0,

where

.. math::
   \nabla_x = (\partial_{x_1},\ldots, \partial_{x_d}), \quad
   \nabla_v = (\partial_{v_1},\ldots, \partial_{v_d}).

To close the system, we need a formula for the acceleration :math:`a`.
In the (single species) Vlasov-Poisson model, the acceleration is
determined by the electrostatic force,

.. math::
   a = -\frac{1}{m}\nabla\phi,

where :math:`m`
is the mass per plasma particle, and :math:`\phi` is the electrostatic
potential determined by the Poisson equation,

.. math::
   -\nabla^2\phi = q\int_{\mathbb{R}^d} f(x,v,t)\mathrm{d} v,

where :math:`q` is the electric charge per plasma particle.
   
As usual, to implement this problem, we start by importing the
Firedrake namespace. ::

  from firedrake import *


A python script version of this demo can be found :demo:`here <vp1d.py>`.
