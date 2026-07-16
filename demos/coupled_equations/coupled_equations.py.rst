Coupled equations
================

1. Introduction and background
2. Describe an example (using MMS)
3. Describe the python implementation
4. Provide example of plotting output and verify accuracy - convergence analysis

------------
Introduction
------------

This tutorial shows how Firedrake can handle coupled equations using coupled meshes with two examples. The first example implements an advection equation coupled to a Helmholtz (OR POISSON) equation using a conforming mesh. In the next example, we repeat this coupling using nonconforming meshes by alterring the starting functions for each equation.

In this tutorial, we provide the relevant weak forms and boundary conditions for the advection and Helmholtz equations. Further details on the theory and implementation of each individual equation can be found in `DG advection equation with upwinding`_ and `Simple Helmholtz equation`_. 

The advection equations 

----------------------------------------------------------------
Coupled Advection and Helmholtz equations with a conforming mesh
----------------------------------------------------------------

The advection equation on a domain :math:`\Omega` is defined as

.. math ::
  
  \frac{\partial q}{\partial t} + (\xrightarrow{u} \cdot \nabla)q = 0

where :math:`\xrightarrow{u}` is a prescribed vector field and :math:`q(\xrightarrow{x}, t)` is an unknown scalar field [TODO: DESCRIBE VARIABLES BETTER]. 

The value of :math:`q` is known initially as 

.. math ::
  `q(\xrightarrow{x}, 0) = q_0(\xrightarrow{x})`
and at the boundary :math:`\Gamma` where :math:`\xrightarrow{x}` is directed towards the interior of the domain
.. math ::
  q(\xrightarrow{x},t) = q_{in}(\xrightarrow{x}, t) on \Gamma_{inflow}.




--------------------------------------
Method of Manufactured Solutions (MMS)
--------------------------------------

The method of manufactured solutions verifies the accuracy of implemented finite element models. This is done by explicitly specifying a solution for the differential equation at hand, ensuring that this solution satisfies all conditions set on the problem. We can compare the approximated solution and the actual solution to 




A python script version of this demo can be found :demo:`here <coupled_equations.py>`.

.. _DG advection equation with upwinding: https://www.firedrakeproject.org/demos/DG_advection.py.html
.. _Simple Helmholtz equation: https://www.firedrakeproject.org/demos/helmholtz.py.html


############################ NOTES: TODO DELETE #####################################################
Can add maths using 
.. math::

   \frac{\partial u}{\partial t} + (u\cdot\nabla) u - \nu\nabla^2 u = 0

   (n\cdot \nabla) u = 0 \ \textrm{on}\ \Gamma

where :math:`\Gamma` is the domain boundary and :math:`\nu` is a
constant scalar viscosity.

Can add python block using::
  from firedrake import *
  n = 30
  mesh = UnitSquareMesh(n, n)

Block ends when indentation finishes.