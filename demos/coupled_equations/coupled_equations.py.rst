Coupled equations
================

1. Introduction and background
2. Describe an example (using MMS)
3. Describe the python implementation
4. Provide example of plotting output and verify accuracy


A python script version of this demo can be found :demo:`here <coupled_equations.py>`.



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