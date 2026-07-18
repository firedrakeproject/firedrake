Coupled equations
================

1. Introduction and background
1.1. Explain the Poisson and Helmholtz equations and Dirichlet-Neumann Method [CITE TEXTBOOK]
1.2. Explain MMS
2. Describe example (using MMS) [conforming/nonconforming meshes]
3. Describe the python implementation
4. Provide example of plotting output and verify accuracy - convergence analysis

------------
Introduction
------------

This tutorial shows how Firedrake can handle coupled equations with coupled meshes. The first example implements a poisson equation coupled to a Helmholtz equation. [......]

----------------------------------------------------------------
Coupled Poisson and Helmholtz equations [To Delete: with a (non)conforming mesh]
----------------------------------------------------------------

Consider unit squares :math:`\Omega_1 = [0,1] \times [0,1]` and :math:`\Omega_2 = [1,2] \times [0,1]` with boundary :math:`\Gamma` and :math:`\Gamma_1 = {x = 1, y \in [0,1]}` be the shared edge between each unit square. The poisson equation is defined on :math:`\Omega_1` as

.. math::

  -\nabla^2 u_1 = f

  u_2 = 0 \textit{ on } \partial \Omega_1\\ \Gamma_1.

The Helmholtz equation is defined on :math:`\Omega_2` as

.. math::
  -\nabla^2 u_2 + u_2 = g

  \nabla u_2 \cdot \bf{n} = 0 \textit{ on } \partial \Omega_2 \\ \Gamma_2. 

:math:`f \text{ and } g` are known functions and :math:`u_1, u_2 \in V` are the solutions to these equations in some function space :math:`V`.


The weak forms for these equations can be found by multiplying by an arbitrary test function :math:`v \in V` and integrating by parts. Further details on this process can be found in `Mixed formulation for the Poisson equation`_ and `Simple Helmholtz equation`_.
[TODO: Add weak form for each equation from Leo's report]


The weak form for the Helmholtz equation is

.. math::

  \int_{\Omega_2}{\nabla u \cdot \nabla v + uv dx} = \int_{\Omega_2}{v f dx} + 



------------------------
Dirichlet-Neumann Method
------------------------

--------------------------------------
Method of Manufactured Solutions (MMS)
--------------------------------------

The method of manufactured solutions verifies the accuracy of implemented finite element models. This is done by explicitly specifying a solution for the differential equation at hand, ensuring that this solution satisfies all conditions set on the problem. We can then analyse the accuracy of the approximated solution. 




A python script version of this demo can be found :demo:`here <coupled_equations.py>`.

.. _DG advection equation with upwinding: https://www.firedrakeproject.org/demos/DG_advection.py.html
.. _Simple Helmholtz equation: https://www.firedrakeproject.org/demos/helmholtz.py.html
.. _Mixed formulation for the Poisson equation: https://www.firedrakeproject.org/demos/poisson_mixed.py.html


############################ NOTES: TODO DELETE #####################################################
Can add python block using::
  from firedrake import *
  n = 30
  mesh = UnitSquareMesh(n, n)

Block ends when indentation finishes.