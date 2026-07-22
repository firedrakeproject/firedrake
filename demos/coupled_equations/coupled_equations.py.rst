Coupled equations
=================

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
Coupled Poisson and Helmholtz equations
----------------------------------------------------------------

Consider unit squares :math:`\Omega_1 = [0,1] \times [0,1]` and :math:`\Omega_2 = [1,2] \times [0,1]` with boundary :math:`\Gamma` and :math:`\Gamma = {(1, y) : y \in [0,1]}` be the shared edge between each unit square. The poisson equation is defined on :math:`\Omega_1` as

.. math::

  -\nabla^2 u_1 &= f

  u_1 &= 0 \ \textrm{on}\ \Omega_1 \setminus \Gamma.

The Helmholtz equation is defined on :math:`\Omega_2` as

.. math::
  -\nabla^2 u_2 + u_2 &= g

  \nabla u_2 \cdot n &= 0 \ \textrm{on}\ \partial \Omega_2 \setminus \Gamma. 

:math:`f \ \textrm{and}\ g` are known functions and :math:`u_1, u_2 \in V^1, V^2` are the solutions to these equations in some function spaces :math:`V^1 \ \textrm{and}\ V^2`.


The weak forms for these equations can be found by multiplying by an arbitrary test function :math:`v \in V` and integrating by parts. Further details on this process can be found in `Mixed formulation for the Poisson equation`_ and `Simple Helmholtz equation`_. From the weak forms, variational problems can be defined. For the poisson equation, the variational problem involves finding :math:`u_1 \in V^1` such that :math:`a_{11}(u_1, v_1) = L_1(v_1) \ \textrm{for all}\ v_1 \in V^1` where

.. math::

  a_{11} (u_1, v_1) &= \int_{\Omega_1}\nabla u_1 \cdot \nabla v_1  \ {\rm d} x - \int_{V_1}v_1 \nabla u_1 \cdot n  \ {\rm d} s,

  L_1 (v_1) &= \int_{\Omega_1}f v_1  \ {\rm d} x.

Similarly, the variational problem for the Helmholtz equation involves finding :math:`u_2 \in V^2` such that :math:`a_{22} (u_2, v_2) = L_2 (v_2) \ \textrm{for all}\ v_2 \in V^2` where

.. math::

  a_{22}(u_2, v_2) &= \int_{\Omega_2}\nabla u_2 \cdot \nabla u_2 + u_2 v_2  \ {\rm d} x - \int_{\Gamma}v_2 \nabla u_2 \cdot n  \ {\rm d} s,

  L_2 (v_2) &= \int_{\Omega_2}g v_2  \ {\rm d} x.

These equations are then coupled along the shared interface :math:`\Gamma` with Nitsche's method to enforce the boundary conditions in the weak form. [TODO: maybe explain what Nitsche's method is] This is done by adding a boundary penalty term to both sides of the weak forms. The poisson and Helmholtz weak form equations are updated to become

.. math::

  a_{11} (u_1, v_1) &= \int_{\Omega_1}\nabla u_1 \cdot \nabla v_1  \ {\rm d} x - \int_{V_1}v_1 \nabla u_1 \cdot n \ {\rm d} s + w_1 \int_{\Gamma}u_1 v_1 \ {\rm d} s,

  a_{22}(u_2, v_2) &= \int_{\Omega_2}\nabla u_2 \cdot \nabla u_2 + u_2 v_2  \ {\rm d} x - \int_{\Gamma}v_2 \nabla u_2 \cdot n \ {\rm d} s + w_2 \int_{\Gamma}u_2 v_2 \ {\rm d} s.


The coupling terms are also defined as

.. math::

  a_{12}(u_2, v_1) &= -w_1 \int_{\Gamma}\mathcal{I}_{V^1} (u_2) v_1 \ {\rm d} s,

  a_{21}(u_1, v_2) &= -w_2 \int_{\Gamma}\mathcal{I}_{V^2} (u_1) v_2 \ {\rm d} s,

where :math:`w_1, w_2 \>\> 0` are penalty parameters and :math:`\mathcal{I}_{V^2}: V^1 \rightarrow V^2` is a cross-mesh interpolation operator. Therefore, the variational problem for the coupled equations is: find :math:`(u_1, u_2) \in V^1 \times V^2` such that

.. math::

  a_{11}(u_1,v_1) + a_{12}(u_2,v_1) + a_{22}(u_2,v_2) + a_{21}(u_1,v_2) = L_1(v_1) + L_2(v_2) \ \textrm{for all}\ (v_1, v_2) \in V^1 \times V^2.

[TODO: Maybe discuss discretisation - how does this become Ax = b]

------------------------
Dirichlet-Neumann Method
------------------------
To computationally approximate the coupled problem, the Dirichlet-Neumann method is applied. This method enforces further conditions on the solution:

.. math::

  \begin{cases}

    L u_1^{(k)} &= f \ \textrm{in}\ \Omega_1,\\
    u_1^{(k)} &= u_2^{(k-1)} \ \textrm{on}\ \Gamma,\\
    u_1^{(k)} &= 0 \ \textrm{on}\ \partial \Omega_1 \setminus \Gamma.
  
  \end{cases}

  \begin{cases}

    L u_2^{(k)} &= f \ \textrm{in}\ \Omega_2,\\
    \frac{\partial u_2^{(k)}}{\partial n} &= \frac{\partial u_1^{(k)}}{\partial n} \ \textrm{on}\ \Gamma,\\
    u_2^{(k)} &= 0 \ \textrm{on}\ \partial \Omega_2 \setminus \Gamma.

  \end{cases}

- TODO: Talk about how this is implemented

--------------------------------------
Method of Manufactured Solutions (MMS)
--------------------------------------

The method of manufactured solutions verifies the accuracy of implemented finite element models. This is done by explicitly specifying a solution for the differential equation at hand, ensuring that this solution satisfies all conditions set on the problem. We can then analyse the accuracy of the approximated solution. 

- TODO: Provide example solution 

--------------
Implementation
--------------

Can add python block using::

  from firedrake import *
  n = 30
  mesh = UnitSquareMesh(n, n)

Block ends when indentation finishes.



A python script version of this demo can be found :demo:`here <coupled_equations.py>`.

.. _DG advection equation with upwinding: https://www.firedrakeproject.org/demos/DG_advection.py.html
.. _Simple Helmholtz equation: https://www.firedrakeproject.org/demos/helmholtz.py.html
.. _Mixed formulation for the Poisson equation: https://www.firedrakeproject.org/demos/poisson_mixed.py.html
