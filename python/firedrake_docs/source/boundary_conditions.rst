Boundary conditions
===================

Mathematical background
-----------------------

To understand how Firedrake applies strong (Dirichlet) boundary
conditions, it is necessary to write the variational problem to be
solved in residual form: find :math:`u \in V` such that::

  F(u; v) = 0 \ \forall v\in V.

This is the natural form of a nonlinear problem. A linear problem is
frequently written: find :math:`u \in V` such that::

  a(u, v) = L(v) \ \forall v in V.

However, this form can trivially be rewritten in residual form by defining::

  F(u; v) = a(u, v) - L(v).

In general, F will be linear in :math:`v` but nonlinear in :math:`u`. 

When we impose a strong (Dirichlet, essential) boundary condition on
:math:`u`, we are imposing the additional constraint::

  u = g(x) \ \textrm{on} \Gamma_D

where :math:`\Gamma_D` is some subset of the domain boundary. To
impose this constraint, we first split the function space :math:`V`::

  V = V_0 + V_\Gamma

where :math:`V_\Gamma` is the space spanned by those functions in the
basis of :math:`V` which are non-zero on :math:`\Gamma_D`, and :math:`V_0` is the space spanned by the remaining 
