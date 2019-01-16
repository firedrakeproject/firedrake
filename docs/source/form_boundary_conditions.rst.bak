.. default-role:: math

.. only:: html

   .. contents::

Form boundary conditions
=============================

Form boundary conditions :class:`~.FormBC` can be regarded as a
generalization of Dirichlet boundary conditions :class:`~.DirichletBC`. 
For :class:`~.FormBC`, instead of prescribing the values of the solution 
on a boundary, we prescribe equations to be satisfied on the boundary.  
This document explains the mathematical formulation of the form boundary 
conditions, and their implementation.


Mathematical background
-----------------------

The discussion given here for :class:`~.FormBC` is similar to that for 
:class:`~.DirichletBC`, but we make appropriate changes emphasizing the 
difference between these two.
We again consider a nonlinear variational problem 
in residual form: find `u \in V` such that:

.. math::

  F(u; v) = 0 \quad \forall v\in V.

A linear problem: find `u \in V` such that:

.. math::

  a(u, v) = L(v) \quad \forall v \in V

is rewritten in residual form by defining:

.. math::

  F(u; v) = a(u, v) - L(v).

In the general case, `F` will be always linear in `v` but
may be nonlinear in `u`.

When we impose a form boundary condition on
`u`, we are substituting the constraint (in variational form):

.. math::

  F_\Gamma(u; v) = 0 \ \text{on}\ \Gamma_F

for the original equation on `\Gamma_F`, where `\Gamma_F`
is some subset of the domain boundary. To impose this constraint, we
first split the function space `V`:

.. math::

  V = V_0 \oplus V_\Gamma

where `V_\Gamma` is the space spanned by those functions in the
basis of `V` which are non-zero on `\Gamma_F`, and
`V_0` is the space spanned by the remaining basis functions (i.e.
those basis functions which vanish on `\Gamma_F`).

In Firedrake we always have a nodal basis for `V`, `\phi_V
= \{\phi_i\}`, and we will write `\phi^0` and
`\phi^\Gamma` for the subsets of that basis which span
`V_0` and `V_\Gamma` respectively.

We can similarly write `v\in V` as `v_0+v_\Gamma` and use the
linearity of `F` in `v`:

.. math::
 
   F(u; v) = F(u; v_0) + F(u; v_\Gamma)

If we impose a Form boundary condition over `\Gamma_F` then we no
longer impose the constraint `F(u; v_\Gamma)=0` for any
`v_\Gamma\in V_\Gamma`, but instead we impose:

.. math::

   F_\Gamma(u; \phi_i) =  0 \quad \phi_i\in \phi^\Gamma.
   
Note that the stipulation that
`F_\Gamma(u; v)` must be linear in `v` is sufficient to
extend the definition to any `v\in V_\Gamma`.

This means that the full statement of the problem in residual form
becomes: find `u\in V` such that:

.. math::

   \hat F(u; v_0 + v_\Gamma) = F(u; v_0) + F_\Gamma(u; v_\Gamma) = 0 \quad \forall v_0\in V_0,
   \forall v_\Gamma \in V_\Gamma.


Solution strategy
-----------------

The system of equations will be solved by a gradient-based nonlinear
solver, of which a simple and illustrative example is a Newton
solver. Firedrake applies this solution strategy to linear equations
too, although in that case only one iteration of the nonlinear solver
will ever be required or executed.

We write `u = u_i\phi_i` as the current iteration of the
solution and write `\mathrm{U}` for the vector whose components
are the coefficients `u_i`. Similarly, we write `u^*` for
the next iterate and `\mathrm{U}^*` for the vector of its
coefficients. Then a single step of Newton is given by:

.. math::

   \mathrm{U}^* = \mathrm{U} - J^{-1} \mathrm{F}(u)

where `\mathrm{F}(u)_i = \hat F(u; \phi_i)` and
`J` is the Jacobian matrix defined by the Gâteaux derivative of
`F`:

.. math::

   dF(u; \tilde{u}, v) = \lim_{h\rightarrow0}
   \frac{\hat F(u+h\tilde u; v) - \hat F(u; v)}{h} \quad \forall v,
   \tilde u \in V

The actual Jacobian matrix is given by:
 
.. math::

   J_{ij} = dF(u; \phi_i, \phi_j)

where `\phi_i`, `\phi_j` are the ith and jth 
basis functions of `V`. Our definition of the modified residual
`\hat F` produces submatrices of distinct structures on the form 
boundary condition rows of `J` and on the remaining rows of `J`.
In other words, the rows of `J` corresponding to the boundary
condition nodes are replaced by Jacobian matrix corresponding to
`F_\Gamma`.
The resulting Jacobian matrix is thus non-symmetric in general.

Contrary to the case of Dirichlet boundary conditions, we do not
know the values of the solution on the boundary condition nodes a priori,
and the whole system of equations are to be solved monolithically
using linear/nonlinear solvers.


Implementation
--------------

Variational problems
~~~~~~~~~~~~~~~~~~~~

Both linear and nonlinear PDEs are solved in residual form in
Firedrake using the `PETSc SNES interface <http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/SNES/>`_. In the case of linear
systems, a single step of Newton is employed. 

In the following we will use ``F`` for the residual :class:`~ufl.form.Form`
and ``J`` for the Jacobian :class:`~ufl.form.Form`. In both cases these
forms do not include the boundary conditions. 
A Form boundary condition :class:`~.FormBC` object separately carries ``F_{Form}`` 
for the boundary residual :class:`~ufl.form.Form` and ``J_{Form}`` for the 
boundary Jacobian :class:`~ufl.form.Form`.
Additionally ``u`` will be the solution :class:`~.Function`.

Form boundary conditions are applied as follows:

1. Each time the solver assembles the Jacobian matrix, the following happens. 

   a) ``J`` is assembled using modified indirection maps in which the
      row indices associated with form boundary condition node have been replaced by negative
      values. PETSc interprets these negative indices as an
      instruction to drop the corresponding entry. 

   b) ``J_{Form}`` is assembled to populate the form boundary node rows
      that are not populated in a).
   
2. Each time the solver assembles the residual, the following happens.
   
   a) ``F`` is assembled using unmodified indirection maps taking no
      account of the boundary conditions. This results in an assembled
      residual which is correct on the non-boundary condition nodes but
      contains spurious values in the boundary condition entries.

   b) The entries of ``F`` corresponding to boundary condition nodes
      are set to zero.

   c) ``F_{Form}`` is assembled to populate the entries corresponding to 
      the form boundary condition nodes that have been zeroed in b).

