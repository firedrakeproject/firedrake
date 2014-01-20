Boundary conditions
===================

Mathematical background
-----------------------

To understand how Firedrake applies strong (Dirichlet) boundary
conditions, it is necessary to write the variational problem to be
solved in residual form: find :math:`u \in V` such that:

.. math::

  F(u; v) = 0 \quad \forall v\in V.

This is the natural form of a nonlinear problem. A linear problem is
frequently written: find :math:`u \in V` such that:

.. math::

  a(u, v) = L(v) \quad \forall v \in V.

However, this form can trivially be rewritten in residual form by defining:

.. math::

  F(u; v) = a(u, v) - L(v).

In general, :math:`F` will be linear in :math:`v` but nonlinear in :math:`u`. 

When we impose a strong (Dirichlet, essential) boundary condition on
:math:`u`, we are substituting constraint:

.. math::

  u = g(x) \ \textrm{on}\ \Gamma_D

for the original equation on :math:`\Gamma_D`, where :math:`\Gamma_D`
is some subset of the domain boundary. To impose this constraint, we
first split the function space :math:`V`:

.. math::

  V = V_0 \oplus V_\Gamma

where :math:`V_\Gamma` is the space spanned by those functions in the
basis of :math:`V` which are non-zero on :math:`\Gamma_D`, and
:math:`V_0` is the space spanned by the remaining basis functions (ie
those basis functions which vanish on :math:`\Gamma_D`).

We can similarly write :math:`v\in V` as :math:`v_0+v_\Gamma` and use the
linearity of :math:`F` in :math:`v`:

.. math::
 
   F(u; v) = F(u; v_0) + F(u; v_\Gamma)

If we impose a Dirichlet condition over :math:`\Gamma_D` then we no
longer impose the constraint :math:`F(u; v_\Gamma)=0` for any
:math:`v_\Gamma\in V_\Gamma`. Instead, we need to impose a term which
is zero when :math:`u` satisfies the boundary conditions, and non-zero
otherwise. We always have a nodal basis for :math:`V`,
:math:`\{\phi_i\}`. So we define for those basis function :math:`\phi_i\in
V_\Gamma`:

.. math::

   F_\Gamma(u; \phi_i) =  u_i - g_i \quad \phi_i\in V_\Gamma
   
where :math:`g_i` indicates the evaluation of :math:`g(x)` at the node
associated with :math:`\phi_i`. Note that the stipulation that
:math:`F_\Gamma(u; v)` must be linear in :math:`v` is sufficient to
extend the definition to any :math:`v\in V_\Gamma`.

This means that the full statement of the problem in residual form
becomes: find :math:`u\in V` such that:

.. math::

   F'(u; v_0 + v_\Gamma) = F(u; v_0) + F_\Gamma(u; v_\Gamma) = 0 \quad \forall v_0\in V_0,
   \forall v_\Gamma \in V_\Gamma.

Solution strategy
-----------------

The system of equations will be solved by a gradient-based nonlinear
solver, of which a simple and illustrative example is a Newton
solver. Firedrake applies this solution strategy to linear equations
too, although in that case only one iteration of the nonlinear solver
will ever be required or executed.

We write :math:`u = u_i\phi_i` as the current interation of the
solution and write :math:`\mathrm{U}` for the vector whose components
are the coefficients :math:`u_i`. Similarly, we write :math:`u^*` for
the next iterate and :math:`\mathrm{U}^*` for the vector of its
coefficients. Then a single step of Newton is given by:

.. math::

   \mathrm{U}^* = \mathrm{U} - J^{-1} \mathrm{F}(\mathrm{U})

where :math:`\mathrm{F}(\mathrm{U})_i = F(u; \phi_i)` and where
:math:`J` is the Jacobian matrix defined by the Gateaux derivative of
:math:`F`:

.. math::

   J(\tilde{u}, v) = dF(u; \tilde{u}, v) = \lim{h\rightarrow0} F(u+h\u\tilde)
