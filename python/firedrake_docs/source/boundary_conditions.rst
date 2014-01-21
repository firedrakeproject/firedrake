.. default-role:: math

Boundary conditions
===================

Mathematical background
-----------------------

To understand how Firedrake applies strong (Dirichlet) boundary
conditions, it is necessary to write the variational problem to be
solved in residual form: find `u \in V` such that:

.. math::

  F(u; v) = 0 \quad \forall v\in V.

This is the natural form of a nonlinear problem. A linear problem is
frequently written: find `u \in V` such that:

.. math::

  a(u, v) = L(v) \quad \forall v \in V.

However, this form can trivially be rewritten in residual form by defining:

.. math::

  F(u; v) = a(u, v) - L(v).

In the general case, `F` will be always linear in `v` but
may be nonlinear in `u`.

When we impose a strong (Dirichlet, essential) boundary condition on
`u`, we are substituting constraint:

.. math::

  u = g(x) \ \textrm{on}\ \Gamma_D

for the original equation on `\Gamma_D`, where `\Gamma_D`
is some subset of the domain boundary. To impose this constraint, we
first split the function space `V`:

.. math::

  V = V_0 \oplus V_\Gamma

where `V_\Gamma` is the space spanned by those functions in the
basis of `V` which are non-zero on `\Gamma_D`, and
`V_0` is the space spanned by the remaining basis functions (ie
those basis functions which vanish on `\Gamma_D`).

In Firedrake we always have a nodal basis for `V`, `\phi_V
= \{\phi_i\}` and we will write `\phi^0` and
`\phi^\Gamma` for the subsets of that basis which span
`V_0` and `V_\Gamma` respectively.

We can similarly write `v\in V` as `v_0+v_\Gamma` and use the
linearity of `F` in `v`:

.. math::
 
   F(u; v) = F(u; v_0) + F(u; v_\Gamma)

If we impose a Dirichlet condition over `\Gamma_D` then we no
longer impose the constraint `F(u; v_\Gamma)=0` for any
`v_\Gamma\in V_\Gamma`. Instead, we need to impose a term which
is zero when `u` satisfies the boundary conditions, and non-zero
otherwise. So we define:

.. math::

   F_\Gamma(u; \phi_i) =  u_i - g_i \quad \phi_i\in \phi^\Gamma
   
where `g_i` indicates the evaluation of `g(x)` at the node
associated with `\phi_i`. Note that the stipulation that
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

We write `u = u_i\phi_i` as the current interation of the
solution and write `\mathrm{U}` for the vector whose components
are the coefficients `u_i`. Similarly, we write `u^*` for
the next iterate and `\mathrm{U}^*` for the vector of its
coefficients. Then a single step of Newton is given by:

.. math::

   \mathrm{U}^* = \mathrm{U} - J^{-1} \mathrm{F}(u)

where `\mathrm{F}(u)_i = \hat F(u; \phi_i)` and where
`J` is the Jacobian matrix defined by the Gâteaux derivative of
`F`:

.. math::

   dF(u; \tilde{u}, v) = \lim_{h\rightarrow0}
   \frac{\hat F(u+h\tilde u; v) - \hat F(u; v)}{h} \quad \forall v,
   \tilde u \in V

The actual Jacobian matrix is given by:
 
.. math::

   J_{ij} = dF(u, \phi_i, \phi_j)

where `\phi_i`, `\phi_j` are the ith and jth 
basis functions of `V`. Our definition of the modified residual
`\hat F` produces some interesting results for the boundary condition
rows of `J`:

.. math::

   J_{ij} = \begin{cases} 1 & i=j\ \textrm{and}\ \phi_j\in \phi^\Gamma\\
   0 & i\neq j\ \textrm{and}\ \phi_j\in \phi^\Gamma\end{cases}
