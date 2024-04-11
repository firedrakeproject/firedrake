.. only:: html

   .. contents::

Solving PDEs
============

Introduction
------------

Now that we have learnt how to define weak variational problems, we
will move on to how to actually solve them using Firedrake.  Let us
consider a weak variational problem

.. math::

   a(u, v) = L(v) \; \forall v \in V \mathrm{on}\: \Omega

   u = u_0 \; \mathrm{on}\: \partial\Omega

we will call the bilinear and linear parts of this form ``a`` and
``L`` respectively.  The strongly imposed boundary condition, :math:`u
= u_0 \;\mathrm{on}\:\partial\Omega` will be represented by a variable
of type :py:class:`~.DirichletBC`, ``bc``.

Now that we have all the pieces of our variational problem, we can
move forward to solving it.

.. _solve_var_problem:

Solving the variational problem
-------------------------------

The function used to solve PDEs defined as above is
:py:func:`~firedrake.solving.solve`.  This is a unified interface for
solving both linear and non-linear variational problems along with
linear systems (where the arguments are already assembled matrices and
vectors, rather than `UFL`_ forms).  We will treat the variational
interface first.

Linear variational problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the problem is linear, that is ``a`` is linear in both the test and
trial functions and ``L`` is linear in the test function, we can use
the linear variational problem interface to ``solve``.  To start, we
need a :py:class:`~.Function` to hold the value of
the solution:

.. code-block:: python3

   s = Function(V)

We can then solve the problem, placing the solution in ``s`` with:

.. code-block:: python3

   solve(a == L, s)

To apply boundary conditions, one passes a list of
:py:class:`~.DirichletBC` objects using the ``bcs``
keyword argument.  For example, if there are two boundary conditions,
in ``bc1`` and ``bc2``, we write:

.. code-block:: python3

   solve(a == L, s, bcs=[bc1, bc2])

Nonlinear variational problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For nonlinear problems, the interface is similar.  In this case, we
solve a problem:

.. math::

    F(u; v) = 0 \; \forall v \in V \mathrm{on}\: \Omega

    u = u_0 \; \mathrm{on}\: \partial\Omega

where the *residual* :math:`F(u; v)` is linear in the test function
:math:`v` but possibly non-linear in the unknown
:py:class:`~.Function` :math:`u`.  To solve such a
problem we write, if ``F`` is the residual form:

.. code-block:: python3

   solve(F == 0, u)

to apply strong boundary conditions, as before, we provide a list of
``DirichletBC`` objects using the ``bcs`` keyword:

.. code-block:: python3

   solve(F == 0, u, bcs=[bc1, bc2])

Nonlinear problems in Firedrake are solved using Newton-like methods.
That is, we compute successive approximations to the solution using

.. math::

   u_{k+1} = u_{k} - J(u_k)^{-1} F(u_k) \; k = 0, 1, \dots

where :math:`u_0` is an initial guess for the solution and
:math:`J(u_k) = \frac{\partial F(u_k)}{\partial u_k}` is the
*Jacobian* of the residual, which should be non-singular at each
iteration.  Notice how in the above examples, we did not explicitly
supply a Jacobian.  If it is not supplied, it will be computed by
automatic differentiation of the residual form ``F`` with respect to the
solution variable ``u``.  However, we may also supply the Jacobian
explicitly, using the keyword argument ``J``:

.. code-block:: python3

   solve(F == 0, u, J=user_supplied_jacobian_form)

The initial guess for the Newton iterations is provided in ``u``, for
example, to provide a non-zero guess that the solution is the value of
the ``x`` coordinate everywhere:

.. code-block:: python3

   x = SpatialCoordinate(m)
   u.interpolate(x[0])

   solve(F == 0, u)

Solving linear systems
----------------------

Often, we might be solving a time-dependent linear system.  In this
case, the bilinear form ``a`` does not change between timesteps, whereas
the linear form ``L`` does.  Since assembly of the bilinear form is a
potentially costly process, Firedrake offers the ability to
"pre-assemble" forms in such systems and then reuse the assembled
operator in successive linear solves.  Again, we use the same ``solve``
interface to do this, but must build slightly different objects to
pass in.  In the pre-assembled case, we are solving a linear system:

.. math::

   A\vec{x} = \vec{b}

Where :math:`A` is a known matrix, :math:`\vec{b}` is a known right
hand side vector and :math:`\vec{x}` is the unknown solution vector.
In Firedrake, :math:`A` is represented as a
:py:class:`~.Matrix`, while :math:`\vec{x}` is a :py:class:`~.Function`, and
:math:`\vec{b}` a :py:class:`~.Cofunction`.
We build these values by calling ``assemble`` on the UFL forms that
define our problem, which, as before are denoted ``a`` and ``L``.
Similarly to the linear variational case, we first need a function in
which to place our solution:

.. code-block:: python3

   x = Function(V)

We then :py:func:`~.assemble` the left hand side
matrix ``A`` and known right hand side ``b`` from the bilinear and
linear forms respectively:

.. code-block:: python3

   A = assemble(a)
   b = assemble(L)

Finally, we can solve the problem placing the solution in ``x``:

.. code-block:: python3

   solve(A, x, b)

to apply boundary conditions to the problem, we can assemble the
linear operator ``A`` with boundary conditions using the ``bcs``
keyword argument to :py:func:`~.assemble` (and then
not supply them in solve call):

.. code-block:: python3

   A = assemble(a, bcs=[bc1, bc2])
   b = assemble(L)
   solve(A, x, b)

.. warning::

   It is no longer possible to apply or change boundary
   conditions after assembling the matrix ``A``; pass any
   necessary boundary conditions to :py:func:`~.assemble`.

Specifying solution methods
---------------------------

Not all linear and non-linear systems defined by PDEs are created
equal, and we therefore need ways of specifying which solvers to use
and options to pass to them.  Firedrake uses `PETSc`_ to solve both
linear and non-linear systems and presents a uniform interface in
``solve`` to set PETSc solver options.  In all cases, we set options
in the solve call by passing a dictionary to the ``solver_parameters``
keyword argument.  To set options we use the same names that PETSc
uses in its command-line option setting interface (having removed the
leading ``-``).  For more complete details on PETSc option naming we
recommend looking in the `PETSc manual`_.  We describe some of the
more common options here.

Configuring solvers from the commandline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As well as specifying solver options in a parameters dict at the call
site for each solve, one can configure solvers by passing options in
the normal PETSc style via the commandline.  To do this, we need to
specify the ``options_prefix`` for each solver that we wish to
configure via the commandline.  This is done by providing a
non-``None`` argument as the ``options_prefix`` keyword argument to
the solver.  The separator between the prefix and the subsequent
options is an underscore, which is automatically appended if the
provided options prefix does end in one.

When using an options prefix, we do not need to specify the prefix in
the solver parameters dictionary (it is automatically added in the
appropriate way).  Also to note is that command line options
*override* parameters set in the dictionary.  This way we can provide
good defaults for solvers, and override them on a case-by-case basis.

For example, suppose we have a file ``pde.py`` that contains

.. code-block:: python3

   ...
   solve(F == 0, u, options_prefix="pde",
         solver_parameters={"ksp_type": "gmres"})

If we run this code as:

.. code-block:: sh

   python pde.py

Then the KSP solver will be GMRES.  Conversely, when running

.. code-block:: sh

   python pde.py -pde_ksp_type cg

we will use conjugate gradients as the KSP solver.

Linear solver options
~~~~~~~~~~~~~~~~~~~~~

We use a PETSc `KSP`_ object to solve linear systems.  This is a
uniform interface for solving linear systems using Krylov subspace
methods.  By default, the solve call will use GMRES using an
incomplete LU factorisation to precondition the problem.  To change
the Krylov method used in solving the problem, we set the
``'ksp_type'`` option.  For example, if we want to solve a modified
Helmholtz equation, we know the operator is symmetric positive
definite, and therefore can choose the conjugate gradient method,
rather than GMRES.

.. code-block:: python3

   solve(a == L, solver_parameters={'ksp_type': 'cg'})

To change the preconditioner used, we set the ``'pc_type'`` option.
For example, if PETSc has been installed with the `Hypre`_ package, we
can use its algebraic multigrid preconditioner, BoomerAMG, to
precondition the system with:

.. code-block:: python3

   solve(a == L,
         solver_parameters={'pc_type': 'hypre',
                            'pc_hypre_type': 'boomeramg'})

Although the `KSP` name suggests that only Krylov methods are
supported, this is not the case.  We may, for example, solve the
system directly by computing an LU factorisation of the problem.  To
do this, we set the ``pc_type`` to ``'lu'`` and tell PETSc to use a
"preconditioner only" Krylov method:

.. code-block:: python3

   solve(a == L,
         solver_parameters={'ksp_type': 'preonly',
                            'pc_type': 'lu'})

In a similar manner, we can use Jacobi preconditioned Richardson
iterations with:

.. code-block:: python3

   solve(a == L,
         solver_parameters={'ksp_type': 'richardson',
                            'pc_type': 'jacobi'}

.. note::

   We note in passing that the method Firedrake utilises internally
   for applying strong boundary conditions does not destroy the
   symmetry of the linear operator.  If the system without boundary
   conditions is symmetric, it will continue to be so after the
   application of any boundary conditions.

.. _linear_solver_tols:

Setting solver tolerances
+++++++++++++++++++++++++

In an iterative solver, such as Krylov method, we iterate until some
specified tolerance is reached.  The measure of how much the current
solution :math:`\vec{x}_i` differs from the true solution is called
the residual and is calculated as:

.. math::

   r = |\vec{b} - A \vec{x}_i|

PETSc allows us to set three different tolerance options for solving
the system.  The *absolute tolerance* tells us we should stop if
:math:`r` drops below some given value.  The *relative tolerance*
tells us we should stop if :math:`\frac{r}{|\vec{b}|}` drops below
some given value.  Finally, PETSc can detect divergence in a linear
solve, that is, if :math:`r` increases above some specified value.
These values are set with the options ``'ksp_atol'`` for the absolute
tolerance, ``'ksp_rtol'`` for the relative tolerance, and
``'ksp_divtol'`` for the divergence tolerance.  The values provided to
these options should be floats.  For example, to set the absolute
tolerance to :math:`10^{-30}`, the relative tolerance to
:math:`10^{-9}` and the divergence tolerance to :math:`10^4` we would
use:

.. code-block:: python3

   solver_parameters={'ksp_atol': 1e-30,
                      'ksp_rtol': 1e-9,
                      'ksp_divtol': 1e4}

.. note::

   By default, PETSc (and hence Firedrake) check for the convergence
   in the preconditioned norm, that is, if the system is
   preconditioned with a matrix :math:`P` the residual is calculated
   as:

   .. math::

       r = |P^{-1}(\vec{b} - A \vec{x}_i)|

   to check for convergence in the unpreconditioned norm set the
   ``'ksp_norm_type'`` option to ``'unpreconditioned'``.


Finally, we can set the maximum allowed number of iterations for the
Krylov method by using the ``'ksp_max_it'`` option.

.. _mixed-preconditioning:

Preconditioning mixed finite element systems
++++++++++++++++++++++++++++++++++++++++++++

PETSc provides an interface to composing "physics-based"
preconditioners for mixed systems which Firedrake exploits when it
assembles linear systems.  In particular, for systems with two
variables (for example Navier-Stokes where we solve for the velocity
and pressure of the fluid), we can exploit PETSc's ability to build
preconditioners from Schur complements.  This is one type of
preconditioner based on PETSc's `fieldsplit`_ technology.  To take a
concrete example, let us consider solving the dual form of the
modified Helmholtz equation:

.. math::

   \langle p, q \rangle - \langle q, \mathrm{div} u \rangle + \lambda
   \langle v, u \rangle + \langle \mathrm{div}v, p \rangle =
   \langle f, q \rangle \; \forall v \in V_1, q \in V_2

This has a stable solution if, for example, :math:`V_1` is the lowest order
Raviart-Thomas space and :math:`V_2` is the lowest order discontinuous
space.

.. code-block:: python3

   V1 = FunctionSpace(mesh, 'RT', 1)
   V2 = FunctionSpace(mesh, 'DG', 0)
   W = V1 * V2
   lmbda = 1
   u, p = TrialFunctions(W)
   v, q = TestFunctions(W)
   f = Function(V2)

   a = (p*q - q*div(u) + lmbda*inner(v, u) + div(v)*p)*dx
   L = f*q*dx

   u = Function(W)
   solve(a == L, u,
         solver_parameters={'ksp_type': 'cg',
                            'pc_type': 'fieldsplit',
                            'pc_fieldsplit_type': 'schur',
                            'pc_fieldsplit_schur_fact_type': 'FULL',
                            'fieldsplit_0_ksp_type': 'cg',
                            'fieldsplit_1_ksp_type': 'cg'})

We refer to section 4.5 of the `PETSc manual`_ for more complete
details, but briefly describe the options in use here.  The monolithic
system is conceptually a :math:`2\times2` block matrix:

.. math::

   \left(\begin{matrix}
         \lambda \langle v, u \rangle & -\langle q, \mathrm{div} u \rangle \\
         \langle \mathrm{div} v, p \rangle & \langle p, q \rangle
         \end{matrix}
   \right) = \left(\begin{matrix} A & B \\ C & D \end{matrix}\right).

We can factor this block matrix in the following way:

.. math::

   \left(\begin{matrix} I & 0 \\ C A^{-1} & I\end{matrix}\right)
   \left(\begin{matrix}A & 0 \\ 0 & S\end{matrix}\right)
   \left(\begin{matrix} I & A^{-1} B \\ 0 & I\end{matrix}\right).

This is the *Schur complement factorisation* of the block system, its
inverse is:

.. math::

   P = \left(\begin{matrix} I & -A^{-1}B \\ 0 & I \end{matrix}\right)
   \left(\begin{matrix} A^{-1} & 0 \\ 0 & S^{-1}\end{matrix}\right)
   \left(\begin{matrix} I & 0 \\ -CA^{-1} & I\end{matrix}\right).

Where :math:`S` is the *Schur complement*:

.. math::

   S = D - C A^{-1} B.

The options in the example above use an approximation to :math:`P` to
precondition the system.  To do so, we tell PETSc that the
preconditioner should be of type ``'fieldsplit'``, and the the
fieldsplit's type should be ``'schur'``.  We then select a
factorisation type for the Schur complement.  The option ``'FULL'`` as
used above preconditions using an approximation to :math:`P`.  We can
also use ``'diag'`` which uses an approximation to:

.. math::

   \left(\begin{matrix} A^{-1} & 0 \\ 0 & -S^{-1} \end{matrix}\right).

Note the minus sign in front of :math:`S^{-1}` which is there such
that this preconditioner is positive definite.  Two other options are
``'lower'``, where the preconditioner is an approximation to:

.. math::

   \left(\begin{matrix}A & 0 \\ C & S\end{matrix}\right)^{-1} =
   \left(\begin{matrix}A^{-1} & 0 \\ 0 & S^{-1}\end{matrix}\right)
   \left(\begin{matrix}I & 0 \\ -C A^{-1} & I\end{matrix}\right)

and ``'upper'`` which uses:

.. math::

   \left(\begin{matrix}A & B \\ 0 & S\end{matrix}\right)^{-1} =
   \left(\begin{matrix}I & -A^{-1}B \\ 0 & I\end{matrix}\right)
   \left(\begin{matrix}A^{-1} & 0 \\ 0 & S^{-1}\end{matrix}\right).

Note that the inverses of :math:`A` and :math:`S` are never formed
explicitly by PETSc, instead their actions are computed approximately
using a Krylov method.  The choice of method is selected using the
``'fieldsplit_0_ksp_type'`` option (for the Krylov solver computing
:math:`A^{-1}`) and ``'fieldsplit_1_ksp_type'`` (for the Krylov solver
computing :math:`S^{-1}`).

.. note::

   If you have given your
   :py:class:`~.FunctionSpace`\s names, then
   instead of 0 and 1, you should use the name of the function space
   in these options.

By default PETSc uses an approximation to :math:`D^{-1}` to
precondition the Krylov system solving for :math:`S`, you can also use
a `least squares commutator <LSC_>`_, see the relevant section of the
`PETSc manual pages <fieldsplit_>`_ for more details.

Specifying assembled matrix types
+++++++++++++++++++++++++++++++++

Firedrake supports the assembly of linear operators in a number of
different formats.  Either as assembled sparse matrices, or as
matrix-free operators that only provide matrix-vector products.  Since
matrix-free actions require special preconditioning, they have
:doc:`their own section in the manual <matrix-free>`.  Even in the
sparse matrix case there are a few options.  Firedrake can build
nested block matrices, or monolithic sparse matrices.  The latter
admit a wider range of preconditioners, but are memory-inefficient
when using fieldsplit preconditioning.  Even monolithic matrices have
choices, specifically, if there is a block structure, whether that
should be exploited or not.  Again the trade-off is between memory
efficiency (in the block case) and access to a slightly smaller range
of preconditioners.

The default matrix type can be set with the global parameter
``parameters["default_matrix_type"]``.  In the case where the matrix
is assembled as a nested matrix, there is a choice as to the type of
the blocks (they may be "aij" or "baij").  The default choice can be
controlled with ``parameters["default_sub_matrix_type"]``.  For
finer-grained control over the matrix type, one can provide it when
calling :func:`~.assemble` through the ``mat_type`` and
``sub_mat_type`` keyword arguments.  When using variational solvers,
the matrix type is controlled through use of the ``solver_parameters``
dictionary by specifying the ``"mat_type"`` entry.

.. note::

   It is not currently possible to control the matrix type of
   sub-matrices through the solving interface.  If you need this
   functionality, please :doc:`get in touch <contact>`

More block preconditioners
++++++++++++++++++++++++++

As well as physics-based Schur complement preconditioners for block
systems, PETSc also allows us to use preconditioners formed from block
Jacobi (``'pc_fieldsplit_type': 'additive'``) and block Gauss-Seidel
(``'multiplicative'`` or ``'symmetric_multiplicative'``) inverses of
the block system.  These work for any number of blocks, whereas the
Schur complement approach mentioned above only works for two by two
blocks.  There is also a :doc:`separate manual section <matrix-free>`
on specifying preconditioners that require auxiliary operators.

Recursive fieldsplits
+++++++++++++++++++++

If your system contains more than two fields, it is possible to
recursively define block preconditioners by specifying the fields
which should belong to each split.  Note that at present this only
works for "monolithically assembled" matrices, so you should set the
solver parameter ``"mat_type"`` to ``"aij"`` when solving your system
or assembling your matrix.  To change the default assembly from nested
matrices to monolithically assembled matrices, set the global
parameter ``parameters["default_matrix_type"] = "aij"``.

As an example, consider a three field system which we wish to
precondition by forming a schur complement of the first two fields
into the third, and then using a multiplicative fieldsplit with LU on
each split for the approximation to :math:`A^{-1}` and ILU to
precondition the schur complement.  The solver parameters we need are
as follows:

.. code-block:: python3

   parameters = {"pc_type": "fieldsplit",
                 "pc_fieldsplit_type": "schur",
                 # first split contains first two fields, second
                 # contains the third
                 "pc_fieldsplit_0_fields": "0, 1",
                 "pc_fieldsplit_1_fields": "2",
                 # Multiplicative fieldsplit for first field
                 "fieldsplit_0_pc_type": "fieldsplit",
                 "fieldsplit_0_pc_fieldsplit_type": "multiplicative",
                 # LU on each field
                 "fieldsplit_0_fieldsplit_0_pc_type": "lu",
                 "fieldsplit_0_fieldsplit_1_pc_type": "lu",
                 # ILU on the schur complement block
                 "fieldsplit_1_pc_type": "ilu"}

In this example, none of the :class:`~.FunctionSpace`\s used had
names, and hence we referred to the fields by number.  If the
function spaces are named, then any time a single field appears as a
split, its options prefix is referred to by the space's *name* (rather
than a number).  Concretely, if the previous example had use a set of
FunctionSpace definitions:

.. code-block:: python3

   V = FunctionSpace(..., name="V")
   P = FunctionSpace(..., name="P")
   T = FunctionSpace(..., name="T")
   W = V*P*T

Then we would have referred to the single (field 1) split using
``fieldsplit_T_pc_type``, rather than ``fieldsplit_1_pc_type``.

.. _nested_options_blocks:

Specifying nested options blocks
++++++++++++++++++++++++++++++++

For complex nested preconditioners, it can be tedious to write out the
same prefix over and over.  Moreover, we may have a block system where
multiple blocks use the same preconditioning options.  It is then
error-prone to type these options out twice.  To alleviate these
problems, one can describe the nesting in the solver parameters
dictionary by using a nested :class:`.dict` as the value.  In this
case, the key is used as an options prefix to all of the key-value
pairs in the nested dictionary.  As an example, the following two
parameter sets are equivalent:

.. code-block:: python3

   {"ksp_type": "cg",
    "pc_type": "fieldsplit",
    "fieldsplit_0": {"ksp_type": "gmres",
                     "pc_type": "hypre",
                      "ksp_rtol": 1e-5},
    "fieldsplit_1": {"ksp_type": "richardson",
                     "pc_type": "ilu"}}

and

.. code-block:: python3

   {"ksp_type": "cg",
    "pc_type": "fieldsplit",
    "fieldsplit_0_ksp_type": "gmres",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_0_ksp_rtol": 1e-5,
    "fieldsplit_1_ksp_type": "richardson",
    "fieldsplit_1_pc_type": "ilu"}

PETSc uses an underscore as a separator between option names, and we
do the same.  For convenience, the prefix key to a nested dict can
omit the trailing underscore, it will be added automatically if
missing.  Hence

.. code-block:: python3

   {"a": {"b": "foo"}}

and

.. code-block:: python3

   {"a_": {"b": "foo"}}

both expand to

.. code-block:: python3

   {"a_b": "foo"}

Nonlinear solver options
~~~~~~~~~~~~~~~~~~~~~~~~

As for linear systems, we use a PETSc object to solve nonlinear
systems.  This time it is a `SNES`_.  This offers a uniform interface
to Newton-like and quasi-Newton solution schemes.  To select the SNES
type to use, we use the ``'snes_type'`` option.  Recall that each
Newton iteration is the solution of a linear system, options for the
inner linear solve may be set in the same way as described above for
linear problems.  For example, to solve a nonlinear problem using
Newton-Krylov iterations using a line search and direct factorisation
to solve the linear system we would write:

.. code-block:: python3

   solve(F == 0, u,
         solver_parameters={'snes_type': 'newtonls',
                            'ksp_type': 'preonly',
                            'pc_type': 'lu'}

.. note::

   Not all of PETSc's SNES types are currently supported by Firedrake,
   since some of them require extra information which we do not
   currently provide.


Setting convergence criteria
++++++++++++++++++++++++++++

In addition to setting the tolerances for the inner, linear solve in a
nonlinear system, which is done in exactly the same way as for
:ref:`linear problems <linear_solver_tols>`, we can also set
convergence tolerances on the outer SNES object.  These are the
*absolute tolerance* (``'snes_atol'``), *relative tolerance*
(``'snes_rtol'``), *step tolerance* (``'snes_stol'``) along with the
maximum number of nonlinear iterations (``'snes_max_it'``) and the
maximum number of allowed function evaluations (``'snes_max_func'``).
The step tolerance checks for convergence due to:

.. math::

   |\Delta x_k| < \mathrm{stol} \, |x_k|

The maximum number of allowed function evaluations limits the number
of times the residual may be evaluated before returning a
non-convergence error, and defaults to 1000.


Providing an operator for preconditioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, Firedrake uses the Jacobian of the residual (or equally
the bilinear form for linear problems) to construct preconditioners
for the linear systems it solves.  That is, it does not directly
solve:

.. math::

   A \vec{x} = \vec{b}

but rather

.. math::

   \tilde{A}^{-1} A \vec{x} = \tilde{A}^{-1} \vec{b}

where :math:`\tilde{A}^{-1}` is an approximation to :math:`A^{-1}`.  If we
know something about the structure of our problem, we may be able to
construct an operator :math:`P` explicitly which is "easy" to invert,
and whose inverse approximates :math:`A^{-1}` well.  Firedrake allows
you to provide this operator when solving variational problems by
passing an explicit ``Jp`` keyword argument to the solve call,
the provided form will then be used to construct an approximate
inverse when preconditioning the problem, rather than the form we're
solving with.

.. code-block:: python3

   a = ...
   L = ...
   Jp = ...
   # Use the approximate inverse of Jp to precondition solves
   solve(a == L, ..., Jp=Jp)

Default solver options
~~~~~~~~~~~~~~~~~~~~~~

If no parameters are passed to a solve call, we use, in most cases,
the defaults that PETSc supplies for solving the linear or nonlinear
system.  We describe the most commonly modified options (along with
their defaults in Firedrake) here.  For linear variational solves we
use:

* ``ksp_type``: GMRES, with a restart (``ksp_gmres_restart``) of 30
* ``ksp_rtol``: 1e-7
* ``ksp_atol``: 1e-50
* ``ksp_divtol`` 1e4
* ``ksp_max_it``: 10000
* ``pc_type``: ILU (Jacobi preconditioning for mixed problems)

For nonlinear variational solves we have:

* ``snes_type``: Newton linesearch
* ``ksp_type``: GMRES, with a restart (``ksp_gmres_restart``) of 30
* ``snes_rtol``: 1e-8
* ``snes_atol``: 1e-50
* ``snes_stol``: 1e-8
* ``snes_max_it``: 50
* ``ksp_rtol``: 1e-5
* ``ksp_atol``: 1e-50
* ``ksp_divtol``: 1e4
* ``ksp_max_it``: 10000
* ``pc_type``: ILU (Jacobi preconditioning for mixed problems)

To see the full view that PETSc has of solver objects, you can pass a
view flag to the solve call.  For linear solves pass:

.. code-block:: python3

   solver_parameters={'ksp_view': None}

For nonlinear solves use:

.. code-block:: python3

   solver_parameters={'snes_view': None}

PETSc will then print its view of the solver objects that Firedrake
has constructed.  This is especially useful for debugging complicated
preconditioner setups for mixed problems.

.. _singular_systems:

Solving singular systems
------------------------

Some systems of PDEs, for example the Poisson equation with pure
Neumann boundary conditions, have an operator which is singular.  That
is, we have :math:`Ae = 0` with :math:`e \neq 0`.  The vector space
spanned by the set of vectors :math:`{e}` for which :math:`Ae = 0` is
termed the *null space* of :math:`A`.  If we wish to solve such a
system, we must remove the null space from the solution.  To do this
in Firedrake, we first must define the null space, and then inform the
solver of its existance.  We use a
:class:`~firedrake.nullspace.VectorSpaceBasis` to hold the vectors
which span the null space.  We must provide a list of
:class:`~.Function`\s or
:class:`~.Vector`\s spanning the space.  Additionally,
since removing a constant null space is such a common operation, we
can pass ``constant=True`` to the constructor (rather than
constructing the constant vector by hand).  Note that the vectors we
pass in must be *orthonormal*.  Once the null space is built, we just
need to inform the solve about it (using the ``nullspace`` keyword
argument).

As an example, consider the Poisson equation with pure Neumann
boundary conditions:

.. math::

   -\nabla^2 u &= 0 \quad \mathrm{in}\;\Omega\\
   \nabla u \cdot n &= g \quad \mathrm{on}\;\Gamma.

We will solve this problem on the unit square applying homogeneous
Neumann boundary conditions on the planes :math:`x = 0` and :math:`x =
1`.  On :math:`y = 0` we set :math:`g = -1` while on :math:`y = 1` we
set :math:`g = 1`.  The null space of the operator we form is the set
of constant functions, and thus the problem has solution
:math:`u(x, y) = y + c` where :math:`c` is a constant.  To solve the
problem, we will inform the solver of this constant null space, fixing
the solution to be :math:`u(x, y) = y - 0.5`.

.. code-block:: python3

   m = UnitSquareMesh(25, 25)
   V = FunctionSpace(m, 'CG', 1)
   u = TrialFunction(V)
   v = TestFunction(V)

   a = inner(grad(u), grad(v))*dx
   L = -v*ds(3) + v*ds(4)

   nullspace = VectorSpaceBasis(constant=True)
   u = Function(V)
   solve(a == L, u, nullspace=nullspace)
   x = SpatialCoordinate(m)
   exact = Function(V).interpolate(x[1] - 0.5)
   print sqrt(assemble((u - exact)*(u - exact)*dx))

For this to work, the provided right hand side must be orthogonal to
the transpose nullspace of the operator as well.  In many cases, we
can arrange for this to occur by careful choice of initial
conditions.  Sometimes this is not possible.  In this case, you can
ask Firedrake to remove the component of the right hand side that is
in the transpose nullspace by providing a
:class:`~firedrake.nullspace.VectorSpaceBasis` with the
``transpose_nullspace`` keyword argument to :func:`~.solve`.

Singular operators in mixed spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have an operator in a mixed space, you may well precondition
the system using a :ref:`Schur complement <mixed-preconditioning>`.  If
the operator is singular, you will therefore have to tell the solver
about the null space of each diagonal block separately.  To do this in
Firedrake, we build a
:class:`~firedrake.nullspace.MixedVectorSpaceBasis` instead of a
:class:`~firedrake.nullspace.VectorSpaceBasis` and then inform the
solver about it as before.  A
:class:`~firedrake.nullspace.MixedVectorSpaceBasis` takes a list of
:class:`~firedrake.nullspace.VectorSpaceBasis` objects defining the
null spaces of each of the diagonal blocks in the mixed operator.  In
addition, as a first argument, you must provide the
:class:`~.MixedFunctionSpace` you're building a basis for.  You do not
have to provide a null space for all blocks.  For those you don't care
about, you can pass an indexed function space at the appropriate
position.  For example, imagine we have a mixed space :math:`W = V
\times Q` and an operator which has a null space of constant functions
in :math:`V` (this occurs, for example, for a discretisation of the
mixed poisson problem on the surface of a sphere).  We can specify the
null space (indicating that we only really care about the constant
function) as:

.. code-block:: python3

   V = ...
   Q = ...
   W = V*Q
   v_basis = VectorSpaceBasis(constant=True)
   nullspace = MixedVectorSpaceBasis(W, [v_basis, W.sub(1)])

Debugging convergence failures
------------------------------

Occasionally, we will set up a problem and call solve only to be
confronted with an error that the solve failed to converge.  Here, we
discuss some useful techniques to try and understand the reason.  Much
of the advice in the `PETSc FAQ`_ is useful here, especially the
sections on `SNES nonconvergence`_ and `KSP nonconvergence`_.  We
first consider linear problems.

Linear convergence failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the linear operator is correct, but the solve fails to converge, it
is likely the case that the problem is badly conditioned (leading to
slow convergence) or a symmetric method is being used (such as
conjugate gradient) where the problem is non-symmetric.  The first
thing to check is what happened to the residual (error) term.  To
monitor this in the solution we pass the "flag" options
``'ksp_converged_reason'`` and ``'ksp_monitor_true_residual'``,
additionally, we pass ``ksp_view`` so that PETSc prints its idea of
what the solver object contains (this is useful to debug the where
options are not being passed in correctly):

.. code-block:: python3

   solver_parameters={'ksp_converged_reason': None,
                      'ksp_monitor_true_residual': None,
                      'ksp_view': None}

If the problem is converging, but only slowly, it may be that it is
badly conditioned.  If the problem is small, we can try using a direct
solve to see if the solution obtained is correct:

.. code-block:: python3

   solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'}

If this approach fails with a "zero-pivot" error, it is likely that
the equations are singular, or nearly so, check to see if boundary
conditions have been imposed correctly.

If the problem converges with a direct method to the correct solution
but does not converge with a Krylov method, it's probable that the
conditioning is bad.  If it's a mixed problem, try using a
physics-based preconditioner as described above, if not maybe try
using an algebraic multigrid preconditioner.  If PETSc was installed
with Hypre use:

.. code-block:: python3

   solver_parameters={'pc_type': 'hypre', 'pc_hypre_type': 'boomeramg'}

If you're using a symmetric method, such as conjugate gradient, check
that the linear operator is actually symmetric, which you can compute
with the following:

.. code-block:: python3

   A = assemble(a)  # use bcs keyword if there are boundary conditions
   print A.M.handle.isSymmetric(tol=1e-13)

If the problem is not symmetric, try using a method such as GMRES
instead.  PETSc uses restarted GMRES with a default restart of 30, for
difficult problems this might be too low, in which case, you can
increase the restart length with:

.. code-block:: python3

   solver_parameters={'ksp_gmres_restart': 100}


Nonlinear convergence failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Much of the advice for linear systems applies to nonlinear systems as
well.  If you have a convergence failure for a nonlinear problem, the
first thing to do is run with monitors to see what is going on, and
view the SNES object with ``snes_view`` to ensure that PETSc is seeing
the correct options:

.. code-block:: python3

   solver_parameters={'snes_monitor': None,
                      'snes_view': None,
                      'ksp_monitor_true_residual': None,
                      'snes_converged_reason': None,
                      'ksp_converged_reason': None}

If the linear solve fails to converge, debug the problem as above for
linear systems.  If the linear solve converges but the outer Newton
iterations do not, the problem is likely a bad Jacobian.  If you
provided the Jacobian by hand, is it correct?  If no Jacobian was
provided in the solve call, it is likely a bug in Firedrake and you
should `report it to us <firedrake_bugs_>`_.

Checking the provided Jacobian
++++++++++++++++++++++++++++++

It is possible to verify that the provided Jacobian is consistent with
the residual we are trying to minimise by comparing it with a finite
differenced Jacobian computed by PETSc.  This is possible using only a
few extra options to the call to :func:`~.solve`.  We just need to
specify that the nonlinear solver we want PETSc to employ should be of
type ``test``.  PETSc will then go away, compute an approximate
Jacobian by finite differencing the residual and compare it to our
provided exact Jacobian.  The only thing we need to be aware of is
that if the problem to be solved is in a mixed space, we need to set
the solver parameter ``"mat_type"`` to ``"aij"`` in the solve call.

To make things concrete, consider the following, somewhat contrived,
example where we attempt to solve a Galerkin projection in a mixed
space, but provide an incorrectly scaled Jacobian to the solve.

.. code-block:: python3

   from firedrake import *
   mesh = UnitSquareMesh(1, 1)
   V = FunctionSpace(mesh, "CG", 1)
   W = V*V
   f = Function(W)
   v = TestFunction(W)
   u = TrialFunction(W)

   F = dot(f, v)*dx - dot(Constant((1, 2)), v)*dx

   J = Constant(4)*dot(u, v)*dx

   solve(F == 0, f, J=J)

When run, this produces the following output:

.. code-block:: python3

   pyop2:INFO Solving nonlinear variational problem...
   Traceback (most recent call last):
       solve(F == 0, u, J=J)
     File "firedrake/solving.py", line 120, in solve
       _solve_varproblem(*args, **kwargs)
     File "firedrake/solving.py", line 162, in _solve_varproblem
       solver.solve()
     File "<string>", line 2, in solve
     File "pyop2/profiling.py", line 203, in wrapper
       return f(*args, **kwargs)
     File "firedrake/variational_solver.py", line 175, in solve
       solving_utils.check_snes_convergence(self.snes)
     File "firedrake/solving_utils.py", line 62, in check_snes_convergence
       """%s""" % (snes.getIterationNumber(), msg))
   RuntimeError: Nonlinear solve failed to converge after 50 nonlinear iterations.
   Reason:
       DIVERGED_MAX_IT

In this example we can notice by inspection of the code that the
provided Jacobian is incorrect.  The Gateaux derivative of :math:`F`
with respect to :math:`f` is :math:`\langle u, v \rangle`, not
:math:`4\langle u, v \rangle`.  In the more general case, it may be
that there is a bug in the assembly of the Jacobian, even if the
symbolic form is correct.  To verify the Jacobian we rerun the solve,
but pass some additional options:

.. code-block:: python3

   solve(F == 0, f, J=J,
         solver_parameters={'snes_type': 'test',
                            'mat_type': 'aij'})

This time we get the following output

.. code-block:: python3

   pyop2:INFO Solving nonlinear variational problem...
   Testing hand-coded Jacobian, if the ratio is
   O(1.e-8), the hand-coded Jacobian is probably correct.
   Run with -snes_test_display to show difference
   of hand-coded and finite difference Jacobian.
   Norm of matrix ratio 0.75, difference 1.32288 (user-defined state)
   Norm of matrix ratio 0.75, difference 1.32288 (constant state -1.0)
   Norm of matrix ratio 0.75, difference 1.32288 (constant state 1.0)
   Traceback (most recent call last):
       solve(F == 0, u, J=J, solver_parameters={'snes_type': 'test', 'mat_type': 'aij'})
     File "firedrake/solving.py", line 120, in solve
       _solve_varproblem(*args, **kwargs)
     File "firedrake/solving.py", line 162, in _solve_varproblem
       solver.solve()
     File "<string>", line 2, in solve
     File "pyop2/profiling.py", line 203, in wrapper
       return f(*args, **kwargs)
     File "firedrake/variational_solver.py", line 173, in solve
       self.snes.solve(None, v)
     File "PETSc/SNES.pyx", line 520, in petsc4py.PETSc.SNES.solve (src/petsc4py.PETSc.c:165224)
   petsc4py.PETSc.Error: error code 73
   [0] SNESSolve() line 3907 in petsc/src/snes/interface/snes.c
   [0] SNESSolve_Test() line 127 in petsc/src/snes/impls/test/snestest.c
   [0] Object is in wrong state
   [0] SNESTest aborts after Jacobian test: it is NORMAL behavior.

The important lines are:

.. code-block:: python3

   Testing hand-coded Jacobian, if the ratio is
   O(1.e-8), the hand-coded Jacobian is probably correct.
   Run with -snes_test_display to show difference
   of hand-coded and finite difference Jacobian.
   Norm of matrix ratio 0.75, difference 1.32288 (user-defined state)
   Norm of matrix ratio 0.75, difference 1.32288 (constant state -1.0)
   Norm of matrix ratio 0.75, difference 1.32288 (constant state 1.0)

Here PETSc is printing information about the difference between the
finite difference and provided Jacobians.  We can see that these
differences are large.  Therefore, we conclude the the provided
"exact" Jacobian is not consistent with the residual, and likely
incorrect.

For comparison, here are the same relevant lines when running with the
correct Jacobian:

.. code-block:: python3

   solve(F == 0, f, solver_parameters={'snes_type': 'test', 'mat_type': 'aij'})

   Testing hand-coded Jacobian, if the ratio is
   O(1.e-8), the hand-coded Jacobian is probably correct.
   Run with -snes_test_display to show difference
   of hand-coded and finite difference Jacobian.
   Norm of matrix ratio 4.98807e-08, difference 2.19953e-08 (user-defined state)
   Norm of matrix ratio 2.91936e-08, difference 1.28732e-08 (constant state -1.0)
   Norm of matrix ratio 1.51242e-08, difference 6.66915e-09 (constant state 1.0)

Notice how now the differences are small (within expected error
tolerances) so we are happy that the Jacobian is correct.


.. _Hypre: https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods
.. _PETSc: https://petsc.org/release/
.. _PETSc manual: https://petsc.org/release/docs/manual/manual.pdf
.. _KSP: https://petsc.org/release/manualpages/KSP/
.. _SNES: https://petsc.org/release/manualpages/SNES/
.. _fieldsplit: https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/
.. _PETSc FAQ: https://petsc.org/release/faq
.. _SNES nonconvergence: https://petsc.org/release/faq/#why-is-newton-s-method-snes-not-converging-or-converges-slowly
.. _KSP nonconvergence: https://petsc.org/release/faq/#why-is-the-linear-solver-ksp-not-converging-or-converges-slowly
.. _LSC: https://petsc.org/release/manualpages/PC/PCLSC/
.. _UFL: https://fenics-ufl.readthedocs.io/en/latest/
.. _firedrake_bugs: mailto:firedrake@imperial.ac.uk
