.. only:: html

  .. contents::

.. _firedrake_interpolation:

Interpolation
=============

Firedrake offers highly flexible capabilities for interpolating expressions
(functions of space) into finite element :py:class:`~.Function`\s.
Interpolation is often used to set up initial conditions and/or boundary
conditions. Mathematically, if :math:`e(x)` is a function of space and
:math:`V` is a finite element function space then
:math:`\operatorname{interpolate}(e, V)` is the :py:class:`~.Function`
:math:`v_i \phi_i\in V` such that:

.. math::

   v_i = \bar{\phi}^*_i(e)

where :math:`\bar{\phi}^*_i` is the :math:`i`-th dual basis function to
:math:`V` suitably extended such that its domain encompasses :math:`e`.

.. note::

   The extension of dual basis functions to :math:`e` usually follows from the
   definition of the dual basis. For example, point evaluation and integral
   nodes can naturally be extended to any expression which is evaluatable at
   the relevant points, or integrable over that domain.

   Firedrake will not impose any constraints on the expression to be
   interpolated beyond that its value shape matches that of the space into
   which it is interpolated. If the user interpolates an expression for which
   the nodes are not well defined (for example point evaluation at a
   discontinuity), the result is implementation-dependent.

The interpolate operator
------------------------

The basic syntax for interpolation is:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 1]
   :end-before: [test_interpolate_operator 2]

Here, the :py:func:`~.interpolate` function returned a **symbolic** UFL_ :py:class:`~ufl.Interpolate`
expression. To calculate a concrete numerical result, we need to call :py:func:`~.assemble` on this expression.

It is also possible to interpolate an expression directly into an existing
:py:class:`~.Function`:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 3]
   :end-before: [test_interpolate_operator 4]

This is a numerical operation, equivalent to:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 5]
   :end-before: [test_interpolate_operator 6]


The source expression can be any UFL_ expression with the correct shape.
UFL produces clear error messages in case of syntax or type errors, yet
UFL expressions have good run-time performance, since they are
translated to C interpolation kernels using TSFC_ technology.
Moreover, UFL offers a rich language for describing expressions,
including:

* The coordinates: in physical space as
  :py:class:`~ufl.SpatialCoordinate`, and in reference space as
  :py:class:`ufl.geometry.CellCoordinate`.
* Firedrake :py:class:`~.Function`\s, derivatives of
  :py:class:`~.Function`\s, and :py:class:`~.Constant`\s.
* Literal numbers, basic arithmetic operations, and also mathematical
  functions such as ``sin``, ``cos``, ``sqrt``, ``abs``, etc.
* Conditional expressions using UFL :py:mod:`~ufl.conditional`.
* Compound expressions involving any of the above.

Here is an example demonstrating some of these features:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 7]
   :end-before: [test_interpolate_operator 8]

This also works when interpolating into a space defined on the facets of the mesh:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 9]
   :end-before: [test_interpolate_operator 10]

.. note::

   Interpolation is supported into most, but not all, of the elements that
   Firedrake provides. In particular it is not currently possible to
   interpolate into spaces defined by higher-continuity elements such as
   Argyris and Hermite.


Semantics of symbolic interpolation
-----------------------------------

Let :math:`U` and :math:`V` be finite element spaces with DoFs :math:`\{\psi^{*}_{i}\}` and :math:`\{\phi^{*}_{i}\}`
and basis functions :math:`\{\psi_{i}\}` and :math:`\{\phi_{i}\}`, respectively.
The interpolation operator between :math:`U` and :math:`V` is defined

.. math::

   \mathcal{I}_{V} : U &\to V \\ \mathcal{I}_{V}(u)(x) &= \phi^{*}_{i}(u)\phi_{i}(x).

We define the following bilinear form

.. math::

   I : U \times V^{*} &\to \mathbb{R} \\ I(u, v^*) &= v^{*}(u)

where :math:`v^{*}\in V^{*}` is a linear functional in the dual space to :math:`V`, extended so that
it can act on functions in :math:`U`. If we choose :math:`v^{*} = \phi^{*}_{i}` then 
:math:`I(u, \phi^{*}_{i}) = \phi^{*}_{i}(u)` gives the coefficients of the interpolation of :math:`u` into :math:`V`.
This allows us to represent the interpolation as a form in UFL_. This is exactly the 
:py:class:`~ufl.Interpolate` UFL_ object. Note that this differs from typical bilinear forms since one of the
arguments is in a dual space. For more information on dual spaces in Firedrake, 
see :ref:`the relevant section of the manual <duals>`.

Interpolation operators
~~~~~~~~~~~~~~~~~~~~~~~

2-forms are assembled into matrices, and we can do the same with the interpolation form.
If we let :math:`u` be a ``TrialFunction(U)`` (i.e. an argument in slot 1) and :math:`v^*` be a
``TestFunction(V.dual())`` (i.e. a :py:class:`~ufl.Coargument` in slot 0) then

.. math::

   I(u, v^*) = I(\psi_{j},\phi_{i}^*)=\phi_{i}^*(\psi_{j})=:A_{ij}

The matrix :math:`A` is the interpolation matrix from :math:`U` to :math:`V`. In Firedrake, we can
assemble this matrix by doing

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 11]
   :end-before: [test_interpolate_operator 12]

Passing a :py:class:`~.FunctionSpace` into the dual slot of :py:func:`~.interpolate` is
syntactic sugar for ``TestFunction(V.dual())``.

If :math:`g\in U` is a :py:class:`~.Function`, then we can write it as :math:`g = g_j \psi_j` for
some coefficients :math:`g_j`. Interpolating :math:`g` into :math:`V` gives

.. math::

   I(g, v^*) = \phi^{*}_{i}(g_j \psi_j)= A_{ij} g_j,

so we can multiply the vector of coefficients of :math:`g` by the interpolation matrix to obtain the
coefficients of the interpolated function. In Firedrake, we can do this by

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 12]
   :end-before: [test_interpolate_operator 13]

:math:`h` is a :py:class:`~.Function` in :math:`V` representing the interpolation of :math:`g` into :math:`V`.

.. note::

   When interpolating a :py:class:`~.Function` directly, for example

   .. code-block:: python3

      assemble(interpolate(Function(U), V))

   Firedrake does not explicitly assemble the interpolation matrix. Instead, the interpolation
   is performed matrix-free.

Adjoint interpolation
~~~~~~~~~~~~~~~~~~~~~
The adjoint of the interpolation operator is defined as

.. math::

   \mathcal{I}_{V}^{*} : V^{*} \to U^{*}.

This operator interpolates :py:class:`~.Cofunction`\s in the dual space :math:`V^{*}` into
the dual space :math:`U^{*}`. The associated form is

.. math::

   I^{*} : V^{*} \times U \to \mathbb{R}.

So to obtain the adjoint interpolation operator, we swap the arguments of the :py:class:`~ufl.Interpolate` 
form. In Firedrake, we can accomplish this in two ways. The first is to swap the argument numbers to the form:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 14]
   :end-before: [test_interpolate_operator 15]

The second way is to use UFL_'s :py:func:`~ufl.adjoint` operator, which takes a form and returns its adjoint:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 15]
   :end-before: [test_interpolate_operator 16]

If :math:`g^*` is a :py:class:`~.Cofunction` in :math:`V^{*}` then we can interpolate it into :math:`U^{*}` by doing

.. math::

   I^{*}(g^*, u) = g^*_i \phi_i^*(\psi_j) = g^*_i A_{ij}.

This is the product of the adjoint interpolation matrix :math:`A^{*}` and the coefficients of :math:`g^*`. 
In Firedrake, we can do this by

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 16]
   :end-before: [test_interpolate_operator 17]

Again, Firedrake does not explicitly assemble the adjoint interpolation matrix, but performs the
interpolation matrix-free. To perform the interpolation with the assembled adjoint interpolation operator,
we can take the :py:func:`~ufl.action` of the operator on the :py:class:`~.Cofunction`:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 17]
   :end-before: [test_interpolate_operator 18]

The final case is when we interpolate a :py:class:`~.Function` into :py:class:`~.Cofunction`:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 19]
   :end-before: [test_interpolate_operator 20]

This interpolation has zero arguments and hence is assembled into a number. Mathematically, we have

.. math::

   I^{*}(g^*, u) = g^*_i \phi_i^*(u_{j}\psi_j) = g^*_i A_{ij} u_j.

which indeed contracts into a number.

Interpolation across meshes
---------------------------

The interpolation API supports interpolation across meshes where the target
function space has any finite element which supports interpolation, as specified in the list of
:ref:`supported elements <supported_elements>`. Vector, tensor, and mixed function
spaces can also be interpolated into from other meshes as long as they are
constructed from these spaces.

.. note::

   The list of supported elements above is only for *target* function spaces.
   Function spaces on the *source* mesh can be built from most of the supported
   elements.

There are few constraints on the meshes involved: the target mesh can have a
different cell shape, topological dimension, or resolution to the source mesh.
There are many use cases for this: For example, two solutions to the same
problem calculated on meshes with different resolutions or cell shapes can be
interpolated onto one another, or onto a third, finer mesh, and be directly
compared.


Interpolating onto sub-domain meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The target mesh for a cross-mesh interpolation need not cover the full domain
of the source mesh. Volume, surface and line integrals can therefore be
calculated by interpolating onto the mesh or
:ref:`immersed manifold <immersed_manifolds>` which defines the volume,
surface or line of interest in the domain. The integral itself is calculated
by calling :py:func:`~.assemble` on an appropriate form over the target mesh
function space:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_line_integral 1]
   :end-before: [test_line_integral 2]

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_line_integral 3]
   :end-before: [test_line_integral 4]

For more on forms, see :ref:`this section of the manual <more_complicated_forms>`.


Interpolating onto other meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the target mesh extends outside the source mesh domain, then cross-mesh
interpolation will raise a :py:class:`~.DofNotDefinedError`.

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 1]
   :end-before: [test_cross_mesh 2]

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 3]
   :end-before: [test_cross_mesh 4]

This can be overridden with the optional ``allow_missing_dofs`` keyword
argument:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 5]
   :end-before: [test_cross_mesh 6]

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 7]
   :end-before: [test_cross_mesh 8]

In this case, the missing degrees of freedom (DoFs, the global basis function
coefficients which could not be set) are, by default, set to zero:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 9]
   :end-before: [test_cross_mesh 10]

If we specify an output :py:class:`~.Function` then the missing DoFs are
unmodified.

We can optionally specify a value to use for our missing DoFs. Here
we set them to be ``nan`` ('not a number') for easy identification:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 11]
   :end-before: [test_cross_mesh 12]

If we specify an output :py:class:`~.Function`, this overwrites the missing
DoFs.

If we don't set ``default_missing_val`` then any missing DoFs are left as 
they were prior to interpolation:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 13]
   :end-before: [test_cross_mesh 14]

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 15]
   :end-before: [test_cross_mesh 16]

Similarly, using the :py:meth:`~.Function.interpolate` method on a :py:class:`~.Function` will not overwrite
the pre-existing values if ``default_missing_val`` is not set:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 17]
   :end-before: [test_cross_mesh 18]

.. _external_interpolation:

Interpolation from external data
--------------------------------

Unfortunately, UFL interpolation is not applicable if some of the
source data is not yet available as a Firedrake :py:class:`~.Function`
or UFL expression.  Here we describe a recipe for moving external data to
Firedrake fields.

Let us assume that there is some function ``mydata(X)`` which takes as
input an :math:`n \times d` array, where :math:`n` is the number of
points at which the data values are needed, and :math:`d` is the
geometric dimension of the mesh.  ``mydata(X)`` shall return a
:math:`n` long vector of the scalar values evaluated at the points
provided.  (Assuming that the target :py:class:`~.FunctionSpace` is
scalar valued, although this recipe can be extended to vector or
tensor valued fields.)  Presumably ``mydata`` works by interpolating
the external data source, but the precise details are not relevant
now.  In this case, interpolation into a target function space ``V``
proceeds as follows:


.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_external 1]
   :end-before: [test_interpolate_external 2]

This will also work in parallel, as the interpolation will occur on
each process, and Firedrake will take care of the halo updates before
the next operation using ``f``.

For interaction with external point data, see the
:ref:`corresponding manual section <external-point-data>`.

Interpolation between mixed function spaces
-------------------------------------------

Assembly of interpolation operators between mixed function spaces is also supported.
Each component of the mixed space may be on different meshes.
For example, consider the following mixed finite element spaces:

.. math::

   W &= V_1 \times V_2 \\
   U &= V_3 \times V_4

where each :math:`V_i` is a finite element space defined on possibly different meshes.
We can assemble the interpolation matrix from :math:`U` to :math:`W` in Firedrake as follows:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_mixed_space_interpolation 1]
   :end-before: [test_mixed_space_interpolation 2]

We specified ``mat_type="nest"`` here to obtain a PETSc MatNest matrix, but Firedrake also
supports assembly of ``mat_type="aij"`` and ``mat_type="matfree"`` interpolation matrices
between mixed function spaces. In this example ``I`` is a block diagonal matrix, with
each block given by

.. math::

   \begin{pmatrix}
   V_3 \rightarrow V_1 & 0 \\
   0 & V_4 \rightarrow V_2
   \end{pmatrix}

The off-diagonal blocks are zero since the dofs are applied component-wise. Firedrake's form
compiler recognises this and avoids assembling the zero blocks.

We can assemble more general interpolation matrices between mixed function spaces by interpolating
vector expressions with arguments. For example, by doing

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_mixed_space_interpolation 3]
   :end-before: [test_mixed_space_interpolation 4]

we can assemble the interpolation matrix with block structure

.. math::

   \begin{pmatrix}
   V_3 \rightarrow V_1 & V_4 \rightarrow V_1 \\
   V_3 \rightarrow V_2 & V_4 \rightarrow V_2
   \end{pmatrix}

Here we obtain non-zero off-diagonal blocks by including both components of the trial function
in each component of the expression.

Generating Functions with randomised values
-------------------------------------------

The :py:mod:`~.randomfunctiongen` module wraps  the external numpy package `numpy.random`_,
which gives Firedrake users an easy access to many stochastically sound random number generators,
including :py:class:`~numpy.random.PCG64`, :py:class:`~numpy.random.Philox`, and :py:class:`~numpy.random.SFC64`, which are parallel-safe.
All distribution methods defined in `numpy.random`_,
are made available, and one can pass a :class:`.FunctionSpace` to most of these methods
to generate a randomised :class:`.Function`.

.. code-block:: python3

    mesh = UnitSquareMesh(2,2)
    V = FunctionSpace(mesh, "CG", 1)
    # PCG64 random number generator
    pcg = PCG64(seed=123456789)
    rg = RandomGenerator(pcg)
    # beta distribution
    f_beta = rg.beta(V, 1.0, 2.0)

    print(f_beta.dat.data)

    # produces:
    # [0.56462514 0.11585311 0.01247943 0.398984 0.19097059 0.5446709 0.1078666 0.2178807 0.64848515]


.. _math.h: http://en.cppreference.com/w/c/numeric/math
.. _UFL: http://fenics-ufl.readthedocs.io/en/latest/
.. _TSFC: https://github.com/firedrakeproject/tsfc
.. _numpy.random: https://numpy.org/doc/stable/reference/random/index.html
