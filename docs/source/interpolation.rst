.. only:: html

  .. contents::

Interpolation
=============

Firedrake offers highly flexible capabilities for interpolating expressions
(functions of space) into finite element :py:class:`~.Function`\s.
Interpolation is often used to set up initial conditions and/or boundary
conditions. Mathematically, if :math:`e(x)` is a function of space and
:math:`V` is a finite element functionspace then
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

.. note::
   The semantics for interpolation in Firedrake are in the course of changing.
   The documentation provided here is for the new behaviour, in which the
   `interpolate` operator is symbolic. In order to access the behaviour
   documented here (which is recommended), users need to use the following
   import line:

   .. code-block:: python3

      from firedrake.__future__ import interpolate


The basic syntax for interpolation is:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_interpolate_operator 1]
   :end-before: [test_interpolate_operator 2]

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

This also works as expected when interpolating into a a space defined on the facets
of the mesh:

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

Interpolation across meshes
---------------------------

The interpolation API supports interpolation between meshes where the target
function space has finite elements (as given in the list of
:ref:`supported elements <supported_elements>`)

* **Lagrange/CG** (also known a Continuous Galerkin or P elements),
* **Q** (i.e. Lagrange/CG on lines, quadrilaterals and hexahedra),
* **Discontinuous Lagrange/DG** (also known as Discontinuous Galerkin or DP elements) and
* **DQ** (i.e. Discontinuous Lagrange/DG on lines, quadrilaterals and hexahedra).

Vector, tensor and mixed function spaces can also be interpolated into from
other meshes as long as they are constructed from these spaces.

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
by calling :py:func:`~.assemble` on an approriate form over the target mesh
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

.. note::

   Interpolation *from* :ref:`high-order meshes <changing_coordinate_fs>` is
   currently not supported.

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

This can be overriden with the optional ``allow_missing_dofs`` keyword
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

When using :py:class:`~.Interpolator`\s, the ``allow_missing_dofs`` keyword
argument is set at construction:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 13]
   :end-before: [test_cross_mesh 14]

The ``default_missing_val`` keyword argument is then set whenever we call
:py:meth:`~.Interpolator.interpolate`:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 15]
   :end-before: [test_cross_mesh 16]

If we supply an output :py:class:`~.Function` and don't set
``default_missing_val`` then any missing DoFs are left as they were prior to
interpolation:

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 17]
   :end-before: [test_cross_mesh 18]

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 19]
   :end-before: [test_cross_mesh 20]

.. literalinclude:: ../../tests/firedrake/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 21]
   :end-before: [test_cross_mesh 22]


Interpolation from external data
--------------------------------

Unfortunately, UFL interpolation is not applicable if some of the
source data is not yet available as a Firedrake :py:class:`~.Function`
or UFL expression.  Here we describe a recipe for moving external to
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
