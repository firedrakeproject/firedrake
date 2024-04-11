.. only:: html

  .. contents::

Interpolation
=============

Firedrake offers various ways to interpolate expressions onto fields
(:py:class:`~.Function`\s).  Interpolation is often used to set up
initial conditions and/or boundary conditions. The basic syntax for
interpolation is:

.. code-block:: python3

   # create new function f on function space V
   f = interpolate(expression, V)

   # alternatively:
   f = Function(V).interpolate(expression)

   # setting the values of an existing function
   f.interpolate(expression)

.. note::

   Interpolation is supported for most, but not all, of the elements
   that Firedrake provides. In particular, higher-continuity elements
   such as Argyris and Hermite do not presently support interpolation.

The recommended way to specify the source expression is UFL.  UFL_
produces clear error messages in case of syntax or type errors, yet
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

.. code-block:: python3

   # g is a vector-valued Function, e.g. on an H(div) function space
   f = interpolate(sqrt(3.2 * div(g)), V)

This also works as expected when interpolating into a a space defined on the facets
of the mesh:

.. code-block:: python3

   # where trace is a trace space on the current mesh:
   f = interpolate(expression, trace)


Interpolator objects
--------------------

Firedrake is also able to generate reusable :py:class:`~.Interpolator`
objects which provide caching of the interpolation operation. The
following line creates an interpolator which will interpolate the
current value of `expression` into the space `V`:

.. code-block:: python3

   interpolator = Interpolator(expression, V)

If `expression` does not contain a :py:func:`~ufl.TestFunction` then
the interpolation can be performed with:

.. code-block:: python3

   f = interpolator.interpolate()

Alternatively, one can use the interpolator to set the value of an existing :py:class:`~.Function`:

.. code-block:: python3

   f = Function(V)
   interpolator.interpolate(output=f)

If `expression` contains a :py:func:`~ufl.TestFunction` then
the interpolator acts to interpolate :py:class:`~.Function`\s in the
test space to those in the target space. For example:

.. code-block:: python3

   w = TestFunction(W)
   interpolator = Interpolator(w, V)

Here, `interpolator` acts as the interpolation matrix from the
:py:func:`~.FunctionSpace` W into the
:py:func:`~.FunctionSpace` V. Such that if `f` is a
:py:class:`~.Function` in `W` then `g = interpolator.interpolate(f)` is its
interpolation into a function `g` in `V`. As before, the `output` parameter can
be used to write into an existing :py:class:`~.Function`. Passing the
`transpose=True` option to :py:meth:`~.Interpolator.interpolate` will
cause the transpose interpolation to occur. This is equivalent to the
multigrid restriction operation which interpolates assembled 1-forms
in the dual space to `V` to assembled 1-forms in the dual space to
`W`.


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

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_line_integral 1]
   :end-before: [test_line_integral 2]

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
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

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 1]
   :end-before: [test_cross_mesh 2]

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 3]
   :end-before: [test_cross_mesh 4]

This can be overriden with the optional ``allow_missing_dofs`` keyword
argument:

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 5]
   :end-before: [test_cross_mesh 6]

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 7]
   :end-before: [test_cross_mesh 8]

In this case, the missing degrees of freedom (DoFs, the global basis function
coefficients which could not be set) are, by default, set to zero:

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 9]
   :end-before: [test_cross_mesh 10]

If we specify an output :py:class:`~.Function` then the missing DoFs are
unmodified.

We can optionally specify a value to use for our missing DoFs. Here
we set them to be ``nan`` ('not a number') for easy identification:

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 11]
   :end-before: [test_cross_mesh 12]

If we specify an output :py:class:`~.Function`, this overwrites the missing
DoFs.

When using :py:class:`~.Interpolator`\s, the ``allow_missing_dofs`` keyword
argument is set at construction:

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 13]
   :end-before: [test_cross_mesh 14]

The ``default_missing_val`` keyword argument is then set whenever we call
:py:meth:`~.Interpolator.interpolate`:

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 15]
   :end-before: [test_cross_mesh 16]

If we supply an output :py:class:`~.Function` and don't set
``default_missing_val`` then any missing DoFs are left as they were prior to
interpolation:

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 17]
   :end-before: [test_cross_mesh 18]

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
   :language: python3
   :dedent:
   :start-after: [test_cross_mesh 19]
   :end-before: [test_cross_mesh 20]

.. literalinclude:: ../../tests/regression/test_interpolation_manual.py
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

.. code-block:: python3

   # First, grab the mesh.
   m = V.mesh()

   # Now make the VectorFunctionSpace corresponding to V.
   W = VectorFunctionSpace(m, V.ufl_element())

   # Next, interpolate the coordinates onto the nodes of W.
   X = interpolate(m.coordinates, W)

   # Make an output function.
   f = Function(V)

   # Use the external data function to interpolate the values of f.
   f.dat.data[:] = mydata(X.dat.data_ro)

This will also work in parallel, as the interpolation will occur on
each process, and Firedrake will take care of the halo updates before
the next operation using ``f``.

For interaction with external point data, see the
:ref:`corresponding manual section <external-point-data>`.


C string expressions
--------------------

.. warning::

   C string expressions were a FEniCS compatibility feature which has
   now been removed. Users should use UFL expressions instead. This
   section only remains to assist in the transition of existing code.

Here are a couple of old-style C string expressions, and their modern replacements.

.. code-block:: python3

   # Expression:
   f = interpolate(Expression("sin(x[0]*pi)"), V)

   # UFL equivalent:
   x = SpatialCoordinate(V.mesh())
   f = interpolate(sin(x[0] * math.pi), V)

   # Expression with a Constant parameter:
   f = interpolate(Expression('sin(x[0]*t)', t=t), V)

   # UFL equivalent:
   x = SpatialCoordinate(V.mesh())
   f = interpolate(sin(x[0] * t), V)


Python expression classes
-------------------------

.. warning::

   Python expression classes were a FEniCS compatibility feature which has
   now been removed. Users should use UFL expressions instead. This
   section only remains to assist in the transition of existing code.

Since Python ``Expression`` classes expressions are
deprecated, below are a few examples on how to replace them with UFL
expressions:

.. code-block:: python3

   # Python expression:
   class MyExpression(Expression):
       def eval(self, value, x):
           value[:] = numpy.dot(x, x)

       def value_shape(self):
           return ()

   f.interpolate(MyExpression())

   # UFL equivalent:
   x = SpatialCoordinate(f.function_space().mesh())
   f.interpolate(dot(x, x))


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
