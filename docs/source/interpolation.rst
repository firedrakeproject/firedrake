.. only:: html

  .. contents::

Interpolation
=============

Firedrake offers various ways to interpolate expressions onto fields
(:py:class:`~.Function`\s).  Interpolation is often used to set up
initial conditions and/or boundary conditions. The basic syntax for
interpolation is:

.. code-block:: python

   # create new function f on function space V
   f = interpolate(expression, V)

   # alternatively:
   f = Function(V).interpolate(expression)

   # setting the values of an existing function
   f.interpolate(expression)

.. warning::

   Interpolation currently only works if all nodes of the target
   finite element are point evaluation nodes.

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
* Conditional expressions using UFL :py:class:`~ufl.conditional`.
* Compound expressions involving any of the above.

Here is an example demonstrating some of these features:

.. code-block:: python

   # g is a vector-valued Function, e.g. on an H(div) function space
   f = interpolate(sqrt(3.2 * div(g)), V)


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

.. code-block:: python

   # First, grab the mesh.
   m = V.ufl_domain()

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


C string expressions
--------------------

.. warning::

   C string expressions were a FEniCS compatibility feature which has
   now been removed. Users should use UFL expressions instead. This
   section only remains to assist in the transition of existing code.

Here are a couple of old-style C string expressions, and their modern replacements.   

.. code-block:: python

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

   This is a deprecated feature, but it remains supported for
   compatibility with FEniCS.

One can subclass :py:class:`~.Expression` and define a Python method
``eval`` on the subclass.  An example usage:

.. code-block:: python

   class MyExpression(Expression):
       def eval(self, value, x):
           value[:] = numpy.dot(x, x)

       def value_shape(self):
           return ()

   f.interpolate(MyExpression())

Here the arguments ``value`` and ``x`` of ``eval`` are `numpy` arrays.
``x`` contains the physical coordinates, and the result of the
expression must be written into ``value``.  One *must not reassign*
the local variable ``value``, but *overwrite* its content.

Since Python :py:class:`~.Expression` classes expressions are
deprecated, below are a few examples on how to replace them with UFL
expressions:

.. code-block:: python

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


.. _math.h: http://en.cppreference.com/w/c/numeric/math
.. _UFL: http://fenics-ufl.readthedocs.io/en/latest/
.. _TSFC: https://github.com/firedrakeproject/tsfc
