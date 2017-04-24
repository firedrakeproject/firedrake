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
Moreover, it offers a rich language for describing expressions,
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


C string expressions
--------------------

.. warning::

   This is a deprecated feature, but it remains supported for
   compatibility with FEniCS.

The :py:class:`~.Expression` class wraps a C string expression,
e.g. ``Expression("sin(x[0]*pi)")``, which is then copy-pasted into
the interpolation kernel.  Consequently, C syntax rules apply inside
the string, with the following "environment":

* The physical spatial coordinate is available as "vector" ``x``
  (array in C), i.e. the coordinates `x`, `y`, and `z` are accessed as
  ``x[0]``, ``x[1]``, and ``x[2]``.
* The mathematical constant :math:`{\pi}` is available as ``pi``.
* Mathematical functions from the C header `math.h`_.

It is possible to augment this environment.  For example,
``Expression('sin(x[0]*t)', t=t)`` takes the value from the Python
variable ``t``, and uses that value for ``t`` inside the C string.

Since C string expressions are deprecated, below are a few examples on
how to replace them with UFL expressions:

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
