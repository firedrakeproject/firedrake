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

Firedrake offers three ways to specify the source expression:


C string expressions
--------------------

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


Python expression classes
-------------------------

One can subclass :py:class:`~.Expression` and define a Python method
``eval`` on the subclass.  An example usage:

.. code-block:: python

   class MyExpression(Expression):
       def eval(self, value, x):
           value[:] = numpy.dot(x, x)

       def value_shape(self):
           return (1,)

   f.interpolate(MyExpression())

Here the arguments ``value`` and ``x`` of ``eval`` are `numpy` arrays.
``x`` contains the physical coordinates, and the result of the
expression must be written into ``value``.  One *must not reassign*
the local variable ``value``, but *overwrite* its content.


UFL expressions
---------------

Using UFL_ expressions is the most general way that Firedrake offers
to specify the source expression of an interpolation.  This option
allows the source expressions to contain:

* other :py:class:`~.Function`\s,
* derivatives of :py:class:`~.Function`\s,
* :py:class:`~.Constant`\s,
* compound expressions involving any of the above.

One can rewrite any of the above examples using UFL_:

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

   # Python expression:
   class MyExpression(Expression):
       def eval(self, value, x):
           value[:] = numpy.dot(x, x)

       def value_shape(self):
           return (1,)

   f.interpolate(MyExpression())

   # UFL equivalent:
   x = SpatialCoordinate(f.function_space().mesh())
   f.interpolate(dot(x, x))

As mentioned above, one can have :py:class:`~.Function`\s in UFL
expressions for interpolation.  Here is an example which has no
equivalent using C strings or Python expression classes:

.. code-block:: python

   # g is a vector-valued Function, e.g. on an H(div) function space
   f = interpolate(sqrt(3.2 * div(g)), V)


.. note::

   UFL expressions are type checked, and thus safer to use than C
   strings, which rely on string manipulation to assemble the
   interpolation kernel.

.. note::

   UFL expressions have good run-time performance (unlike Python
   expression classes), since they are translated to C interpolation
   kernels using TSFC_ technology.


.. _math.h: http://en.cppreference.com/w/c/numeric/math
.. _UFL: http://fenics-ufl.readthedocs.org/en/latest/
.. _TSFC: https://github.com/firedrakeproject/tsfc
