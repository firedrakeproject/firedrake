.. only:: html

  .. contents::

Changing mesh coordinates
=========================

Users may want to change the coordinates of an existing mesh object
for certain reasons. The coordinates can be accessed as a
:py:class:`~.Function` through ``mesh.coordinates`` where ``mesh`` is
a mesh object. For example,

.. code-block:: python

   mesh.coordinates.dat.data[:, 1] *= 2.0

streches the mesh in the *y*-direction. Another possibility is to use
:func:`~.Function.assign`:

.. code-block:: python

   Vc = mesh.coordinates.function_space()
   f = Function(Vc).interpolate(Expression(("x[0]", "x[1]*2.0")))
   mesh.coordinates.assign(f)

This can also be used if `f` is a solution to a PDE.

.. note::

   Unfortunately, the following is currently broken:

   .. code-block:: python

      mesh.coordinates.interpolate(Expression(("x[0]", "x[1]*2.0")))


Changing the coordinate function space
--------------------------------------

For more complicated situations, one might wish to replace the mesh
coordinates with a field which lives on a different
:py:class:`~.FunctionSpace` (e.g. higher-order meshes).

.. note::

   Re-assigning the ``coordinates`` property of a mesh used to be an
   undocumented feature. However, this no longer works:

   .. code-block:: python

      mesh.coordinates = f  # Raises an exception

Instead of re-assigning the coordinates of a mesh, one can create new
mesh object from a field `f`:

.. code-block:: python

   new_mesh = Mesh(f)

``new_mesh`` has the same mesh topology as the original mesh, but its
coordinate values and coordinate function space are from `f`. The
coordinate function space must be a :py:class:`~.VectorFunctionSpace`.
For efficiency, the new mesh object shares data with `f`. That is,
changing the values of `f` will change the coordinate values of the
mesh, and *vice versa*.  If this behaviour is undesired, one should
explicitly copy:

.. code-block:: python

   g = Function(f)  # creates a copy of f
   new_mesh = Mesh(g)

Or simply:

.. code-block:: python

   new_mesh = Mesh(Function(f))


Replacing the mesh geometry of an existing function
---------------------------------------------------

Creating a new mesh geometry object, as described above, leaves any
existing :py:class:`~.Function`\s untouched -- they continue to live
on their original mesh geometries.  One may wish to move these
functions over to the new mesh.  To move `f` over to ``mesh``, use:

.. code-block:: python

   g = Function(functionspace.WithGeometry(f.function_space(), mesh), val=f.topological)

This creates a :py:class:`~.Function` `g` which shares data with `f`,
but its mesh geometry is ``mesh``.

.. warning::

   The example above uses Firedrake internal APIs, which might change in the future.
