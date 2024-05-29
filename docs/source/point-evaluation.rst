.. only:: html

  .. contents::

Point evaluation
================

Firedrake can evaluate :py:class:`~.Function`\s at arbitrary physical
points.  This feature can be useful for the evaluation of the result
of a simulation, or for creating expressions which contain point evaluations.
Three APIs are offered to this feature: two Firedrake-specific ones, and one
from UFL.


Firedrake convenience function
------------------------------

Firedrake's first API for evaluating functions at arbitrary points,
:meth:`~.Function.at`, is designed for simple interrogation of a function with
a few points.

.. code-block:: python3

   # evaluate f at a 1-dimensional point
   f.at(0.3)

   # evaluate f at two 1-dimensional points, or at one 2-dimensional point
   # (depending on f's geometric dimension)
   f.at(0.2, 0.4)

   # evaluate f at one 2-dimensional point
   f.at([0.2, 0.4])

   # evaluate f at two 2-dimensional point
   f.at([0.2, 0.4], [1.2, 0.5])

   # evaluate f at two 2-dimensional point (same as above)
   f.at([[0.2, 0.4], [1.2, 0.5]])

While in these examples we have only shown lists, other *iterables*
such as tuples and ``numpy`` arrays are also accepted. The following
are equivalent:

.. code-block:: python3

   f.at(0.2, 0.4)
   f.at((0.2, 0.4))
   f.at([0.2, 0.4])
   f.at(numpy.array([0.2, 0.4]))

For a single point, the result is a ``numpy`` array, or a tuple of
``numpy`` arrays in case of *mixed* functions.  When evaluating
multiple points, the result is a list of values for each point.
To summarise:

* Single point, non-mixed: ``numpy`` array
* Single point, mixed: tuple of ``numpy`` arrays
* Multiple points, non-mixed: list of ``numpy`` arrays
* Multiple points, mixed: list of tuples of ``numpy`` arrays


Points outside the domain
~~~~~~~~~~~~~~~~~~~~~~~~~

When any point is outside the domain of the function,
:py:class:`.PointNotInDomainError` exception is raised. If
``dont_raise=True`` is passed to :meth:`~.Function.at`, the result is
``None`` for those points which fall outside the domain.

.. code-block:: python3

   mesh = UnitIntervalMesh(8)
   f = mesh.coordinates

   f.at(1.2)                   # raises exception
   f.at(1.2, dont_raise=True)  # returns None

   f.at(0.5, 1.2)                   # raises exception
   f.at(0.5, 1.2, dont_raise=True)  # returns [0.5, None]


Evaluation on a moving mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you move the mesh, by :doc:`changing the mesh coordinates
<mesh-coordinates>`, then the bounding box tree that Firedrake
maintains to ensure fast point evaluation must be rebuilt.  To do
this, after moving the mesh, call
:meth:`~.MeshGeometry.clear_spatial_index` on the mesh you have just
moved.

Evaluation with a distributed mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is limited support for :meth:`~.Function.at` when running Firedrake
in parallel. There is no special API, but there are some restrictions:

* Point evaluation is a *collective* operation.
* Each process must ask for the same list of points.
* Each process will get the same values.

If ``RuntimeError: Point evaluation gave different results across processes.``
is raised, try lowering the :ref:`mesh tolerance <tolerance>`.


.. _primary-api:

Primary API: Interpolation onto a vertex-only mesh
--------------------------------------------------

Firedrake's principal API for evaluating functions at arbitrary points,
interpolation onto a :func:`~.VertexOnlyMesh`, is designed for evaluating a
function at many points, or repeatedly, and for creating expressions which
contain point evaluations. It is parallel-safe. Whilst :meth:`~.Function.at`
produces a list of values, cross-mesh interpolation onto
:func:`~.VertexOnlyMesh` gives Firedrake :py:class:`~.Function`\s.

This is discussed in detail in :cite:`nixonhill2023consistent` but, briefly,
the idea is that the :func:`~.VertexOnlyMesh` is a mesh that represents a
point cloud domain. Each cell of the mesh is a vertex at a chosen location in
space. As usual for a mesh, we represent values by creating functions in
function spaces on it. The only function space that makes sense for a mesh
whose cells are vertices is the space of piecewise constant functions, also
known as the Polynomial degree 0 Discontinuous Galerkin (P0DG) space.

Our vertex-only meshes are immersed in some 'parent' mesh. We perform point
evaluation of a function :math:`f` defined in a function space
:math:`V` on the parent mesh by interpolating into the P0DG space on the
:func:`~.VertexOnlyMesh`. For example:

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_vertex_only_mesh_manual_example 1]
   :end-before: [test_vertex_only_mesh_manual_example 3]

will print ``[0.02, 0.08, 0.18]`` when running in serial, the values of
:math:`x^2 + y^2` at the points :math:`(0.1, 0.1)`, :math:`(0.2, 0.2)` and
:math:`(0.3, 0.3)`. For details on viewing the outputs in parallel, see the
:ref:`section on the input ordering property. <input_ordering>`

Note that ``f_at_points`` is a :py:class:`~.Function` which takes
on *all* the values of ``f`` evaluated at ``points``. The cell ordering of a
:func:`~.VertexOnlyMesh` follows the ordering of the list of points it is given
at construction. In general :func:`~.VertexOnlyMesh` accepts any numpy array of
shape ``(num_points, point_dim)`` (or equivalent list) as the set of points to
create disconnected vertices at.

The operator for evaluation at the points specified can be
created by making an :py:class:`~.Interpolator` acting on a
:py:func:`~.TestFunction`

.. code-block:: python3

   u = TestFunction(V)
   Interpolator(u, P0DG)

For more on :py:class:`~.Interpolator`\s and interpolation see the
:doc:`interpolation <interpolation>` section.


Vector and tensor valued function spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When interpolating from vector or tensor valued function spaces, the P0DG
function space on the vertex-only mesh must be a
:py:func:`~.VectorFunctionSpace` or :py:func:`~.TensorFunctionSpace`
respectively. For example:

.. code-block:: python3

   V = VectorFunctionSpace(parent_mesh, "CG", 2)

or

.. code-block:: python3

   V = FunctionSpace(parent_mesh, "N1curl", 2)

each require

.. code-block:: python3

   vom = VertexOnlyMesh(parent_mesh, points)
   P0DG_vec = VectorFunctionSpace(vom, "DG", 0)

for successful interpolation.


Parallel behaviour
~~~~~~~~~~~~~~~~~~

In parallel the ``points`` given to :func:`~.VertexOnlyMesh` are assumed to be
the same on each MPI process and are taken from rank 0. To let different ranks
provide different points to the vertex-only mesh set the keyword argument
``redundant = False``

.. code-block:: python3

   # Default behaviour
   vom = VertexOnlyMesh(parent_mesh, points, redundant = True)

   # Different points on each MPI rank to add to the vertex-only mesh
   vom = VertexOnlyMesh(parent_mesh, points, redundant = False)

In this case, ``points`` will redistribute to the mesh partition where they are
located. This means that if rank A has ``points`` :math:`\{X\}` that are not
found in the mesh cells owned by rank A but are found in the mesh cells owned
by rank B then they will be moved to rank B.

If the same coordinates are supplied more than once, they are always assumed to
be a new vertex: this is true for both ``redundant = True`` and
``redunant = False``. So if we have the same set of points on all MPI processes
and switch from ``redundant = True`` to ``redundant = False`` we will get point
duplication.


Points outside the domain
~~~~~~~~~~~~~~~~~~~~~~~~~

Be default points outside the domain by more than the :ref:`specified
tolerance <tolerance>` will generate
a :class:`~.VertexOnlyMeshMissingPointsError`. This can be switched to a
warning or switched off entirely:

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_vom_manual_points_outside_domain 1]
   :end-before: [test_vom_manual_points_outside_domain 2]

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_vom_manual_points_outside_domain 3]
   :end-before: [test_vom_manual_points_outside_domain 4]

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_vom_manual_points_outside_domain 5]
   :end-before: [test_vom_manual_points_outside_domain 6]


Expressions with point evaluations
----------------------------------

Integrating over a vertex-only mesh is equivalent to summing over
it. So if we have a vertex-only mesh :math:`\Omega_v` with :math:`N` vertices
at points :math:`\{x_i\}_{i=0}^{N-1}` and we have interpolated a function
:math:`f` onto it giving a new function :math:`f_v` then

.. math::

   \int_{\Omega_v} f_v \, dx = \sum_{i=0}^{N-1} f(x_i).

These equivalent expressions for point evaluation

.. math::

   \sum_{i=0}^{N-1} f(x_i) = \sum_{i=0}^{N-1} \int_\Omega f(x) \delta(x - x_i) \, dx

where :math:`N` is the number of points, :math:`x_i` is the :math:`i`\th point,
:math:`\Omega` is a 'parent' mesh, :math:`f` is a function on that mesh,
:math:`\delta` is a dirac delta distribition can therefore be written in
Firedrake using :func:`~.VertexOnlyMesh` and :func:`~.interpolate` as

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_vom_manual_keyword_arguments 1]
   :end-before: [test_vom_manual_keyword_arguments 2]

.. _external-point-data:

Interacting with external point data
------------------------------------

.. _input_ordering:

Using the input ordering property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any set of points with associated data in our domain can be expressed as a
P0DG function on a :func:`~.VertexOnlyMesh`. The recommended way to import data
from an external source is via the :py:attr:`~.VertexOnlyMeshTopology.input_ordering`
property: this produces another vertex-only mesh which has points in the order
and MPI rank that they were specified when first creating the original
vertex-only mesh. For example:

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_input_ordering_input 1]
   :end-before: [test_input_ordering_input 2]

This is entirely parallel safe.

Similarly, we can use :py:attr:`~.VertexOnlyMeshTopology.input_ordering` to get data out
of a vertex-only mesh in a parallel-safe way. If we return to our example from
:ref:`the section where we introduced vertex only meshes <primary-api>`, we
had

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_vertex_only_mesh_manual_example 2]
   :end-before: [test_vertex_only_mesh_manual_example 3]

In parallel, this will print the values of ``f`` at the given ``points`` list
**after the points have been distributed over the parent mesh**. If we want the
values of ``f`` at the ``points`` list **before the points have been
distributed** we can use :py:attr:`~.VertexOnlyMeshTopology.input_ordering` as follows:

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_vertex_only_mesh_manual_example 4]
   :end-before: [test_vertex_only_mesh_manual_example 5]

.. note::

   When a a vertex-only mesh is created with ``redundant = True`` (which is the
   default when creating a :func:`~.VertexOnlyMesh`) the
   :py:attr:`~.VertexOnlyMeshTopology.input_ordering` method will return a vertex-only
   mesh with all points on rank 0.

If we ran the example in parallel, the above code would print
``[0.02, 0.08, 0.18]`` on rank 0 and ``[]`` on all other ranks. If we set
``redundant = False`` when creating the vertex-only mesh, the above code would
print ``[0.02, 0.08, 0.18]`` on all ranks and we would have point duplication.

If any of the specified points were not found in the mesh, the value on the
input ordering vertex-only mesh will not be effected by the interpolation from
the original vertex-only mesh. In the above example, the values would be zero
at those points. To make it more obvious that those points were not found, it's
a good idea to set the values to ``nan`` before the interpolation:

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_vertex_only_mesh_manual_example 6]
   :end-before: [test_vertex_only_mesh_manual_example 7]


More ways to interact with external data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aside from :py:attr:`~.VertexOnlyMeshTopology.input_ordering`, we can use
:func:`~.interpolate` to interact with external data to, for example,
compare a PDE solution with the point data. The :math:`l_2` error norm
(euclidean norm) of a function :math:`f` (which may be a PDE solution)
evaluated against a set of point data :math:`\{y_i\}_{i=0}^{N-1}` at points
:math:`\{x_i\}_{i=0}^{N-1}` is defined as

.. math::

   \sqrt{ \sum_{i=0}^{N-1} (f(x_i) - y_i)^2 }.

We can express this in Firedrake as

.. code-block:: python3

   error = sqrt(assemble((interpolate(f, P0DG) - y_pts)**2*dx))

   # or equivalently
   error = errornorm(interpolate(f, P0DG), y_pts)

We can then use the :py:attr:`~.VertexOnlyMeshTopology.input_ordering` vertex-only mesh
to safely check the values of ``error`` at the points
:math:`\{x_i\}_{i=0}^{N-1}`.

.. _tolerance:

Mesh tolerance
--------------

If points are outside the mesh domain but ought to still be found a
``tolerance`` parameter can be set. The tolerance is relative to the size of
the mesh cells and is a property of the mesh itself

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_mesh_tolerance 1]
   :end-before: [test_mesh_tolerance 2]

Keyword arguments
~~~~~~~~~~~~~~~~~

Alternatively we can modify the tolerance by providing it as an argument to the
vertex-only mesh. This will modify the tolerance property of the parent mesh.

.. warning::

   To avoid confusion, it is recommended that the tolerance be set for a mesh
   before any point evaluations are performed, rather than making use of these
   keyword arguments.

.. literalinclude:: ../../tests/vertexonly/test_vertex_only_manual.py
   :language: python3
   :dedent:
   :start-after: [test_mesh_tolerance_change 1]
   :end-before: [test_mesh_tolerance_change 2]

Note that since our tolerance is relative, the number of cells in a mesh
dictates the point loss behaviour close to cell edges. So the mesh
``UnitSquareMesh(5, 5, quadrilateral = True)`` will include the point
:math:`(1.1, 1.0)` by default.

Changing mesh tolerance only affects point location after it has been changed.
To apply the new tolerance to a vertex-only mesh, a new vertex-only mesh must
be created. Any existing immersed vertex-only mesh will have been created
using the previous tolerance and will be unaffected by the change.


UFL API
-------

UFL reserves the function call operator for evaluation:

.. code-block:: python3

   f([0.2, 0.4])

will evaluate :math:`f` at :math:`(0.2, 0.4)`. UFL does not accept
multiple points at once, and cannot configure what to do with a point
which is not in the domain. The advantage of this syntax is that it
works on any :py:class:`~.ufl.core.expr.Expr`, for example:

.. code-block:: python3

   (f*sin(f)([0.2, 0.4])

will evaluate :math:`f \cdot \sin(f)` at :math:`(0.2, 0.4)`.

.. note::

   The expression itself is not translated into C code.  While the
   evaluation of a function uses the same infrastructure as the
   Firedrake APIs, which use generated C code, the expression tree is
   evaluated by UFL in Python.
