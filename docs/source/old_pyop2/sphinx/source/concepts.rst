.. _concepts:

PyOP2 Concepts
==============

Many numerical algorithms and scientific computations on unstructured meshes
can be viewed as the *independent application* of a *local operation*
everywhere on a mesh. This local operation is often called a computational
*kernel* and its independent application lends itself naturally to parallel
computation. An unstructured mesh can be described by *sets of entities*
(vertices, edges, cells) and the connectivity between those sets forming the
topology of the mesh.

PyOP2 is a domain-specific language (DSL) for the parallel executions of
computational kernels on unstructured meshes or graphs.

.. _sets:

Sets and mappings
-----------------

A mesh is defined by :class:`sets <pyop2.Set>` of entities and
:class:`mappings <pyop2.Map>` between these sets. Sets are used to represent
entities in the mesh (nodes in the graph) or degrees of freedom of data
(fields) living "on" the mesh (graph), while maps define the connectivity
between entities (links in the graph) or degrees of freedom, for example
associating an edge with its incident vertices. Sets of mesh entities may
coincide with sets of degrees of freedom, but this is not necessarily the case
e.g. the set of degrees of freedom for a field may be defined on the vertices
of the mesh and the midpoints of edges connecting the vertices.

.. note ::
  There is a requirement for the map to be of *constant arity*, that is each
  element in the source set must be associated with a constant number of
  elements in the target set. There is no requirement for the map to be
  injective or surjective. This restriction excludes certain kinds of mappings
  e.g. a map from vertices to incident egdes or cells is only possible on a
  very regular mesh where the multiplicity of any vertex is constant.

In the following we declare a :class:`~pyop2.Set` ``vertices``, a
:class:`~pyop2.Set` ``edges`` and a :class:`~pyop2.Map` ``edges2vertices``
between them, which associates the two incident vertices with each edge: ::

    vertices = op2.Set(4)
    edges = op2.Set(3)
    edges2vertices = op2.Map(edges, vertices, 2, [[0, 1], [1, 2], [2, 3]])

.. _data:

Data
----

PyOP2 distinguishes three kinds of user provided data: data that lives on a
set (often referred to as a field) is represented by a :class:`~pyop2.Dat`,
data that has no association with a set by a :class:`~pyop2.Global` and data
that is visible globally and referred to by a unique identifier is declared as
:class:`~pyop2.Const`. Examples of the use of these data types are given in
the :ref:`par_loops` section below.

.. _data_dat:

Dat
~~~

Since a set does not have any type but only a cardinality, data declared on a
set through a :class:`~pyop2.Dat` needs additional metadata to allow PyOP2 to
interpret the data and to specify how much memory is required to store it. This
metadata is the *datatype* and the *shape* of the data associated with any
given set element. The shape is not associated with the :class:`~pyop2.Dat`
directly, but with a :class:`~pyop2.DataSet`. One can associate a scalar with
each element of the set or a one- or higher-dimensional vector. Similar to the
restriction on maps, the shape and therefore the size of the data associated
which each element needs to be uniform. PyOP2 supports all common primitive
data types supported by `NumPy`_.  Custom datatypes are supported insofar as
the user implements the serialisation and deserialisation of that type into
primitive data that can be handled by PyOP2.

Declaring coordinate data on the ``vertices`` defined above, where two float
coordinates are associated with each vertex, is done like this: ::

    dvertices = op2.DataSet(vertices, dim=2)
    coordinates = op2.Dat(dvertices,
                          [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
                          dtype=float)

.. _data_global:

Global
~~~~~~

In contrast to a :class:`~pyop2.Dat`, a :class:`~pyop2.Global` has no
association to a set and the shape and type of the data are declared directly
on the :class:`~pyop2.Global`. A 2x2 elasticity tensor would be defined as
follows: ::

    elasticity = op2.Global((2, 2), [[1.0, 0.0], [0.0, 1.0]], dtype=float)

.. _data_const:

Const
~~~~~

Data that is globally visible and read-only to kernels is declared with a
:class:`~pyop2.Const` and needs to have a globally unique identifier.  It does
not need to be declared as an argument to a :func:`~pyop2.par_loop`, but is
accessible in a kernel by name. A globally visible parameter ``eps`` would be
declared as follows: ::

    eps = op2.Const(1, 1e-14, name="eps", dtype=float)

.. _data_mat:

Mat
~~~

In a PyOP2 context, a (sparse) matrix is a linear operator from one set to
another. In other words, it is a linear function which takes a
:class:`~pyop2.Dat` on one set :math:`A` and returns the value of a
:class:`~pyop2.Dat` on another set :math:`B`. Of course, in particular,
:math:`A` may be the same set as :math:`B`. This makes the operation of at
least some matrices equivalent to the operation of a particular PyOP2 kernel.

PyOP2 can be used to assemble :class:`matrices <pyop2.Mat>`, which are defined
on a :class:`sparsity pattern <pyop2.Sparsity>` which is built from a pair of
:class:`DataSets <pyop2.DataSet>` defining the row and column spaces the
sparsity maps between and one or more pairs of maps, one for the row and one
for the column space of the matrix respectively. The sparsity uniquely defines
the non-zero structure of the sparse matrix and can be constructed purely from
those mappings. To declare a :class:`~pyop2.Mat` on a :class:`~pyop2.Sparsity`
only the data type needs to be given.

Since the construction of large sparsity patterns is a very expensive
operation, the decoupling of :class:`~pyop2.Mat` and :class:`~pyop2.Sparsity`
allows the reuse of sparsity patterns for a number of matrices without
recomputation. In fact PyOP2 takes care of caching sparsity patterns on behalf
of the user, so declaring a sparsity on the same maps as a previously declared
sparsity yields the cached object instead of building another one.

Defining a matrix of floats on a sparsity which spans from the space of
vertices to the space of vertices via the edges is done as follows: ::

    sparsity = op2.Sparsity((dvertices, dvertices),
                            [(edges2vertices, edges2vertices)])
    matrix = op2.Mat(sparsity, float)

.. _par_loops:

Parallel loops
--------------

Computations in PyOP2 are executed as :func:`parallel loops <pyop2.par_loop>`
of a :class:`~pyop2.Kernel` over an *iteration set*. Parallel loops are the
core construct of PyOP2 and hide most of its complexity such as parallel
scheduling, partitioning, colouring, data transfer from and to device and
staging of the data into on chip memory. Computations in a parallel loop must
be independent of the order in which they are executed over the set to allow
PyOP2 maximum flexibility to schedule the computation in the most efficient
way. Kernels are described in more detail in :doc:`kernels`.

.. _loop-invocations:

Loop invocations
~~~~~~~~~~~~~~~~

A parallel loop invocation requires as arguments, other than the iteration set
and the kernel to operate on, the data the kernel reads and/or writes. A
parallel loop argument is constructed by calling the underlying data object
(i.e. the :class:`~pyop2.Dat` or :class:`~pyop2.Global`) and passing an
*access descriptor* and the mapping to be used when accessing the data. The
mapping is required for an *indirectly accessed* :class:`~pyop2.Dat` not
declared on the same set as the iteration set of the parallel loop. In the
case of *directly accessed* data defined on the same set as the iteration set
the map is omitted and only an access descriptor given.

Consider a parallel loop that translates the ``coordinate`` field by a
constant offset given by the :class:`~pyop2.Const` ``offset``. Note how the
kernel has access to the local variable ``offset`` even though it has not been
passed as an argument to the :func:`~pyop2.par_loop`. This loop is direct and
the argument ``coordinates`` is read and written: ::

    op2.Const(2, [1.0, 1.0], dtype=float, name="offset");

    translate = op2.Kernel("""void translate(double * coords) {
      coords[0] += offset[0];
      coords[1] += offset[1];
    }""", "translate")

    op2.par_loop(translate, vertices, coordinates(op2.RW))

.. _access-descriptors:

Access descriptors
~~~~~~~~~~~~~~~~~~

Access descriptors define how the data is accessed by the kernel and give
PyOP2 crucial information as to how the data needs to be treated during
staging in before and staging out after kernel execution. They must be one of
:data:`pyop2.READ` (read-only), :data:`pyop2.WRITE` (write-only),
:data:`pyop2.RW` (read-write), :data:`pyop2.INC` (increment),
:data:`pyop2.MIN` (minimum reduction) or :data:`pyop2.MAX` (maximum
reduction).

Not all of these descriptors apply to all PyOP2 data types. A
:class:`~pyop2.Dat` can have modes :data:`~pyop2.READ`, :data:`~pyop2.WRITE`,
:data:`~pyop2.RW` and :data:`~pyop2.INC`. For a :class:`~pyop2.Global` the
valid modes are :data:`~pyop2.READ`, :data:`~pyop2.INC`, :data:`~pyop2.MIN` and
:data:`~pyop2.MAX` and for a :class:`~pyop2.Mat` only :data:`~pyop2.WRITE` and
:data:`~pyop2.INC` are allowed.

.. _matrix-loops:

Loops assembling matrices
~~~~~~~~~~~~~~~~~~~~~~~~~

We declare a parallel loop assembling the ``matrix`` via a given ``kernel``
which we'll assume has been defined before over the ``edges`` and with
``coordinates`` as input data. The ``matrix`` is the output argument of this
parallel loop and therefore has the access descriptor :data:`~pyop2.INC` since
the assembly accumulates contributions from different vertices via the
``edges2vertices`` mapping. Note that the mappings are being indexed with the
:class:`iteration indices <pyop2.base.IterationIndex>` ``op2.i[0]`` and
``op2.i[1]`` respectively. This means that PyOP2 generates a :ref:`local
iteration space <local-iteration-spaces>` of size ``arity * arity`` with the
``arity`` of the :class:`~pyop2.Map` ``edges2vertices`` for any given element
of the iteration set.  This local iteration space is then iterated over using
the iteration indices on the maps.  The kernel is assumed to only apply to a
single point in that local iteration space. The ``coordinates`` are accessed
via the same mapping, but are a read-only input argument to the kernel and
therefore use the access descriptor :data:`~pyop2.READ`: ::

    op2.par_loop(kernel, edges,
                 matrix(op2.INC, (edges2vertices[op2.i[0]],
                                  edges2vertices[op2.i[1]])),
                 coordinates(op2.READ, edges2vertices))

You can stack up multiple successive parallel loops that add values to
a matrix, before you use the resulting values, you must explicitly
tell PyOP2 that you want to do so, by calling
:meth:`~pyop2.Mat.assemble` on the matrix.  Note that executing a
:func:`~pyop2.solve` will do this automatically for you.

.. _reduction-loops:

Loops with global reductions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`Globals <pyop2.Global>` are used primarily for reductions where a
given quantity on a field is reduced to a single number by summation or
finding the minimum or maximum. Consider a kernel computing the `L2 norm`_ of
the ``pressure`` field defined on the set of ``vertices`` as ``l2norm``. Note
that the :class:`~pyop2.Dat` constructor automatically creates an anonymous
:class:`~pyop2.DataSet` of dimension 1 if a :class:`~pyop2.Set` is passed as
the first argument. We assume ``pressure`` is the result of some prior
computation and only give the declaration for context. ::

    pressure = op2.Dat(vertices, [...], dtype=float)
    l2norm = op2.Global(dim=1, data=[0.0])

    norm = op2.Kernel("""void norm(double * out, double * field) {
      *out += field[0] * field[0];
    }""", "norm")

    op2.par_loop(pressure, vertices,
                 l2norm(op2.INC),
                 vertices(op2.READ))

.. _NumPy: http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
.. _L2 norm: https://en.wikipedia.org/wiki/L2_norm#Euclidean_norm
