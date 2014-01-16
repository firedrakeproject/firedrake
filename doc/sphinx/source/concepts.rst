PyOP2 Concepts
==============

PyOP2 is a domain-specific language (DSL) for the parallel executions of
computational kernels on unstructured meshes or graphs.

Sets and mappings
-----------------

A mesh is defined by :class:`sets <pyop2.Set>` of entities and
:class:`mappings <pyop2.Map>` between these sets. Sets are used to represent
entities in the mesh (nodes in the graph) while maps define the connectivity
between entities (links in the graph), for example associating an edge with
its incident vertices.

.. note ::
  There is a requirement for the map to be of *constant arity*, that is each
  element in the source set must be associated with a constant number of
  elements in the target set. There is no requirement for the map to be
  injective or surjective. This restriction excludes certain kinds of mappings
  e.g. a map from vertices to incident egdes or cells is only possible on a
  very regular mesh where the multiplicity of any vertex is constant.

Data
----

Data can be declared on a set through a :class:`Dat <pyop2.Dat>` or globally
through a :class:`Global <pyop2.Global>` and can be of arbitrary but constant
shape. When declaring data on a set one can associate a scalar with each
element of the set or a one- or higher-dimensional vector. Similar to the
restriction on maps, the shape and therefore the size of the data associated
which each element needs to be uniform. PyOP2 supports all common primitive
data types. The shape and data type are defined through a :class:`DataSet
<pyop2.DataSet>` declared on a given set, which fully describes the in-memory
size of any :class:`Dat <pyop2.Dat>` declared on this :class:`DataSet
<pyop2.DataSet>`. Custom datatypes are supported insofar as the user
implements the serialisation and deserialisation of that type into primitive
data that can be handled by PyOP2.

PyOP2 can also be used to assemble :class:`matrices <pyop2.Mat>`, which are
defined on a :class:`sparsity pattern <pyop2.Sparsity>` which is built from a
pair of :class:`DataSets <pyop2.DataSet>` defining the row and column spaces
the sparsity maps between and one or more pairs of maps, one for the row and
one for the column space of the matrix respectively. The sparsity uniquely
defines the non-zero structure of the sparse matrix and can be constructed
purely from mappings. To declare a :class:`Mat <pyop2.Mat>` on a
:class:`Sparsity <pyop2.Sparsity>` only the data type needs to be given.

Parallel loops
--------------

Computations in PyOP2 are executed as :func:`parallel loops <pyop2.par_loop>`
of a :class:`kernel <pyop2.Kernel>` over an *iteration set*. A parallel loop
invocation requires as arguments, other than the iteration set and the kernel
to operate on, the data the kernel reads and/or writes. A parallel loop
argument is constructed by calling the underlying data object (i.e. the
:class:`Dat <pyop2.Dat>` or :class:`Global <pyop2.Global>`) and passing an
*access descriptor* and the mapping to be used when accessing the data. The
mapping is required for an *indirectly accessed* :class:`Dat <pyop2.Dat>` not
declared on the same set as the iteration set of the parallel loop. In the
case of *directly accessed* data defined on the same set as the iteration set
the map is omitted and only an access descriptor given.

Access descriptors define how the data is accessed by the kernel and must be
one of :data:`pyop2.READ` (read-only), :data:`pyop2.WRITE` (write-only),
:data:`pyop2.RW` (read-write), :data:`pyop2.INC` (increment),
:data:`pyop2.MIN` (minimum reduction) or :data:`pyop2.MAX` (maximum
reduction).
