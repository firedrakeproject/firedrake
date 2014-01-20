Extruded Meshes in Firedrake
============================

Introduction
------------

Solving Partial Differential Equations (PDEs) on high aspect ratio domains
often leads to treating the short (or *vertical*) direction differently from a
numerical point of view. This implies that certain properties of the domain
can be easily exploited in the presence of a structured vertical direction.
This allows for the numerical algorithms behind domain-wide mathematical
operations (such as integration) to achieve good performance.

Firedrake supports extruded meshes and aims to deliver an automatic way of
exploiting the benefits exposed by these domain types.


Types of Meshes
---------------

The two main categories of meshes are structured and unstructured meshes.

The characteristic of the two types of meshes which has the biggest impact on
performance is the way the topology is specified.

In the *structured* case, the identifiers of mesh components can be inferred
(computed) based on the current location in the mesh (given the identifier of
the current component (cell), the identifiers of some neighbouring components
(vertices) can be computed using a closed form mathematical formula). This is
known as direct addressing (A[i]).

In the *unstructured* case, the mesh components have to be explicitly
enumerated in the form of maps, from one mesh component type to another (from
cells to vertices for example, for each cell the map will contain a list of
vertices). This is known as indirect addressing (A[B[i]]).

Memory latency makes indirect addressing more expensive than direct addressing
(as it is often more effecient to compute than to perform a fetch from
memory).

An *extruded mesh* in the current Firedrake implementation, combines the two
types of meshes: an unstructured mesh is used to specify the topology of the
widest part of the domain while the short (*vertical*) is structured.

More details on the meshes Firedrake can handle can be found in the
:doc:`Variational Problems<variational-problems>` section.

Implementation of the Extrusion Process in Firedrake
----------------------------------------------------

Non-extruded meshes
~~~~~~~~~~~~~~~~~~~

A Firedrake :py:class:`~firedrake.core_types.Mesh` object contains the topolgy
and geometric data required to fully define a mesh. In Firedrake this means:

- the mesh knows about the shape of the cells (interval, triangle, etc).
- the vertices of the mesh have a coordinate field associated to them (2D or
  3D coordinates).
- each mesh component (cell, vertex) is assigned a unique identifier (global
  numbering).
- a map is needed to define the connectivity of the mesh (cell to vertices).

The size of the mesh is given by the number of cells on the sides of the
square, in this case two cells on each side.

<INSERT PICTURE HERE OF 2D MESH>

Conceptually, an extruded mesh requires a generic base mesh and a number of
layers. The number of layers specifies the multiplicity of the base mesh in
the extruded mesh.

<INSERT PICTURE OF EXTRUDED MESH HERE>

As we can see here, the base mesh remains unstructured but each vertical
column of 3D elements is structured.

Extrusion also changes the type of the cell. For example, a triangle cell
becomes a wedge-shaped cell.

<INSERT PICTURE OF TRIANGLE TO WEDGE TRANSFORMATION>

The extrusion process can add another dimension to the mesh. A planar 2D mesh
becomes a 3D mesh through extrusion. This is not necessarily true for
manifolds, their dimension may remain the same even after extrusion. The
radially-extruded version of a 3D spherical surface remains in 3D space.

Extruded Meshes
~~~~~~~~~~~~~~~

In Firedrake, a :py:class:`~firedrake.core_types.ExtrudedMesh` object is a
subclass of :py:class:`~firedrake.core_types.Mesh`. It mimics the
:py:class:`~firedrake.core_types.Mesh` object behaviour in the presence of an
extra inferred dimension (we call this the *vertical*). If direct addressing
techniques are to be used in the *vertical* then the mesh topology need not be
explicit for all the elements of the *vertical* direction.

In Firedrake, an :py:class:`~firedrake.core_types.ExtrudedMesh` object is a
subclass of :py:class:`~firedrake.core_types.Mesh`. It mimics the
:py:class:`~firedrake.core_types.Mesh` object behaviour in the presence of an
extra inferred dimension (we call this the *vertical*). If direct addressing
techqniues are to be used in the *vertical* then the mesh topology need not be
explicit for all the elements of the *vertical* direction.

Let the *base layer* be the bottom-most layer of extruded cells.

Only the *base layer* elements require explicit maps. The remainder of the
topology information can be computed by the addition of an *offset* to the
*base layer* information. We can name the the *inferred* part of the mesh.

The extruded mesh is therefore not fully constructed, it is simply an enhanced
version of the unstructured base mesh with the following modifications:

- the mesh contains the number of layers (this was not present in the
  :py:class:`~firedrake.core_types.Mesh` object).
- the shape of the cells changes (triangles become wedges).
- the vertex coordinates are (re)computed for each vertex (including the
  inferred vertices) of the mesh based on the type of the extrusion (uniform,
  radial).
- each mesh component, inferred or not, is assigned a unique identifier
  (global numbering).
- the map contains explicit indirections of the *base layer* only.

Using Extruded Meshes in Firedrake
--------------------------------------

The current Firedrake implementation only supports evenly-spaced extruded meshes.

As mentioned in the implementation section above, the extrusion process needs
to start from a *base mesh*. Any mesh can be used as a base mesh, the more
common case being meshes with 2D coordinate systems.

The following code creates a unit square mesh with triangular cells.

.. code-block:: python

	mesh = UnitSquareMesh(2, 2)

This is one of the built-in functions which can be used to create Firedrake
pre-defined meshes of different sizes. A more detailed descritpion of other
meshes available in Firedrake can be found in the :doc:`Variational
Problems<variational-problems>` section on mesh construction.

Based on the assumptions in the section above, the construction of an
:py:class:`~firedrake.core_types.ExtrudedMesh` object:

- must include a :py:class:`~firedrake.core_types.Mesh` object to be used as
  a base for the extrusion.
- must include a number of layers (the base mesh multiplicity factor).
- may include the ``layer_height`` (the current implementation assumes even
  spacing between layers).
- may include the ``extrusion_type`` uniform (default) or radial.

The default ``layer_height`` is obtained by dividing the unit length equally
between all layers (the sum of all the distances between subsequent layers
equals 1).

Uniform Extrusion
~~~~~~~~~~~~~~~~~

Uniform extrusion is a form of extrusion which adds another dimesnion to the
coordinate field (2D coordinates become 3D for example). The computation of
the coordinates in the new direction is based on the assumption that the
layers are evenly spaced (hence the word uniform).

Let ``mesh`` be the previously constructed unit square mesh defined above.
Uniformly extruding ``mesh`` with 11 base mesh layers and a distance of
:math:`0.1` between them can be done in the following way:

.. code-block:: python

	extruded_mesh = ExtrudedMesh(mesh, 11, layer_height=0.1, extrusion_type='uniform')

As uniform extrusion is the default type of of extrusion, the call can be
simplified to:

.. code-block:: python

	extruded_mesh = ExtrudedMesh(mesh, 11, layer_height=0.1)

A further simplification can be made as the provided layer height in this case
is equal to the default value :math:`1/(11 - 1) = 0.1`:

.. code-block:: python

	extruded_mesh = ExtrudedMesh(mesh, 11)

Radial Extrusion
~~~~~~~~~~~~~~~~

Given a mesh, every point is extruded in the outwards direction from the
origin.

.. code-block:: python

	extruded_mesh = ExtrudedMesh(mesh, 11, layer_height=0.1, extrusion_type='radial')

Radial extrusion has been developed as a way of extruding spherical surfaces.
The following code radially extrudes a spherical mesh:

.. code-block:: python

	mesh = IcosahedralSphereMesh(radius=1000, refinement_level=2)
	extruded_mesh = ExtrudedMesh(mesh, 11, layer_height=0.1, extrusion_type='radial')

In the above example the layer height can be omitted as it is the same as the
default value.

Custom Extrusion
~~~~~~~~~~~~~~~~

In order to perform the computation of the coordinates effeciently (because
this is a mesh-wide operation), a PyOP2-style parallel loop is constructed
by the Firedrake backend.

The kernels to be used for the coordinate field computation of the extruded
mesh are either automatically generated (uniform or radial extrusion) or can
be provided by the user as constant strings.

.. code-block:: python

	kernel = """
	   void extrusion_kernel(double *extruded_coords[],
                             double *two_d_coords[],
                             int *layer_number[]) {
           extruded_coords[0][0] = two_d_coords[0][0]; // X
           extruded_coords[0][1] = two_d_coords[0][1]; // Y
           extruded_coords[0][2] = 0.1 * layer_number[0][0]; // Z
       }
	"""
	extruded_mesh = ExtrudedMesh(mesh, layers, kernel=kernel)


Function Spaces on Extruded Meshes
----------------------------------

Building a :py:class:`~firedrake.core_types.FunctionSpace` or
:py:class:`~firedrake.core_types.VectorFunctionSpace` on extruded meshes
