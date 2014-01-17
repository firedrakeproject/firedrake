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


The Extrusion Process
---------------------

A Firedrake :py:class:`~firedrake.core_types.Mesh` object contains the topolgy
and geometric data required to fully define a mesh. In Firedrake this means:

- the mesh knows about the shape of the cells (interval, triangle, etc).
- the vertices of the mesh have a coordinate field associated to them (2D or
  3D coordinates).
- each mesh component (cell, vertex) is assigned a unique identifier (global
  numbering).
- a map is needed to define the connectivity of the mesh (cell to vertices).

The following code creates a unit square triangle mesh.

.. code-block:: python

	mesh = UnitSquareMesh(2, 2)

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

The extrusion process can add another dimension to the mesh (a planar 2D mesh
becomes a 3D mesh through extrusion). This is not necessarily true for
manifolds, their dimension may remain the same even after extrusion (the
extruded version of a 3D spherical surface remains in 3D space).

Extrusion in Firedrake
~~~~~~~~~~~~~~~~~~~~~~

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

Building an Extruded Mesh in Firedrake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current Firedrake implementation only supports evenly-spaced extruded meshes.

Based on the assumptions in the section above, the construction of an
:py:class:`~firedrake.core_types.ExtrudedMesh` object:

- must include a :py:class:`~firedrake.core_types.Mesh` object to be used as
  a base for the extrusion.
- must include a number of layers (the base mesh multiplicity factor).
- may include the ``layer_height`` (the current implementation assumes even
  spacing between layers).
- may include the ``extrusion_type`` uniform (default) or radial.

The default ``layer_height`` is obtained by dividing the unit length equally
between all layers.

Uniform Extrusion
~~~~~~~~~~~~~~~~~

Uniform extrusion is given, or it computes by default the layer spacing.

.. code-block:: python

	extruded_mesh = ExtrudedMesh(mesh, layers, layer_height=layer_height)

Radial Extrusion
~~~~~~~~~~~~~~~~

Given a mesh, every point is extruded in the outwards direction from the
origin.

.. code-block:: python

	extruded_mesh = ExtrudedMesh(mesh, layers, layer_height=layer_height, extrusion_type='radial')
