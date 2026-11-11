.. only:: html

  .. contents::

.. _supported_elements:

Supported finite elements
=========================

A function space is built from a mesh, an element *family*, and a
polynomial *degree*, for example

.. code-block:: python

   V = FunctionSpace(mesh, "Lagrange", 2)

This page lists the element families which Firedrake supports, and it shows
where to find more information about each one.

Family names
------------

Each family has a full name, and most also have one or more other names,
such as an abbreviation.  You can use **any** of them as the family
argument; they select the same element.  For example, these three lines
are equivalent:

.. code-block:: python

   V = FunctionSpace(mesh, "Raviart-Thomas", 1)
   V = FunctionSpace(mesh, "RT", 1)
   V = FunctionSpace(mesh, "N1div", 1)

If the "Other names" column is empty, the family has no other name, and
you must use the full name.  Names are case sensitive.

List of supported elements
--------------------------

.. csv-table::
    :header: "Name", "Other names", "Value shape", "Valid cells"
    :widths: 20, 20, 10, 30
    :file: element_list.csv

The table above is generated from the element list in FIAT, specifically
from the file ``finat/ufl/elementlist.py`` in the `FIAT repository <https://github.com/firedrakeproject/fiat>`__.
The file also defines names in the notation of finite element exterior
calculus, for example ``"P-"``, ``"Q-"`` and ``"S Lambda"``.  These
names are not in the table, because the family that they select depends
on the cell dimension and on the form degree.

In addition, the
:py:class:`~finat.ufl.tensorproductelement.TensorProductElement`
operator can be used to create product elements on extruded meshes.
See :doc:`extruded-meshes`.


Getting more information about an element
-----------------------------------------

To see what an element actually is (its degrees of freedom, its
polynomial space, and diagrams of its basis functions), look it up on
`DefElement <https://defelement.org>`__.  In the table above, the name
of each family that has a DefElement page is a link to that page.

The page
`Implemented in FIAT <https://defelement.org/lists/implementations/fiat.html>`__
lists the elements that DefElement knows are implemented in FIAT, which
is the library that Firedrake gets its elements from.


.. _element_variants:

Element variants
----------------

Some finite element spaces offer more than one choice of nodes.  For Q, DQ, DQ
L2, RTCE, RTCF, NCE, and NCF spaces on intervals, quadrilaterals and hexahedra,
Firedrake offers both equispaced points and better conditioned Legendre points.
For discontinuous elements these are the Gauss-Legendre points, and for
continuous elements these are the Gauss-Lobatto-Legendre points.
For CG and DG spaces on simplices, Firedrake offers both equispaced points and
the better conditioned recursive Legendre points from :cite:`Isaac2020` via the
`recursivenodes`_ module. These are selected by passing ``variant="equispaced"``
or ``variant="spectral"`` to the :py:class:`~finat.ufl.finiteelement.FiniteElement` or
:py:func:`~.FunctionSpace` constructors. For example:

.. code-block:: python

    fe = FiniteElement("RTCE", quadrilateral, 2, variant="equispaced")

For CG and DG spaces, and most tensor-product elements, the default is ``variant="spectral"``.

Integral-type degrees of freedom are also available through ``variant="integral"``. These
enable orthogonal bases for DG on any cell type:

.. code-block:: python

    V = FunctionSpace(mesh, "DG", 1, variant="integral")

In this case, the degrees of freedom are integral moments against a basis of polynomials,
which need to be computed with a quadrature rule. The degree of accuracy of the quadrature
can be increased also through the variant option. For instance, ``variant="integral(4)"``
will use a quadrature that exactly evaluates the degrees of freedom on polynomials of
4 degrees higher than that of the space.
Furthermore, the quadrature rule can be specified via the ``"quad_scheme"`` keyword argument.
For more details, see the manual section on :ref:`quadrature rules <element_quad_scheme>`.

Integral-type degrees of freedom are useful to nearly preserve curl-free or
divergence-free functions when interpolating into the finite element space.
H(curl) and H(div) elements, such as Nédélec, Brezzi-Douglas-Marini, and Raviart-Thomas,
have ``variant="integral"`` as default. These elements also support ``variant="point"``,
where the degrees of freedom are vector components evaluated at equispaced points.

Macroelements can be constructed by splitting each cell into subcells and tiling
the element on each subcell, without refining the mesh. For example,
the continuous Lagrange space where each simplex is subdivided
by connecting each of its vertices to the barycenter is constructed as

.. code-block:: python

    V = FunctionSpace(mesh, "CG", 1, variant="alfeld")

The space P1-iso-P2 may be similarly constructed with ``variant="iso(2)"``.
Macroelements also support integral-type degrees of freedom,
for example ``variant="alfeld,integral(2)"``.

.. _recursivenodes: https://tisaac.gitlab.io/recursivenodes/
