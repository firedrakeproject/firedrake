.. only:: html

  .. contents::

Defining variational problems
=============================

Firedrake uses a high-level language, `UFL`_, to describe variational
problems.  To do this, we need a number of pieces.  We need a
representation of the domain we're solving the :abbr:`PDE (partial
differential equation)` on: Firedrake uses a
:py:func:`~.Mesh` for this.  On top of this mesh,
we build :py:class:`~.FunctionSpace`\s which
define the space in which the solutions to our equation live.  Finally
we define :py:class:`~.Function`\s in those
function spaces to actually hold the solutions.

Constructing meshes
-------------------

Firedrake can read meshes in `Gmsh`_, `triangle`_, `CGNS`_, and
`Exodus`_ formats.  To build a mesh one uses the :py:func:`~.Mesh`
constructor, passing the name of the file as an argument, which see
for more details.  The mesh type is determined by the file extension,
for example if the provided filename is ``coastline.msh`` the mesh is
assumed to be in Gmsh format, in which case you can construct a mesh
object like so:

.. code-block:: python3

   coastline = Mesh("coastline.msh")

This works in both serial and parallel, Firedrake takes care of
decomposing the mesh among processors transparently.

Reordering meshes for better performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most mesh generators produce badly numbered meshes (with bad data
locality) which can reduce the performance of assembling and solving
finite element problems.  By default then, Firedrake reorders input
meshes to improve data locality by performing reverse Cuthill-McKee
reordering on the adjacency matrix of the input mesh.  If you know
your mesh has a good numbering (perhaps your mesh generator uses space
filling curves to number entities) then you can switch off this
reordering by passing ``reorder=False`` to the appropriate
:py:func:`~.Mesh` constructor.  You can control Firedrake's default
behaviour in reordering meshes with the ``"reorder_meshes"``
parameter.  For example, to turn off mesh reordering globally:

.. code-block:: python3

   from firedrake import *
   parameters["reorder_meshes"] = False

The parameter passed in to the mesh constructor overrides this default
value.

.. note::

   Firedrake numbers degrees of freedom in a function space by
   visiting each cell in order and performing a depth first numbering
   of all degrees of freedom on that cell.  Hence, if your mesh has a
   good numbering, the degrees of freedom will too.

.. _utility_mesh_functions:

Utility mesh functions
~~~~~~~~~~~~~~~~~~~~~~

As well as offering the ability to read mesh information from a file,
Firedrake also provides a number of built in mesh types for a number
of standard shapes.  1-dimensional intervals may be constructed with
:func:`~.IntervalMesh`; 2-dimensional rectangles with
:func:`~.RectangleMesh`; and 3-dimensional boxes with
:func:`~.BoxMesh`.  There are also more specific constructors (for
example to build unit square meshes).  See
:mod:`~firedrake.utility_meshes` for full details.

.. _immersed_manifolds:

Immersed manifolds
~~~~~~~~~~~~~~~~~~

In addition to the simple meshes described above, Firedrake also has
support for solving problems on orientable `immersed manifolds
<submanifold_>`_.  That is, meshes in which the entities are
*immersed* in a higher dimensional space.  For example, the surface of
a sphere in 3D.

If your mesh is such an immersed manifold, you need to tell Firedrake
that the geometric dimension of the coordinate field (defining where
the points in mesh are) is not the same as the topological dimension
of the mesh entities.  This is done by passing an optional second
argument to the mesh constructor which specifies the geometric
dimension.  For example, for the surface of a sphere embedded in 3D we
use:

.. code-block:: python3

   sphere_mesh = Mesh('sphere_mesh.node', dim=3)

Firedrake provides utility meshes for the surfaces of spheres immersed
in 3D that are approximated using an `icosahedral mesh`_.  You can
either build a mesh of the unit sphere with
:py:func:`~.UnitIcosahedralSphereMesh`, or a mesh of a
sphere with specified radius using
:py:func:`~.IcosahedralSphereMesh`.  The meshes are
constructed by recursively refining a `regular icosahedron
<icosahedron_>`_, you can specify the refinement level by passing a
non-zero ``refinement_level`` to the constructor.  For example, to
build a sphere mesh that approximates the surface of the Earth (with a
radius of 6371 km) that has subdivided the original icosahedron 7
times we would write:

.. code-block:: python3

   earth = IcosahedralSphereMesh(radius=6371, refinement_level=7)

Ensuring consistent cell orientations
+++++++++++++++++++++++++++++++++++++

Variational forms that include particular function spaces (those
requiring a *contravariant Piola transform*), require information
about the orientation of the cells.  For normal meshes, this can be
deduced automatically. However, when using immersed meshes, Firedrake
needs extra information to calculate the orientation of each cell
relative to some global orientation. This
is used by Firedrake to ensure that the cell normal on,
say, the surface of a sphere, uniformly points outwards.  To do this,
after constructing an immersed mesh, we must initialise the cell
orientation information.  This is carried out with the function
``~.Mesh.init_cell_orientations``, which
takes a UFL expression used to produce
the reference normal direction.  For example, on the sphere mesh of
the earth defined above we can initialise the cell orientations
relative to vector pointing out from the origin:

.. code-block:: python3

   earth.init_cell_orientations(SpatialCoordinate(earth))

However, a more complicated expression would be needed to initialise
the cell orientations on a toroidal mesh.


Semi-structured extruded meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firedrake has special support for solving PDEs on high-aspect ratio
domains, such as in the ocean or atmosphere, where the numerics
dictate that the "short" dimension should be structured.  These are
termed *extruded meshes* and have a :doc:`separate section
<extruded-meshes>` in the manual.

Building function spaces
------------------------

Now that we have a mesh of our domain, we need to build the function
spaces the solution to our :abbr:`PDE (partial differential equation)`
will live in, along with the spaces for the trial and test functions.
To do so, we use the :py:func:`~.FunctionSpace` constructor.
This is the only way to obtain a function space for a scalar variable,
such as pressure, which has a single value at each point in the
domain.

To construct a function space, you must specify its family and
polynomial degree. To build a scalar-valued function space of
continuous piecewise-cubic polynomials, we write:

.. code-block:: python3

   V = FunctionSpace(mesh, "Lagrange", 3)

There are three main routes to obtaining a function space for a
vector-valued variable such as velocity. Firstly, you can pass the
:py:func:`~.FunctionSpace` constructor a natively *vector-valued*
family such as ``"Raviart-Thomas"``. Secondly, you may use the
:py:func:`~.VectorFunctionSpace` constructor with a *scalar-valued*
family, which gives a vector-valued space where each component is
identical to the appropriate scalar-valued
:py:class:`~.FunctionSpace`.  Thirdly, you can create a
:py:class:`~ufl.classes.VectorElement` directly (which is itself
*vector-valued* and pass that to the :py:func:`~.FunctionSpace`
constructor).

To build a vector-valued function space using the lowest-order
``Raviart-Thomas`` elements, we write

.. code-block:: python3

   V = FunctionSpace(mesh, "Raviart-Thomas", 1)

To build a vector-valued function space for which each component
is a discontinuous piecewise-quadratic polynomial, we can write either

.. code-block:: python3

   V = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 2)

or

.. code-block:: python3

   Vele = VectorElement("Discontinuous Lagrange", cell=mesh.ufl_cell(), degree=2)
   V = FunctionSpace(mesh, Vele)


Advanced usage of ``VectorFunctionSpace``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the number of components of a
:py:func:`~.VectorFunctionSpace` is the geometric dimension of the
mesh (e.g. 3, if the mesh is 3D). However, sometimes we might want
the number of components in the vector to differ from the geometric
dimension of the mesh. We can do this by passing a value for the
``dim`` argument to the :py:func:`~.VectorFunctionSpace` constructor.
For example, if we wanted a vector-valued function space on the surface
of a unit sphere mesh with only 2 components, we might write:

.. code-block:: python3

   mesh = UnitIcosahedralSphereMesh(refinement_level=3)
   V = VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)


Mixed function spaces
~~~~~~~~~~~~~~~~~~~~~

Many :abbr:`PDE (partial differential equation)`\s are posed in terms
of multiple, coupled, variables. The variational problem for such a
PDE uses a so-called *mixed* function space. In Firedrake, this is
represented by a :py:class:`~.MixedFunctionSpace`.  We can either
build such a space by invoking the :py:func:`constructor directly
<.MixedFunctionSpace>`, or, more readably, by taking existing function
spaces and multiplying them together using the ``*`` operator.  For
example:

.. code-block:: python3

   V = FunctionSpace(mesh, 'RT', 1)
   Q = FunctionSpace(mesh, 'DG', 0)
   W = V*Q

is equivalent to:

.. code-block:: python3

   V = FunctionSpace(mesh, 'RT', 1)
   Q = FunctionSpace(mesh, 'DG', 0)
   W = MixedFunctionSpace([V, Q])


Function spaces on extruded meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On :doc:`extruded meshes <extruded-meshes>`, we build function spaces
by taking a tensor product of the base ("horizontal") space and the
extruded ("vertical") space.  Firedrake allows us to separately choose
the horizontal and vertical spaces when building a function space on
an extruded mesh.  We refer the reader to the :doc:`manual section on
extrusion <extruded-meshes>` for details.


.. _supported_elements:

Supported finite elements
-------------------------

Firedrake supports the use of the following finite elements.

.. csv-table::
    :header: "Name", "Short name", "Value shape", "Valid cells"
    :widths: 20, 10, 10, 40
    :file: element_list.csv

In addition, the
:py:class:`~ufl.finiteelement.tensorproductelement.TensorProductElement`
operator can be used to create product elements on extruded meshes.

Element variants
~~~~~~~~~~~~~~~~

Some finite element spaces offer more than one choice of nodes.  For Q, DQ, DQ
L2, RTCE, RTCF, NCE, and NCF spaces on intervals, quadrilaterals and hexahedra,
Firedrake offers both equispaced points and better conditioned Legendre points.
For discontinuous elements these are the Gauss-Legendre points, and for
continuous elements these are the Gauss-Lobatto-Legendre points.
For CG and DG spaces on simplices, Firedrake offers both equispaced points and
the better conditioned recursive Legendre points from :cite:`Isaac2020` via the
`recursivenodes`_ module. These are selected by passing `variant="equispaced"`
or `variant="spectral"` to the :py:class:`~ufl.classes.FiniteElement` or
:py:func:`~.FunctionSpace` constructors. For example:

.. code-block:: python3

    fe = FiniteElement("RTCE", quadrilateral, 2, variant="equispaced")

The default is the spectral variant.


Expressing a variational problem
--------------------------------

Firedrake uses the UFL language to express variational problems.  For
complete documentation, we refer the reader to `the UFL package
documentation <UFL_package_>`_ and the description of the language in
`TOMS <UFL_>`_.  We present a brief overview of the syntax here,
for a more didactic introduction, we refer the reader to the
:ref:`Firedrake tutorial examples <firedrake_tutorials>`.

There are two ways to express a variational problem in UFL. A linear
variational problem is defined in terms of a bilinear form
:math:`a(u,v)` and a linear form :math:`L[v]`, seeking :math:`u\in V`
such that :math:`a(u,v)=L[v]\,\forall v\in V` for some finite element
space :math:`V`. The following section of the notes describes how such
problems can be expressed in UFL and solved using Firedrake. We shall
see that the solve is invoked by writing

.. code-block:: python3

   solve(a == L, s)

but solver reuse can be achieved using :py:class:`~.LinearVariationalSolver`,
which is usually the most efficient option for timestepping problems.
   
A nonlinear variational problem is defined in terms of a linear form
:math:`F[u;v]` which is linear in the test function :math:`v` but may
be nonlinear in the coefficient :math:`u`. The nonlinear variational
problem seeks :math:`u\in V` such that :math:`F[u;v]=0\, \forall v\in
V`. In UFL, the solution variable should be of type :py:class:`~.Function`
instead of :py:class:`~firedrake.ufl_expr.TrialFunction`.

.. code-block:: python3

   solve(F == 0, s)

but solver reuse can be achieved using
:py:class:`~.NonlinearVariationalSolver`, which is usually the most
efficient option for timestepping problems. The solution approach for
this problems is some form of Newton's method. UFL automates the
symbolic differentiation of :math:`F` to obtain the Jacobian expressed
as a bilinear form, which is then solved. Note that nonlinear problems
can be linear (Firedrake and UFL will make no effort to detect this),
in which case Newton's method will converge in one iteration.

For more details about nonlinear
variational problems, see the `UFL manual
<UFL_package_>`_ as well as the :ref:`Firedrake tutorials
<firedrake_tutorials>`.

Building test and trial spaces for linear variational problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have function spaces that our solution will live in, the
next step is to actually write down the variational form of the
problem we wish to solve.  To do this, we will need a test function in
an appropriate space along with a function to hold the solution and
perhaps a trial function.  Test functions are obtained via a call to
:py:class:`~firedrake.ufl_expr.TestFunction`, trial functions via
:py:class:`~firedrake.ufl_expr.TrialFunction` and functions with
:py:class:`~.Function`.  The former two are purely
symbolic objects, the latter contains storage for the coefficients of
the basis functions in the function space.  We use them as follows:

.. code-block:: python3

   u = TrialFunction(V)
   v = TestFunction(V)
   f = Function(V)

.. note::

   A newly allocated :py:class:`~.Function` has
   coefficients which are all zero.

If ``V`` above were a
:py:class:`~.MixedFunctionSpace`, the test and
trial functions we obtain are for the combined mixed space.  Often, we
would like to have test and trial functions for the subspaces of the
mixed space.  We can do this by asking for
:py:class:`~firedrake.ufl_expr.TrialFunctions` and
:py:class:`~firedrake.ufl_expr.TestFunctions`, which return an ordered
tuple of test and trial functions for the underlying spaces.  For
example, if we write:

.. code-block:: python3

   V = FunctionSpace(mesh, 'RT', 1)
   Q = FunctionSpace(mesh, 'DG', 0)
   W = V * Q

   u, p = TrialFunctions(W)
   v, q = TestFunctions(W)

then ``u`` and ``v`` will be, respectively, trial and test
functions for ``V``, while ``p`` and ``q`` will be trial and test
functions for ``Q``.

.. note::

   If we intend to build a variational problem on a mixed space, we
   cannot build the individual test and trial functions on the
   function spaces that were used to construct the mixed space
   directly.  The functions that we build must "know" that they come
   from a mixed space or else Firedrake will not be able to assemble
   the correct system of equations.


A first variational form
~~~~~~~~~~~~~~~~~~~~~~~~

With our test and trial functions defined, we can write down our first
variational form.  Let us consider solving the identity equation:

.. math::

   u = f \quad \mathrm{on} \, \Omega

where :math:`\Omega` is the unit square, using piecewise linear
polynomials for our solution.  We start with a mesh and build a
function space on it:

.. code-block:: python3

   mesh = UnitSquareMesh(10, 10)
   V = FunctionSpace(mesh, "CG", 1)

now we need a test function, and since ``u`` is unknown, a trial
function:

.. code-block:: python3

   u = TrialFunction(V)
   v = TestFunction(V)

finally we need a function to hold the right hand side :math:`f` which
we will populate with the x component of the coordinate field.

.. code-block:: python3

   f = Function(V)
   x = SpatialCoordinate(mesh)
   f.interpolate(x[0])

For details on how :py:meth:`~.Function.interpolate` works, see the
:doc:`appropriate section in the manual <interpolation>`.  The
variational problem is to find :math:`u \in V` such that

.. math::

   \int_\Omega \! u v \, \mathrm{d}x = \int_\Omega \! f v \, \mathrm{d}x \quad
   \forall v \in V

we define the variational problem in UFL with:

.. code-block:: python3

   a = u*v*dx
   L = f*v*dx

Where the ``dx`` indicates that the integration should be carried out
over the cells of the mesh.  UFL can also express integrals over the
boundary of the domain, using ``ds``, and the interior facets of the
domain, using ``dS``.

How to solve such variational problems is the subject of the
:doc:`next section <solving-interface>`, but for completeness we show
how to do it here.  First we define a function to hold the solution

.. code-block:: python3

   s = Function(V)

and call :py:func:`~.solve` to solve the variational
problem:

.. code-block:: python3

   solve(a == L, s)


Forms with constant coefficients
--------------------------------

Many PDEs will contain values that are constant over the whole mesh,
but may vary in time.  For example, a time-varying diffusivity, or a
time-dependent forcing function.  Although you can create a new form
for each new value of this constant, this will not be efficient, since
Firedrake must generate new code each time the value changes.  A
better option is to use a :py:class:`~.Constant` coefficient.  This
object behaves exactly like a :py:class:`~.Function`, except that it
has a single value over the whole mesh.  One may assign a new value to
the :py:class:`~.Constant` using the :py:meth:`~.Constant.assign`
method.  As an example, let us consider a form which contains a time
varying constant which we wish to assemble in a time loop.  We can use
a :py:class:`~.Constant` to do this:

.. code-block:: python3

   ...
   t = 0
   dt = 0.1
   from math import exp
   c = Constant(exp(-t))
   # Exponentially decaying RHS
   L = f*v*c*dx
   while t < tend:
       solve(a == L, ...)
       t += dt
       c.assign(exp(-t))


.. warning::

   Although UFL supports computing the derivative of a form with
   respect to a :py:class:`~.Constant`, the resulting form will have
   an unknown in the reals, which is currently unsupported by
   Firedrake.

Incorporating boundary conditions
---------------------------------

Boundary conditions enter the variational problem in one of two ways.
`Natural` (often termed `Neumann` or `weak`) boundary conditions,
which prescribe values of the derivative of the solution, are
incorporated into the variational form.  `Essential` (often termed
`Dirichlet` or `strong`) boundary conditions, which prescribe values
of the solution, become prescriptions on the function space.  In
Firedrake, the former are naturally expressed as part of the
formulation of the variational problem, the latter are represented as
:py:class:`~.DirichletBC` objects and are applied when
solving the variational problem.  Construction of such a strong
boundary condition requires a function space (to impose the boundary
condition in), a value and a subdomain to apply the boundary condition
over:

.. code-block:: python3

   bc = DirichletBC(V, value, subdomain_id)

The ``subdomain_id`` is an integer indicating which section of the
mesh the boundary condition should be applied to.  The subdomain ids
for the various :ref:`utility meshes <utility_mesh_functions>` are
described in their respective constructor documentation.  For
externally generated meshes, Firedrake just uses whichever ids the
mesh generator provided.  The ``value`` may be either a scalar, or
more generally a UFL expression, for example a :class:`~.Function` or
:py:class:`~.Constant`, of the appropriate shape.  You may also supply
an iterable of literal constants:

.. code-block:: python3

   bc = DirichletBC(V, (1.0, 2.0), 1)

Strong boundary conditions are applied in the solve by passing a list
of boundary condition objects:

.. code-block:: python3

   solve(a == L, bcs=[bc])

See the :doc:`next section <solving-interface>` for a more complete
description of the interface Firedrake provides to solve PDEs.  The
details of how Firedrake applies strong boundary conditions are
slightly involved and therefore have :doc:`their own section
<boundary_conditions>` in the manual.

Boundary conditions on interior facets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to apply strong boundary conditions to interior facets of
your mesh, this is transparently supported. You should arrange that
your mesh generator marks those facets on which you wish to apply
boundary conditions, and just use the subdomain ids as usual.

Special subdomain ids
~~~~~~~~~~~~~~~~~~~~~

As well as integer subdomain ids that come from marked portions of the
mesh, Firedrake also supports the magic string ``"on_boundary"`` to
apply a boundary condition to all exterior facets of the mesh.
Further, on :doc`:extruded meshes <extruded-meshes>` the special
strings ``"top"`` and ``"bottom"`` can be used to apply a boundary
condition on respectively the top and bottom of the extruded domain.

.. note::

   These special strings cannot be combined with integer ids, so if
   you want to apply boundary data on an extruded mesh on (say) ids
   ``1`` and ``2`` as well as the top of the domain you would write

   .. code-block:: python3

      bcs = [DirichletBC(V, ..., (1, 2)), DirichletBC(V, ..., "top")]

Specifying conditions on components of a space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When solving a problem defined on either a
:class:`~.MixedFunctionSpace` or a rank-1 :class:`~.FunctionSpace`, it is
common to want to specify boundary values for only some of the
components.  In the former case, this is the only supported method of
setting boundary values, the latter also supports setting the value
for all components.  In both cases, the syntax is the same.  When
defining the :py:class:`~.DirichletBC` we must index the function space
used.  For example, to specify that the third component of a
:py:func:`~.VectorFunctionSpace` should take the boundary value 0, we write:

.. code-block:: python3

   V = VectorFunctionSpace(mesh, ...)
   bc = DirichletBC(V.sub(2), Constant(0), boundary_ids)

Note that when indexing a :py:class:`~.MixedFunctionSpace` in this
manner, one pulls out the indexed sub-space, rather than a component.
For example, to specify the velocity values in a Taylor-Hood
discretisation we write:

.. code-block:: python3

   V = VectorFunctionSpace(mesh, "CG", 2)
   P = FunctionSpace(mesh, "CG", 1)
   W = V*P

   bcv = DirichletBC(W.sub(0), Constant((0, 0)), boundary_ids)

If we only wanted to specify a single component, we would have to
index twice.  For example, specifying that the x-component of the
velocity is zero, using the same function space definitions:

.. code-block:: python3

   bcv_x = DirichletBC(W.sub(0).sub(0), Constant(0), boundary_ids)

Boundary conditions in discontinuous spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firedrake uses the topological association of nodes to facets to
determine where to apply strong boundary conditions. For spaces where
nodes are not topologically associated with the boundary facets, such
as discontinuous Galerkin spaces, you should instead apply boundary
conditions weakly.

Time dependent boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Imposition of time-dependent boundary conditions can by carried out by
modifying the value in the appropriate :py:class:`~.DirichletBC`
object.  Note that if you use a literal value to initialise the
boundary condition object within the timestepping loop, this will
necessitate a recompilation of code every time the boundary condition
changes.  For this reason we either recommend using a
:py:class:`~.Constant` if the boundary condition is spatially uniform,
or a UFL expression if it has both space and
time-dependence.  For example, a purely time-varying boundary
condition might be implemented as:

.. code-block:: python3

   c = Constant(sin(t))
   bc = DirichletBC(V, c, 1)
   while t < T:
       solve(F == 0, bcs=[bc])
       t += dt
       c.assign(sin(t))

If the boundary condition instead has both space and time dependence
we can write:

.. code-block:: python3

   c = Constant(t)
   e = sin(x[0]*c)
   bc = DirichletBC(V, e, 1)
   while t < T:
       solve(F == 0, bcs=[bc])
       t += dt
       c.assign(t)

.. _more_complicated_forms:

More complicated forms
----------------------

UFL is a fully-fledged language for expressing variational problems,
and hence has operators for all appropriate vector calculus operations
along with special support for discontinuous galerkin methods in the
form of symbolic expressions for facet averages and jumps.  For an
introduction to these concepts we refer the user to the `UFL manual
<UFL_package_>`_ as well as the :ref:`Firedrake tutorials
<firedrake_tutorials>` which cover a wider variety of different
problems.


.. _icosahedral mesh: https://en.wikipedia.org/wiki/Geodesic_grid
.. _icosahedron: https://en.wikipedia.org/wiki/Icosahedron
.. _triangle: http://www.cs.cmu.edu/~quake/triangle.html
.. _Gmsh: http://gmsh.info/
.. _CGNS: http://cgns.github.io/
.. _Exodus: https://sandialabs.github.io/seacas-docs/sphinx/html/
.. _UFL: https://arxiv.org/abs/1211.4047
.. _UFL_package: http://fenics-ufl.readthedocs.io/en/latest/
.. _FIAT: https://github.com/firedrakeproject/fiat
.. _submanifold: https://en.wikipedia.org/wiki/Submanifold
.. _recursivenodes: https://tisaac.gitlab.io/recursivenodes/
