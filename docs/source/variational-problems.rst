.. only:: html

  .. contents::

Defining variational problems
=============================

Firedrake uses a high-level language, `UFL`_, to describe variational
problems.  To do this, we need a number of pieces.  We need a
representation of the domain we're solving the :abbr:`PDE (partial
differential equation)` on: Firedrake uses a
:py:class:`~.Mesh` for this.  On top of this mesh,
we build :py:class:`~.FunctionSpace`\s which
define the space in which the solutions to our equation live.  Finally
we define :py:class:`~.Function`\s in those
function spaces to actually hold the solutions.

Constructing meshes
-------------------

Firedrake can read meshes in `Gmsh`_ and `triangle`_ format.  To build
a mesh one uses the :py:class:`~.Mesh`
constructor, passing the name of the file as an argument.  If your
mesh is in triangle format, you should pass the name of ``node`` file,
if it is in Gmsh format, use the name of the ``msh`` file.  For
example, if your mesh comes from a Gmsh ``geo`` file called
``coastline.geo``, you can generate a ``Mesh`` object in Firedrake
with:

.. code-block:: python

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
reordering by passing :py:data:`reorder=False` to the appropriate
:py:class:`~.Mesh` constructor.  You can control Firedrake's default
behaviour in reordering meshes with the :py:data:`"reorder_meshes"`
parameter.  For example, to turn off mesh reordering globally:

.. code-block:: python

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
of standard shapes.  The simplest is a
:py:class:`~.IntervalMesh` which is a regularly
subdivided unit line of specified length.  We may also build square
meshes with the :py:class:`~.UnitSquareMesh`
constructor, and cube meshes with
:py:class:`~.UnitCubeMesh`.  The API documentation for
the :py:mod:`~firedrake.mesh` module has full usage descriptions.

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

.. code-block:: python
   
   sphere_mesh = Mesh('sphere_mesh.node', 3)

Firedrake provides utility meshes for the surfaces of spheres immersed
in 3D that are approximated using an `icosahedral mesh`_.  You can
either build a mesh of the unit sphere with
:py:class:`~.UnitIcosahedralSphereMesh`, or a mesh of a
sphere with specified radius using
:py:class:`~.IcosahedralSphereMesh`.  The meshes are
constructed by recursively refining a `regular icosahedron
<icosahedron_>`_, you can specify the refinement level by passing a
non-zero ``refinement_level`` to the constructor.  For example, to
build a sphere mesh that approximates the surface of the Earth (with a
radius of 6371 km) that has subdivided the original icosahedron 7
times we would write:

.. code-block:: python

   earth = IcosahedralSphereMesh(radius=6371, refinement_level=7)

Ensuring consistent cell orientations
+++++++++++++++++++++++++++++++++++++

Variational forms that contain facet normals, for example problems
where a non-zero boundary condition is applied to the normal
derivative of the solution, require information about the orientation
of the cells.  For normal meshes, this is does not pose a problem,
however for immersed meshes we must tell Firedrake about the
orientation of each cell relative to some global orientation.  This
information is used by Firedrake to ensure that the facet normal on,
say, the surface of a sphere, uniformly points outwards.  To do this,
after constructing an immersed mesh, we must initialise the cell
orientation information.  This is carried out with the function
:py:meth:`~.Mesh.init_cell_orientations`, which
takes an :py:class:`~.Expression` used to produce
the reference normal direction.  For example, on the sphere mesh of
the earth defined above we can initialise the cell orientations
relative to vector pointing out from the origin:

.. code-block:: python

   earth.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]')))


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
To do so, we use the :py:class:`~.FunctionSpace`
or :py:class:`~.VectorFunctionSpace` constructors.
The former may be used to define a function space for a scalar
variable, for example pressure, which has a single value at each point
in the domain; the latter is for vector-valued variables, such as
velocity, whose value is a vector at each point in the domain.  To
construct a function space, you must decide on its family and its
degree.  For example, to build a function space of piecewise cubic
polynomials we write:

.. code-block:: python

   V = FunctionSpace(mesh, "Lagrange", 3)

Firedrake supports all function spaces that are allowed by `FIAT`_.

Function spaces on immersed manifolds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the number of components of each vector in a vector
function space is the geometric dimension of the mesh (e.g. 3, if the
mesh is 3D).  However, sometimes we might want that the number of
components in the vector differs from the geometric dimension of the
mesh.  We can do this by passing a value for the ``dim`` argument to
the :py:class:`~.VectorFunctionSpace` constructor.
For example, if we wanted a 2D vector-valued function space on the
surface of a unit sphere mesh we might write:

.. code-block:: python

   mesh = UnitIcosahedralSphereMesh(refinement_level=3)
   V = VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)


Mixed function spaces
~~~~~~~~~~~~~~~~~~~~~

Many :abbr:`PDE (partial differential equation)`\s are posed in terms
of more than one, coupled, variable.  The function space for the
variational problem for such a PDE is termed a *mixed* function space.
Such a space is represented in Firedrake by a
:py:class:`~.MixedFunctionSpace`.  We can either
build such a space by invoking the constructor directly, or, more
readably, by taking existing function spaces and multiplying them
together using the ``*`` operator.  For example:

.. code-block:: python

   V = FunctionSpace(mesh, 'RT', 1)
   Q = FunctionSpace(mesh, 'DG', 0)
   W = V*Q

is equivalent to:

.. code-block:: python

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


Expressing a variational problem
--------------------------------

Firedrake uses the UFL language to express variational problems.  For
complete documentation, we refer the reader to `the UFL package
documentation <UFL_package_>`_ and the description of the language in
`TOMS <UFL_>`_.  We present a brief overview of the syntax here,
for a more didactic introduction, we refer the reader to the
:ref:`Firedrake tutorial examples <firedrake_tutorials>`.

Building test and trial spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. code-block:: python

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

.. code-block:: python

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

   u = f \; \mathrm{on} \, \Omega

where :math:`\Omega` is the unit square, using piecewise linear
polynomials for our solution.  We start with a mesh and build a
function space on it:

.. code-block:: python

   mesh = UnitSquareMesh(10, 10)
   V = FunctionSpace(mesh, "CG", 1)

now we need a test function, and since ``u`` is unknown, a trial
function:

.. code-block:: python

   u = TrialFunction(V)
   v = TestFunction(V)

finally we need a function to hold the right hand side :math:`f` which
we will populate with the x component of the coordinate field.

.. code-block:: python

   f = Function(V)
   f.interpolate(Expression('x[0]'))

For details on how :py:class:`~.Expression`\s and
:py:meth:`~.Function.interpolate` work, see the
:doc:`appropriate section in the manual <expressions>`.  The
variational problem is to find :math:`u \in V` such that

.. math::

   \int_\Omega u v \mathrm{d}x = \int_\Omega f v \mathrm{d}x \;
   \forall v \in V

we define the variational problem in UFL with:

.. code-block:: python

   a = u*v*dx
   L = f*v*dx

Where the ``dx`` indicates that the integration should be carried out
over the cells of the mesh.  UFL can also express integrals over the
boundary of the domain, using ``ds``, and the interior facets of the
domain, using ``dS``.

How to solve such variational problems is the subject of the
:doc:`next section <solving-interface>`, but for completeness we show
how to do it here.  First we define a function to hold the solution

.. code-block:: python

   s = Function(V)

and call :py:func:`~.solve` to solve the variational
problem:

.. code-block:: python

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

.. code-block:: python

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

.. code-block:: python

   bc = DirichletBC(V, value, subdomain_id)

The ``subdomain_id`` is an integer indicating which section of the
mesh the boundary condition should be applied to.  The subdomain ids
for the various :ref:`utility meshes <utility_mesh_functions>` are
described in their respective constructor documentation.  For
externally generated meshes, Firedrake just uses whichever ids the
mesh generator provided.  The ``value`` may be either a scalar, or
more generally an :py:class:`~.Expression`, :py:class:`~.Function` or
:py:class:`~.Constant` of the appropriate shape.  You may also supply
an iterable of literal constants, which will be converted to an
:py:class:`~.Expression`.  Hence the following two are equivalent:

.. code-block:: python

   bc1 = DirichletBC(V, Expression(('1.0', '2.0')), 1)
   bc2 = DirichletBC(V, (1.0, 2.0), 1)

Strong boundary conditions are applied in the solve by passing a list
of boundary condition objects:

.. code-block:: python

   solve(a == L, bcs=[bc])

See the :doc:`next section <solving-interface>` for a more complete
description of the interface Firedrake provides to solve PDEs.  The
details of how Firedrake applies strong boundary conditions are
slightly involved and therefore have :doc:`their own section
<boundary_conditions>` in the manual.

Boundary conditions in discontinuous spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default method Firedrake uses to determine where to apply strong
boundary conditions is :py:data:`"topological"`, meaning that nodes
topologically associated with a boundary facet will be included.  In
discontinuous spaces, however, the nodes to be included do not all
live on boundary facets, in this case, you should use the
:py:data:`"geometric"` method for determining boundary condition
nodes.  In this case, nodes associated with basis functions that do
not vanish on the boundary are included.  This method can be used to
impose strong boundary conditions on discontinuous galerkin spaces, or
no-slip conditions on HDiv spaces.  To select the method used for
determining boundary condition nodes, use the :py:data:`method`
argument to the :py:class:`DirichletBC` constructor.  For example, to
select geometric boundary node determination we would write:

.. code-block:: python

   V = FunctionSpace(mesh, 'DG', 2)
   bc = DirichletBC(V, 1.0, subdomain_id, method="geometric")
   ...


Time dependent boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Imposition of time-dependent boundary conditions can by carried out by
modifying the value in the appropriate :py:class:`~.DirichletBC`
object.  Note that if you use a literal value to initialise the
boundary condition object within the timestepping loop, this will
necessitate a recompilation of code every time the boundary condition
changes.  For this reason we either recommend using a
:py:class:`~.Constant` if the boundary condition is spatially uniform,
or a :py:class:`~.Expression` if it has both space and
time-dependence.  For example, a purely time-varying boundary
condition might be implemented as:

.. code-block:: python

   c = Constant(sin(t))
   bc = DirichletBC(V, c, 1)
   while t < T:
       solve(F == 0, bcs=[bc])
       t += dt
       c.assign(sin(t))

If the boundary condition instead has both space and time dependence
we can write:

.. code-block:: python

   e = Expression('sin(x[0]*t'), t=t)
   bc = DirichletBC(V, e, 1)
   while t < T:
       solve(F == 0, bcs=[bc])
       t += dt
       e.t = t

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


.. _icosahedral mesh: http://en.wikipedia.org/wiki/Geodesic_grid
.. _icosahedron: http://en.wikipedia.org/wiki/Icosahedron
.. _triangle: http://www.cs.cmu.edu/~quake/triangle.html
.. _Gmsh: http://geuz.org/gmsh/
.. _UFL: http://arxiv.org/abs/1211.4047
.. _UFL_package: http://fenicsproject.org/documentation/ufl/1.2.0/ufl.html
.. _FIAT: https://bitbucket.org/mapdes/fiat
.. _submanifold: http://en.wikipedia.org/wiki/Submanifold
