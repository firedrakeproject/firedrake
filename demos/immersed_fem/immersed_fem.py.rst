Creating Firedrake-compatible meshes in Gmsh
============================================

The purpose of this demo is to summarize the
key structure of a ``gmsh.geo`` file that creates
Firedrake-compatible mesh. For more details about Gmsh, please
refer to the `Gmsh documentation <http://gmsh.info/#Documentation>`_.

The first thing we define are four corners of a rectangle.
We specify the x,y, and z(=0) coordinates, as well as the target
element size at these corner (which we set to 0.5). ::

  Point(1) = {-6, 2, 0, 0.5};
  Point(2) = {-6, -2, 0, 0.5};
  Point(3) = {6, -2, 0, 0.5};
  Point(4) = {6, 2, 0, 0.5};

Then, we define 5 points to describe a circle. ::

  Point(5) = {0, 0, 0, 0.1};
  Point(6) = {0.5, 0, 0, 0.1};
  Point(7) = {-0.5, 0, 0, 0.1};
  Point(8) = {0, 0.5, 0, 0.1};
  Point(9) = {0, -0.5, 0, 0.1};

Then, we create 8 edges: 4 for the rectangle and 4 for the circle.
Recall that the Gmsh command ``Circle`` requires the arc to be
strictly smaller than :math:`\pi`. ::

  Line(1) = {1, 4};
  Line(2) = {4, 3};
  Line(3) = {3, 2};
  Line(4) = {2, 1};
  Circle(5) = {8, 5, 6};
  Circle(6) = {6, 5, 9};
  Circle(7) = {9, 5, 7};
  Circle(8) = {7, 5, 8};

Then, we glue together the rectangle edges and, separately, the circle edges.
Note that ``Line``, ``Circle``, and ``Curve Loop`` (as well as ``Physical Curve`` below)
are all curves in Gmsh and must possess a unique tag. ::

  Curve Loop( 9) = {1, 2, 3, 4};
  Curve Loop(10) = {8, 5, 6, 7};

Then, we define two plane surfaces: the rectangle without the disc first, and the disc itself then. ::

  Plane Surface(1) = {9, 10};
  Plane Surface(2) = {10};

Finally, we group together some edges and define ``Physical`` entities, which
can be accessed in Firedrake. ::

  Physical Curve("HorEdges", 11) = {1, 3};
  Physical Curve("VerEdges", 12) = {2, 4};
  Physical Curve("Circle", 13) = {8, 7, 6, 5};
  Physical Surface("PunchedDom", 3) = {1};
  Physical Surface("Disc", 4) = {2};

For simplicity, we have gathered all this commands in the file
`punched_domain.geo <punched_domain.geo>`__. To generate a mesh using this file,
you can type the following command in the terminal::

    gmsh -2 punched_domain.geo -format msh2

To illustrate how to access all these features within Firedrake,
we consider the following interface problem. Denoting by
:math:`\Omega` the filled rectangle and by :math:`D` the disc,
we seek a function :math:`u\in H^1_0(\Omega)` such that

.. math::

   -\nabla \cdot (\sigma \nabla  u) + u = 5 \quad \textrm{in } D

where :math:`\sigma = 1` in :math:`\Omega \setminus D` and :math:`\sigma = 2`
in :math:`D`. Since :math:`sigma` attains different values across :math:`\partial \Omega`,
we need to prescribe the behavior of :math:`u` across the interface. This is
implicitely done by imposing :math:`u\in H^1_0(\Omega)`: the function :math:`u` must be continuous
across :math:`\partial \Omega`. This allows us to employ Lagrangian finite elements
to approximate :math:`u`. However, something that we need to specify is the the jump
of :math:`\sigma \nabla u \cdot \vec{n}` on :math:`\partial \Omega`. This terms arises
natuarlly in the weak formulation of the problem under consideration. In this demo
we simply set

.. math::

   [\sigma \nabla u \cdot \vec{n}] = 3 \quad \textrm{on}\ \partial D

.. note::
   In in practical applications, it is imporant to specify whether the normal vector
   :math:`\vec{n}` points inward or outward. One should also specify the order in which
   the jump operator is computed. This is irrelevant for the purpose of our demo and may
   only lead to replacing the sign of boundary integral in the weak formulation.

The resulting weak formulation reads as follows:

.. math::

   \int_D \sigma \nabla u \cdot \nabla v + uv \mathrm{d}\mathbf{x} - \int_{\partial D} 3v \mathrm{d}S = \int_{\Omega} 5v \mathrm{d}\mathbf{x} \quad \text{for every } v\in H^1_0(\Omega)\,.

The following Firedrake code shows how to solve this variational problem
using linear Lagrangian finite elements. ::

   from firedrake import *
   mesh = Mesh('punched_domain.msh')
   V = FunctionSpace(mesh, "CG", 1)
   u = TrialFunction(V)
   v = TestFunction(V)
   a = 2*dot(grad(v), grad(u))*dx((4,)) + dot(grad(v), grad(u))*dx((3,)) + v*u*dx
   L = Constant(5.) * v * dx + Constant(3.)*v*ds((13,))
   DirBC = DirichletBC(V, 0, [11, 12])
   u = Function(V)
   solve(a == L, u, bcs=DirBC, solver_parameters={'ksp_type': 'cg'})
