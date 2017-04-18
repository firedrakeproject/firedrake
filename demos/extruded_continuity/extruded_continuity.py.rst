Steady-state continuity equation on an extruded mesh
====================================================

This demo showcases the use of extruded meshes, including the new regions of
integration and the construction of sophisticated finite element spaces.

We now consider the equation

.. math::

   \nabla\cdot(\vec{u}q) &= 0 \\
   q &= q_\mathrm{in} \quad \text{on} \ \Gamma_\mathrm{inflow},

in a domain :math:`\Omega`, where :math:`\vec{u}` is a prescribed vector field,
and :math:`q` is an unknown scalar field. The value of :math:`q` is known on the
'inflow' part of the boundary :math:`\Gamma`, where :math:`\vec{u}` is directed
towards the interior of the domain. :math:`q` can be interpreted as the
steady-state distribution of a passive tracer carried by a fluid with velocity
field :math:`\vec{u}`.

We apply an upwind DG method, as we saw in the previous example.  Denoting the
upwind value of :math:`q` on interior facets by :math:`\widetilde{q}`, the full
set of equations are then

.. math::

   -\int_\Omega \! q \vec{u_0} \cdot \nabla \phi \, \mathrm{d} x
   + \int_{\Gamma_\rlap{\mathrm{ext, outflow}}} \! \phi q \vec{u} \cdot \vec{n}
   \, \mathrm{d} s
   + \int_{\Gamma_\mathrm{int}} \! (\phi_+ \vec{u} \cdot \vec{n}_+ +
     \phi_- \vec{u} \cdot \vec{n}_-) \widetilde{q} \, \mathrm{d} S
   \quad = \quad
   -\int_{\Gamma_\rlap{\mathrm{ext, inflow}}} \phi q_\mathrm{in} \vec{u} \cdot
   \vec{n} \, \mathrm{d} s \quad \forall \ \phi \in V,

We will take the domain :math:`\Omega` to be the cuboid
:math:`\Omega = [0,1] \times [0,1] \times [0,0.2]`. We will use the uniform
velocity field :math:`\vec{u} = (0, 0, 1)`. :math:`\Gamma_\mathrm{inflow}`
is therefore the base of the cuboid, while :math:`\Gamma_\mathrm{outflow}`
is the top. The four vertical sides can be ignored, since
:math:`\vec{u} \cdot \vec{n} = 0` on these faces.

We use an *extruded* mesh, where the base mesh is a 20 by 20 unit square,
divided into triangles, with 10 evenly-spaced vertical layers. This gives
prism-shaped cells. ::

  from firedrake import *
  m = UnitSquareMesh(20, 20)
  mesh = ExtrudedMesh(m, layers=10, layer_height=0.02)

We will use a simple piecewise-constant function space for the unknown scalar
:math:`q`: ::

  V = FunctionSpace(mesh, "DG", 0)

Our velocity will live in a low-order Raviart-Thomas space. The construction of
this is more complicated than element spaces that have appeared previously. The
horizontal and vertical components of the field are specified separately. They
are combined into a single element which is used to build a FunctionSpace. ::

  # RT1 element on a prism
  W0_h = FiniteElement("RT", "triangle", 1)
  W0_v = FiniteElement("DG", "interval", 0)
  W0 = HDivElement(TensorProductElement(W0_h, W0_v))
  W1_h = FiniteElement("DG", "triangle", 0)
  W1_v = FiniteElement("CG", "interval", 1)
  W1 = HDivElement(TensorProductElement(W1_h, W1_v))
  W_elt = W0 + W1
  W = FunctionSpace(mesh, W_elt)

As an aside, since our prescibed velocity is purely in the vertical direction, a
simpler space would have sufficed: ::

  # Vertical part of RT1 element
  # W_h = FiniteElement("DG", "triangle", 0)
  # W_v = FiniteElement("CG", "interval", 1)
  # W_elt = HDivElement(TensorProductElement(W_h, W_v))
  # W = FunctionSpace(mesh, W_elt)

Or even: ::

  # Why can't everything in life be this easy?
  # W = VectorFunctionSpace(mesh, "CG", 1)

Next, we set the prescribed velocity field: ::

  velocity = as_vector((0.0, 0.0, 1.0))
  u = project(velocity, W)

  # if we had used W = VectorFunctionSpace(mesh, "CG", 1), we could have done
  # u = Function(W)
  # u.interpolate(velocity)

Next, we will set the boundary value on our scalar to be a simple indicator
function over part of the bottom of the domain: ::

  x, y, z = SpatialCoordinate(mesh)
  inflow = conditional(And(z < 0.02, x > 0.5), 1.0, -1.0)
  q_in = Function(V)
  q_in.interpolate(inflow)

Now we will define our forms.  We use the same trick of defining ``un`` to aid
with the upwind terms: ::

  n = FacetNormal(mesh)
  un = 0.5*(dot(u, n) + abs(dot(u, n)))

We define our trial and test functions in the usual way: ::

  q = TrialFunction(V)
  phi = TestFunction(V)

Since we are on an extruded mesh, we have several new integral types at our
disposal. An integral over the cells of the domain is still denoted by ``dx``.
Boundary integrals now come in several varieties: ``ds_b`` denotes an integral
over the base of the mesh, while ``ds_t`` denotes an integral over the top of
the mesh. ``ds_v`` denotes an integral over the sides of a mesh, though we will
not use that here.

Similiarly, interior facet integrals are split into ``dS_h`` and ``dS_v``, over
*horizontal* interior facets and *vertical* interior facets respectively. Since
our velocity field is purely in the vertical direction, we will omit the
integral over vertical interior facets, since we know
:math:`\vec{u} \cdot \vec{n}` is zero for these. ::

  a1 = -q*dot(u, grad(phi))*dx
  a2 = dot(jump(phi), un('+')*q('+') - un('-')*q('-'))*dS_h
  a3 = dot(phi, un*q)*ds_t  # outflow at top wall
  a = a1 + a2 + a3

  L = -q_in*phi*dot(u, n)*ds_b  # inflow at bottom wall

Finally, we will compute the solution: ::

  out = Function(V)
  solve(a == L, out)

By construction, the exact solution is quite simple: ::

  exact = Function(V)
  exact.interpolate(conditional(x > 0.5, 1.0, -1.0))

We finally compare our solution to the expected solution: ::

  assert max(abs(out.dat.data - exact.dat.data)) < 1e-10

This demo can be found as a script in
`extruded_continuity.py <extruded_continuity.py>`__.
