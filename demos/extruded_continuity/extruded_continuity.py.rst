Steady-state continuity equation on an extruded mesh
====================================================

We next consider the equation

.. math::

   \nabla\cdot(\vec{u_0}D) = 0

in a domain :math:`\Omega`, where :math:`\vec{u_0}` is a prescribed vector
field, and :math:`D` is an unknown scalar field. The value of :math:`D` is known
on the subset of the boundary :math:`\Gamma` in which :math:`\vec{u_0}` is
directed towards the interior of the domain:

.. math::

  D = D_0 \quad \mathrm{on} \ \Gamma_\mathrm{inflow}

where :math:`\Gamma_\mathrm{inflow}` is defined appropriately. :math:`D` can
be interpreted as the steady-state distribution of a passive tracer carried by a
fluid with velocity field :math:`\vec{u_0}`.

A weak form of the continuous equation is

.. math::

   \int_\Omega \! \phi \nabla \cdot (\vec{u_0} D) \, \mathrm{d} x &= 0 \quad
   \forall \ \phi \in D(\Omega), \\
   
   D &= D_0 \quad \mathrm{on} \ \Gamma_\mathrm{inflow}

where :math:`D(\Omega)` is the space of smooth *test functions* with compact
support in :math:`\Omega`. We will look for a solution :math:`D` in a space of
*discontinuous* functions :math:`V`. This suggests using integration by parts to
avoid taking the derivative of a discontinuous quantity:

.. math::

   \sum_e \left( \int_{\partial e} \! \phi_e D \vec{u_0} \cdot \vec{n} \,
   \mathrm{d} S - \int_e \! D \vec{u_0} \cdot \nabla \phi \, \mathrm{d} x \right) = 0
   \quad \forall \ \phi \in V, \\
   
   D = D_0 \quad \mathrm{on} \ \Gamma_\mathrm{inflow}

where the sum is taken over all elements. Since :math:`D` is discontinuous, we
have to make a choice about how it is defined on facets in order to evaluate
the first integral. We will use upwinding: the *upstream* value of :math:`D` is
used on the facet. In light of this, there are three distinct situations we may
encounter:

1. Boundary facets where :math:`\vec{u_0}` points towards the interior of the
   domain. Here, the prescribed boundary value :math:`D_0` is used.
2. Boundary facets where :math:`\vec{u_0}` points away from the interior of the
   domain. Here, the (unknown) interior solution value :math:`D` is used.
3. Interior facets. Here, the upstream value of :math:`D`,
   :math:`\widetilde{D}`, is used.

Note that each of the interior facets contributes to the integral twice. The two
contributions differ in the choice of test function: a subscript :math:`\phi_e`
was used to make this explicit. The full set of equations are then

.. math::

   -\int_\Omega \! D \vec{u_0} \cdot \nabla \phi \, \mathrm{d} x 
   + \int_{\Gamma_\rlap{\mathrm{ext, outflow}}} \! \phi D \vec{u_0} \cdot \vec{n}
   \, \mathrm{d} s 
   + \int_{\Gamma_\mathrm{int}} \! (\phi_+ - \phi_-) \widetilde{D}
   \vec{u_0} \cdot \vec{n} \, \mathrm{d} S
   \quad = \quad
   -\int_{\Gamma_\rlap{\mathrm{ext, inflow}}} \phi D_0 \vec{u_0} \cdot
   \vec{n} \, \mathrm{d} s \quad \forall \ \phi \in V,

   D = D_0 \quad \mathrm{on} \ \Gamma_\mathrm{inflow}

In this worked example, we will take the domain :math:`\Omega` to be the cuboid
:math:`\Omega = [0,1] \times [0,1] \times [0,0.2]`. We will use the constant
velocity field :math:`\vec{u_0} = (0, 0, 1)`. :math:`\Gamma_\mathrm{inflow}`
is therefore the base of the cuboid, while :math:`\Gamma_\mathrm{outflow}`
is the top. The four vertical sides can be ignored, since
:math:`\vec{u_0} \cdot \vec{n} = 0` on these faces.

Firedrake code for this example is as follows:

We will use an *extruded* mesh, where the base mesh is a 20 by 20 unit square,
with 10 evenly-spaced vertical layers. This gives prism-shaped cells. ::

  from firedrake import *
  m = UnitSquareMesh(20, 20)
  mesh = ExtrudedMesh(m, layers=10, layer_height=0.02)

We will use a simple piecewise-constant function space for the unknown scalar
:math:`D`: ::

  V = FunctionSpace(mesh, "DG", 0)

Our velocity will live in a low-order Raviart-Thomas space. The construction of
this is more complicated than element spaces you will have seen previously. The
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

  velocity = as_vector([0.0, 0.0, 1.0])
  u0 = project(velocity, W)
  
  # if we had used W = VectorFunctionSpace(mesh, "CG", 1), we could have done
  # u0 = Function(W)
  # u0.interpolate(velocity)

Next, we will set the boundary value on our scalar to be a simple indicator
function over part of the bottom of the domain: ::

  x = SpatialCoordinate(mesh)
  inflow = conditional(And(x[2] < 0.02, x[0] > 0.5), 1.0, -1.0)
  D0 = Function(V)
  D0.interpolate(inflow)

Now we will define our forms. There are several new concepts here. Firstly, we
will define a new variable ``un`` which takes the value
:math:`\vec{u_0} \cdot \vec{n}` when this is positive, otherwise `0`. This
will be useful for our upwind terms. ::

  n = FacetNormal(mesh)
  un = 0.5*(dot(u0, n) + abs(dot(u0, n)))

We define our trial and test functions in the usual way: ::

  D = TrialFunction(V)
  phi = TestFunction(V)

Since we are on an extruded mesh, we have several new integral types at our
disposal. An integral over the interior of the domain is still denoted by
``dx``. Boundary integrals now come in several varieties: ``ds_b`` denotes an
integral over the base of the mesh, while ``ds_t`` denotes an integral over the
top of the mesh. ``ds_v`` denotes an integral over the sides of a mesh, though
we will not use that here.

Similiarly, interior facet integrals are split into ``dS_h`` and ``dS_v``, over
*horizontal* interior facets and *vertical* interior facets respectively. Since
our velocity field is purely in the vertical direction, we will omit the
integral over vertical interior facets, since we know
:math:`\vec{u_0} \cdot \vec{n}` is zero for these. ::

  a1 = -D*dot(u0, grad(phi))*dx
  a2 = dot(jump(phi), un('+')*D('+') - un('-')*D('-'))*dS_h
  a3 = dot(phi, un*D)*ds_t  # outflow at top wall
  a = a1 + a2 + a3

  L = -D0*phi*dot(u0, n)*ds_b  # inflow at bottom wall

Finally, we will compute the solution: ::

  out = Function(V)
  solve(a == L, out)

By construction, the exact solution is quite simple: ::
  
  exact = Function(V)
  exact.interpolate(conditional(x[0] > 0.5, 1.0, -1.0))

We finally compare our solution to the expected solution: ::

  assert max(abs(out.dat.data - exact.dat.data)) < 1e-10

This demo can be found as a script in
`extruded_continuity.py <extruded_continuity.py>`__.
