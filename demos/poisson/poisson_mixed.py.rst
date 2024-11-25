Mixed formulation for the Poisson equation
==========================================

We're considering the Poisson equation :math:`\nabla^2 u = -f` using a mixed
formulation on two coupled fields. We start by introducing the negative flux
:math:`\sigma = \nabla u` as an auxiliary vector-valued variable. This leaves
us with the PDE on a unit square :math:`\Omega = [0,1] \times [0,1]` with
boundary :math:`\Gamma`

.. math::

   \sigma - \nabla u = 0 \ \textrm{on}\ \Omega

   \nabla \cdot \sigma = -f \ \textrm{on}\ \Omega

   u = u_0  \ \textrm{on}\ \Gamma_D

   \sigma \cdot n = g  \ \textrm{on}\ \Gamma_N

for some known function :math:`f`. The solution to this equation will be some
functions :math:`u\in V` and :math:`\sigma\in \Sigma` for some suitable
function space :math:`V` and :math:`\Sigma` that satisfy these equations. We
multiply by arbitrary test functions :math:`\tau \in \Sigma` and :math:`\nu \in
V`, integrate over the domain and then integrate by parts to obtain a
weak formulation of the variational problem: find :math:`\sigma\in \Sigma` and
:math:`\nu\in V` such that:

.. math::

   \int_{\Omega} (\sigma \cdot \tau + \nabla \cdot \tau \ u) \ {\rm d} x
   &= \int_{\Gamma} \tau \cdot n \ u \ {\rm d} s
   \quad \forall \ \tau \in \Sigma, \\

   \int_{\Omega} \nabla \cdot \sigma v \ {\rm d} x
   &= - \int_{\Omega} f \ v \ {\rm d} x
   \quad \forall \ v \in V.

The flux boundary condition :math:`\sigma \cdot n = g` becomes an *essential*
boundary condition to be enforced on the function space, while the boundary
condition :math:`u = u_0` turn into a *natural* boundary condition which
enters into the variational form, such that the variational problem can be
written as: find :math:`(\sigma, u)\in \Sigma_g \times V` such that

.. math::

   a((\sigma, u), (\tau, v)) = L((\tau, v))
   \quad \forall \ (\tau, v) \in \Sigma_0 \times V

with the variational forms :math:`a` and :math:`L` defined as

.. math::

   a((\sigma, u), (\tau, v)) &=
     \int_{\Omega} \sigma \cdot \tau + \nabla \cdot \tau \ u
   + \nabla \cdot \sigma \ v \ {\rm d} x \\
   L((\tau, v)) &= - \int_{\Omega} f v \ {\rm d} x
   + \int_{\Gamma_D} u_0 \tau \cdot n  \ {\rm d} s

The essential boundary condition is reflected in function spaces
:math:`\Sigma_g = \{ \tau \in H({\rm div}) \text{ such that } \tau \cdot
n|_{\Gamma_N} = g \}` and :math:`V = L^2(\Omega)`.

We need to choose a stable combination of discrete function spaces
:math:`\Sigma_h \subset \Sigma` and :math:`V_h \subset V` to form a mixed
function space :math:`\Sigma_h \times V_h`. One such choice is
Brezzi-Douglas-Marini elements of polynomial order :math:`k` for
:math:`\Sigma_h` and discontinuous elements of polynomial order :math:`k-1`
for :math:`V_h`.

For the remaining functions and boundaries we choose:

.. math::

  \Gamma_{D} = \{(0, y) \cup (1, y) \in \partial \Omega\},
  \Gamma_{N} = \{(x, 0) \cup (x, 1) \in \partial \Omega\}

  u_0 = 0,
  g = \sin(5x)

  f = 10~e^{-\frac{(x - 0.5)^2 + (y - 0.5)^2}{0.02}}

To produce a numerical solution to this PDE in Firedrake we procede as
follows:

The mesh is chosen as a :math:`32\times32` element unit square. ::

  from firedrake import *
  mesh = UnitSquareMesh(32, 32)

As argued above, a stable choice of function spaces for our problem is the
combination of order :math:`k` Brezzi-Douglas-Marini (BDM) elements and order
:math:`k - 1` discontinuous Galerkin elements (DG). We use :math:`k = 1` and
combine the BDM and DG spaces into a mixed function space ``W``. ::

  BDM = FunctionSpace(mesh, "BDM", 1)
  DG = FunctionSpace(mesh, "DG", 0)
  W = BDM * DG

We obtain test and trial functions on the subspaces of the mixed function
spaces as follows: ::

  sigma, u = TrialFunctions(W)
  tau, v = TestFunctions(W)

Next we declare our source function ``f`` over the DG space and initialise it
with our chosen right hand side function value. ::

  x, y = SpatialCoordinate(mesh)
  f = Function(DG).interpolate(
      10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

After dropping the vanishing boundary term on the right hand side, the
bilinear and linear forms of the variational problem are defined as: ::

  a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx
  L = - f*v*dx

The strongly enforced boundary conditions on the BDM space on the top and
bottom of the domain are declared as: ::

  bc0 = DirichletBC(W.sub(0), as_vector([0.0, -sin(5*x)]), 3)
  bc1 = DirichletBC(W.sub(0), as_vector([0.0, sin(5*x)]), 4)

Note that it is necessary to apply these boundary conditions to the first
subspace of the mixed function space using ``W.sub(0)``. This way the
association with the mixed space is preserved. Declaring it on the BDM space
directly is *not* the same and would in fact cause the application of the
boundary condition during the later solve to fail.

Now we're ready to solve the variational problem. We define `w` to be a function
to hold the solution on the mixed space. ::

  w = Function(W)

Then we solve the linear variational problem ``a == L`` for ``w`` under the
given boundary conditions ``bc0`` and ``bc1`` using Firedrake's default
solver parameters. Afterwards we extract the components ``sigma`` and ``u``
on each of the subspaces with ``split``. ::

  solve(a == L, w, bcs=[bc0, bc1])
  sigma, u = w.subfunctions

Lastly we write the component of the solution corresponding to the primal
variable on the DG space to a file in VTK format for later inspection with a
visualisation tool such as `ParaView <http://www.paraview.org/>`__ ::

  VTKFile("poisson_mixed.pvd").write(u)

We could use the built in plot function of firedrake by calling
:func:`plot <firedrake.pyplot.tripcolor>` to plot a surface graph. Before that,
matplotlib.pyplot should be installed and imported::

  try:
    import matplotlib.pyplot as plt
  except:
    warning("Matplotlib not imported")

  try:
    from firedrake.pyplot import tripcolor
    fig, axes = plt.subplots()
    colors = tripcolor(u, axes=axes)
    fig.colorbar(colors)
  except Exception as e:
    warning("Cannot plot figure. Error msg '%s'" % e)

Don't forget to show the image::

  try:
    plt.show()
  except Exception as e:
    warning("Cannot show figure. Error msg '%s'" % e)

This demo is based on the corresponding `DOLFIN mixed Poisson demo
<http://fenicsproject.org/olddocs/dolfin/1.3.0/python/demo/documented/mixed-poisson/python/documentation.html>`__
and can be found as a script in :demo:`poisson_mixed.py <poisson_mixed.py>`.
