Poisson equation on a mixed-cell-type mesh
==========================================

We consider solving the Poisson equation :math:`-\nabla^2 u = f` for :math:`u`
on a square mesh :math:`\Omega = [0,1] \times [0,1]` with boundary :math:`\Gamma`

.. math::

   - \nabla^2 u = f \ \textrm{on}\ \Omega

   u = g  \ \textrm{on}\ \Gamma

for some known function :math:`f` and :math:`g`.
We suppose that the domain can be decomposed as :math:`\Omega = \Omega_q\cup\Omega_t`,
where :math:`\Omega_q` is a quadrilateral mesh and :math:`Omega_t` is a triangular mesh.
composed of quadrilaterals and triangles.
We define suitable function spaces :math:`V_q` and :math:`V_t` respectively on
:math:`\Omega_q` and :math:`\Omega_t`, and recast the above as the following problem
solved for :math:`u_q, u_t \in V_q \times V_t`

.. math::

   - \nabla^2 u_q &= f_q \ \textrm{on}\ \Omega_q

   - \nabla^2 u_t &= f_t \ \textrm{on}\ \Omega_t

   u_q &= g_q  \ \textrm{on}\ \Gamma_q

   u_t &= g_t  \ \textrm{on}\ \Gamma_t

   u_q - u_t &= 0  \ \textrm{on}\ \Gamma_i

   \nabla u_q \cdot n_q + \nabla u_t \cdot n_t &= 0  \ \textrm{on}\ \Gamma_i

where
:math:`f_q` and :math:`f_t` are restrictions of :math:`f` on :math:`\Omega_q` and :math:`\Omega_t`,
:math:`g_q` and :math:`g_t` are restrictions of :math:`g` on :math:`\Gamma_q` and :math:`\Gamma_t`,
where :math:`\Gamma_q = \Gamma\cap\bar\Omega_q`, :math:`\Gamma_t = \Gamma\cap\bar\Omega_t`.
:math:`\Gamma_i = \bar\Omega_q\cap\bar\Omega_t` is the interface between the quadrilateral and triangular
meshes, and :math:`n_q` and :math:`n_t` are unit outward normal to the :math:`\Omega_q` and :math:`\Omega_t`.

We recast the above problem as a weak form solved for :math:`(u_q, u_t)\in V_q\times V_t`

.. math::

   a((u_q, u_v), (v_q, v_t)) = L((v_q, v_t))
   \quad \forall \ (v_q, v_t) \in V_q \times V_t

   u_q &= g_q  \ \textrm{on}\ \Gamma_q

   u_t &= g_t  \ \textrm{on}\ \Gamma_t

where the variational forms :math:`a` and :math:`L` defined as

.. math::

    a((u_q, u_v), (v_q, v_t)) & = \int_{\Omega_q} \nabla u_q \cdot \nabla v_q \ {\rm d} x + \int_{\Omega_t} \nabla u_t \cdot \nabla v_t \ {\rm d} x

                              & - \int_{\Gamma_i} (\nabla u_q + \nabla u_t) / 2 \cdot (v_q n_q + v_t n_t) \ {\rm d} s

                              & - \int_{\Gamma_i} (u_q n_q + u_t n_t) \cdot (\nabla v_q + \nabla v_t) / 2 \ {\rm d} s

                              & + \frac{C}{h} \int_{\Gamma_i} (u_q - u_t)(v_q - v_t) \ {\rm d} s

    L((v_q, v_t)) & = \int_{\Omega_q} f_q v_q \ {\rm d} x + \int_{\Omega_t} f_t v_t \ {\rm d} x

where the conditions on the interface :math:``
The flux boundary condition :math:`\sigma \cdot n = g` becomes an *essential*
boundary condition to be enforced on the function space, while the boundary
condition :math:`u = u_0` turn into a *natural* boundary condition which
enters into the variational form, such that the variational problem can be
written as: find :math:`(\sigma, u)\in \Sigma_g \times V` such that

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
  dim = 2
  label_ext = 1
  label_interf = 2
  mesh = Mesh(os.path.join(cwd, "..", "meshes", "mixed_cell_unit_square.msh"))
  h = 0.1  # roughly
  mesh.topology_dm.markBoundaryFaces(dmcommon.FACE_SETS_LABEL, label_ext)
  mesh_t = Submesh(mesh, dim, PETSc.DM.PolytopeType.TRIANGLE, label_name="celltype", name="mesh_tri")
  x_t, y_t = SpatialCoordinate(mesh_t)
  n_t = FacetNormal(mesh_t)
  mesh_q = Submesh(mesh, dim, PETSc.DM.PolytopeType.QUADRILATERAL, label_name="celltype", name="mesh_quad")
  x_q, y_q = SpatialCoordinate(mesh_q)
  n_q = FacetNormal(mesh_q)
  V_t = FunctionSpace(mesh_t, "P", 8)
  V_q = FunctionSpace(mesh_q, "Q", 7)
  V = V_t * V_q
  u = TrialFunction(V)
  v = TestFunction(V)
  u_t, u_q = split(u)
  v_t, v_q = split(v)
  dx_t = Measure("dx", mesh_t)
  dx_q = Measure("dx", mesh_q)
  ds_t = Measure("ds", mesh_t)
  ds_q = Measure("ds", mesh_q)
  g_t = cos(2 * pi * x_t) * cos(2 * pi * y_t)
  g_q = cos(2 * pi * x_q) * cos(2 * pi * y_q)
  f_t = 8 * pi**2 * g_t
  f_q = 8 * pi**2 * g_q
  a = (
      inner(grad(u_t), grad(v_t)) * dx_t +
      inner(grad(u_q), grad(v_q)) * dx_q
      -inner(
          (grad(u_q) + grad(u_t)) / 2,
          (v_q * n_q + v_t * n_t)
      ) * ds_t(label_interf)
      -inner(
          (u_q * n_q + u_t * n_t),
          (grad(v_q) + grad(v_t)) / 2
      ) * ds_t(label_interf)
      + 100 / h * inner(u_q - u_t, v_q - v_t) * ds_t(label_interf)
  )
  L = (
      inner(f_t, v_t) * dx_t +
      inner(f_q, v_q) * dx_q
  )
  sol = Function(V)
  bc_q = DirichletBC(V.sub(1), g_q, label_ext)
  solve(a == L, sol, bcs=[bc_q])
  sol_t, sol_q = split(sol)
  error_t = assemble(inner(sol_t - g_t, sol_t - g_t) * dx_t)
  error_q = assemble(inner(sol_q - g_q, sol_q - g_q) * dx_q)

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
<https://olddocs.fenicsproject.org/dolfin/1.3.0/python/demo/documented/mixed-poisson/python/documentation.html>`__
and can be found as a script in :demo:`poisson_mixed.py <poisson_mixed.py>`.
