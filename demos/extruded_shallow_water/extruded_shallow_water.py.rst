Linear Shallow Water Equations on an Extruded Mesh
==================================================

This demo solves the linear shallow water equations on an extruded mesh
using a Strang timestepping scheme.

The equations are solved in a domain $\Omega$ utilizing a 2D base mesh
that is extruded vertically to form a 3D volume.

As usual, we start by importing Firedrake: ::

  from firedrake import *

Mesh Generation
----------------

We use an *extruded* mesh, where the base mesh is a $2^5 \times 2^5$ unit square
with 5 evenly-spaced vertical layers. This results in a 3D volume composed of 
prisms. ::

    power = 5
    m = UnitSquareMesh(2 ** power, 2 ** power)
    layers = 5

    mesh = ExtrudedMesh(m, layers, layer_height=0.25)

Function Spaces
----------------
For the velocity field, we use an $H(\mathrm{div})$-conforming function space
constructed as the outer product of a 2D BDM space and a 1D DG space. This ensures
that the normal component of the velocity is continuous across element boundaries
in the horizontal directions, which is important for accurately capturing fluxes. ::

  horiz = FiniteElement("BDM", "triangle", 1)
  vert = FiniteElement("DG", "interval", 0)
  prod = HDiv(OuterProductElement(horiz, vert))
  W = FunctionSpace(mesh, prod)

We also define a pressure space  $X$ using piecewise constant discontinuous
Galerkin elements, and a plotting space $X_{\text{plot}}$ using continuous
Galerkin elements for better visualization. ::

  X = FunctionSpace(mesh, "DG", 0, vfamily="DG", vdegree=0)
  Xplot = FunctionSpace(mesh, "CG", 1, vfamily="Lagrange", vdegree=1)

Initial Conditions
-------------------

We define our functions for velocity and pressure fields. The initial pressure field
is set to a prescribed sine function to create a wave-like disturbance. ::

  # Define starting field
  u_0 = Function(W)
  u_h = Function(W)
  u_1 = Function(W)
  p_0 = Function(X)
  p_1 = Function(X)
  p_plot = Function(Xplot)
  x, y = SpatialCoordinate(m)
  p_0.interpolate(sin(4*pi*x)*sin(2*pi*x))

  T = 0.5
  t = 0
  dt = 0.0025

  file = VTKFile("lsw3d.pvd")

Before starting the time-stepping loop, we project the initial pressure field
into the plotting space for visualization. ::

  p_trial = TrialFunction(Xplot)
  p_test = TestFunction(Xplot)
  solve(p_trial * p_test * dx == p_0 * p_test * dx, p_plot)
  file << p_plot, t

  E_0 = assemble(0.5 * p_0 * p_0 * dx + 0.5 * dot(u_0, u_0) * dx)

Time-Stepping Loop
--------------------

We evolve the system in time using a Strang splitting method. In each time step,
we perform a half-step update of the velocity field, a full-step update of the pressure field,
and then another half-step update of the velocity field. This approach helps to
maintain stability and accuracy. ::

  while t < T:
    u = TrialFunction(W)
    w = TestFunction(W)
    a_1 = dot(w, u) * dx
    L_1 = dot(w, u_0) * dx + 0.5 * dt * div(w) * p_0 * dx
    solve(a_1 == L_1, u_h)

    p = TrialFunction(X)
    phi = TestFunction(X)
    a_2 = phi * p * dx
    L_2 = phi * p_0 * dx - dt * phi * div(u_h) * dx
    solve(a_2 == L_2, p_1)

    u = TrialFunction(W)
    w = TestFunction(W)
    a_3 = dot(w, u) * dx
    L_3 = dot(w, u_h) * dx + 0.5 * dt * div(w) * p_1 * dx
    solve(a_3 == L_3, u_1)

    u_0.assign(u_1)
    p_0.assign(p_1)
    t += dt

    # project into P1 x P1 for plotting
    p_trial = TrialFunction(Xplot)
    p_test = TestFunction(Xplot)
    solve(p_trial * p_test * dx == p_0 * p_test * dx, p_plot)
    file << p_plot, t
    print(t)

Energy Calculation
------------------

Finally, we compute and print the total energy of the system at the end of the simulation
and compare it to the initial energy to assess conservation properties. ::

  E_1 = assemble(0.5 * p_0 * p_0 * dx + 0.5 * dot(u_0, u_0) * dx)
  print('Initial energy', E_0)
  print('Final energy', E_1)