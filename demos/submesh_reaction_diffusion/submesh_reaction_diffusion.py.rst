Coupled volume-surface reaction-diffusion on a torus
=====================================================

In many biological and physical processes, chemical species diffuse through
a volume and exchange with a surface species, coupled by an interfacial
transfer mechanism.  A prototypical model of this class is the coupled
volume-surface reaction-diffusion system analysed by :cite:`Egger:2018`, which
we consider here on a solid torus :math:`\Omega \subset \mathbb{R}^3` with
surface :math:`\Gamma = \partial\Omega`.

The model
---------

Let :math:`L : \Omega \times (0, T] \to \mathbb{R}` denote the volume
concentration and :math:`\ell : \Gamma \times (0, T] \to \mathbb{R}` the
surface concentration.  The evolution is governed by

.. math::

   \partial_t L - d_L \Delta L = 0
   \qquad \text{in } \Omega,

   \partial_t \ell - d_\ell \Delta_\Gamma \ell = \lambda L - \gamma \ell
   \qquad \text{on } \Gamma,

with the interface condition

.. math::

   d_L \,\partial_n L = \gamma \ell - \lambda L
   \qquad \text{on } \Gamma.

Here :math:`d_L` and :math:`d_\ell` are diffusion coefficients, while
:math:`\lambda` and :math:`\gamma` are positive transfer constants.  The
interface condition states that the normal flux of :math:`L` out of the volume
equals the net transfer from surface to volume; mass is conserved globally:

.. math::

   M(t) = \int_\Omega L \,\mathrm{d}x + \int_\Gamma \ell \,\mathrm{d}s
        = M(0) \qquad \text{for all } t > 0.

Weak formulation
----------------

Multiplying the volume equation by :math:`v \in H^1(\Omega)`, integrating over
:math:`\Omega`, and using the interface condition in the boundary term gives

.. math::

   (\partial_t L, v)_\Omega
   + d_L (\nabla L, \nabla v)_\Omega
   + (\lambda L - \gamma \ell,\, v)_\Gamma = 0.

Multiplying the surface equation by :math:`w \in H^1(\Gamma)` and integrating
over :math:`\Gamma` gives

.. math::

   (\partial_t \ell, w)_\Gamma
   + d_\ell (\nabla_\Gamma \ell, \nabla_\Gamma w)_\Gamma
   - (\lambda L - \gamma \ell,\, w)_\Gamma = 0.

The coupling terms :math:`(\lambda L - \gamma \ell, v)_\Gamma` and
:math:`(\lambda L - \gamma \ell, w)_\Gamma` involve both the volume function
:math:`L` (restricted to :math:`\Gamma`) and the surface function :math:`\ell`,
integrated over the same surface.  In Firedrake this is handled by a
*cross-mesh measure* on the submesh.

We discretise in time with the Crank–Nicolson scheme, setting
:math:`L^\mathrm{mid} = (L^{n+1} + L^n)/2` and
:math:`\ell^\mathrm{mid} = (\ell^{n+1} + \ell^n)/2`.

Implementation
--------------

We begin by importing Firedrake and, optionally, ngsPETSc for the torus mesh.
::

  from firedrake import *
  try:
      import netgen
  except ImportError:
      import sys
      warning("Unable to import Netgen.")
      sys.exit(0)

Building the torus mesh with ngsPETSc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We construct a solid torus using Open CASCADE Technology via Netgen.  The
generating circle has major radius :math:`R = 3` and minor radius :math:`r = 1`
and is swept around the :math:`z`-axis. ::

  from netgen.occ import *
  import math

  n_pts = 80
  def _curve(t):
      return Pnt(0, 3 + math.cos(t), math.sin(t))

  pnts = [_curve(2 * math.pi * k / n_pts) for k in range(n_pts + 1)]
  spline = SplineApproximation(pnts)
  face   = Face(Wire(spline))
  torus  = face.Revolve(Axis((0, 0, 0), Z), 360)

  ngmesh = OCCGeometry(torus).GenerateMesh(maxh=0.5)
  base   = Mesh(ngmesh)

Mesh hierarchy and submesh hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geometric multigrid requires a hierarchy of uniformly refined meshes.  We
build the volume hierarchy first, then use :func:`~.MeshHierarchy` on the
surface submesh — Firedrake re-applies the same label filter to every level of
the parent hierarchy to produce the matching submesh hierarchy. ::

  nref    = 1
  mh      = MeshHierarchy(base, nref)
  mesh    = mh[-1]

  # The exterior facet marker is unique on this closed surface.
  marker  = mesh.exterior_facets.unique_markers[0]
  smesh_base = Submesh(base, subdim=2, subdomain_id=marker)
  smh     = MeshHierarchy(smesh_base, nref)
  smesh   = smh[-1]

Function spaces and parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use continuous piecewise-linear elements on both the volume and the surface,
collected into a :func:`~.MixedFunctionSpace`. ::

  V  = FunctionSpace(mesh,  "CG", 1)
  sV = FunctionSpace(smesh, "CG", 1)
  Z  = MixedFunctionSpace([V, sV])

The model parameters are ::

  d_L   = Constant(1.0)    # volume diffusion
  d_ell = Constant(1.0)    # surface diffusion
  lam   = Constant(1.0)    # transfer rate: volume → surface
  gam   = Constant(1.0)    # transfer rate: surface → volume

Integration measures
~~~~~~~~~~~~~~~~~~~~~

Three measures are needed.  ``dV`` integrates over :math:`\Omega`, ``dA`` over
:math:`\Gamma`, and ``dC`` is the *cross-mesh measure* that integrates over the
submesh but also queries degrees of freedom from the parent volume mesh.
Firedrake computes the intersection automatically from ``intersect_measures``. ::

  dV = Measure("dx", domain=mesh)
  dA = Measure("dx", domain=smesh)
  dC = Measure("dx", smesh, intersect_measures=(Measure("ds", mesh),))

Initial data and time-stepping setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We initialise with smooth functions on the torus. ::

  dt    = Constant(0.1)
  T     = 1.0

  z     = Function(Z)   # solution at t^{n+1}
  z_old = Function(Z)   # solution at t^n

  L,     l     = split(z)
  L_old, l_old = split(z_old)
  v,     w     = split(TestFunction(Z))

  L_mid = (L + L_old) / 2
  l_mid = (l + l_old) / 2

  X  = SpatialCoordinate(mesh)
  Xs = SpatialCoordinate(smesh)

  z_old.subfunctions[0].interpolate(sin(X[0]) * cos(X[1]))
  z_old.subfunctions[1].interpolate(cos(Xs[2]))
  z.assign(z_old)

The variational form
~~~~~~~~~~~~~~~~~~~~~

The Crank–Nicolson residual assembles straightforwardly.  The cross-mesh
coupling terms use ``dC``: the first argument to ``inner`` may come from the
volume space :math:`V` (``L_mid``) or the surface space :math:`sV`
(``l_mid``), and the test functions ``v`` and ``w`` live on their respective
spaces. ::

  transfer = lam * L_mid - gam * l_mid

  F = (
        inner((L - L_old) / dt, v) * dV
      + d_L * inner(grad(L_mid), grad(v)) * dV
      + inner(transfer, v) * dC
      + inner((l - l_old) / dt, w) * dA
      + d_ell * inner(grad(l_mid), grad(w)) * dA
      - inner(transfer, w) * dC
  )

Fieldsplit geometric-multigrid solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system is not symmetric because :math:`\lambda \neq \gamma` in general, so
we use FGMRES as the outer solver.  The two diagonal blocks — a volume elliptic
problem and a surface elliptic problem — are each preconditioned independently
by a full-cycle geometric multigrid solver.  Firedrake automatically
rediscretises the operators on each level using the mesh hierarchies we built
above. ::

  gmg_block = {
      "ksp_type": "preonly",
      "pc_type":  "mg",
      "pc_mg_type": "full",
      "mg_levels_ksp_type": "chebyshev",
      "mg_levels_pc_type":  "jacobi",
      "mg_levels_ksp_max_it": 2,
  }

  sp = {
      "snes_type": "ksponly",
      "ksp_type":  "fgmres",
      "ksp_rtol":  1.0e-8,
      "pc_type":   "fieldsplit",
      "pc_fieldsplit_type": "additive",
      **{f"fieldsplit_{i}_{k}": v
         for i in (0, 1) for k, v in gmg_block.items()},
  }

We create the problem and solver once, then step in time. ::

  problem = NonlinearVariationalProblem(F, z)
  solver  = NonlinearVariationalSolver(problem, solver_parameters=sp)

Time loop
~~~~~~~~~

We advance the solution and write VTK output at each step. ::

  L_out, l_out = z_old.subfunctions
  L_out.rename("Volume")
  l_out.rename("Surface")

  pvd_V = VTKFile("output/volume.pvd")
  pvd_S = VTKFile("output/surface.pvd")

  t = 0.0
  pvd_V.write(L_out, time=t)
  pvd_S.write(l_out, time=t)

  while t < T - 1e-8:
      solver.solve()
      z_old.assign(z)
      t += float(dt)
      pvd_V.write(L_out, time=t)
      pvd_S.write(l_out, time=t)

A python script version of this demo can be found :demo:`here
<submesh_reaction_diffusion.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
