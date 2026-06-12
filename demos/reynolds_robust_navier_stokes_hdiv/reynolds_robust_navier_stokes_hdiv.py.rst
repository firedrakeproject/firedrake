Reynolds-robust solvers for the stationary Navier-Stokes equations using H(div)–L² elements
===========================================================================================

.. rst-class:: emphasis

    This demo shows a discretisation using H(div)-L² elements and preconditioner for the stationary incompressible
    Navier-Stokes equations that exhibit Reynolds-robustness. For the discretisation,
    this means that its error estimates do not depend explicitly on the Reynolds number;
    for the preconditioner, this means that the number of Krylov iterations per Newton
    step barely changes as the Reynolds number is varied.

    The demo was contributed by `Patrick Farrell
    <mailto:patrick.farrell@maths.ox.ac.uk>`__.

The stationary incompressible Navier-Stokes equations are a fundamental
model for viscous fluid flow. A notorious difficulty is that classical
iterative solvers degrade badly as the Reynolds number :math:`\mathrm{Re}` grows:
standard block preconditioners such as pressure convection-diffusion
(PCD) or least-squares commutator (LSC) require iteration counts that
grow with :math:`\mathrm{Re}` :cite:`Elman:2014`.  This demo shows how to construct a
solver whose iteration count remains *independent of both* :math:`\mathrm{Re}` *and the
mesh size*, following the approach of :cite:`Benzi:2006,Hong:2015,Farrell:2019`.

The strategy combines three ingredients: an H(div)-conforming
discretisation that captures the divergence-free constraint exactly at
the discrete level; the augmented Lagrangian technique of
:cite:`Fortin:1983` to control the pressure Schur complement at any
Reynolds number; and a parameter-robust geometric multigrid
preconditioner based on vertex-star space decomposition
:cite:`Schoberl:1999` to handle the augmented velocity block
efficiently.

The problem
-----------

We solve the classical lid-driven cavity problem for the stationary
incompressible Navier-Stokes equations on the unit square
:math:`\Omega = (0,1)^2`:

.. math::

   -\nabla \cdot \bigl(2\,\mathrm{Re}^{-1}\,\varepsilon(u)\bigr)
   + \nabla \cdot (u \otimes u) + \nabla p &= 0
   \quad \text{in } \Omega,

   \nabla \cdot u &= 0
   \quad \text{in } \Omega,

where :math:`\varepsilon(u) = \tfrac{1}{2}(\nabla u + \nabla u^\top)`
is the symmetric velocity gradient.  We impose no-slip conditions
:math:`u = 0` on the left, right, and bottom walls (:math:`\Gamma_1, \Gamma_2, \Gamma_3`) and a lid velocity
:math:`u = (16 x^2(1-x)^2,\, 0)` on the top wall (:math:`\Gamma_4`).

After discretisation this system becomes a saddle-point linear algebra
problem for the update at each Newton step:

.. math::

   \begin{pmatrix} A & B^\top \\ B & 0 \end{pmatrix}
   \begin{pmatrix} \delta u \\ \delta p \end{pmatrix}
   = \begin{pmatrix} f \\ 0 \end{pmatrix}.

Here :math:`A` is the linearised velocity block and :math:`B` encodes
the divergence constraint.  The key difficulty for preconditioning is
the pressure Schur complement :math:`S = -B A^{-1} B^\top`.  For the
Stokes equations (no advection), the Schur complement is spectrally
equivalent to a scaled pressure mass matrix :math:`Q`
:cite:`Fortin:1983,Silvester:1994,Elman:2014`, but this equivalence
quickly degrades once the convection term is introduced.

The augmented Lagrangian
------------------------

In the context of the Stokes equations, the augmented Lagrangian method
:cite:`Fortin:1983` adds a penalty term to the Lagrangian for the
incompressibility constraint:

.. math::

   \mathcal{L}_\gamma(u, p)
   = \mathcal{L}(u, p) + \frac{\gamma}{2} \int_\Omega (\nabla \cdot u)^2 \,\mathrm{d}x,

where :math:`\gamma \geq 0` is the augmentation parameter.  Since the
exact solution satisfies :math:`\nabla \cdot u = 0`, this term does not
alter the solution. Applying this term to the Stokes equations, the corresponding augmented
velocity block is

.. math::

   A_\gamma = 2\,\mathrm{Re}^{-1}
   \int_\Omega \varepsilon(\varphi_j) : \varepsilon(\varphi_i)\,\mathrm{d}x
   + \gamma \int_\Omega (\nabla \cdot \varphi_j)(\nabla \cdot \varphi_i)\,\mathrm{d}x,

and the Schur complement approximation becomes :math:`S_\gamma \approx
(2\,\mathrm{Re}^{-1} + \gamma)^{-1} Q`.  For Stokes, this is helpful but
not essential, since the Schur complement is already controlled; but for
Navier-Stokes (with the advection terms), it is extremely useful, since
for large :math:`\gamma` the Schur complement becomes perfectly
controlled by something that is straightforward to solve. The tradeoff
is that large :math:`\gamma` makes :math:`A_\gamma` harder to solve, so
the specialised multigrid strategy described below is critical.

To solve :math:`A_\gamma` in a :math:`\gamma`-robust way, the multigrid
smoother must capture the kernel of the divergence operator in a certain
sense.  The theoretical foundation is due to Schöberl
:cite:`Schoberl:1999`: put coarsely, a multigrid method is parameter-robust if the
relaxation captures the kernel and its prolongation maps coarse kernel
functions to (nearly) fine kernel functions.  It is not known how to
devise multigrid components that satisfy these requirements for arbitrary
discretisations; the :math:`\mathrm{Re}`-robust solver we present here can (at present)
be applied to Scott-Vogelius discretisations
:cite:`Farrell:2021,Farrell:2021b`, high-order Taylor-Hood
discretisations, H(div)-L2 discretisations (as coded in this demo), the Mardal-Tai-Winther
discretisation :cite:`Mardal:2002`, and
a modification of the Bernardi-Raugel discretisation
:cite:`Farrell:2019`. Its application to other discretisations is a matter of ongoing
research. In particular, it is not known how to apply these ideas to
the popular low-order Taylor--Hood, MINI, or stabilised equal-order discretisations.
In any case these latter discretisations do not give exactly incompressible
velocity approximations, and their error estimates therefore depend
on the Reynolds number, which is undesirable.

H(div) conforming discretisation
---------------------------------

We use the Brezzi–Douglas–Marini (BDM) space of degree :math:`k = 2`
for velocity and a DG space of degree :math:`k-1` for pressure, using
the integral variant so that the pressure mass matrix is diagonal (and
hence trivially inverted). This element pair yields a strongly divergence
free velocity approximation, which causes the discretisation error estimates
to become independent of the Reynolds number :cite:`John:2017`. Because BDM functions have only normal
continuity across edges, the viscous term must be discretised using a
symmetric interior penalty DG formulation.  The convective term uses
standard DG upwinding.

We build a mesh hierarchy for geometric multigrid; this can be refined to make the
simulation more accurate/expensive. The distribution parameters
configure the mesh partitioning to enable greater overlap than usual,
which is necessary for vertex-star relaxation in parallel. ::

  from firedrake import *
  from firedrake.petsc import PETSc
  from collections.abc import Iterable

  print = PETSc.Sys.Print

  distribution_parameters = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
  num_refinements = 2
  base = UnitSquareMesh(16, 16, diagonal="crossed", distribution_parameters=distribution_parameters)
  mh = MeshHierarchy(base, num_refinements)
  mesh = mh[-1]
  n = FacetNormal(mesh)
  (x, y) = SpatialCoordinate(mesh)

We define the BDM–DG mixed space. ::

  k = 2
  V = FunctionSpace(mesh, "BDM", k)
  Q = FunctionSpace(mesh, "DG", k-1, variant="integral")
  W = MixedFunctionSpace([V, Q])

The boundary conditions impose the no-slip and lid conditions on the
velocity component of the mixed space. ::

  Re = Constant(1)
  bcs = [DirichletBC(W.sub(0), 0, (1, 2, 3)),
         DirichletBC(W.sub(0), as_vector([16 * x**2 * (1-x)**2, 0]), (4,))]

We set up the solution function and test functions. ::

  w = Function(W, name="Solution")
  (u, p) = split(w)
  z = TestFunction(W)
  (v, q) = split(z)

Variational form
----------------

The full weak form consists of three groups of terms.  The first group
discretises the viscous term
:math:`-\nabla \cdot (2\,\mathrm{Re}^{-1}\varepsilon(u))` using a
symmetric interior penalty DG method: a consistency term, a symmetry
term, and a penalty term scaled by :math:`\sigma/h` (with
:math:`\sigma = 5(k+1)^2`).

The second group is the DG upwind discretisation of the convective term
:math:`\nabla \cdot (u \otimes u)`.  On interior facets the upwind flux
is :math:`\hat{F}(u) = \tfrac{1}{2}(u \cdot n + |u \cdot n|) u`.

The final group implements the pressure gradient, divergence constraint,
and augmented Lagrangian term. ::

  gamma = Constant(10000)
  sigma = Constant(5 * (k+1)**2)
  h = CellDiameter(mesh)
  uflux_int = 0.5*(dot(u, n) + abs(dot(u, n)))*u

  F = (
        2/Re                 * inner(sym(grad(u)), sym(grad(v)))*dx
      - 2/Re                 * inner(avg(sym(grad(u))), 2*avg(outer(v, n)))*dS
      - 2/Re                 * inner(avg(sym(grad(v))), 2*avg(outer(u, n)))*dS
      + 2/Re * sigma/avg(h)  * inner(avg(outer(u, n)), 2*avg(outer(v, n)))*dS
      -                        inner(u, div(outer(v, u)))*dx
      +                        inner(jump(uflux_int), jump(v))*dS
      -                        inner(p, div(v))*dx
      -                        inner(div(u), q)*dx
      + gamma                * inner(div(u), div(v))*dx  # The augmented Lagrangian term
      )

Boundary conditions are imposed weakly via two helper functions.
``a_bc`` contributes the viscous boundary terms (analogues of the
interior penalty terms restricted to boundary facets), and ``c_bc``
contributes the convective upwind flux on boundary facets. ::

  def a_bc(u, v, bid, g):
      return (
             - 2/Re * inner(outer(v, n), sym(grad(u)))*ds(bid)
             - 2/Re * inner(outer(u-g, n), sym(grad(v)))*ds(bid)
             + 1/Re * (sigma/h) * inner(v, u-g)*ds(bid)
             )

  def c_bc(u, v, bid, g):
      uflux_ext = 0.5*(dot(u, n) + abs(dot(u, n)))*u
      if g is not None:
          uflux_ext += 0.5*(dot(u, n) - abs(dot(u, n)))*g
      return inner(uflux_ext, v)*ds(bid)

We loop over the Dirichlet boundary conditions, adding the weak
boundary terms for each marked wall, and separately handle any
remaining exterior facets with a zero-inflow flux. ::

  exterior_markers = set(mesh.exterior_facets.unique_markers)
  for bc in bcs:
      g = bc.function_arg
      bid = bc.sub_domain
      if isinstance(bid, Iterable):
          [exterior_markers.remove(_) for _ in bid]
      else:
          exterior_markers.remove(bid)
      F += a_bc(u, v, bid, g)
      F += c_bc(u, v, bid, g)
  for bid in exterior_markers:
      F += c_bc(u, v, bid, None)

Solver
------

The outer solver is FGMRES (flexible GMRES is needed because the
preconditioner is nonlinear—it contains inner iterative solves)
preconditioned by a full Schur factorisation fieldsplit.

The velocity block uses one Richardson step with a full-cycle geometric
multigrid preconditioner. At each multigrid level, five steps of FGMRES
are applied, preconditioned by the additive Schwarz method with
vertex-star patches (:class:`~.ASMStarPC`).  A star patch around a
vertex consists of all cells sharing that vertex; these patches together
stably partition the divergence-free subspace, ensuring that the
smoother captures the kernel of :math:`\nabla \cdot` as required by
Schöberl's theory :cite:`Schoberl:1999`.  The coarse-grid problem is
solved exactly with LU factorisation.

The pressure block uses one Richardson step preconditioned by
:class:`~.MassInvPC`, which inverts the pressure mass matrix.  For the
integral-variant DG space the mass matrix is diagonal, so Jacobi is
exact. We set the scale factor of the mass matrix in `appctx["nu"]`
to :math:`-(2\,\mathrm{Re}^{-1} + \gamma)`,
coming from the Schur complement
approximation :math:`S_\gamma \approx (2\,\mathrm{Re}^{-1} + \gamma)^{-1} Q`,
so that :math:`S_\gamma^{-1} \approx (2\,\mathrm{Re}^{-1} + \gamma)\, Q^{-1}`. ::

  appctx = {"nu": -1 / (2/Re + gamma)}

  sp = {
      'mat_type': 'nest',
      'snes_monitor': None,
      'snes_converged_reason': None,
      'snes_max_it': 20,
      'snes_atol': 1e-8,
      'snes_rtol': 1e-12,
      'snes_stol': 1e-06,
      'ksp_type': 'fgmres',
      'ksp_converged_reason': None,
      'ksp_monitor_true_residual': None,
      'ksp_max_it': 500,
      'ksp_atol': 1e-08,
      'ksp_rtol': 1e-10,
      'pc_type': 'fieldsplit',
      'pc_fieldsplit_type': 'schur',
      'pc_fieldsplit_schur_factorization_type': 'full',
      'fieldsplit_0': {
          'ksp_convergence_test': 'skip',
          'ksp_max_it': 1,
          'ksp_norm_type': 'unpreconditioned',
          'ksp_richardson_self_scale': False,
          'ksp_type': 'richardson',
          'pc_type': 'mg',
          'pc_mg_type': 'full',
          'mg_coarse_mat_type': 'aij',
          'mg_coarse_pc_type': 'lu',
          'mg_coarse_pc_factor_mat_solver_type': 'mumps',
          'mg_levels': {
              'ksp_convergence_test': 'skip',
              'ksp_max_it': 5,
              'ksp_type': 'fgmres',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.ASMStarPC',
          },
      },
      'fieldsplit_1': {
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.MassInvPC',
          'Mp_pc_type': 'jacobi',
      },
  }

Solving over a range of Reynolds numbers
-----------------------------------------

We perform continuation in Reynolds number, using as initial guess the
converged solution from the previous :math:`\mathrm{Re}`.
We report the total Krylov iterations and the average
per Newton step, which should remain nearly constant as :math:`\mathrm{Re}` grows.  As a
diagnostic, we also print :math:`\|\nabla \cdot u\|_{L^2}`. Since the
discretisation is exactly incompressible, this should remain close to machine
precision. ::

  (u_, p_) = w.subfunctions
  u_.rename("Velocity")
  p_.rename("Pressure")
  pvd = VTKFile("output/navier_stokes.pvd")

  problem = NonlinearVariationalProblem(F, w, bcs)
  solver = NonlinearVariationalSolver(problem, solver_parameters=sp, appctx=appctx)

  for Re_ in [1, 100, 500] + list(range(1000, 5100, 500)):
      Re.assign(Re_)

      # Solve
      print(BLUE % f"Solving for Re = {Re_}")
      solver.solve()

      # Diagnostics
      linear_its = solver.snes.getLinearSolveIterations()
      nonlinear_its = solver.snes.getIterationNumber()
      print(f"  Krylov iterations: {linear_its} ({linear_its/nonlinear_its:.1f} per Newton step)")
      print(f"  ||div u||: {norm(div(u_), 'L2'):.2e}")
      pvd.write(u_, p_)

The average number of Krylov iterations per Newton step varies from 3.8--5.3 over the whole range of Reynolds numbers considered.

A python script version of this demo can be found :demo:`here
<reynolds_robust_navier_stokes_hdiv.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
