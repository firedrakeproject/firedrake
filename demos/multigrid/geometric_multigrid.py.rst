Using geometric multigrid solvers in Firedrake
==============================================

In addition to the full gamut of algebraic solvers offered by PETSc,
Firedrake also provides access to multilevel solvers with geometric
hierarchies.  In this demo, we will see how to use this
functionality.  We first solve the prototypical elliptic problem, the
Poisson equation.  We move on to a multi-field example, the Stokes
equations, demonstrating how the multigrid functionality composes with
fieldsplit preconditioning.

Creating a geometric hierarchy
------------------------------

Geometric multigrid requires a geometric hierarchy of meshes on which
the equations will be discretised.  To create a hierarchy, we use
:func:`~.MeshHierarchy` to create a hierarchy of meshes, the resulting
object remembers the relationships between them.  Currently, these
hierarchies are constructed using regular bisection refinement, so we
must create a coarse mesh. ::

  from firedrake import *

  mesh = UnitSquareMesh(8, 8)

Now we will create the mesh hierarchy, providing the coarse mesh and
the number of refinements we would like.  Here, we request four
refinements, going from 128 cells on the coarse mesh to 32768 cells on
the finest. ::

  hierarchy = MeshHierarchy(mesh, 4)

Defining the problem: the Poisson equation
------------------------------------------

Having defined the hierarchy we now need to set up our problem.  The
most transparent way to do this is to set up the problem on the finest
mesh, Firedrake then manages the rediscretised operators by providing
appropriate callbacks to PETSc.  In this way, we can control the
behaviour of the solver entirely through runtime options.  So our next
step is just to grab the finest mesh and define the problem.  ::

  mesh = hierarchy[-1]

  V = FunctionSpace(mesh, "CG", 1)

  u = TrialFunction(V)
  v = TestFunction(V)

  a = dot(grad(u), grad(v))*dx

  bcs = DirichletBC(V, zero(), (1, 2, 3, 4))

For a forcing function, we will use a product of sines such that we
know the exact solution and can compute an error. ::

  x, y = SpatialCoordinate(mesh)

  f = -0.5*pi*pi*(4*cos(pi*x) - 5*cos(pi*x*0.5) + 2)*sin(pi*y)

  L = f*v*dx

The exact solution is::

  exact = sin(pi*x)*tan(pi*x*0.25)*sin(pi*y)

We'll demonstrate a few different sets of solver parameters, so let's define a
function that takes in set of parameters and returns the solution ::

  def run_solve(parameters):
      u = Function(V)
      solve(a == L, u, bcs=bcs, solver_parameters=parameters)
      return u

and another to compute the error. ::

  def error(u):
      expect = Function(V).interpolate(exact)
      return norm(assemble(u - expect))

Specifying the solver
~~~~~~~~~~~~~~~~~~~~~

Let's start with our first test.  We'll confirm a working solve by
using a direct method. ::

  u = run_solve({"ksp_type": "preonly", "pc_type": "lu"})
  print('LU solve error', error(u))

Next we'll use the conjugate gradient method preconditioned by a
geometric multigrid V-cycle.  Firedrake automatically takes care of
rediscretising the operator on coarse grids, and providing the number
of levels to PETSc. ::

  u = run_solve({"ksp_type": "cg", "pc_type": "mg"})
  print('MG V-cycle + CG error', error(u))

For such a simple problem, an appropriately configured multigrid solve
can achieve algebraic error equal to discretisation error in one
cycle, without the application of a Krylov accelerator.  In
particular, for the Poisson equation with constant coefficients, a
single full multigrid cycle with appropriately chosen smoothers achieves
discretisation error.  As ever, PETSc allows us to configure the
appropriate settings using solver parameters. ::

  parameters = {
     "ksp_type": "preonly",
     "pc_type": "mg",
     "pc_mg_type": "full",
     "mg_levels_ksp_type": "chebyshev",
     "mg_levels_ksp_max_it": 2,
     "mg_levels_pc_type": "jacobi"
  }

  u = run_solve(parameters)
  print('MG F-cycle error', error(u))

A saddle-point system: The Stokes equations
-------------------------------------------

Having demonstrated basic usage, we'll now move on to an example where
the configuration of the multigrid solver is somewhat more complex.
This demonstrates how the multigrid functionality composes with the
other aspects of solver configuration, like fieldsplit
preconditioning.  We'll use Taylor-Hood elements and solve a problem
with specified velocity inflow and outflow conditions. ::

  mesh = RectangleMesh(15, 10, 1.5, 1)

  hierarchy = MeshHierarchy(mesh, 3)

  mesh = hierarchy[-1]

  V = VectorFunctionSpace(mesh, "CG", 2)
  W = FunctionSpace(mesh, "CG", 1)
  Z = V * W

  u, p = TrialFunctions(Z)
  v, q = TestFunctions(Z)
  nu = Constant(1)

  a = (nu*inner(grad(u), grad(v)) - p * div(v) + div(u) * q)*dx

  L = inner(Constant((0, 0)), v) * dx

  x, y = SpatialCoordinate(mesh)

  t = conditional(y < 0.5, y - 0.25, y - 0.75)
  l = 1.0/6.0
  gbar = conditional(Or(And(0.25 - l/2 < y,
  y < 0.25 + l/2),
  And(0.75 - l/2 < y,
  y < 0.75 + l/2)),
  Constant(1.0), Constant(0.0))

  value = gbar*(1 - (2*t/l)**2)
  inflowoutflow = Function(V).interpolate(as_vector([value, 0]))
  bcs = [DirichletBC(Z.sub(0), inflowoutflow, (1, 2)),
  DirichletBC(Z.sub(0), zero(2), (3, 4))]

First up, we'll use an algebraic preconditioner, with a direct solve,
remembering to tell PETSc to use pivoting in the factorisation. ::

  u = Function(Z)
  solve(a == L, u, bcs=bcs, solver_parameters={"ksp_type": "preonly",
                                               "pc_type": "lu",
                                               "pc_factor_shift_type": "inblocks",
                                               "ksp_monitor": None,
                                               "pmat_type": "aij"})

Next we'll use a Schur complement solver, using geometric multigrid to
invert the velocity block. The Schur complement is spectrally equivalent
to the viscosity-weighted pressure mass matrix. Since the pressure mass
matrix does not appear in the original form, we need to supply its
bilinear form to the solver ourselves: ::

  class Mass(AuxiliaryOperatorPC):

      def form(self, pc, test, trial):
          a = 1/nu * inner(test, trial)*dx
          bcs = None
          return (a, bcs)

  parameters = {
      "ksp_type": "gmres",
      "ksp_monitor": None,
      "pc_type": "fieldsplit",
      "pc_fieldsplit_type": "schur",
      "pc_fieldsplit_schur_fact_type": "lower",
      "fieldsplit_0_ksp_type": "preonly",
      "fieldsplit_0_pc_type": "mg",
      "fieldsplit_1_ksp_type": "preonly",
      "fieldsplit_1_pc_type": "python",
      "fieldsplit_1_pc_python_type": "__main__.Mass",
      "fieldsplit_1_aux_pc_type": "bjacobi",
      "fieldsplit_1_aux_sub_pc_type": "icc",
  }

  u = Function(Z)
  solve(a == L, u, bcs=bcs, solver_parameters=parameters)

Finally, we'll use coupled geometric multigrid on the full problem,
using Schur complement "smoothers" on each level. On the coarse grid
we use a full factorisation for the velocity and Schur complement
approximations, whereas on the finer levels we use incomplete
factorisations for the velocity block and Schur complement
approximations.

.. note::

   If we wanted to just use LU for the velocity-pressure system on the
   coarse grid we would have to say ``"mat_type": "aij"``, rather than
   ``"mat_type": "nest"``.

::

  parameters = {
        "ksp_type": "gcr",
        "ksp_monitor": None,
        "mat_type": "nest",
        "pc_type": "mg",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "fieldsplit",
        "mg_coarse_pc_fieldsplit_type": "schur",
        "mg_coarse_pc_fieldsplit_schur_fact_type": "full",
        "mg_coarse_fieldsplit_0_ksp_type": "preonly",
        "mg_coarse_fieldsplit_0_pc_type": "lu",
        "mg_coarse_fieldsplit_1_ksp_type": "preonly",
        "mg_coarse_fieldsplit_1_pc_type": "python",
        "mg_coarse_fieldsplit_1_pc_python_type": "__main__.Mass",
        "mg_coarse_fieldsplit_1_aux_pc_type": "cholesky",
        "mg_levels_ksp_type": "richardson",
        "mg_levels_ksp_max_it": 1,
        "mg_levels_pc_type": "fieldsplit",
        "mg_levels_pc_fieldsplit_type": "schur",
        "mg_levels_pc_fieldsplit_schur_fact_type": "upper",
        "mg_levels_fieldsplit_0_ksp_type": "richardson",
        "mg_levels_fieldsplit_0_ksp_convergence_test": "skip",
        "mg_levels_fieldsplit_0_ksp_max_it": 2,
        "mg_levels_fieldsplit_0_ksp_richardson_self_scale": None,
        "mg_levels_fieldsplit_0_pc_type": "bjacobi",
        "mg_levels_fieldsplit_0_sub_pc_type": "ilu",
        "mg_levels_fieldsplit_1_ksp_type": "richardson",
        "mg_levels_fieldsplit_1_ksp_convergence_test": "skip",
        "mg_levels_fieldsplit_1_ksp_richardson_self_scale": None,
        "mg_levels_fieldsplit_1_ksp_max_it": 3,
        "mg_levels_fieldsplit_1_pc_type": "python",
        "mg_levels_fieldsplit_1_pc_python_type": "__main__.Mass",
        "mg_levels_fieldsplit_1_aux_pc_type": "bjacobi",
        "mg_levels_fieldsplit_1_aux_sub_pc_type": "icc",
  }

  u = Function(Z)
  solve(a == L, u, bcs=bcs, solver_parameters=parameters)

Finally, we'll write the solution for visualisation with Paraview. ::

  u, p = u.subfunctions
  u.rename("Velocity")
  p.rename("Pressure")

  VTKFile("stokes.pvd").write(u, p)

A runnable python version of this demo can be found :demo:`here
<geometric_multigrid.py>`.
