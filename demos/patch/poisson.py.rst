Using patch relaxation for multigrid
====================================

Simple relaxation like point Jacobi are not optimal or even suitable
smoothers for all applications.  Firedrake supports additive Schwarz methods
based on local patch-based decompositions through two different paths.

This demonstration illustrates basic usage of these methods for the Poisson
problem.  Here, multigrid with point Jacobi relaxation works, but the iteration
count degrades with polynomial degree, while vertex star patches give
degree-indendent iteration counts.

For many problems, point Jacobi is even worse, and patches are required even to
get a convergent method.  We refer the reader to other demos.

We start by importing firedrake and setting up a mesh hierarchy and the
exact solution and forcing data.::

  from firedrake import *

  mesh = UnitSquareMesh(4, 4)
  mh = MeshHierarchy(mesh, 1)
  mesh = mh[-1]
  x, y = SpatialCoordinate(mesh)

Next, this function solves the Poisson equation discretized with
a user-provided degree of Lagrange elements and set of solver
parameters and returns the iteration count required for convergence.
To stress-test the solver, the forcing function is taken as a randomly
generated cofunction.::

  def run_solve(deg, params):
      V = FunctionSpace(mesh, "CG", deg)
      u = TrialFunction(V)
      v = TestFunction(V)
      uu = Function(V)
      a = inner(grad(u), grad(v)) * dx
      rg = RandomGenerator(PCG64(seed=123456789))
      L = rg.uniform(V.dual(), -1, 1)
      bcs = DirichletBC(V, 0, "on_boundary")

      problem = LinearVariationalProblem(a, L, uu, bcs)
      solver = LinearVariationalSolver(problem, solver_parameters=params)

      solver.solve()

      return solver.snes.getLinearSolveIterations()


These two dictionaries specify parameters for sparse direct method, to be used
on the coarsest level of the multigrid hierarchy.::

  lu = {
      "ksp_type": "preonly",
      "pc_type": "lu"
  }

When we use a matrix-free method, there will not be an assembled matrix to factor
This forces the matrix to be assembled.::

  assembled_lu = {
      "ksp_type": "preonly",
      "pc_type": "python",
      "pc_python_type": "firedrake.AssembledPC",
      "assembled": lu
  }

This function creates multigrid parameters using a given set of
relaxation options and matrix assembled type.::

  def mg_params(relax, mat_type="aij"):
      if mat_type == "aij":
          coarse = lu
      else:
          coarse = assembled_lu

      return {
          "mat_type": mat_type,
          "ksp_type": "cg",
          "pc_type": "mg",
          "mg_levels": {
              "ksp_type": "chebyshev",
              "ksp_max_it": 1,
              **relax
          },
          "mg_coarse": coarse
      }

The simplest parameter case will use point Jacobi smoothing on each level.
Here, a matrix-free implementation is appropriate, and Firedrake will
automatically assembly the diagonal for us.
Point Jacobi, however, will require more multigrid iterations as the polynomial
degree increases.::

  jacobi_relax = mg_params({"pc_type": "jacobi"}, mat_type="matfree")

These options specify an additive Schwarz relaxation through PatchPC.
PatchPC builds the patch operators by assembling the bilineary form over
each subdomain.  Hence, it does not require the global stiffness
matrix to be assembled.::

  patch_relax = mg_params(
      {"pc_type": "python",
       "pc_python_type": "firedrake.PatchPC",
       "patch": {

These two options specify the star (all cells that contain) around each vertex::

           "pc_patch_construct_type": "star",
           "pc_patch_construct_dim": 0,

Store the local patch matrices as dense matrices::

           "pc_patch_sub_mat_type": "seqdense",

Solve the local patch problems with LU factorization.::

           "sub_ksp_type": "preonly",
           "sub_pc_type": "lu",

These options tell the system to precompute the patch matrices, save them,
and keep the inverses as dense matrices.::

           "pc_patch_dense_inverse": True,
           "pc_patch_save_operators": True,
           "pc_patch_precompute_element_tensors": None}},
      mat_type="matfree")

ASMStarPC, on the other hand, does no re-discretization, but extracts the
patch operators for each patch from the already-assembled global stiffness matrix.::

  asm_relax = mg_params(
      {"pc_type": "python",
       "pc_python_type": "firedrake.ASMStarPC",

The tinyasm backend uses LAPACK to invert all the patch operators.  If this option
is not specified, PETSc's ASM framework will set up a small KSP for each patch.
This can be useful when the patches become larger and one wants to use a sparse
direct or Krylov method on each one.::

      "pc_star_backend_type": "tinyasm"
       })

Now, for each parameter choice, we report the iteration count for the Poisson problem
over a range of polynomial degrees.  We see that the Jacobi relaxation leads to growth
in iteration count, while both PatchPC and ASMStarPC do not.  Mathematically, the two
latter options do the same operations, just via different code paths.::

  names = {"Jacobi": jacobi_relax,
           "Patch": patch_relax,
           "ASM Star": asm_relax}

  for name, method in names.items():
      print(name)
      print("Degree | Iterations")
      print("-------------------")
      for deg in range(1, 8):
          its = run_solve(deg, method)
          print(f"     {deg} |    {its}")

A runnable python version of this demo can be found :demo:`here<poisson.py>`.
