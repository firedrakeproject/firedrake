Using patch relaxation for H(div) and H(curl)
=============================================

Multigrid in H(div) and H(curl) also requires relaxation based on topological patches.
Here, we demonstrate how to do this in a few cases.::

  from firedrake import *

  mesh = UnitCubeMesh(2, 2, 2)
  mh = MeshHierarchy(mesh, 3)
  mesh = mh[-1]

First, we consider the Riesz map on H(div), discretized with lowest order
Raviart--Thomas elements.  We force the system with a random right-hand side and
impose homogeneous Dirichlet boundary conditions::
  
  def run_solve(mesh, params):
      V = FunctionSpace(mesh, "RT", 1)
      u = TrialFunction(V)
      v = TestFunction(V)
      uu = Function(V)
      a = inner(div(u), div(v)) * dx + inner(u, v) * dx
      rg = RandomGenerator(PCG64(seed=123456789))
      L = rg.uniform(V.dual(), -1, 1)
      bcs = DirichletBC(V, 0, "on_boundary")

      problem = LinearVariationalProblem(a, L, uu, bcs)
      solver = LinearVariationalSolver(problem, solver_parameters=params)

      solver.solve()

      return solver.snes.getLinearSolveIterations()

Having done both ASMStarPC and PatchPC in other demos, here we simply opt for the former.
Arnold, Falk, and Winther show that either vertex (`construct_dim=0`) or edge patches (`construct_dim=1`)  will be acceptable in three dimensions.::

  def asm_params(construct_dim):
      return {
          "ksp_type": "cg",
	  "pc_type": "mg",
	  "mg_levels": {
	      "ksp_type": "chebyshev",
	      "ksp_max_it": 1,
	      "pc_type": "python",
              "pc_python_type": "firedrake.ASMStarPC",
              "pc_star_construct_dim": construct_dim,
              "pc_star_backend_type": "tinyasm"
	  },
	  "mg_coarse": {
	      "ksp_type": "preonly",
	      "pc_type": "cholesky",
	  }
      }

Now, for each parameter choice, we report the iteration count for the Poisson problem
over a range of polynomial degrees.  We see that the Jacobi relaxation leads to growth
in iteration count, while both PatchPC and ASMStarPC do not.  Mathematically, the two
latter options do the same operations, just via different code paths.::

  for cdim in (0, 1):
      print(f"Relaxation with patches of dimension {cdim}")
      print("Level | Iterations")
      for lvl, msh in enumerate(mh[1:], start=1):
          its = run_solve(msh, asm_params(cdim))
          print(f"{lvl}     | {its}")

For vertex patches, we expect output of the form
======== ============
 Level    Iterations    
======== ============
  1        9
  2        10
  3        13
======== ============

and with edge patches
======== ============
 Level    Iterations    
======== ============
  1        25
  2        29
  3        32
======== ============

and we expect further leveling off with mesh refinement.
	  
A runnable python version of this demo can be found :demo:`here<riesz.py>`.
