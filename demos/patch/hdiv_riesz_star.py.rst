Using patch relaxation for H(div)
=================================

Contributed by `Robert Kirby <https://sites.baylor.edu/robert_kirby/>`_
and `Pablo Brubeck <https://www.maths.ox.ac.uk/people/pablo.brubeckmartinez/>`_.

Multigrid in H(div) and H(curl) also requires relaxation based on topological patches.
Here, we demonstrate how to do this in the former case. ::

  from firedrake import *

  dparams = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
  base = UnitCubeMesh(2, 2, 2, distribution_parameters=dparams)
  mh = MeshHierarchy(base, 3)
  mesh = mh[-1]

We consider the Riesz map on H(div), discretized with lowest order
Raviart--Thomas elements.  We force the system with a random right-hand side and
impose homogeneous Dirichlet boundary conditions::


  def run_solve(mesh, params):
      V = FunctionSpace(mesh, "RT", 1)
      u = TrialFunction(V)
      v = TestFunction(V)
      uh = Function(V)
      a = inner(div(u), div(v)) * dx + inner(u, v) * dx
      rg = RandomGenerator(PCG64(seed=123456789))
      L = rg.uniform(V.dual(), -1, 1)
      bcs = DirichletBC(V, 0, "on_boundary")

      problem = LinearVariationalProblem(a, L, uh, bcs)
      solver = LinearVariationalSolver(problem, solver_parameters=params)

      solver.solve()

      return solver.snes.getLinearSolveIterations()

Having done both :class:`~.ASMStarPC` and :class:`~.PatchPC` in other demos, here we simply opt for the former.
Arnold, Falk, and Winther show that either vertex (`construct_dim=0`) or edge patches (`construct_dim=1`)  will be acceptable in three dimensions. ::


  def mg_params(relax, mat_type="aij"):
      return {
          "mat_type": mat_type,
          "ksp_type": "cg",
          "pc_type": "mg",
          "mg_levels": {
              "ksp_type": "chebyshev",
              "ksp_max_it": 1,
              **relax
          },
          "mg_coarse": {
              "mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "cholesky"
          }
      }


  def asm_params(construct_dim):
      return {
          "pc_type": "python",
          "pc_python_type": "firedrake.ASMStarPC",
          "pc_star_construct_dim": construct_dim,
          "pc_star_backend": "tinyasm"
      }

Now, for each parameter choice, we report the iteration count for the Riesz map
over a range of meshes.  We see that vertex patches give lower iteration counts than
edge patches, but they are more expensive. ::


  for cdim in (0, 1):
      params = mg_params(asm_params(cdim))
      print(f"Relaxation with patches of dimension {cdim}")
      print("Level | Iterations")
      for lvl, msh in enumerate(mh[1:], start=1):
          its = run_solve(msh, params)
          print(f"{lvl}     | {its}")

For vertex patches, we expect output like,

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

and additional mesh refinement will lead to these numbers leveling off.

A runnable python version of this demo can be found :demo:`here<hdiv_riesz_star.py>`.
