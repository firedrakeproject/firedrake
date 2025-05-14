Using patch relaxation for H(div)
=================================

Contributed by `Robert Kirby <https://sites.baylor.edu/robert_kirby/>`_
and `Pablo Brubeck <https://www.maths.ox.ac.uk/people/pablo.brubeckmartinez/>`_.

Multigrid in H(div) and H(curl) also requires relaxation based on topological patches.
Here, we demonstrate how to do this in the latter case.::

  from firedrake import *

  mesh = UnitCubeMesh(2, 2, 2)
  mh = MeshHierarchy(mesh, 3)
  mesh = mh[-1]

We consider the Riesz map on H(div), discretized with lowest order
Nedelec elements.  We force the system with a random right-hand side and
impose homogeneous Dirichlet boundary conditions::


  def run_solve(mesh, params):
      V = FunctionSpace(mesh, "N1curl", 1)
      u = TrialFunction(V)
      v = TestFunction(V)
      uu = Function(V)
      a = inner(curl(u), curl(v)) * dx + inner(u, v) * dx
      rg = RandomGenerator(PCG64(seed=123456789))
      L = rg.uniform(V.dual(), -1, 1)
      bcs = DirichletBC(V, 0, "on_boundary")

      problem = LinearVariationalProblem(a, L, uu, bcs)
      solver = LinearVariationalSolver(problem, solver_parameters=params)

      solver.solve()

      return solver.snes.getLinearSolveIterations()

Having done both :class:`~.ASMStarPC` and :class:`~.PatchPC` in other demos,
here we simply opt for the former. Arnold, Falk, and Winther show that vertex
patches yield a robust method.::


  def mg_params(relax):
      return {
          "ksp_type": "cg",
          "pc_type": "mg",
          "mg_levels": {
               "ksp_type": "chebyshev",
               "ksp_max_it": 1,
                **relax
          },
          "mg_coarse": {
              "ksp_type": "preonly",
               "pc_type": "cholesky"
          }
      }


  def asm_params(construct_dim):
      return {
	        "pc_type": "python",
     "pc_python_type": "firedrake.ASMStarPC",
     "pc_star_construct_dim": construct_dim,
     "pc_star_backend_type": "tinyasm"
     }

Hiptmair proposed a finer space decomposition for Nedelec elements using edge
patches and vertex patches on the gradient of a Lagrange space. The python type
preconditioner :class:`~.HiptmairPC` automatically sets up a two-level method
using the auxiliary Lagrange space in a multigrid hierarchy. ::


  def hiptmair_params():
      return {
           "pc_type": "python",
     "pc_python_type": "firedrake.HiptmairPC",
     "hiptmair_mg_levels": asm_params(1),
     "hiptmair_mg_coarse": asm_params(0),
      }


Now, for each parameter choice, we report the iteration count for the Riesz map
over a range of meshes.  We see that the auxiliary space approach gives lower
iteration counts than vertex patches, while being cheaper to invert.::

  names = {
     "Vertex Star": mg_params(asm_params(0)),
     "Hiptmair": mg_params(hiptmair_params()),
  }

  for name, parameters in names.items():
      print(f"{name}")
      print("Level | Iterations")
      for lvl, msh in enumerate(mh[1:], start=1):
          its = run_solve(msh, parameters)
          print(f"{lvl}     | {its}")

For vertex patches, we expect output like,

======== ============
 Level    Iterations
======== ============
  1        10
  2        14
  3        16
======== ============

and with Hiptmair (edge patches + vertex patches on gradients of H1)

======== ============
 Level    Iterations
======== ============
  1        10
  2        12
  3        13
======== ============

and additional mesh refinement will lead to these numbers leveling off.

A runnable python version of this demo can be found :demo:`here<hdiv_riesz_star.py>`.
