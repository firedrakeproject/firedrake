Using fast diagonalisation solvers in Firedrake
===============================================

Contributed by `Pablo Brubeck <https://www.maths.ox.ac.uk/people/pablo.brubeckmartinez/>`_.

In this demo we show how to efficiently solve the Poisson equation using
high-order tensor-product elements. This is accomplished through a special
basis, obtained from the fast diagonalisation method (FDM). We first construct
an auxiliary operator that is sparse in this basis, with as many zeros as a
low-order method.  We then combine this with an additive Schwarz method.
Finally, we show how to do static condensation using fieldsplit. A detailed
description of these solvers is found in :cite:`Brubeck2022` and
:cite:`Brubeck2024`.


Creating an extruded mesh
-------------------------

The fast diagonalisation method produces a basis of discrete eigenfunctions.
These are polynomials, and can be efficiently computed on tensor
product-elements by solving an eigenproblem on the interval. Therefore, we will
require quadrilateral or hexahedral meshes.  Currently, the solver only supports
extruded hexahedral meshes, so we must create an :func:`~.ExtrudedMesh`. ::

  from firedrake import *

  base = UnitSquareMesh(8, 8, quadrilateral=True)
  mesh = ExtrudedMesh(base, 8)


Defining the problem: the Poisson equation
------------------------------------------

Having defined the mesh we now need to set up our problem.  The crucial step
for fast diagonalisation is a special choice of basis functions. We obtain them
by passing ``variant="fdm"`` to the :func:`~.FunctionSpace` constructor.  The
solvers in this demo work also with other element variants, but each iteration
would involve an additional a basis transformation.  To stress-test the solver,
we prescribe a random :class:`~.Cofunction` as right-hand side.

We'll demonstrate a few different sets of solver parameters, so let's define a
function that takes in set of parameters and uses them on a
:class:`~.LinearVariationalSolver`. ::


  def run_solve(degree, parameters):
      V = FunctionSpace(mesh, "Q", degree, variant="fdm")
      u = TrialFunction(V)
      v = TestFunction(V)
      uh = Function(V)
      a = inner(grad(u), grad(v)) * dx
      bcs = [DirichletBC(V, 0, sub) for sub in ("on_boundary", "top", "bottom")]

      rg = RandomGenerator(PCG64(seed=123456789))
      L = rg.uniform(V.dual(), -1, 1)

      problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
      solver = LinearVariationalSolver(problem, solver_parameters=parameters)
      solver.solve()
      return solver.snes.getLinearSolveIterations()

Specifying the solver
---------------------

The solver avoids the assembly of a matrix with dense element submatrices, and
instead applies a matrix-free conjugate gradient method with a preconditioner
obtained by assembling a sparse matrix.  This is done through the python type
preconditioner :class:`~.FDMPC`.  We define a function that enables us to
compose :class:`~.FDMPC` with an inner relaxation. ::


  def fdm_params(relax):
      return {
          "mat_type": "matfree",
          "ksp_type": "cg",
          "pc_type": "python",
          "pc_python_type": "firedrake.FDMPC",
          "fdm": relax,
      }

Let's start with our first test.  We'll confirm a working solve by
using a sparse direct LU factorization. ::


  lu_params = {
      "pc_type": "lu",
      "pc_factor_mat_solver_type": "mumps",
  }

  fdm_lu_params = fdm_params(lu_params)
  its = run_solve(5, fdm_lu_params)
  print(f"LU iterations {its}")


.. note::
    On this Cartesian mesh, the sparse operator constructed by :class:`~.FDMPC`
    corresponds to the original operator. This is no longer the case with non-Cartesian
    meshes or more general PDEs, as the FDM basis will fail to diagonalise the
    problem.  For such cases, :class:`~.FDMPC` will produce a sparse
    approximation of the original operator.

Moving on to a more complicated solver, we'll employ a two-level solver with
the lowest-order coarse space via :class:`~.P1PC`.  As the fine level
relaxation we define an additive Schwarz method on vertex-star patches
implemented via :class:`~.ASMExtrudedStarPC` as we have an extruded mesh::

  asm_params = {
      "pc_type": "python",
      "pc_python_type": "firedrake.P1PC",
      "pmg_mg_coarse_mat_type": "aij",
      "pmg_mg_coarse": lu_params,
      "pmg_mg_levels": {
          "ksp_max_it": 1,
          "ksp_type": "chebyshev",
          "pc_type": "python",
          "pc_python_type": "firedrake.ASMExtrudedStarPC",
          "sub_sub_pc_type": "lu",
      },
  }

  fdm_asm_params = fdm_params(asm_params)

  print("FDM + ASM")
  print("Degree\tIterations")
  for degree in range(3, 6):
      its = run_solve(degree, fdm_asm_params)
      print(f"{degree}\t{its}")

We observe degree-independent iteration counts:

======== ============
 Degree   Iterations
======== ============
  3        13
  4        13
  5        12
======== ============

Static condensation
-------------------

Finally, we construct :class:`~.FDMPC` solver parameters using static
condensation.  The fast diagonalisation basis diagonalises the operator on cell
interiors. So we define a solver that splits the interior and facet degrees of
freedom via :class:`~.FacetSplitPC` and fieldsplit options.  We set the option
``fdm_static_condensation`` to tell :class:`~.FDMPC` to assemble a 2-by-2 block
preconditioner where the lower-right block is replaced by the Schur complement
resulting from eliminating the interior degrees of freedom.  The Krylov
solver is posed on the full set of degrees of freedom, and the preconditioner
applies a symmetrized multiplicative sweep on the interior and the facet
degrees of freedom. In general, we are not able to fully eliminate the
interior, as the sparse operator constructed by :class:`~.FDMPC` is only an
approximation on non-Cartesian meshes.  We apply point-Jacobi on the interior
block, and the two-level additive Schwarz method on the facets. ::


  def fdm_static_condensation_params(relax):
      return {
          "mat_type": "matfree",
          "ksp_type": "cg",
          "pc_type": "python",
          "pc_python_type": "firedrake.FacetSplitPC",
          "facet_pc_type": "python",
          "facet_pc_python_type": "firedrake.FDMPC",
          "facet_fdm_static_condensation": True,
          "facet_fdm_pc_use_amat": False,
          "facet_fdm_pc_type": "fieldsplit",
          "facet_fdm_pc_fieldsplit_type": "symmetric_multiplicative",
          "facet_fdm_fieldsplit_ksp_type": "preonly",
          "facet_fdm_fieldsplit_0_pc_type": "jacobi",
          "facet_fdm_fieldsplit_1": relax,
      }


  fdm_sc_asm_params = fdm_static_condensation_params(asm_params)

  print('FDM + SC + ASM')
  print("Degree\tIterations")
  for degree in range(3, 6):
      its = run_solve(degree, fdm_sc_asm_params)
      print(f"{degree}\t{its}")

We also observe degree-independent iteration counts:

======== ============
 Degree   Iterations
======== ============
  3        10
  4        10
  5        10
======== ============

A runnable python version of this demo can be found :demo:`here
<fast_diagonalisation_poisson.py>`.


.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
