import pytest
import numpy
from firedrake import *


def stokes_solver(mesh, mu):
    """Set up a Stokes solver preconditioned by a monolithic multigrid cycle.

    The relaxation on each level is an additive fieldsplit.  It applies Jacobi
    to the velocity block, and the inverse of a viscosity-weighted mass matrix
    to the pressure block.  Both blocks depend on the viscosity ``mu``, which
    the multigrid hierarchy obtains by coarsening the application context.
    """
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q

    up = Function(Z, name="solution")
    u, p = split(up)
    v, q = TestFunctions(Z)

    x, y = SpatialCoordinate(mesh)
    f = as_vector([sin(pi*x)*cos(pi*y), -cos(pi*x)*sin(pi*y)])
    F = (inner(2*mu*sym(grad(u)), sym(grad(v)))
         - inner(p, div(v))
         - inner(div(u), q)
         - inner(f, v))*dx

    # The pressure is fixed by the natural boundary condition on side 4.
    bcs = [DirichletBC(Z.sub(0), 0, (1, 2, 3))]

    parameters = {
        "mat_type": "aij",
        "snes_type": "ksponly",
        "ksp_type": "fgmres",
        "ksp_rtol": 1E-8,
        "pc_type": "mg",
        "mg_levels_ksp_type": "gmres",
        "mg_levels_ksp_max_it": 4,
        "mg_levels_ksp_convergence_test": "skip",
        "mg_levels_pc_type": "fieldsplit",
        "mg_levels_pc_fieldsplit_type": "additive",
        "mg_levels_fieldsplit_0_ksp_type": "preonly",
        "mg_levels_fieldsplit_0_pc_type": "jacobi",
        "mg_levels_fieldsplit_1_ksp_type": "preonly",
        "mg_levels_fieldsplit_1_pc_type": "python",
        "mg_levels_fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
        "mg_levels_fieldsplit_1_Mp_pc_type": "jacobi",
        "mg_coarse_pc_type": "lu",
        "mg_coarse_pc_factor_mat_solver_type": "mumps",
    }

    # MassInvPC reads the viscosity from the key "mu".
    appctx = {"mu": mu}
    problem = NonlinearVariationalProblem(F, up, bcs=bcs)
    return NonlinearVariationalSolver(problem, solver_parameters=parameters,
                                      appctx=appctx)


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 3])
def test_stokes_appctx_coarsening():
    base = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(base, 2)
    mesh = mh[-1]

    DG = FunctionSpace(mesh, "DG", 0)
    rg = RandomGenerator(PCG64(seed=123456789))
    mu = rg.uniform(DG, 1.0, 2.0)
    mu.rename("mu")
    # Injection must carry a genuinely variable viscosity down the hierarchy,
    # so check that the random sample does vary.
    with mu.dat.vec_ro as v:
        assert v.max()[1] - v.min()[1] > 0.5

    solver = stokes_solver(mesh, mu)
    solver.solve()
    assert solver.snes.ksp.getIterationNumber() < 40

    # Every coarse level must hold its own viscosity, injected from the level
    # above.
    transfer = TransferManager()
    ctx = solver._ctx
    fine_mu = mu
    for level in reversed(range(len(mh) - 1)):
        ctx = ctx._coarse
        coarse_mu = ctx.appctx["mu"]
        assert coarse_mu.function_space().mesh() is mh[level]

        expected = Function(coarse_mu.function_space())
        transfer.inject(fine_mu, expected)
        assert numpy.allclose(coarse_mu.dat.data_ro, expected.dat.data_ro)
        fine_mu = coarse_mu
