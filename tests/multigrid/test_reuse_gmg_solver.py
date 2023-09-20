from firedrake import *
import pytest


def run_helmholtz(typ):
    if typ == "mg":
        parameters = {"ksp_type": "cg",
                      "pc_type": "mg",
                      "mg_levels_ksp_type": "chebyshev",
                      "mg_levels_ksp_max_it": 2,
                      "mg_levels_pc_type": "jacobi"}
    elif typ == "mgmatfree":
        parameters = {"ksp_type": "cg",
                      "mat_type": "matfree",
                      "pc_type": "mg",
                      "mg_coarse_ksp_type": "preonly",
                      "mg_coarse_pc_type": "python",
                      "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                      "mg_coarse_assembled_pc_type": "lu",
                      "mg_levels_ksp_type": "chebyshev",
                      "mg_levels_ksp_max_it": 2,
                      "mg_levels_pc_type": "jacobi"}
    else:
        raise RuntimeError("Unknown parameter set '%s' request", typ)

    mesh = UnitSquareMesh(10, 10)

    nlevel = 4

    mh = MeshHierarchy(mesh, nlevel)
    mesh = mh[-1]
    R = FunctionSpace(mesh, 'R', 0)
    V = FunctionSpace(mesh, 'CG', 1)
    v = TestFunction(V)
    u = TrialFunction(V)
    uh = function.Function(V)
    alpha = function.Function(R)

    # Choose a forcing function such that the exact solution is not an
    # eigenmode.  This stresses the preconditioner much more.  e.g. 10
    # iterations of ilu fails to converge this problem sufficiently.
    x = SpatialCoordinate(V.mesh())
    uexact = sin(pi*x[0])*tan(pi*x[0]*0.25)*sin(pi*x[1])

    # The problem is parametrized such that
    # alpha = 0 gives the mass matrix, and alpha = 1 gives Poisson
    a = inner(v, (1-alpha)*u)*dx + inner(grad(v), alpha*grad(u))*dx
    L = a(v, uexact)
    bcs = DirichletBC(V, 0.0, (1, 2, 3, 4))

    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters)
    for val in (0.0, 1.0):
        alpha.assign(val)
        uh.assign(0)
        solver.solve()
        its_reused = solver.snes.ksp.getIterationNumber()

    new_solver = LinearVariationalSolver(problem, solver_parameters=parameters)
    uh.assign(0)
    new_solver.solve()
    its_new = new_solver.snes.ksp.getIterationNumber()
    assert its_reused == its_new


@pytest.mark.parametrize("typ",
                         ["mg", "mgmatfree"])
def test_poisson_gmg(typ):
    run_helmholtz(typ)
