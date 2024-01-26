from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER
import pytest


def run_poisson():
    deg = 4
    coarse_deg = 2
    test_params = {"snes_type": "ksponly",
                   "ksp_type": "cg",
                   "ksp_monitor": None,
                   "pc_type": "python",
                   "pc_python_type": "firedrake.P1PC",
                   "pmg_mg_levels": {
                       "ksp_type": "chebyshev",
                       "ksp_max_it": 2,
                       "pc_type": "jacobi"},
                   "pmg_mg_coarse": {
                       "degree": coarse_deg,
                       "ksp_type": "preonly",
                       "pc_type": "lu",
                       "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER
                   }}

    N = 2
    base_msh = UnitSquareMesh(N, N, quadrilateral=True)
    base_mh = MeshHierarchy(base_msh, 2)
    mh = ExtrudedMeshHierarchy(base_mh, height=1, base_layer=N)
    msh = mh[-1]

    V = FunctionSpace(msh, "S", deg)
    v = TestFunction(V)
    u = Function(V, name="Potential")
    gg = Function(V)

    bcs = [DirichletBC(V, gg, blah) for blah in ("on_boundary", "top", "bottom")]

    x, y, z = SpatialCoordinate(msh)
    uex = x * (1 - x) * y * (1 - y) * z * (1 - z) * exp(x)
    f = -div(grad(uex))

    F = inner(grad(u), grad(v))*dx - inner(f, v)*dx(metadata={"quadrature_degree": 2*deg})

    problem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=test_params)
    solver.solve()

    assert solver.snes.ksp.its <= 10
    ppc = solver.snes.ksp.pc.getPythonContext().ppc
    levels = ppc.getMGLevels()
    assert levels == 2

    def get_int_degree(context):
        V = context._problem.u.ufl_function_space()
        N = V.ufl_element().degree()
        try:
            N, = set(N)
        except TypeError:
            pass
        return N

    fine = solver._ctx
    coarse = fine._coarse
    assert get_int_degree(fine) == deg
    assert get_int_degree(coarse) == coarse_deg

    err = errornorm(uex, u)
    return err


@pytest.mark.skipcomplex
@pytest.mark.parallel
def test_poisson_gmg():
    assert run_poisson() < 1e-3
