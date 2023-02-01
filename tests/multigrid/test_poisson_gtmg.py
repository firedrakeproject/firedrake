from firedrake import *
import pytest


def run_gtmg_mixed_poisson():

    m = UnitSquareMesh(10, 10)
    nlevels = 2
    mh = MeshHierarchy(m, nlevels)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)

    def get_p1_space():
        return FunctionSpace(mesh, "CG", 1)

    def get_p1_prb_bcs():
        return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")

    def p1_callback():
        P1 = get_p1_space()
        p = TrialFunction(P1)
        q = TestFunction(P1)
        return inner(grad(p), grad(q))*dx

    degree = 1
    RT = FunctionSpace(mesh, "RT", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = RT * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    f = Function(DG)
    f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    a = (inner(sigma, tau) - inner(u, div(tau)) + inner(div(sigma), v))*dx
    L = inner(f, v)*dx

    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'mat_type': 'matfree',
                                'pc_type': 'python',
                                'pc_python_type': 'firedrake.GTMGPC',
                                'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                     'pc_type': 'jacobi',
                                                     'ksp_max_it': 3},
                                       'mg_coarse': {'ksp_type': 'preonly',
                                                     'pc_type': 'mg',
                                                     'pc_mg_type': 'full',
                                                     'mg_levels': {'ksp_type': 'chebyshev',
                                                                   'pc_type': 'jacobi',
                                                                   'ksp_max_it': 3}}}}}
    appctx = {'get_coarse_operator': p1_callback,
              'get_coarse_space': get_p1_space,
              'coarse_space_bcs': get_p1_prb_bcs()}

    solve(a == L, w, solver_parameters=params, appctx=appctx)
    _, uh = w.subfunctions

    # Analytical solution
    f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

    return errornorm(f, uh, norm_type="L2")


def run_gtmg_scpc_mixed_poisson():

    m = UnitSquareMesh(10, 10)
    nlevels = 2
    mh = MeshHierarchy(m, nlevels)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)

    def get_p1_space():
        return FunctionSpace(mesh, "CG", 1)

    def get_p1_prb_bcs():
        return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")

    def p1_callback():
        P1 = get_p1_space()
        p = TrialFunction(P1)
        q = TestFunction(P1)
        return inner(grad(p), grad(q))*dx

    degree = 1
    n = FacetNormal(mesh)
    U = FunctionSpace(mesh, "DRT", degree)
    V = FunctionSpace(mesh, "DG", degree - 1)
    T = FunctionSpace(mesh, "DGT", degree - 1)
    W = U * V * T

    sigma, u, lambdar = TrialFunctions(W)
    tau, v, gammar = TestFunctions(W)

    f = Function(V)
    f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])

    a = (inner(sigma, tau)*dx - inner(u, div(tau))*dx
         + inner(div(sigma), v)*dx
         + inner(lambdar('+'), jump(tau, n=n))*dS
         # Multiply transmission equation by -1 to ensure
         # SCPC produces the SPD operator after statically
         # condensing
         - inner(jump(sigma, n=n), gammar('+'))*dS)
    L = inner(f, v)*dx

    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.SCPC',
              'pc_sc_eliminate_fields': '0, 1',
              'condensed_field': {'ksp_type': 'cg',
                                  'mat_type': 'matfree',
                                  'pc_type': 'python',
                                  'pc_python_type': 'firedrake.GTMGPC',
                                  'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                       'pc_type': 'jacobi',
                                                       'ksp_max_it': 3},
                                         'mg_coarse': {'ksp_type': 'preonly',
                                                       'pc_type': 'mg',
                                                       'pc_mg_type': 'full',
                                                       'mg_levels': {'ksp_type': 'chebyshev',
                                                                     'pc_type': 'jacobi',
                                                                     'ksp_max_it': 3}}}}}
    appctx = {'get_coarse_operator': p1_callback,
              'get_coarse_space': get_p1_space,
              'coarse_space_bcs': get_p1_prb_bcs()}

    bcs = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

    solve(a == L, w, bcs=bcs, solver_parameters=params, appctx=appctx)
    _, uh, _ = w.subfunctions

    # Analytical solution
    f.interpolate(x[0]*(1-x[0])*x[1]*(1-x[1]))

    return errornorm(f, uh, norm_type="L2")


@pytest.mark.skipcomplexnoslate
def test_mixed_poisson_gtmg():
    assert run_gtmg_mixed_poisson() < 1e-5


@pytest.mark.skipcomplexnoslate
def test_scpc_mixed_poisson_gtmg():
    assert run_gtmg_scpc_mixed_poisson() < 1e-5
