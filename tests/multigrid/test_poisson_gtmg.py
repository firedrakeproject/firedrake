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

    degree = 2
    RT = FunctionSpace(mesh, "RT", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = RT * DG

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    f = Function(DG)
    f.interpolate(-0.5*pi*pi*(4*cos(pi*x[0]) - 5*cos(pi*x[0]*0.5) + 2)*sin(pi*x[1]))

    a = (dot(sigma, tau) - div(tau)*u + v*div(sigma)) * dx
    L = f * v * dx

    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'mat_type': 'matfree',
                                'ksp_rtol': 1e-8,
                                'pc_type': 'python',
                                'pc_python_type': 'firedrake.GTMGPC',
                                'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                     'pc_type': 'bjacobi',
                                                     'sub_pc_type': 'ilu',
                                                     'ksp_max_it': 2},
                                       'mg_coarse': {'ksp_type': 'preonly',
                                                     'pc_type': 'mg',
                                                     'pc_mg_type': 'full',
                                                     'mg_levels': {'ksp_type': 'chebyshev',
                                                                   'pc_type': 'bjacobi',
                                                                   'sub_pc_type': 'ilu',
                                                                   'ksp_max_it': 2}}}}}
    appctx = {'get_coarse_operator': p1_callback,
              'get_coarse_space': get_p1_space,
              'coarse_space_bcs': get_p1_prb_bcs()}

    solve(a == L, w, solver_parameters=params, appctx=appctx)
    _, uh = w.split()
    exact = sin(pi*x[0])*tan(pi*x[0]*0.25)*sin(pi*x[1])
    return errornorm(exact, uh, norm_type="L2")


@pytest.mark.parallel
def test_mixed_poisson_gtmg():
    assert run_gtmg_mixed_poisson() < 3e-4
