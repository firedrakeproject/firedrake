import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


@pytest.mark.parametrize('quad', [False, True])
def test_hybrid_extr_helmholtz(quad):
    """Hybridize the lowest order HDiv conforming method using
    both triangular prism and hexahedron elements.
    """
    base = UnitSquareMesh(5, 5, quadrilateral=quad)
    mesh = ExtrudedMesh(base, layers=5, layer_height=0.2)

    if quad:
        RT = FiniteElement("RTCF", quadrilateral, 1)
        DG_v = FiniteElement("DG", interval, 0)
        DG_h = FiniteElement("DQ", quadrilateral, 0)
        CG = FiniteElement("CG", interval, 1)

    else:
        RT = FiniteElement("RT", triangle, 1)
        DG_v = FiniteElement("DG", interval, 0)
        DG_h = FiniteElement("DG", triangle, 0)
        CG = FiniteElement("CG", interval, 1)

    HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                               HDiv(TensorProductElement(DG_h, CG)))
    V = FunctionSpace(mesh, HDiv_ele)
    U = FunctionSpace(mesh, "DG", 0)
    W = V * U

    x, y, z = SpatialCoordinate(mesh)
    f = Function(U)
    expr = (1+12*pi*pi)*cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z)
    f.interpolate(expr)

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    a = inner(sigma, tau)*dx + inner(u, v)*dx + inner(div(sigma), v)*dx - inner(u, div(tau))*dx
    L = inner(f, v)*dx
    w = Function(W)
    params = {
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': DEFAULT_DIRECT_SOLVER
        }
    }

    solve(a == L, w, solver_parameters=params)
    sigma_h, u_h = w.subfunctions

    w2 = Function(W)
    params2 = {'pc_type': 'fieldsplit',
               'pc_fieldsplit_type': 'schur',
               'ksp_type': 'cg',
               'ksp_rtol': 1e-8,
               'pc_fieldsplit_schur_fact_type': 'FULL',
               'fieldsplit_0': {'ksp_type': 'cg'},
               'fieldsplit_1': {'ksp_type': 'cg'}}
    solve(a == L, w2, solver_parameters=params2)
    nh_sigma, nh_u = w2.subfunctions

    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 5e-8
    assert u_err < 1e-8
