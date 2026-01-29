from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER
import pytest


@pytest.mark.parametrize(("degree", "hdiv_family"),
                         [(1, "RT"), (1, "BDM")])
def test_darcy_flow_hybridization(degree, hdiv_family):
    """
    Solves Darcy's equation:

    sigma - grad(u) = 0; div(sigma) = -f, in [0, 1] x [0, 1]
    u = 0 on {(0, y).union(1, y)}
    sigma.n = sin(5x) on {(0, x).union(1, x)}
    """
    # Create a mesh
    mesh = UnitSquareMesh(6, 6)
    U = FunctionSpace(mesh, hdiv_family, degree)
    V = FunctionSpace(mesh, "DG", degree - 1)
    W = U * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    n = FacetNormal(mesh)

    # Define the source function
    x, y = SpatialCoordinate(mesh)
    f = Function(V)
    f.interpolate(10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

    # Define the variational forms
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
    L = -inner(f, v) * dx + Constant(0.0) * inner(n, tau) * (ds(3) + ds(4))

    # Compare hybridized solution with non-hybridized
    w = Function(W)
    bc1 = DirichletBC(W[0], as_vector([0.0, -sin(5*x)]), 1)
    bc2 = DirichletBC(W[0], as_vector([0.0, sin(5*y)]), 2)
    bcs = [bc1, bc2]

    hybrid_params = {'mat_type': 'matfree',
                     'ksp_type': 'preonly',
                     'pc_type': 'python',
                     'pc_python_type': 'firedrake.HybridizationPC',
                     'hybridization': {'ksp_type': 'preonly',
                                       'pc_type': 'lu'}}
    solve(a == L, w, bcs=bcs, solver_parameters=hybrid_params)
    sigma_h, u_h = w.subfunctions

    w2 = Function(W)
    sc_params = {'mat_type': 'aij',
                 'ksp_type': 'preonly',
                 'pc_type': 'lu',
                 'pc_factor_mat_solver_type': DEFAULT_DIRECT_SOLVER}
    solve(a == L, w2, bcs=bcs, solver_parameters=sc_params)
    nh_sigma, nh_u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8
