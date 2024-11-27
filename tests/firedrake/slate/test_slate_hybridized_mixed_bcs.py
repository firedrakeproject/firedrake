import pytest
from firedrake import *


@pytest.mark.parametrize(("degree", "hdiv_family", "quadrilateral"),
                         [(1, "RT", False), (1, "RTCF", True),
                          (2, "RT", False), (2, "RTCF", True)])
def test_slate_hybridized_on_boundary(degree, hdiv_family, quadrilateral):
    # Create a mesh
    mesh = UnitSquareMesh(6, 6, quadrilateral=quadrilateral)
    RT = FunctionSpace(mesh, hdiv_family, degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = RT * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

    # Define the variational forms
    a = (inner(sigma, tau) - inner(u, div(tau)) + inner(u, v) + inner(div(sigma), v)) * dx
    L = inner(f, v) * dx

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner, pc_type slate.HybridizationPC
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu'}}
    bcs = [DirichletBC(W[0], zero(), "on_boundary")]

    solve(a == L, w, solver_parameters=params, bcs=bcs)
    sigma_h, u_h = w.subfunctions

    # (Non-hybrid)
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'pc_type': 'lu',
                             'mat_type': 'aij',
                             'ksp_type': 'preonly'}, bcs=bcs)
    nh_sigma, nh_u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-11
    assert u_err < 1e-11


@pytest.mark.parametrize(("degree", "hdiv_family"),
                         [(1, "RTCF"),
                          (2, "RTCF")])
def test_slate_hybridized_extruded_bcs(degree, hdiv_family):
    # Create a mesh
    m = UnitIntervalMesh(6)
    mesh = ExtrudedMesh(m, 6)
    RT = FunctionSpace(mesh, hdiv_family, degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = RT * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

    # Define the variational forms
    a = (inner(sigma, tau) - inner(u, div(tau)) + inner(u, v) + inner(div(sigma), v)) * dx
    L = inner(f, v) * dx

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner, pc_type slate.HybridizationPC
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu'}}
    bcs = [DirichletBC(W[0], zero(), "top"),
           DirichletBC(W[0], zero(), "bottom")]
    solve(a == L, w, solver_parameters=params, bcs=bcs)
    sigma_h, u_h = w.subfunctions

    # (Non-hybrid)
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'pc_type': 'lu',
                             'mat_type': 'aij',
                             'ksp_type': 'preonly'}, bcs=bcs)
    nh_sigma, nh_u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-11
    assert u_err < 1e-11
