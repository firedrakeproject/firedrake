import pytest
from firedrake import *


@pytest.mark.parametrize("subdomain", ["on_boundary",
                                       pytest.mark.xfail(reason="Subdomains not yet implemented in Slate", strict=True)((1, 2))])
@pytest.mark.parametrize(("degree", "hdiv_family", "quadrilateral"),
                         [(1, "RT", False), (1, "RTCF", True),
                          (2, "RT", False), (2, "RTCF", True)])
def test_slate_hybridized_on_boundary(degree, hdiv_family, quadrilateral, subdomain):
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
    a = (dot(sigma, tau) - div(tau) * u + u * v + v * div(sigma)) * dx
    L = f * v * dx

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner, pc_type slate.HybridizationPC
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'ksp_rtol': 1e-14},
                                'hdiv_projection': {'ksp_type': 'cg',
                                                    'ksp_rtol': 1e-14}}}
    bcs = [DirichletBC(W[0], Constant((0., 0.)), subdomain)]

    solve(a == L, w, solver_parameters=params, bcs=bcs)
    sigma_h, u_h = w.split()

    # (Non-hybrid)
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'pc_type': 'lu',
                             'mat_type': 'aij',
                             'ksp_type': 'preonly'}, bcs=bcs)
    nh_sigma, nh_u = w2.split()

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
    a = (dot(sigma, tau) - div(tau) * u + u * v + v * div(sigma)) * dx
    L = f * v * dx

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner, pc_type slate.HybridizationPC
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'ksp_rtol': 1e-14},
                                'hdiv_projection': {'ksp_type': 'cg',
                                                    'ksp_rtol': 1e-14}}}
    bcs = [DirichletBC(W[0], Constant((0., 0.)), "top"),
           DirichletBC(W[0], Constant((0., 0.)), "bottom")]
    solve(a == L, w, solver_parameters=params, bcs=bcs)
    sigma_h, u_h = w.split()

    # (Non-hybrid)
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'pc_type': 'lu',
                             'mat_type': 'aij',
                             'ksp_type': 'preonly'}, bcs=bcs)
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-11
    assert u_err < 1e-11


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
