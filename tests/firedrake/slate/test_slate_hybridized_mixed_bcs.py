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


def test_slate_hybridized_fieldsplit_bcs():
    mesh = UnitSquareMesh(6, 6)
    R = FunctionSpace(mesh, "DG", 0)
    RT = FunctionSpace(mesh, "RT", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    Z = R * RT * DG
    r, sigma, u = TrialFunctions(Z)
    s, tau, v = TestFunctions(Z)

    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

    a = (inner(r, s) + inner(sigma, tau) - inner(u, div(tau)) + inner(u, v) + inner(div(sigma), v)) * dx
    L = inner(f, v) * dx + inner(f, s) * dx
    bcs = [DirichletBC(Z.sub(1), zero(), "on_boundary")]

    # The hybridized block is the subspace Z.sub((1, 2)), so the boundary
    # condition on Z.sub(1) is on the H(div) space of that block.
    z = Function(Z)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'fieldsplit',
              'pc_fieldsplit_type': 'additive',
              'pc_fieldsplit_0_fields': '0',
              'pc_fieldsplit_1_fields': '1,2',
              'fieldsplit_0': {'ksp_type': 'preonly',
                               'pc_type': 'python',
                               'pc_python_type': 'firedrake.AssembledPC',
                               'assembled_pc_type': 'lu'},
              'fieldsplit_1': {'ksp_type': 'preonly',
                               'pc_type': 'python',
                               'pc_python_type': 'firedrake.HybridizationPC',
                               'hybridization': {'ksp_type': 'preonly',
                                                 'pc_type': 'lu'}}}
    solve(a == L, z, solver_parameters=params, bcs=bcs)

    z2 = Function(Z)
    solve(a == L, z2,
          solver_parameters={'pc_type': 'lu',
                             'mat_type': 'aij',
                             'ksp_type': 'preonly'}, bcs=bcs)

    for z_h, nh_z in zip(z.subfunctions, z2.subfunctions):
        assert errornorm(z_h, nh_z) < 1e-11


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
