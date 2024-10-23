from firedrake import *
import pytest
import numpy as np


# This test solves a mixed formulation of the Poisson equation with
# inhomogeneous Neumann boundary conditions such that the exact
# solution is p(x, y) = x - 0.5.  First on a 2D mesh, and then again
# on a 2D mesh embedded in 3D.
def run_no_manifold():
    mesh = UnitSquareMesh(3, 3)
    x = SpatialCoordinate(mesh)

    V0 = FunctionSpace(mesh, "RT", 2)
    V1 = FunctionSpace(mesh, "DG", 1)

    V = V0 * V1

    bc_arg = Function(V0).project(Constant((-1, 0)))
    bc = DirichletBC(V.sub(0), bc_arg, (1, 2, 3, 4))

    u, p = TrialFunctions(V)
    v, q = TestFunctions(V)

    a = (inner(u, v) - inner(p, div(v)) - inner(div(u), q))*dx

    f = Function(V1)
    f.assign(0)
    L = -inner(f, q)*dx

    up = Function(V)

    nullspace = MixedVectorSpaceBasis(V, [V.sub(0), VectorSpaceBasis(constant=True)])

    params = {'ksp_type': 'gmres',
              'ksp_view': None,
              'pc_type': 'svd',
              'ksp_converged_reason': None,
              'ksp_monitor': None}
    solve(a == L, up, bcs=bc, nullspace=nullspace, solver_parameters=params)
    exact = Function(V1).interpolate(x[0] - 0.5)

    u, p = up.subfunctions
    assert errornorm(exact, p, degree_rise=0) < 1e-8


def run_manifold():
    # Make a mesh embedded in 3D with zero z coordinate.
    mesh = UnitSquareMesh(3, 3)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    X = Function(V)
    x, y = SpatialCoordinate(mesh)
    X.interpolate(as_vector([x, y, 0]))
    mesh = Mesh(X)
    mesh.init_cell_orientations(Constant((0, 0, 1)))
    x_n = SpatialCoordinate(mesh)
    V0 = FunctionSpace(mesh, "RT", 2)
    V1 = FunctionSpace(mesh, "DG", 1)

    V = V0 * V1

    bc_arg = Function(V0).project(Constant((-1, 0, 0)))
    bc = DirichletBC(V.sub(0), bc_arg, (1, 2, 3, 4))

    u, p = TrialFunctions(V)
    v, q = TestFunctions(V)

    a = (inner(u, v) - inner(p, div(v)) - inner(div(u), q))*dx

    f = Function(V1)
    f.assign(0)
    L = -inner(f, q)*dx

    up = Function(V)

    nullspace = MixedVectorSpaceBasis(V, [V.sub(0), VectorSpaceBasis(constant=True)])

    params = {'ksp_type': 'gmres',
              'pc_type': 'svd',
              'ksp_converged_reason': None,
              'ksp_monitor': None}
    solve(a == L, up, bcs=bc, nullspace=nullspace, solver_parameters=params)
    exact = Function(V1).interpolate(x_n[0] - 0.5)

    u, p = up.subfunctions
    assert errornorm(exact, p, degree_rise=0) < 1e-8


def test_no_manifold_serial():
    run_no_manifold()


def test_manifold_serial():
    run_manifold()


@pytest.mark.parallel(nprocs=2)
def test_no_manifold_parallel():
    run_no_manifold()


@pytest.mark.parallel(nprocs=2)
def test_manifold_parallel():
    run_manifold()


@pytest.mark.parametrize('space', ["RT", "BDM", "RTCF"])
def test_contravariant_piola_facet_integral(space):
    if space == "RTCF":
        mesh = UnitCubedSphereMesh(refinement_level=2)
    else:
        mesh = UnitIcosahedralSphereMesh(refinement_level=2)
    x = SpatialCoordinate(mesh)
    global_normal = as_vector((x[0], x[1], x[2]))
    mesh.init_cell_orientations(global_normal)
    V = FunctionSpace(mesh, space, 1)
    # Some non-zero function
    u = project(as_vector((x[0], -x[1], 0)), V)
    n = FacetNormal(mesh)

    pos = inner(u('+'), n('+'))*dS
    neg = inner(u('-'), n('-'))*dS

    assert np.allclose(assemble(pos) + assemble(neg), 0)
    assert np.allclose(assemble(pos + neg), 0)


@pytest.mark.parametrize('space', ["N1curl", "N2curl", "RTCE"])
def test_covariant_piola_facet_integral(space):
    if space == "RTCE":
        mesh = UnitCubedSphereMesh(refinement_level=2)
    else:
        mesh = UnitIcosahedralSphereMesh(refinement_level=2)
    x = SpatialCoordinate(mesh)
    global_normal = as_vector((x[0], x[1], x[2]))
    mesh.init_cell_orientations(global_normal)
    V = FunctionSpace(mesh, space, 1)
    # Some non-zero function
    u = project(as_vector((x[0], -x[1], 0)), V)
    n = FacetNormal(mesh)

    pos = inner(u('+'), n('+'))*dS
    neg = inner(u('-'), n('-'))*dS

    assert np.allclose(assemble(pos) + assemble(neg), 0, atol=1e-7)
    assert np.allclose(assemble(pos + neg), 0, atol=1e-7)
