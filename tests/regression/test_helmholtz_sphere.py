# Solve -laplace u + u = f on the surface of the sphere
# with forcing function xyz, this has exact solution xyz/13
import pytest
import numpy as np
from firedrake import *


def run_helmholtz_sphere(MeshClass, r, d):
    m = MeshClass(refinement_level=r, degree=d)
    x = SpatialCoordinate(m)
    m.init_cell_orientations(x)
    V = FunctionSpace(m, 'CG', d)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V).interpolate(x[0]*x[1]*x[2])

    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx

    u = Function(V)
    solve(a == L, u, solver_parameters={"ksp_type": "cg"})

    f.interpolate(x[0]*x[1]*x[2]/13.0)
    return errornorm(f, u, degree_rise=0)


def run_helmholtz_mixed_sphere(MeshClass, r, meshd, eltd):
    m = MeshClass(refinement_level=r, degree=meshd)
    x = SpatialCoordinate(m)
    m.init_cell_orientations(x)
    if m.ufl_cell().cellname() == "triangle":
        V = FunctionSpace(m, 'RT', eltd+1)
    elif m.ufl_cell().cellname() == "quadrilateral":
        V = FunctionSpace(m, 'RTCF', eltd+1)
    Q = FunctionSpace(m, 'DG', eltd)
    W = V*Q

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    f = Function(Q)
    f.interpolate(x[0]*x[1]*x[2])
    a = (inner(p, q) - inner(div(u), q) + inner(u, v) + inner(p, div(v))) * dx
    L = inner(f, q) * dx

    soln = Function(W)

    solve(a == L, soln, solver_parameters={'pc_type': 'fieldsplit',
                                           'pc_fieldsplit_type': 'schur',
                                           'ksp_type': 'cg',
                                           'pc_fieldsplit_schur_fact_type': 'FULL',
                                           'fieldsplit_0_ksp_type': 'cg',
                                           'fieldsplit_1_ksp_type': 'cg'})

    _, u = soln.subfunctions
    f.interpolate(x[0]*x[1]*x[2]/13.0)
    return errornorm(f, u, degree_rise=0)


@pytest.mark.parametrize('MeshClass', [UnitIcosahedralSphereMesh, UnitCubedSphereMesh])
def test_helmholtz_sphere_lowestorder(MeshClass):
    errors = [run_helmholtz_sphere(MeshClass, r, 1) for r in range(2, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])

    assert (l2conv > 1.7).all()


@pytest.mark.parametrize('MeshClass', [UnitIcosahedralSphereMesh, UnitCubedSphereMesh])
def test_helmholtz_sphere_higherorder(MeshClass):
    errors = [run_helmholtz_sphere(MeshClass, r, 2) for r in range(2, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])

    assert (l2conv > 2.99).all()


@pytest.mark.parametrize('MeshClass', [UnitIcosahedralSphereMesh, UnitCubedSphereMesh])
def test_helmholtz_mixed_sphere_lowestorder(MeshClass):
    errors = [run_helmholtz_mixed_sphere(MeshClass, r, 1, 0) for r in range(2, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])

    # Note, due to "magic hybridisation stuff" we expect the numerical
    # solution to converge to the projection of the exact solution to
    # DG0 at second order (ccotter, pers comm).
    assert (l2conv > 1.7).all()


@pytest.mark.parametrize('MeshClass', [UnitIcosahedralSphereMesh, UnitCubedSphereMesh])
def test_helmholtz_mixed_sphere_higherorder(MeshClass):
    errors = [run_helmholtz_mixed_sphere(MeshClass, r, 2, 2) for r in range(2, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])

    assert (l2conv > 2.7).all()


@pytest.mark.parallel
def test_helmholtz_mixed_cubedsphere_parallel():
    errors = [run_helmholtz_mixed_sphere(UnitCubedSphereMesh, r, 2, 2) for r in range(2, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])

    assert (l2conv > 2.7).all()
