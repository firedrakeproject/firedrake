import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module')
def cg1(mesh):
    return FunctionSpace(mesh, "CG", 1)


def test_eviction(cg1):
    cache = assembly_cache.AssemblyCache()
    cache.clear()

    old_limit = parameters["assembly_cache"]["max_bytes"]
    try:
        parameters["assembly_cache"]["max_bytes"] = 5000
        u = TrialFunction(cg1)
        v = TestFunction(cg1)

        # The mass matrix should be 1648 bytes, so 3 of them fit in
        # cache, and inserting a 4th will cause two to be evicted.
        for i in range(1, 5):
            # Scaling the mass matrix by i causes cache misses.
            assemble(i*u*v*dx).M

    finally:
        parameters["assembly_cache"]["max_bytes"] = old_limit

    assert 3000 < cache.nbytes < 5000
    assert cache.num_objects == 2


@pytest.mark.parallel(nprocs=2)
def test_eviction_parallel():
    cache = assembly_cache.AssemblyCache()
    cache.clear()

    mesh = UnitSquareMesh(5, 5)
    cg1 = FunctionSpace(mesh, "Lagrange", 1)

    old_limit = parameters["assembly_cache"]["max_bytes"]
    try:
        parameters["assembly_cache"]["max_bytes"] = 5000
        u = TrialFunction(cg1)
        v = TestFunction(cg1)

        # In the parallel case it's harder to ascertain exactly how
        # much cache we will use, so we do this enough times that we
        # can prove that we must have triggered eviction.
        for i in range(1, 15):
            # Scaling the mass matrix by i causes cache misses.
            assemble(i*u*v*dx).M

    finally:
        parameters["assembly_cache"]["max_bytes"] = old_limit

    assert 3000 < cache.nbytes < 5000


def test_hit(cg1):
    cache = assembly_cache.AssemblyCache()
    cache.clear()

    u = TrialFunction(cg1)
    v = TestFunction(cg1)

    assemble(u*v*dx).M
    assemble(u*v*dx).M

    assert cache.num_objects == 1
    assert cache._hits == 1


def test_assemble_rhs_with_without_constant(cg1):
    cache = assembly_cache.AssemblyCache()
    cache.clear()
    v = TestFunction(cg1)
    f = Function(cg1)

    f = assemble(v*dx, f)
    f = assemble(Constant(2)*v*dx, f)

    assert cache.num_objects == 2
    assert np.allclose(f.dat.data_ro, 2 * assemble(v*dx).dat.data_ro)
    assert cache.num_objects == 2


def test_repeated_assign(cg1):
    cache = assembly_cache.AssemblyCache()
    cache.clear()
    u = Function(cg1)
    g = Function(cg1)
    f = Function(cg1)

    assert np.allclose(assemble(g*g*dx), 0)
    assert cache.num_objects == 1
    f.assign(1)
    u.assign(g)
    u.assign(f)
    g.assign(u)
    assert cache.num_objects == 1
    assert np.allclose(assemble(g*g*dx), 1.0)
    assert cache.num_objects == 1


@pytest.mark.parallel
def test_repeated_project():
    cache = assembly_cache.AssemblyCache()
    cache.clear()
    mesh = UnitCubeMesh(2, 2, 2)
    V2 = FunctionSpace(mesh, "DG", 0)
    D0 = project(Expression('x[0]'), V2)

    assert cache.num_objects == 2
    D1 = project(Expression('x[0]'), V2)

    assert cache.num_objects == 2
    assert np.allclose(assemble((D0 - D1)*(D0 - D1)*dx), 0)
    assert cache.num_objects == 3


@pytest.mark.parallel
def test_repeated_mixed_solve():
    cache = assembly_cache.AssemblyCache()
    cache.clear()
    n = 4
    mesh = UnitSquareMesh(n, n)
    V1 = FunctionSpace(mesh, 'RT', 1)
    V2 = FunctionSpace(mesh, 'DG', 0)
    W = V1 * V2
    lmbda = 1
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Function(V2)
    f.interpolate(Expression('1e-7'))

    a = (p*q - q*div(u) + lmbda*inner(v, u) + div(v)*p)*dx
    L = f*q*dx

    solver_parameters = {'ksp_type': 'cg',
                         'pc_type': 'fieldsplit',
                         'pc_fieldsplit_type': 'schur',
                         'pc_fieldsplit_schur_fact_type': 'FULL',
                         'fieldsplit_0_ksp_type': 'cg',
                         'fieldsplit_1_ksp_type': 'cg'}

    solution1 = Function(W)
    solve(a == L, solution1, solver_parameters=solver_parameters)

    assert cache.num_objects == 2
    solution2 = Function(W)
    solve(a == L, solution2, solver_parameters=solver_parameters)
    assert cache.num_objects == 2

    assert np.allclose(errornorm(solution1, solution2, degree_rise=0), 0)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
