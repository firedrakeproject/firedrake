import pytest
import numpy as np
from firedrake import *


def run_vector_valued_test(x, degree=1, family='RT'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(m, family, degree)
    expr = ['cos(x[0]*pi*2)*sin(x[1]*pi*2)']*2
    e = Expression(expr)
    exact = Function(VectorFunctionSpace(m, 'CG', 5))
    exact.interpolate(e)

    # Solve to machine precision.
    ret = project(e, V, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner((ret - exact), (ret - exact)) * dx))


def run_vector_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = VectorFunctionSpace(m, family, degree)
    expr = ['cos(x[0]*pi*2)*sin(x[1]*pi*2)']*2
    e = Expression(expr)
    exact = Function(VectorFunctionSpace(m, 'CG', 5))
    exact.interpolate(e)

    # Solve to machine precision.  This version of the test uses the
    # alternate syntax in which the target Function is already
    # available.
    ret = Function(V)
    project(e, ret, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble(inner((ret - exact), (ret - exact)) * dx))


def run_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(m, family, degree)
    e = Expression('cos(x[0]*pi*2)*sin(x[1]*pi*2)')
    exact = Function(FunctionSpace(m, 'CG', 5))
    exact.interpolate(e)

    # Solve to machine precision. This version of the test uses the
    # method version of project.
    ret = Function(V)
    ret.project(e, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble((ret - exact) * (ret - exact) * dx))


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (1, 'CG', 1.8),
    (2, 'CG', 2.6),
    (3, 'CG', 3.8),
    (0, 'DG', 0.8),
    (1, 'DG', 1.8),
    (2, 'DG', 2.8)])
def test_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_test(x, degree, family) for x in range(2, 5)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (1, 'CG', 1.8),
    (2, 'CG', 2.6),
    (3, 'CG', 3.8),
    (0, 'DG', 0.8),
    (1, 'DG', 1.8),
    (2, 'DG', 2.8)])
def test_vector_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_vector_test(x, degree, family) for x in range(2, 5)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (1, 'RT', 0.75),
    (2, 'RT', 1.8),
    (3, 'RT', 2.8),
    (1, 'BDM', 1.8),
    (2, 'BDM', 2.8),
    (3, 'BDM', 3.8)])
def test_vector_valued_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_vector_valued_test(x, degree, family)
                        for x in range(2, 6)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()


def test_project_mismatched_rank():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'CG', 1)
    U = FunctionSpace(m, 'RT', 1)
    v = Function(V)
    u = Function(U)
    ev = Expression('')
    eu = Expression(('', ''))
    with pytest.raises(RuntimeError):
        project(v, U)
    with pytest.raises(RuntimeError):
        project(u, V)
    with pytest.raises(RuntimeError):
        project(ev, U)
    with pytest.raises(RuntimeError):
        project(eu, V)


def test_project_mismatched_mesh():
    m2 = UnitSquareMesh(2, 2)
    m3 = UnitCubeMesh(2, 2, 2)

    U = FunctionSpace(m2, 'CG', 1)
    V = FunctionSpace(m3, 'CG', 1)

    u = Function(U)
    v = Function(V)

    with pytest.raises(RuntimeError):
        project(u, V)

    with pytest.raises(RuntimeError):
        project(v, U)


def test_project_mismatched_shape():
    m = UnitSquareMesh(2, 2)

    U = VectorFunctionSpace(m, 'CG', 1, dim=3)
    V = VectorFunctionSpace(m, 'CG', 1, dim=2)

    u = Function(U)
    v = Function(V)

    with pytest.raises(RuntimeError):
        project(u, V)

    with pytest.raises(RuntimeError):
        project(v, U)


def test_repeatable():
    mesh = UnitSquareMesh(1, 1)
    Q = FunctionSpace(mesh, 'DG', 1)

    V2 = FunctionSpace(mesh, 'DG', 0)
    V3 = FunctionSpace(mesh, 'DG', 0)
    W = V2 * V3
    expr = Expression('1.0')
    old = project(expr, Q)

    f = project(Expression(('-1.0', '-1.0')), W)  # noqa
    new = project(expr, Q)

    for fd, ud in zip(new.dat.data, old.dat.data):
        assert (fd == ud).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
