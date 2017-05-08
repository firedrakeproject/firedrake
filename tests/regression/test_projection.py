from __future__ import absolute_import, print_function, division
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


def run_tensor_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = TensorFunctionSpace(m, family, degree)
    expr = [['cos(x[0]*pi*2)*sin(x[1]*pi*2)', 'cos(x[0]*pi*2)*sin(x[1]*pi*2)'],
            ['cos(x[0]*pi*2)*sin(x[1]*pi*2)', 'cos(x[0]*pi*2)*sin(x[1]*pi*2)']]
    e = Expression(expr)
    exact = Function(TensorFunctionSpace(m, 'CG', 5))
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
    (1, 'CG', 1.8),
    (2, 'CG', 2.6),
    (3, 'CG', 3.8),
    (0, 'DG', 0.8),
    (1, 'DG', 1.8),
    (2, 'DG', 2.8)])
def test_tensor_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_tensor_test(x, degree, family) for x in range(2, 5)])
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
    ev = Expression('x[0]')
    eu = Expression(('x[0]', 'x[1]'))
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


def test_projector():
    m = UnitSquareMesh(2, 2)
    Vc = FunctionSpace(m, "CG", 2)
    v = Function(Vc).interpolate(Expression("x[0]*x[1] + cos(x[0]+x[1])"))
    mass1 = assemble(v*dx)

    Vd = FunctionSpace(m, "DG", 1)
    vo = Function(Vd)

    P = Projector(v, vo)
    P.project()

    mass2 = assemble(vo*dx)
    assert(np.abs(mass1-mass2) < 1.0e-10)

    v.interpolate(Expression("x[1] + exp(x[0]+x[1])"))
    mass1 = assemble(v*dx)

    P.project()
    mass2 = assemble(vo*dx)
    assert(np.abs(mass1-mass2) < 1.0e-10)


def test_trivial_projector():
    m = UnitSquareMesh(2, 2)
    Vc = FunctionSpace(m, "CG", 2)
    v = Function(Vc).interpolate(Expression("x[0]*x[1] + cos(x[0]+x[1])"))
    mass1 = assemble(v*dx)

    vo = Function(Vc)

    P = Projector(v, vo)
    P.project()

    mass2 = assemble(vo*dx)
    assert(np.abs(mass1-mass2) < 1.0e-10)

    v.interpolate(Expression("x[1] + exp(x[0]+x[1])"))
    mass1 = assemble(v*dx)

    P.project()
    mass2 = assemble(vo*dx)
    assert(np.abs(mass1-mass2) < 1.0e-10)


def test_projector_expression():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)
    vo = Function(V)
    expr = Expression("1")
    with pytest.raises(ValueError):
        Projector(expr, vo)


@pytest.mark.parametrize('tensor', ['scalar', 'vector', 'tensor'])
def test_projector_bcs(tensor):
    mesh = UnitSquareMesh(2, 2)
    if tensor == 'scalar':
        V = FunctionSpace(mesh, "CG", 1)
        V_ho = FunctionSpace(mesh, "CG", 5)
        bc = DirichletBC(V_ho, Constant(42.0), (1, 2, 3, 4))
    elif tensor == 'vector':
        V = VectorFunctionSpace(mesh, "CG", 1)
        V_ho = VectorFunctionSpace(mesh, "CG", 5)
        bc = DirichletBC(V_ho, Expression(("42.0", "42.0")),
                         (1, 2, 3, 4))
    elif tensor == 'tensor':
        V = TensorFunctionSpace(mesh, "CG", 1)
        V_ho = TensorFunctionSpace(mesh, "CG", 5)
        bc = DirichletBC(V_ho, Expression(("42.0", "42.0",
                                           "42.0", "42.0")),
                         (1, 2, 3, 4))

    exact = Function(V_ho)
    bc.apply(exact)

    v = Function(V)
    ret = Function(V_ho)
    projector = Projector(v, ret, bcs=bc, solver_parameters={"ksp_type": "preonly",
                                                             "pc_type": "lu"})
    projector.project()
    assert sqrt(assemble(inner((ret - exact), (ret - exact)) * dx)) < 1e-10


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
