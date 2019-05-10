import pytest
from firedrake import *
import numpy as np


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


def test_pointwise_operator(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    v = Function(V).interpolate(sin(x))
    u = Function(V).interpolate(cos(x))

    m = u*v
    a1 = m*dx

    p = point_expr(lambda x, y: x*y)
    p2 = p(u, v, eval_space=P)
    a2 = p2*dx

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2._ufl_function_space == P
    assert p2.derivatives == (0, 0)
    assert p2.ufl_shape == ()
    assert p2.operator_data(u, v) == u*v

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    assert abs(assemble_a1 - assemble_a2) < 1.0e-3  # Not evaluate on the same space whence the lack of precision


def test_pointwise_solver(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    uexact = Function(P).assign(1)

    m = uexact
    a1 = m*dx

    a = Function(V).assign(0)
    b = Function(V).assign(1)
    c = Function(V).assign(1)

    x0 = Function(V).assign(1.3)
    fprime = lambda x: 2*(x-1) + 400*x*(x**2 - 1)
    newton_params = {'x0': x0, 'fprime': fprime, 'maxiter': 30}

    # Conflict with ufl if we use directly cos()
    p = point_solve(lambda x, y, m1, m2: np.cos(m1)*(1-x)**2 + 100*m2*(y-x**2)**2, solver=newton_params, params=('y', 'm2', 'm1'))
    p2 = p(c, b, a, eval_space=P)  # Rosenbrock function for (m1,m2) = (0,1), the global minimum is reached in (1,1)
    a2 = p2*dx

    assert p2.ufl_operands == (c, b, a)
    assert p2._ufl_function_space == P
    assert p2.derivatives == (0, 0, 0)
    assert p2.ufl_shape == ()
    assert p2.params == ('y', 'm2', 'm1')
    assert p2.solver == newton_params

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    assert abs(assemble_a1 - assemble_a2) < 1.0e-7


def test_pointwise_neuralnet(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    u = Function(V)

    nP = neuralnet('PyTorch')
    nT = neuralnet('TensorFlow') 
    nK = neuralnet('Keras') 
    
    #nP2 = nP(u, eval_space=P, model='test')
    #nT2 = nT(u, eval_space=P, model='test')
    #nK2 = nK(u, eval_space=P, model='test')
    
    #assert nP.framework == 'PyTorch'
    #assert nT.framework == 'TensorFlow'
    #assert nK.framework == 'Keras'


def test_compute_derivatives(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    v = Function(V).interpolate(sin(x))
    u = Function(V).interpolate(cos(x))

    m = u*v
    a1 = m*dx

    p = point_expr(lambda x, y: 0.5*x**2*y)
    p2 = p(u, v, eval_space=P, derivatives=(1, 0))
    a2 = p2*dx

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2._ufl_function_space == P
    assert p2.derivatives == (1, 0)
    assert p2.ufl_shape == ()
    assert p2.operator_data(u, v) == 0.5*u**2*v

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    assert abs(assemble_a1 - assemble_a2) < 1.0e-3  # Not evaluate on the same space whence the lack of precision

    a = Function(V).assign(0)
    b = Function(V).assign(1)

    x0 = Function(V).assign(1.1)
    p = point_solve(lambda x, y, m1, m2: x - y**2 + m1*m2, solver={'x0': x0}, params=('m2', 'm1', 'y'))
    p2 = p(a, a, b, eval_space=P, derivatives=(0, 0, 1))
    a3 = p2*dx

    a4 = 2*b*dx  # dp2/db

    assemble_a3 = assemble(a3)
    assemble_a4 = assemble(a4)

    assert abs(assemble_a3 - assemble_a4) < 1.0e-7
