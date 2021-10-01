import pytest
from firedrake import *
import numpy as np


@pytest.fixture
def mesh():
    return UnitSquareMesh(2, 2, True)


@pytest.fixture
def A(mesh):
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    T = FunctionSpace(mesh, "HDiv Trace", 0)
    n = FacetNormal(mesh)
    W = U * V * T
    u, p, lambdar = TrialFunctions(W)
    w, q, gammar = TestFunctions(W)

    return Tensor(inner(u, w)*dx + p*q*dx - div(w)*p*dx + q*div(u)*dx
                  + lambdar('+')*jump(w, n=n)*dS + gammar('+')*jump(u, n=n)*dS
                  + lambdar*gammar*ds)


@pytest.fixture
def KF(mesh):
    V = VectorFunctionSpace(mesh, "DG", 1)
    U = FunctionSpace(mesh, "DG", 1)
    T = FunctionSpace(mesh, "DG", 0)
    W = V * U * T
    x = SpatialCoordinate(mesh)
    q = Function(V).project(grad(sin(pi*x[0])*cos(pi*x[1])))
    p = Function(U).interpolate(-x[0]*exp(-x[1]**2))
    r = Function(T).assign(42.0)
    u, phi, eta = TrialFunctions(W)
    v, psi, nu = TestFunctions(W)

    K = Tensor(inner(u, v)*dx + inner(phi, psi)*dx + inner(eta, nu)*dx)
    F = Tensor(inner(q, v)*dx + inner(p, psi)*dx + inner(r, nu)*dx)
    return K, K.inv * F


@pytest.fixture
def TC(mesh):
    V = VectorFunctionSpace(mesh, "DG", 1)
    U = FunctionSpace(mesh, "DG", 1)
    T = FunctionSpace(mesh, "DG", 0)
    W = V * U * T
    x = SpatialCoordinate(mesh)
    q = Function(V).project(grad(sin(pi*x[0])*cos(pi*x[1])))
    p = Function(U).interpolate(-x[0]*exp(-x[1]**2))
    r = Function(T).assign(42.0)
    u, phi, eta = TrialFunctions(W)
    v, psi, nu = TestFunctions(W)

    K = Tensor(inner(u, v)*dx + inner(phi, psi)*dx + inner(eta, nu)*dx)
    f = assemble(inner(q, v)*dx + inner(p, psi)*dx + inner(r, nu)*dx)
    return K, AssembledVector(f)


def test_push_tensor_blocks_individual(A):
    """Test Optimisers's ability to handle individual blocks of 2-Tensors."""
    expressions = [(A+A).blocks[0, 0], (A*A).blocks[0, 0], (-A).blocks[0, 0],
                   (A.T).blocks[0, 0], (A.inv).blocks[0, 0]]
    compare_tensor_expressions(expressions)


def test_push_tensor_blocks_mixed(A):
    """Test Optimisers's ability to handle mixed blocks of 2-Tensors."""
    expressions = [(A+A).blocks[:2, :2], (A*A).blocks[:2, :2], (-A).blocks[:2, :2],
                   (A.T).blocks[:2, :2], (A.inv).blocks[:2, :2],
                   (A+A).blocks[1:3, 1:3], (A*A).blocks[1:3, 1:3],
                   (-A).blocks[1:3, 1:3], (A.T).blocks[1:3, 1:3], (A.inv).blocks[1:3, 1:3]]
    compare_tensor_expressions(expressions)


def test_push_tensor_blocks_on_blocks(A):
    """Test Optimisers's ability to handle blocks of blocks of 2-Tensors."""
    expressions = [(A+A).blocks[:2, :2].blocks[0, 0], (A*A).blocks[:2, :2].blocks[0, 0],
                   (-A).blocks[:2, :2].blocks[0, 0], (A.T).blocks[:2, :2].blocks[0, 0],
                   (A.inv).blocks[:2, :2].blocks[0, 0]]
    # test for blocks with too few indices too
    expressions += [(A+A).blocks[:2].blocks[0, 0]]
    compare_tensor_expressions(expressions)


def test_push_vector_block_individual(KF):
    """Test Optimisers's ability to handle individual blocks of 1-Tensors."""
    K, F = KF
    expressions = [(F+F).blocks[0], (K*F).blocks[0], (-F).blocks[0]]
    compare_vector_expressions(expressions)


def test_push_vector_block_mixed(KF):
    """Test Optimisers's ability to handle mixed blocks of 1-Tensors."""
    K, F = KF
    expressions = [(F+F).blocks[:2], (K*F).blocks[:2], (-F).blocks[:2],
                   (F+F).blocks[1:3], (K*F).blocks[1:3], (-F).blocks[1:3]]
    compare_vector_expressions_mixed(expressions)


def test_push_vector_blocks_on_blocks(KF):
    """Test Optimisers's ability to handle blocks of blocks of 1-Tensors."""
    K, F = KF
    expressions = [(F+F).blocks[:2].blocks[0], (K*F).blocks[:2].blocks[0],
                   (-F).blocks[:2].blocks[0]]
    compare_vector_expressions(expressions)


def test_push_assembled_vector_blocks_individual(TC):
    """Test Optimisers's ability to handle individual blocks of AssembledVectors."""
    T, C = TC
    expressions = [C.blocks[0], (C+C).blocks[0], (T*C).blocks[0], (-C).blocks[0], (C.T).blocks[0]]
    compare_vector_expressions(expressions)


def test_push_block_aggressive_unaryop_nesting():
    """Test Optimisers's ability to handle extremely nested expressions."""
    V = FunctionSpace(UnitSquareMesh(1, 1), "DG", 3)
    f = Function(V)
    g = Function(V)
    f.assign(1.0)
    g.assign(0.5)
    F = AssembledVector(f)
    G = AssembledVector(g)
    u = TrialFunction(V)
    v = TestFunction(V)

    A = Tensor(inner(u, v)*dx)
    B = Tensor(2.0*inner(u, v)*dx)

    # This is a very silly way to write the vector of ones
    expressions = [((B.T*A.inv).T*G + (-A.inv.T*B.T).inv*F + B.inv*(A.T).T*F).blocks[0]]
    compare_vector_expressions(expressions)


def compare_tensor_expressions(expressions):
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": False}}).M.values
        opt = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": True}}).M.values
        assert np.allclose(opt, ref, rtol=1e-14)


def compare_vector_expressions(expressions):
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": False}}).dat.data
        opt = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": True}}).dat.data
        assert np.allclose(opt, ref, rtol=1e-14)


def compare_vector_expressions_mixed(expressions):
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": False}}).dat.data
        opt = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": True}}).dat.data
        for r, o in zip(ref, opt):
            assert np.allclose(o, r, rtol=1e-14)
