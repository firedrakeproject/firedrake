import pytest
from firedrake import *
import numpy as np


@pytest.fixture
def mesh():
    return UnitSquareMesh(2, 2, True)


@pytest.fixture
def dg(mesh):
    return [FunctionSpace(mesh, "DG", p) for p in range(3)]


@pytest.fixture
def vector_dg1(mesh):
    return VectorFunctionSpace(mesh, "DG", 1)


@pytest.fixture
def T_hyb(mesh, dg):
    """Fixture for a tensor of a form with a hybridisation like discretisation."""
    rt = FunctionSpace(mesh, "RT", 1)
    hdivt = FunctionSpace(mesh, "HDiv Trace", 0)
    W = rt * dg[0] * hdivt

    u, p, lambdar = TrialFunctions(W)
    w, q, gammar = TestFunctions(W)
    n = FacetNormal(mesh)

    return Tensor(inner(u, w)*dx + p*q*dx - div(w)*p*dx + q*div(u)*dx
                  + lambdar('+')*jump(w, n=n)*dS + gammar('+')*jump(u, n=n)*dS
                  + lambdar*gammar*ds)


@pytest.fixture
def dg1_dg1_dg0(dg, vector_dg1):
    return vector_dg1 * dg[1] * dg[0]


@pytest.fixture
def functions(mesh, dg, vector_dg1):
    x = SpatialCoordinate(mesh)
    q = Function(vector_dg1).project(grad(sin(pi*x[0])*cos(pi*x[1])))
    p = Function(dg[1]).interpolate(-x[0]*exp(-x[1]**2))
    r = Function(dg[0]).assign(42.0)
    return [q, p, r]


@pytest.fixture
def KF(dg1_dg1_dg0, functions):
    """Fixture for a tensor of a form on a 2 and a 1-form with a DG discretisation."""
    q, p, r = functions
    u, phi, eta = TrialFunctions(dg1_dg1_dg0)
    v, psi, nu = TestFunctions(dg1_dg1_dg0)
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


#######################################
# Test block optimisation pass
#######################################
def test_push_tensor_blocks_individual(T_hyb):
    """Test Optimisers's ability to handle individual blocks of Tensors on 2-forms."""
    expressions = [(T_hyb+T_hyb).blocks[0, 0], (T_hyb*T_hyb).blocks[0, 0], (-T_hyb).blocks[0, 0],
                   (T_hyb.T).blocks[0, 0], (T_hyb.inv).blocks[0, 0]]
    compare_tensor_expressions(expressions)


def test_push_tensor_blocks_mixed(T_hyb):
    """Test Optimisers's ability to handle mixed blocks of Tensors on 2-forms."""
    expressions = [(T_hyb+T_hyb).blocks[:2, :2], (T_hyb*T_hyb).blocks[:2, :2], (-T_hyb).blocks[:2, :2],
                   (T_hyb.T).blocks[:2, :2], (T_hyb.inv).blocks[:2, :2],
                   (T_hyb+T_hyb).blocks[1:3, 1:3], (T_hyb*T_hyb).blocks[1:3, 1:3],
                   (-T_hyb).blocks[1:3, 1:3], (T_hyb.T).blocks[1:3, 1:3], (T_hyb.inv).blocks[1:3, 1:3]]
    compare_tensor_expressions(expressions)


def test_push_tensor_blocks_on_blocks(T_hyb):
    """Test Optimisers's ability to handle blocks of blocks of Tensors on 2-forms."""
    expressions = [(T_hyb+T_hyb).blocks[:2, :2].blocks[0, 0], (T_hyb*T_hyb).blocks[:2, :2].blocks[0, 0],
                   (-T_hyb).blocks[:2, :2].blocks[0, 0], (T_hyb.T).blocks[:2, :2].blocks[0, 0],
                   (T_hyb.inv).blocks[:2, :2].blocks[0, 0]]
    # test for blocks with too few indices too
    expressions += [(T_hyb+T_hyb).blocks[:2].blocks[0, 0]]
    compare_tensor_expressions(expressions)


def test_push_vector_block_individual(KF):
    """Test Optimisers's ability to handle individual blocks of Tensors on 1-forms."""
    K, F = KF
    expressions = [(F+F).blocks[0], (K*F).blocks[0], (-F).blocks[0]]
    compare_vector_expressions(expressions)


def test_push_vector_block_mixed(KF):
    """Test Optimisers's ability to handle mixed blocks of Tensors on 1-forms"""
    K, F = KF
    expressions = [(F+F).blocks[:2], (K*F).blocks[:2], (-F).blocks[:2],
                   (F+F).blocks[1:3], (K*F).blocks[1:3], (-F).blocks[1:3]]
    compare_vector_expressions_mixed(expressions)


def test_push_vector_blocks_on_blocks(KF):
    """Test Optimisers's ability to handle blocks of blocks of Tensors on 1-forms."""
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


#######################################
# Helper functions
#######################################
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
