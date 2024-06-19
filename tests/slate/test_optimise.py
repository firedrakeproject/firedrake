import pytest
from firedrake import *
import numpy as np
from firedrake.slate.slac.optimise import optimise
from firedrake.parameters import parameters


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
def dg1_dg1(dg):
    return dg[1] * dg[1]


@pytest.fixture
def functions(mesh, dg, vector_dg1):
    x = SpatialCoordinate(mesh)
    q = Function(vector_dg1).project(grad(sin(pi*x[0])*cos(pi*x[1])))
    p = Function(dg[1]).interpolate(-x[0]*exp(-x[1]**2))
    r = Function(dg[0]).assign(42.0)
    return [q, p, r]


@pytest.fixture
def LM(dg1_dg1):
    u, p = TrialFunctions(dg1_dg1)
    v, q = TestFunctions(dg1_dg1)
    A = Tensor(inner(u, v)*dx + inner(p, q)*dx)
    f = Function(dg1_dg1)
    f.sub(0).assign(2)
    f.sub(1).assign(1)
    F = AssembledVector(f)
    return A, F


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
def TC(dg1_dg1_dg0, functions):
    """Fixture for a tensor of a form with a DG discretisation and a corresponding coefficient."""
    q, p, r = functions
    u, phi, eta = TrialFunctions(dg1_dg1_dg0)
    v, psi, nu = TestFunctions(dg1_dg1_dg0)
    K = Tensor(Constant(2)*inner(u, v)*dx + inner(phi, psi)*dx + inner(eta, nu)*dx)
    f = AssembledVector(assemble(inner(q, v)*dx + inner(p, psi)*dx + inner(r, nu)*dx))
    f2 = AssembledVector(assemble(Constant(2)*inner(q, v)*dx + inner(p, psi)*dx + inner(r, nu)*dx))
    return K, f, f2


@pytest.fixture
def TC_double_mass(dg1_dg1_dg0, functions):
    """Fixture for a tensor on a different form (double mass), but the same discretisation as TC."""
    q, p, r = functions
    u, phi, eta = TrialFunctions(dg1_dg1_dg0)
    v, psi, nu = TestFunctions(dg1_dg1_dg0)
    K = Tensor((inner(u, v)*dx + inner(u, v)*dx + inner(phi, psi)*dx + inner(eta, nu)*dx))
    f = AssembledVector(assemble(inner(q, v)*dx + inner(p, psi)*dx + inner(r, nu)*dx))
    return K, f


@pytest.fixture
def TC_without_trace(dg1_dg1_dg0, functions):
    """Fixture for a tensor on a different form (no trace), but the same discretisation as TC."""
    q, p, _ = functions
    u, phi, _ = TrialFunctions(dg1_dg1_dg0)
    v, psi, _ = TestFunctions(dg1_dg1_dg0)
    K = Tensor(inner(u, v)*dx + inner(phi, psi)*dx)
    f = assemble(inner(q, v)*dx + inner(p, psi)*dx)
    return K, AssembledVector(f)


@pytest.fixture
def TC_non_symm(mesh, dg):
    """Fixture for a non-symmetric tensor and a corresponding coefficient."""
    p3 = VectorElement("CG", triangle, 3)
    p2 = FiniteElement("CG", triangle, 2)
    p3p2 = MixedElement([p3, p2])

    velo = FunctionSpace(mesh, p3)
    pres = FunctionSpace(mesh, p2)
    mixed = FunctionSpace(mesh, p3p2)

    w = Function(mixed)
    x = SpatialCoordinate(mesh)
    velo = Function(velo).project(as_vector([10*sin(pi*x[0]), 0]))
    w.sub(0).assign(velo)
    pres = Function(pres).assign(10.)
    w.sub(1).assign(pres)

    T = TrialFunction(dg[2])
    v = TestFunction(dg[2])

    n = FacetNormal(mesh)
    u = split(w)[0]
    un = abs(dot(u('+'), n('+')))
    jump_v = v('+')*n('+') + v('-')*n('-')
    jump_T = T('+')*n('+') + T('-')*n('-')
    x, y = SpatialCoordinate(mesh)

    T = Tensor(-dot(u*T, grad(v))*dx + (dot(u('+'), jump_v)*avg(T))*dS + dot(v, dot(u, n)*T)*ds + 0.5*un*dot(jump_T, jump_v)*dS)
    C = AssembledVector(Function(dg[2]).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2)))
    return T, C


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


def test_push_assembled_vector_blocks_individual(TC, LM):
    """Test Optimisers's ability to handle individual blocks of AssembledVectors."""
    T, C, _ = TC
    L, M = LM
    expressions = [C.blocks[0], (C+C).blocks[0], (T*C).blocks[0], (-C).blocks[0], (C.T).blocks[0],
                   (T*C).blocks[0] + C.blocks[0], C.blocks[0] + (T*C).blocks[0],
                   C.blocks[1], (C+C).blocks[1], (T*C).blocks[1], (-C).blocks[1], (C.T).blocks[1],
                   (T*C).blocks[1] + C.blocks[1], C.blocks[1] + (T*C).blocks[1]]
    compare_vector_expressions(expressions)

    expressions = [M.blocks[0] + M.blocks[1]]
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


def test_inner_blocks(TC):
    """Test Optimisers's ability to handle expressions where the blocks is not outermost."""
    K, F, F2 = TC
    expressions = [F.blocks[1:2] - K.blocks[1:2, 0:1] * K.blocks[0:1, 0:1].inv * F.blocks[0:1],
                   F.blocks[2:3] - K.blocks[2:3, 0:2] * K.blocks[0:2, 0:2].inv * F.blocks[0:2],
                   K.blocks[0:1, 0:1].solve(F2.blocks[0:1] - K.blocks[0:1, 1:2] * F.blocks[1:2]),
                   (K.blocks[2, 2] - K.blocks[2, :2] * K.blocks[0:2, 0:2].inv * K.blocks[:2, 2]).solve(F2.blocks[2]),
                   (K.blocks[2, 2] - K.blocks[2, :2] * K.blocks[0:2, 0:2].inv * K.blocks[:2, 2]).solve(F2.blocks[2] - K.blocks[2, :2] * K.blocks[0:2, 0:2].inv * F.blocks[:2]),
                   K.blocks[0, 0].solve(F.blocks[0] - K.blocks[0, 1] * F.blocks[1] - K.blocks[0, 2] * F.blocks[2])]
    compare_vector_expressions_mixed(expressions)


#######################################
# Test multiplication optimisation pass
#######################################
def test_push_mul_simple(TC):
    """Test Optimisers's ability to handle multiplications nested with simple expressions."""
    T, C, _ = TC
    expressions = [(T+T)*C, (T*T)*C, (-T)*C, T.T*C, T.inv*C]
    opt_expressions = [T*C+T*C, T*(T*C), -(T*C), (C.T*T).T, T.solve(C)]
    compare_vector_expressions_mixed(expressions)
    compare_slate_tensors(expressions, opt_expressions)


def test_push_mul_nested(TC, TC_without_trace, TC_non_symm):
    """Test Optimisers's ability to handle multiplications nested with nested expressions."""
    T, C, _ = TC
    T2, _ = TC_without_trace
    T3, C3 = TC_non_symm
    expressions = [(T+T+T2)*C, (T+T2+T)*C, (T-T+T2)*C, (T+T2-T)*C,
                   (T*T.inv)*C, (T.inv*T)*C, (T2*T.inv)*C, (T2*T.inv*T)*C,
                   (C.T*T.inv)*(T.inv*T), C3*(T3.inv*T3.T), (C3.T*T3.inv)*(T3.inv*T3)]
    opt_expressions = [T*C+T*C+T2*C, T*C+T2*C+T*C, T*C-(T*C)+T2*C, T*C+T2*C-(T*C),
                       T*T.solve(C), T.solve(T*C), T2*T.solve(C), T2*T.solve(T*C),
                       (T.T.solve(T.T.solve(C))).T*T, (T3*(T3.T.solve(C3.T))).T, (T3.T.solve(T3.T.solve(C3))).T*T3]
    compare_vector_expressions_mixed(expressions, rtol=1e-10)
    compare_slate_tensors(expressions, opt_expressions)

    # Make sure replacing inverse by solves does not introduce errors
    opt = assemble((T3*T3.T.solve(C3.T)).T, form_compiler_parameters={"optimise": False}).dat.data
    ref = assemble(C3*(T3.inv*T3.T), form_compiler_parameters={"optimise": False}).dat.data
    for r, o in zip(ref, opt):
        assert np.allclose(o, r, rtol=1e-12)

    opt = assemble((T3.T.solve(T3.T.solve(C3))).T*T3, form_compiler_parameters={"optimise": False}).dat.data
    ref = assemble((C3.T*T3.inv)*(T3.inv*T3), form_compiler_parameters={"optimise": False}).dat.data
    for r, o in zip(ref, opt):
        assert np.allclose(o, r, rtol=1e-12)


def test_push_mul_schurlike(TC, TC_without_trace):
    """Test Optimisers's ability to handle schur complement like expressions."""
    T, C, _ = TC
    T2, _ = TC_without_trace
    expressions = [(T-T.inv*T)*C, (T+T.inv*T)*C, (T+T-T2*T.inv*T)*C]
    opt_expressions = [T*C-T.solve(T*C), T*C+T.solve(T*C), T*C+T*C-T2*T.solve(T*C)]
    compare_vector_expressions_mixed(expressions)
    compare_slate_tensors(expressions, opt_expressions)


def test_push_mul_non_symm(TC_non_symm):
    """Test Optimisers's ability to handle transponses on tensors
    that are actually not symmetric."""
    T, C = TC_non_symm
    ref = assemble(T, form_compiler_parameters={"optimise": False}).M.values
    opt = assemble(T.T, form_compiler_parameters={"optimise": False}).M.values
    assert not np.allclose(opt, ref, rtol=1e-12)  # go sure problem is non-symmetric
    expressions = [T*C, T.inv*C, T.T*C]
    compare_vector_expressions_mixed(expressions)  # only testing data is sufficient here


def test_push_mul_multiple_coeffs(TC, TC_without_trace):
    """Test Optimisers's ability to handle expression with multiple coefficients."""
    T, C, _ = TC
    T2, C2 = TC_without_trace
    expressions = [(T+T)*C+(T2+T2)*C2]
    opt_expressions = [(T*C+T*C)+(T2*C2+T2*C2)]
    compare_vector_expressions_mixed(expressions)
    compare_slate_tensors(expressions, opt_expressions)


def test_partially_optimised(TC_non_symm, TC_double_mass, TC):
    """Test Optimisers's ability to handle partially optimised expressions."""
    A, C = TC_non_symm
    T2, C2, _ = TC
    T3, _ = TC_double_mass

    # Test some non symmetric, non mixed, nested expressions
    expressions = [A.inv*C+A.inv*C, (A+A)*A.solve(C),
                   (A+A)*A.solve((A+A)*C), C*(A.inv*(A.inv*(A)))]
    opt_expressions = [A.solve(C)+A.solve(C), A*A.solve(C)+A*A.solve(C),
                       A*A.solve(A*C+A*C)+A*A.solve(A*C+A*C),
                       (A.T.solve(A.T.solve(C.T))).T*A]

    compare_vector_expressions(expressions, rtol=1e-10)
    compare_slate_tensors(expressions, opt_expressions)

    # Make sure optimised solve gives same answer as expression with inverses
    opt = assemble((C*(A.solve(A.solve(A)))), form_compiler_parameters={"optimise": True}).dat.data
    ref = assemble((C*(A.inv*(A.inv*(A)))), form_compiler_parameters={"optimise": False}).dat.data
    for r, o in zip(ref, opt):
        assert np.allclose(o, r, rtol=1e-12)
    compare_slate_tensors([(C*(A.solve(A.solve(A))))], [(A.T.solve(A.T.solve(C.T))).T*A])

    # Test some symmetric, mixed, nested expressions
    expressions = [T2.inv*C2+T2.inv*C2,
                   C2*(T2.inv*(T2.inv*(T2))),
                   C2*(T2.inv*(T3.inv*(T2))),
                   (T2.inv*(T3.inv*(T2)))*C2]
    opt_expressions = [T2.solve(C2)+T2.solve(C2),
                       (T2.T.solve(T2.T.solve(C2.T))).T*T2,
                       (T3.T.solve(T2.T.solve(C2.T))).T*T2,
                       T2.solve(T3.solve(T2*C2))]
    compare_vector_expressions_mixed(expressions)
    compare_slate_tensors(expressions, opt_expressions)


#######################################
# Test transposition optimisation pass
#######################################
def test_drop_transposes(TC_non_symm):
    """Test Optimisers's ability to drop double transposes."""
    A, C = TC_non_symm

    expressions = [A.T.T, A.T.T.inv, A.T.T+A.T.T]
    opt_expressions = [A, A.inv, A+A]
    compare_tensor_expressions(expressions)
    compare_slate_tensors(expressions, opt_expressions)

    expressions = [A.solve(A.T.T*C)]
    opt_expressions = [A.solve(A*C)]
    compare_vector_expressions(expressions)
    compare_slate_tensors(expressions, opt_expressions)

    assert A != A.T
    from firedrake.slate.slac.optimise import optimise
    T_opt = optimise(A.T, {"optimise": True})
    assert isinstance(T_opt, Tensor)
    assert np.allclose(assemble(T_opt.T).M.values,
                       assemble(adjoint(T_opt.form)).M.values)


#######################################
# Test diagonal optimisation pass
#######################################
def test_push_diagonal(TC_non_symm):
    """Test Optimisers's ability to push DiagonalTensors inside expressions."""
    A, C = TC_non_symm

    expressions = [DiagonalTensor(A), DiagonalTensor(A+A),
                   DiagonalTensor(-A), DiagonalTensor(A*A),
                   DiagonalTensor(A).inv]
    opt_expressions = [DiagonalTensor(A), DiagonalTensor(A)+DiagonalTensor(A),
                       -DiagonalTensor(A), DiagonalTensor(A*A),
                       DiagonalTensor(A).inv]
    compare_tensor_expressions(expressions)
    compare_slate_tensors(expressions, opt_expressions)

    expressions = [DiagonalTensor(A+A).solve(C)]
    opt_expressions = [(DiagonalTensor(A)+DiagonalTensor(A)).solve(C)]
    compare_vector_expressions(expressions)
    compare_slate_tensors(expressions, opt_expressions)


#######################################
# Helper functions
#######################################
def compare_tensor_expressions(expressions, rtol=1e-12):
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": False}}).M.values
        opt = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": True}}).M.values
        assert np.allclose(opt, ref, rtol=rtol)


def compare_vector_expressions(expressions, rtol=1e-12):
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": False}}).dat.data
        opt = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": True}}).dat.data
        assert np.allclose(opt, ref, rtol=rtol)


def compare_vector_expressions_mixed(expressions, rtol=1e-12):
    for expr in expressions:
        ref = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": False}}).dat.data
        opt = assemble(expr, form_compiler_parameters={"slate_compiler": {"optimise": True}}).dat.data
        for r, o in zip(ref, opt):
            assert np.allclose(o, r, rtol=rtol)


def compare_slate_tensors(expressions, opt_expressions):
    for expr, opt_expr_sol in zip(expressions, opt_expressions):
        opt_expr = optimise(expr, parameters["slate_compiler"])
        assert opt_expr == opt_expr_sol
