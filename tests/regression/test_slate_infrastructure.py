import pytest
from firedrake import *


def function_space():
    mesh = UnitSquareMesh(1, 1)
    return FunctionSpace(mesh, "CG", 1)


def mass(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    return slate.Matrix(u * v * dx)


def stiffness(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    return slate.Matrix(inner(grad(u), grad(v)) * dx)


def load(fs):
    f = Function(fs)
    f.interpolate(Expression("cos(x[0]*pi*2)"))
    v = TestFunction(fs)
    return slate.Vector(f * v * dx)


def boundary_load(fs):
    f = Function(fs)
    f.interpolate(Expression("cos(x[1]*pi*2)"))
    v = TestFunction(fs)
    return slate.Vector(f * v * ds)


def test_arguments():
    V = function_space()
    M = mass(V)
    N = stiffness(V)
    v, u = M.arguments()
    F = load(V)
    f, = F.coefficients()
    S = slate.Scalar(f * dx)

    assert len(N.arguments()) == 2
    assert len(M.arguments()) == 2
    assert N.arguments() == (v, u)
    assert len(F.arguments()) == 1
    assert F.arguments() == (v,)
    assert len(S.arguments()) == 0
    assert S.arguments() == ()

    assert slate.Vector(v * dx).arguments() == (v,)
    assert (slate.Vector(v * dx) + slate.Vector(f * v * ds)).arguments() == (v,)
    assert (M + N).arguments() == (v, u)
    assert (slate.Matrix((f * v) * u * dx) + slate.Matrix((u * 3) * (v / 2) * dx)).arguments() == (v, u)


def test_coefficients():
    V = function_space()
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    g = Function(V)
    f.interpolate(Expression("cos(x[0]*pi*2)"))
    g.interpolate(Expression("cos(x[1]*pi*2)"))

    assert slate.Scalar(f * dx).coefficients() == (f,)
    assert (slate.Scalar(f * dx) + slate.Scalar(f * ds)).coefficients() == (f,)
    assert (slate.Scalar(f * dx) + slate.Scalar(g * dS)).coefficients() == (f, g)
    assert slate.Vector(f * v * dx).coefficients() == (f,)
    assert (slate.Vector(f * v * ds) + slate.Vector(f * v * dS)).coefficients() == (f,)
    assert (slate.Vector(f * v * dx) + slate.Vector(g * v * ds)).coefficients() == (f, g)
    assert slate.Matrix(f * u * v * dx).coefficients() == (f,)
    assert (slate.Matrix(f * u * v * dx) + slate.Matrix(f * inner(grad(u), grad(v)) * dx)).coefficients() == (f,)
    assert (slate.Matrix(f * u * v * dx) + slate.Matrix(g * inner(grad(u), grad(v)) * dx)).coefficients() == (f, g)


def test_integrals():
    V = function_space()
    A = mass(V)
    B = A + stiffness(V)
    F = load(V)
    f, = F.coefficients()
    G = boundary_load(V)

    assert isinstance(A.integrals(), tuple)
    assert len(A.integrals()) == 1
    assert isinstance(A.integrals()[0], slate.SlateIntegral)
    assert A.integrals()[0].integral_type() == "cell"
    assert A.integrals_by_type("cell") == A.integrals()
    assert A.integrals_by_type("interior_facet") == ()
    assert isinstance(B.integrals(), tuple)
    assert len(B.integrals()) == 2
    assert [isinstance(B.integrals()[i], slate.SlateIntegral) for i in range(len(B.integrals()))]
    assert B.integrals_by_type("cell") == B.integrals()
    assert B.integrals_by_type("exterior_facet") == ()
    assert isinstance(F.integrals(), tuple)
    assert len(F.integrals()) == 1
    assert F.integrals()[0].integral_type() == "cell"
    assert isinstance(G.integrals(), tuple)
    assert len(G.integrals()) == 1
    assert G.integrals()[0].integral_type() == "exterior_facet"
    assert len(G.integrals_by_type("exterior_facet")) == 1


def test_unary_ops():
    V = function_space()
    A = mass(V)
    B = stiffness(V)
    F = load(V)
    G = boundary_load(V)

    assert isinstance(A.T, slate.Transpose)
    assert isinstance(B.inv, slate.Inverse)
    assert isinstance(-A, slate.Negative)
    assert isinstance(+B, slate.Positive)
    assert isinstance(-F, slate.Negative)
    assert isinstance(+G, slate.Positive)


def test_binary_ops():
    V = function_space()
    A = mass(V)
    B = stiffness(V)
    F = load(V)
    G = boundary_load(V)

    assert isinstance(A*F, slate.TensorMul)
    assert isinstance(B*G, slate.TensorMul)
    assert isinstance(A*B, slate.TensorMul)
    assert isinstance(A + B, slate.TensorAdd)
    assert isinstance(B - A, slate.TensorSub)
    assert isinstance(F + G, slate.TensorAdd)
    assert isinstance(G - F, slate.TensorSub)


def test_slate_expression_args():
    V = function_space()
    A = mass(V)
    B = stiffness(V)
    F = load(V)
    G = boundary_load(V)
    u = TrialFunction(V)
    v = TestFunction(V)

    assert (A.T).arguments() == (u, v)
    assert (A.inv).arguments() == (u, v)
    assert (A.T + B.inv).arguments() == (u, v)
    assert (F.T).arguments() == (v,)
    assert (F.T + G.T).arguments() == (v,)
    assert (A*F).arguments() == (v,)
    assert (B*G).arguments() == (v,)
    assert ((A + B) * (F - G)).arguments() == (v,)


def test_slate_expression_coeffs():
    V = function_space()
    A = mass(V)
    B = stiffness(V)
    F = load(V)
    G = boundary_load(V)
    f, = F.coefficients()
    g, = G.coefficients()

    assert (A.T).coefficients() == ()
    assert (A.inv).coefficients() == ()
    assert (A.T + B.inv).coefficients() == ()
    assert (F.T).coefficients() == (f,)
    assert (G.T).coefficients() == (g,)
    assert (F + G).coefficients() == (f, g)
    assert (F.T - G.T).coefficients() == (f, g)
    assert (A*F).coefficients() == (f,)
    assert (B*G).coefficients() == (g,)
    assert (A*G + B*F).coefficients() == (g, f)


def test_slate_eq():
    V = function_space()
    u = TrialFunction(V)
    v = TestFunction(V)
    A = slate.Matrix(inner(grad(u), grad(v)) * dx + u * v * dx)
    B = A
    C = B
    F = load(V)
    G = boundary_load(V)
    M = slate.Matrix((u/2)**2*(v + 1)**2 * dx)
    f, = F.coefficients()
    N = slate.Matrix(f * inner(grad(u), grad(v)) * dx + u * v * dx)

    assert A == B
    assert B == A
    assert C == B
    assert B == C
    assert A == C
    assert C == A
    assert A + B == B + A
    assert F != G
    assert A*F == A*F
    assert B*G == A*G
    assert A*F + A*G == C*F + B*G
    assert M*A != A*M
    assert A*G != M*G
    assert M*A == M*C
    assert A*(N + M) == C*(N + M)
    assert A*G == C*G
    assert A != M
    assert A != N
    assert A != mass(V)
    assert A != F
    assert A != u * v * dx


def test_rank_error_scalar():
    V = function_space()
    v = TestFunction(V)
    with pytest.raises(Exception):
        slate.Scalar(v * dx)


def test_rank_error_vector():
    with pytest.raises(Exception):
        slate.Vector(1 * dx)


def test_rank_error_matrix():
    with pytest.raises(Exception):
        slate.Matrix(1 * dx)


def test_dimension_error_matvecadd():
    V = function_space()
    A = mass(V)
    F = load(V)
    with pytest.raises(Exception):
        A + F


def test_dimension_error_vecmatsub():
    V = function_space()
    A = stiffness(V)
    G = boundary_load(V)
    with pytest.raises(Exception):
        G - A


def test_dimension_error_incompat_matadd():
    V = function_space()
    W = FunctionSpace(UnitSquareMesh(1, 1), "CG", 3)
    A = mass(V)
    B = stiffness(W)
    with pytest.raises(Exception):
        A + B


def test_dimension_error_incompat_matmul():
    V = function_space()
    W = FunctionSpace(UnitSquareMesh(1, 1), "CG", 3)
    A = mass(V)
    B = stiffness(W)
    with pytest.raises(Exception):
        B * A


def test_illegal_inverse():
    mesh = UnitSquareMesh(1, 1)
    RT = FunctionSpace(mesh, "RT", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    sigma = TrialFunction(RT)
    v = TestFunction(DG)
    A = slate.Matrix(v * div(sigma) * dx)
    with pytest.raises(AssertionError):
        A.inv


def test_illegal_compile():
    V = function_space()
    v = TestFunction(V)
    form = v * dx
    with pytest.raises(Exception):
        slate.slac.compile_slate_expression(form)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
