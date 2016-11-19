import pytest
from firedrake import *


def function_space():
    mesh = UnitSquareMesh(1, 1)
    return FunctionSpace(mesh, "CG", 1)


def mass(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    return Tensor(u * v * dx)


def stiffness(fs):
    u = TrialFunction(fs)
    v = TestFunction(fs)
    return Tensor(inner(grad(u), grad(v)) * dx)


def load(fs):
    f = Function(fs)
    f.interpolate(Expression("cos(x[0]*pi*2)"))
    v = TestFunction(fs)
    return Tensor(f * v * dx)


def boundary_load(fs):
    f = Function(fs)
    f.interpolate(Expression("cos(x[1]*pi*2)"))
    v = TestFunction(fs)
    return Tensor(f * v * ds)


def test_arguments():
    V = function_space()
    M = mass(V)
    N = stiffness(V)
    v, u = M.arguments()
    F = load(V)
    f, = F.coefficients()
    S = Tensor(f * dx)

    assert len(N.arguments()) == 2
    assert len(M.arguments()) == 2
    assert N.arguments() == (v, u)
    assert len(F.arguments()) == 1
    assert F.arguments() == (v,)
    assert len(S.arguments()) == 0
    assert S.arguments() == ()

    assert Tensor(v * dx).arguments() == (v,)
    assert (Tensor(v * dx) + Tensor(f * v * ds)).arguments() == (v,)
    assert (M + N).arguments() == (v, u)
    assert (Tensor((f * v) * u * dx) + Tensor((u * 3) * (v / 2) * dx)).arguments() == (v, u)


def test_coefficients():
    V = function_space()
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    g = Function(V)
    f.interpolate(Expression("cos(x[0]*pi*2)"))
    g.interpolate(Expression("cos(x[1]*pi*2)"))

    assert Tensor(f * dx).coefficients() == (f,)
    assert (Tensor(f * dx) + Tensor(f * ds)).coefficients() == (f,)
    assert (Tensor(f * dx) + Tensor(g * dS)).coefficients() == (f, g)
    assert Tensor(f * v * dx).coefficients() == (f,)
    assert (Tensor(f * v * ds) + Tensor(f * v * dS)).coefficients() == (f,)
    assert (Tensor(f * v * dx) + Tensor(g * v * ds)).coefficients() == (f, g)
    assert Tensor(f * u * v * dx).coefficients() == (f,)
    assert (Tensor(f * u * v * dx) + Tensor(f * inner(grad(u), grad(v)) * dx)).coefficients() == (f,)
    assert (Tensor(f * u * v * dx) + Tensor(g * inner(grad(u), grad(v)) * dx)).coefficients() == (f, g)


def test_unary_ops():
    V = function_space()
    A = mass(V)
    B = stiffness(V)
    F = load(V)

    assert isinstance(A.T, slate.Transpose)
    assert isinstance(B.inv, slate.Inverse)
    assert isinstance(-A, slate.Negative)
    assert isinstance(-F, slate.Negative)


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
    A = Tensor(v * div(sigma) * dx)
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
