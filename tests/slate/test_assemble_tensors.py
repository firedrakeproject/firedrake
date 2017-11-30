import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    return UnitSquareMesh(2, 2, quadrilateral=request.param)


@pytest.fixture(scope='module', params=['cg1', 'cg2', 'dg0', 'dg1',
                                        'vcg1', 'vcg2', 'tcg1', 'tcg2'])
def function_space(request, mesh):
    """Generates function spaces for testing SLATE tensor assembly."""
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    vcg1 = VectorFunctionSpace(mesh, "CG", 1)
    vcg2 = VectorFunctionSpace(mesh, "CG", 2)
    tcg1 = TensorFunctionSpace(mesh, "CG", 1)
    tcg2 = TensorFunctionSpace(mesh, "CG", 2)
    return {'cg1': cg1,
            'cg2': cg2,
            'dg0': dg0,
            'dg1': dg1,
            'vcg1': vcg1,
            'vcg2': vcg2,
            'tcg1': tcg1,
            'tcg2': tcg2}[request.param]


@pytest.fixture
def f(function_space):
    """Generate a Firedrake function given a particular function space."""
    f = Function(function_space)
    if function_space.rank >= 1:
        f.interpolate(Expression(("x[0]*x[1]",) * function_space.value_size))
    else:
        f.interpolate(Expression("x[0]*x[1]"))
    return f


@pytest.fixture
def g(function_space):
    """Generates a Firedrake function given a particular function space."""
    g = Function(function_space)
    if function_space.rank >= 1:
        g.interpolate(Expression(("x[0]*sin(x[1])",) * function_space.value_size))
    else:
        g.interpolate(Expression("x[0]*sin(x[1])"))
    return g


@pytest.fixture
def mass(function_space):
    """Generate a generic mass form."""
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    return inner(u, v) * dx


@pytest.fixture
def rank_one_tensor(mass, f):
    return Tensor(action(mass, f))


@pytest.fixture
def rank_two_tensor(mass):
    return Tensor(mass)


def test_tensor_action(mass, f):
    V = assemble(Tensor(mass) * AssembledVector(f))
    ref = assemble(action(mass, f))
    assert isinstance(V, Function)
    assert np.allclose(V.dat.data, ref.dat.data, rtol=1e-14)


def test_sum_tensor_actions(mass, f, g):
    V = assemble(Tensor(mass) * AssembledVector(f)
                 + Tensor(0.5*mass) * AssembledVector(g))
    ref = assemble(action(mass, f) + action(0.5*mass, g))
    assert isinstance(V, Function)
    assert np.allclose(V.dat.data, ref.dat.data, rtol=1e-14)


def test_assemble_vector(rank_one_tensor):
    V = assemble(rank_one_tensor)
    assert isinstance(V, Function)
    assert np.allclose(V.dat.data, assemble(rank_one_tensor.form).dat.data, rtol=1e-14)


def test_assemble_matrix(rank_two_tensor):
    M = assemble(rank_two_tensor)
    assert np.allclose(M.M.values, assemble(rank_two_tensor.form).M.values, rtol=1e-14)


def test_assemble_vector_into_tensor(mesh):
    V = FunctionSpace(mesh, "DG", 1)
    v = TestFunction(V)
    f = Function(V)
    # Assemble a SLATE tensor into f
    f = assemble(Tensor(v * dx), f)
    # Assemble a different tensor into f
    f = assemble(Tensor(Constant(2) * v * dx), f)
    assert np.allclose(f.dat.data, 2*assemble(Tensor(v * dx)).dat.data, rtol=1e-14)


def test_assemble_matrix_into_tensor(mesh):
    V = FunctionSpace(mesh, "DG", 0)
    u = TestFunction(V)
    v = TrialFunction(V)
    M = assemble(Tensor(u * v * dx))
    # Assemble a different SLATE tensor into M
    M = assemble(Tensor(Constant(2) * u * v * dx), M)
    assert np.allclose(M.M.values, 2*assemble(Tensor(u * v * dx)).M.values, rtol=1e-14)


def test_mixed_coefficient_matrix(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = V * U
    f = Function(W)
    f.assign(1)
    u = TrialFunction(V)
    v = TestFunction(V)
    T = Tensor((f[0] + f[1]) * u * v * dx)
    ref = assemble((f[0] + f[1]) * u * v * dx)

    assert np.allclose(assemble(T).M.values, ref.M.values, rtol=1e-14)


def test_mixed_coefficient_scalar(mesh):
    V = FunctionSpace(mesh, "DG", 0)
    W = V * V
    f = Function(W)
    g, h = f.split()
    f.assign(1)
    assert np.allclose(assemble(Tensor((g + f[0] + h + f[1])*dx)), 4.0)


def test_nested_coefficients_matrix(mesh):
    V = VectorFunctionSpace(mesh, "CG", 1)
    U = FunctionSpace(mesh, "CG", 1)
    f = Function(U).assign(2.0)
    n = FacetNormal(mesh)

    def T(arg):
        k = Constant([0.0, 1.0])
        return k*inner(arg, k)

    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(v, f*u)*dx - div(T(v))*inner(u, n)*ds
    A = Tensor(form)
    M = assemble(A)

    assert np.allclose(M.M.values, assemble(form).M.values, rtol=1e-14)


@pytest.mark.xfail(raises=NotImplementedError)
def test_mixed_argument_tensor(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = V * U
    sigma, _ = TrialFunctions(W)
    tau, _ = TestFunctions(W)
    T = Tensor(sigma * tau * dx)
    assemble(T)


def test_assemble_indexed_vectors(mesh):
    V = FunctionSpace(mesh, "DG", 4)
    U = FunctionSpace(mesh, "DG", 2)
    W = V * U
    q = Function(V).assign(10.0)
    p = Function(U).assign(42.0)
    u, phi = TrialFunctions(W)
    v, psi = TestFunctions(W)

    K = Tensor(inner(u, v)*dx + inner(phi, psi)*dx)
    F = Tensor(inner(q, v)*dx + inner(p, psi)*dx)
    E = K.inv * F
    foo = assemble(E[0])
    bar = assemble(E[1])

    assert np.allclose(foo.dat.data, q.dat.data, rtol=1e-14)
    assert np.allclose(bar.dat.data, p.dat.data, rtol=1e-14)


def test_assemble_indexed_matrices(mesh):
    U = VectorFunctionSpace(mesh, "CG", 2)
    V = FunctionSpace(mesh, "DG", 1)
    W = U * V
    u, p = TrialFunctions(W)
    w, q = TestFunctions(W)

    A = Tensor(inner(u, w)*dx + p*q*dx + div(w)*p*dx + q*div(u)*dx)
    A00 = assemble(A[0, 0])
    A01 = assemble(A[0, 1])
    A10 = assemble(A[1, 0])
    A11 = assemble(A[1, 1])

    u = TrialFunction(U)
    w = TestFunction(U)
    p = TrialFunction(V)
    q = TestFunction(V)
    A00ref = assemble(inner(u, w)*dx)
    A01ref = assemble(div(w)*p*dx)
    A10ref = assemble(q*div(u)*dx)
    A11ref = assemble(p*q*dx)

    assert np.allclose(A00.M.values, A00ref.M.values, rtol=1e-14)
    assert np.allclose(A01.M.values, A01ref.M.values, rtol=1e-14)
    assert np.allclose(A10.M.values, A10ref.M.values, rtol=1e-14)
    assert np.allclose(A11.M.values, A11ref.M.values, rtol=1e-14)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
