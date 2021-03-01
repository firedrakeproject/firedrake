import pytest
import numpy as np
from firedrake import *
from firedrake.formmanipulation import split_form


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
    f_split = f.split()
    x = SpatialCoordinate(function_space.mesh())

    # NOTE: interpolation of UFL expressions into mixed
    # function spaces is not yet implemented
    for fi in f_split:
        fs_i = fi.function_space()
        if fs_i.rank == 1:
            fi.interpolate(as_vector((x[0]*x[1],) * fs_i.value_size))
        elif fs_i.rank == 2:
            fi.interpolate(as_tensor([[x[0]*x[1] for i in range(fs_i.mesh().geometric_dimension())]
                                      for j in range(fs_i.rank)]))
        else:
            fi.interpolate(x[0]*x[1])
    return f


@pytest.fixture
def g(function_space):
    """Generates a Firedrake function given a particular function space."""
    g = Function(function_space)
    g_split = g.split()
    x = SpatialCoordinate(function_space.mesh())

    # NOTE: interpolation of UFL expressions into mixed
    # function spaces is not yet implemented
    for gi in g_split:
        fs_i = gi.function_space()
        if fs_i.rank == 1:
            gi.interpolate(as_vector((x[0]*sin(x[1]),) * fs_i.value_size))
        elif fs_i.rank == 2:
            gi.interpolate(as_tensor([[x[0]*sin(x[1]) for i in range(fs_i.mesh().geometric_dimension())]
                                      for j in range(fs_i.rank)]))
        else:
            gi.interpolate(x[0]*sin(x[1]))
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


def test_mixed_argument_tensor(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = V * U
    sigma, _ = TrialFunctions(W)
    tau, _ = TestFunctions(W)
    T = Tensor(sigma * tau * dx)
    As = assemble(T)
    A = assemble(sigma * tau * dx)
    for ms, m in zip(As.M, A.M):
        assert np.allclose(ms.values, m.values)


def test_vector_subblocks(mesh):
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
    E = K.inv * F
    _E = E.blocks
    items = [(_E[0], q), (_E[1], p), (_E[2], r)]

    for tensor, ref in items:
        assert np.allclose(assemble(tensor).dat.data, ref.dat.data, rtol=1e-14)


def test_matrix_subblocks(mesh):
    if mesh.ufl_cell() == quadrilateral:
        U = FunctionSpace(mesh, "RTCF", 1)
    else:
        U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    T = FunctionSpace(mesh, "HDiv Trace", 0)
    n = FacetNormal(mesh)
    W = U * V * T
    u, p, lambdar = TrialFunctions(W)
    w, q, gammar = TestFunctions(W)

    A = Tensor(inner(u, w)*dx + p*q*dx - div(w)*p*dx + q*div(u)*dx
               + lambdar('+')*jump(w, n=n)*dS + gammar('+')*jump(u, n=n)*dS
               + lambdar*gammar*ds)

    # Test individual blocks
    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
    refs = dict(split_form(A.form))
    _A = A.blocks
    for x, y in indices:
        ref = assemble(refs[x, y]).M.values
        block = _A[x, y]
        assert np.allclose(assemble(block).M.values, ref, rtol=1e-14)

    # Mixed blocks
    A0101 = _A[:2, :2]
    A1212 = _A[1:3, 1:3]

    _A0101 = A0101.blocks
    _A1212 = A1212.blocks

    # Block of blocks
    A0101_00 = _A0101[0, 0]
    A0101_11 = _A0101[1, 1]
    A0101_01 = _A0101[0, 1]
    A0101_10 = _A0101[1, 0]
    A1212_00 = _A1212[0, 0]
    A1212_11 = _A1212[1, 1]
    A1212_01 = _A1212[0, 1]
    A1212_10 = _A1212[1, 0]

    items = [(A0101_00, refs[(0, 0)]),
             (A0101_11, refs[(1, 1)]),
             (A0101_01, refs[(0, 1)]),
             (A0101_10, refs[(1, 0)]),
             (A1212_00, refs[(1, 1)]),
             (A1212_11, refs[(2, 2)]),
             (A1212_01, refs[(1, 2)]),
             (A1212_10, refs[(2, 1)])]

    # Test assembly of blocks of mixed blocks
    for tensor, form in items:
        ref = assemble(form).M.values
        assert np.allclose(assemble(tensor).M.values, ref, rtol=1e-14)
