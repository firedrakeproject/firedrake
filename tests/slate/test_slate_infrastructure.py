import pytest
from firedrake import *
from firedrake.formmanipulation import ExtractSubBlock
import math


@pytest.fixture(scope='module', params=[interval, triangle, quadrilateral])
def mesh(request):
    """Generate a mesh according to the cell provided."""
    cell = request.param
    if cell == interval:
        return UnitIntervalMesh(1)
    elif cell == triangle:
        return UnitSquareMesh(1, 1)
    elif cell == quadrilateral:
        return UnitSquareMesh(1, 1, quadrilateral=True)
    else:
        raise ValueError("%s cell not recognized" % cell)


@pytest.fixture(scope='module', params=['cg1', 'cg2', 'dg0', 'dg1'])
def function_space(request, mesh):
    """Generates function spaces for testing SLATE tensor assembly."""
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    return {'cg1': cg1,
            'cg2': cg2,
            'dg0': dg0,
            'dg1': dg1}[request.param]


@pytest.fixture(scope='module', params=['cg2-cg1-dg0', 'cg1-dg1-dg0'])
def mixed_space(request, mesh):
    """Generates a mixed function space."""
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    return {'cg2-cg1-dg0': cg2*cg1*dg0,
            'cg1-dg1-dg0': cg1*dg1*dg0}[request.param]


@pytest.fixture
def mass(function_space):
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    return Tensor(u * v * dx)


@pytest.fixture
def stiffness(function_space):
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    return Tensor(inner(grad(u), grad(v)) * dx)


@pytest.fixture
def load(function_space):
    f = Function(function_space)
    x = SpatialCoordinate(function_space.mesh())
    f.interpolate(cos(x[0])*math.pi*2)
    v = TestFunction(function_space)
    return Tensor(f * v * dx)


@pytest.fixture
def boundary_load(function_space):
    f = Function(function_space)
    x = SpatialCoordinate(function_space.mesh())
    if function_space.mesh().cell_dimension() == 1:
        f.interpolate(cos(x[0]*math.pi*2))
    else:
        f.interpolate(cos(x[1] * math.pi*2))
    v = TestFunction(function_space)
    return Tensor(f * v * ds)


@pytest.fixture
def zero_rank_tensor(function_space):
    c = Function(function_space)
    x = SpatialCoordinate(function_space.mesh())
    if function_space.mesh().cell_dimension() == 1:
        c.interpolate(x[0]*x[0])
    else:
        c.interpolate(x[0]*x[1])
    return Tensor(c * dx)


@pytest.fixture
def mixed_matrix(mixed_space):
    u, p, r = TrialFunctions(mixed_space)
    v, q, s = TestFunctions(mixed_space)
    return Tensor(u*v*dx + p*q*dx + r*s*dx)


@pytest.fixture
def mixed_vector(mixed_space):
    v, q, s = TestFunctions(mixed_space)
    return Tensor(v*dx + q*dx + s*dx)


def test_arguments(mass, stiffness, load, boundary_load, zero_rank_tensor):
    S = zero_rank_tensor
    M = mass
    N = stiffness
    F = load
    G = boundary_load

    c, = S.form.coefficients()
    f, = F.form.coefficients()
    g, = G.form.coefficients()
    v, u = M.form.arguments()

    assert len(N.arguments()) == N.rank
    assert len(M.arguments()) == M.rank
    assert N.arguments() == (v, u)
    assert len(F.arguments()) == F.rank
    assert len(G.arguments()) == G.rank
    assert F.arguments() == (v,)
    assert G.arguments() == F.arguments()
    assert len(S.arguments()) == S.rank
    assert S.arguments() == ()
    assert (M.T).arguments() == (u, v)
    assert (N.inv).arguments() == (u, v)
    assert (N.T + M.inv).arguments() == (u, v)
    assert (F.T).arguments() == (v,)
    assert (F.T + G.T).arguments() == (v,)
    assert (M*F).arguments() == (v,)
    assert (N*G).arguments() == (v,)
    assert ((M + N) * (F - G)).arguments() == (v,)

    assert Tensor(v * dx).arguments() == (v,)
    assert (Tensor(v * dx) + Tensor(f * v * ds)).arguments() == (v,)
    assert (M + N).arguments() == (v, u)
    assert (Tensor((f * v) * u * dx) + Tensor((u * 3) * (v / 2) * dx)).arguments() == (v, u)
    assert (G - F).arguments() == (v,)


def test_coefficients(mass, stiffness, load, boundary_load, zero_rank_tensor):
    S = zero_rank_tensor
    M = mass
    N = stiffness
    F = load
    G = boundary_load

    c, = S.form.coefficients()
    f, = F.form.coefficients()
    g, = G.form.coefficients()
    v, u = M.form.arguments()

    assert S.coefficients() == (c,)
    assert F.coefficients() == (f,)
    assert G.coefficients() == (g,)
    assert (M*F).coefficients() == (f,)
    assert (N*G).coefficients() == (g,)
    assert (N*F + M*G).coefficients() == (f, g)
    assert (M.T).coefficients() == ()
    assert (M.inv).coefficients() == ()
    assert (M.T + N.inv).coefficients() == ()
    assert (F.T).coefficients() == (f,)
    assert (G.T).coefficients() == (g,)
    assert (F + G).coefficients() == (f, g)
    assert (F.T - G.T).coefficients() == (f, g)

    assert Tensor(f * dx).coefficients() == (f,)
    assert (Tensor(f * dx) + Tensor(f * ds)).coefficients() == (f,)
    assert (Tensor(f * dx) + Tensor(g * dS)).coefficients() == (f, g)
    assert Tensor(f * v * dx).coefficients() == (f,)
    assert (Tensor(f * v * ds) + Tensor(f * v * dS)).coefficients() == (f,)
    assert (Tensor(f * v * dx) + Tensor(g * v * ds)).coefficients() == (f, g)
    assert Tensor(f * u * v * dx).coefficients() == (f,)
    assert (Tensor(f * u * v * dx) + Tensor(f * inner(grad(u), grad(v)) * dx)).coefficients() == (f,)
    assert (Tensor(f * u * v * dx) + Tensor(g * inner(grad(u), grad(v)) * dx)).coefficients() == (f, g)


def test_integral_information(mass, stiffness, load, boundary_load, zero_rank_tensor):
    S = zero_rank_tensor
    M = mass
    N = stiffness
    F = load
    G = boundary_load

    # Checks the generated information of the tensor agrees with the original
    # data directly in its associated `ufl.Form` object
    assert S.ufl_domain() == S.form.ufl_domain()
    assert M.ufl_domain() == M.form.ufl_domain()
    assert N.ufl_domain() == N.form.ufl_domain()
    assert F.ufl_domain() == F.form.ufl_domain()
    assert G.ufl_domain() == G.form.ufl_domain()
    assert M.inv.ufl_domain() == M.form.ufl_domain()
    assert M.T.ufl_domain() == M.form.ufl_domain()
    assert (-N).ufl_domain() == N.form.ufl_domain()
    assert (F + G).ufl_domain() == (F.form + G.form).ufl_domain()
    assert (M + N).ufl_domain() == (M.form + N.form).ufl_domain()

    assert _is_equal_subdomain_data(S, S.form)
    assert _is_equal_subdomain_data(N, N.form)
    assert _is_equal_subdomain_data(M, M.form)
    assert _is_equal_subdomain_data(F, F.form)
    assert _is_equal_subdomain_data(N.inv, N.form)
    assert _is_equal_subdomain_data(-M, M.form)
    assert _is_equal_subdomain_data((M + N).T, (M.form + N.form))
    assert _is_equal_subdomain_data((F + G), (F.form + G.form))


def test_equality_relations(function_space):
    # Small test to check hash functions
    V = function_space
    u = TrialFunction(V)
    v = TestFunction(V)

    f = AssembledVector(Function(V))
    A = Tensor(u * v * dx)
    B = Tensor(inner(grad(u), grad(v)) * dx)

    assert A == Tensor(u * v * dx)
    assert B != A
    assert B * f != A * f
    assert A + B == Tensor(u * v * dx) + Tensor(inner(grad(u), grad(v)) * dx)
    assert A*B != B*A
    assert B.T != B.inv
    assert A != -A


def test_blocks(zero_rank_tensor, mixed_matrix, mixed_vector):
    S = zero_rank_tensor
    M = mixed_matrix
    F = mixed_vector
    a = M.form
    L = F.form
    splitter = ExtractSubBlock()
    _M = M.blocks
    M00 = _M[0, 0]
    M11 = _M[1, 1]
    M22 = _M[2, 2]
    M0101 = _M[:2, :2]
    M012 = _M[:2, 2]
    M201 = _M[2, :2]

    _F = F.blocks
    F0 = _F[0]
    F1 = _F[1]
    F2 = _F[2]
    F01 = _F[:2]
    F12 = _F[1:3]

    # Test make indexing with too few indices legal
    assert _M[2] == _M[2, :3]

    # Test index checking
    with pytest.raises(ValueError):
        S.blocks[0]

    with pytest.raises(ValueError):
        _F[0, 1]

    with pytest.raises(ValueError):
        _M[0:5, 2]

    with pytest.raises(ValueError):
        _M[3, 3]

    with pytest.raises(ValueError):
        _F[3]

    # Check Tensor is (not) mixed where appropriate
    assert not M00.is_mixed
    assert not M11.is_mixed
    assert not M22.is_mixed
    assert not F0.is_mixed
    assert not F1.is_mixed
    assert not F2.is_mixed
    assert M0101.is_mixed
    assert M012.is_mixed
    assert M201.is_mixed
    assert F01.is_mixed
    assert F12.is_mixed

    # Taking blocks of non-mixed block (or scalars) should induce a no-op
    assert S.blocks[None] == S   # This is silly, but it's technically a no-op
    assert M00.blocks[0, 0] == M00
    assert M11.blocks[0, 0] == M11
    assert M22.blocks[0, 0] == M22
    assert F0.blocks[0] == F0
    assert F1.blocks[0] == F1
    assert F2.blocks[0] == F2

    # Test arguments
    assert M00.arguments() == splitter.split(a, (0, 0)).arguments()
    assert M11.arguments() == splitter.split(a, (1, 1)).arguments()
    assert M22.arguments() == splitter.split(a, (2, 2)).arguments()
    assert F0.arguments() == splitter.split(L, (0,)).arguments()
    assert F1.arguments() == splitter.split(L, (1,)).arguments()
    assert F2.arguments() == splitter.split(L, (2,)).arguments()
    assert M0101.arguments() == splitter.split(a, ((0, 1), (0, 1))).arguments()
    assert M012.arguments() == splitter.split(a, ((0, 1), (2,))).arguments()
    assert M201.arguments() == splitter.split(a, ((2,), (0, 1))).arguments()
    assert F01.arguments() == splitter.split(L, ((0, 1),)).arguments()
    assert F12.arguments() == splitter.split(L, ((1, 2),)).arguments()


def test_illegal_add_sub():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = Tensor(u * v * dx)
    b = Tensor(v * dx)
    c = Function(V)
    c.interpolate(Constant(1))
    s = Tensor(c * dx)

    with pytest.raises(ValueError):
        A + b

    with pytest.raises(ValueError):
        s + b

    with pytest.raises(ValueError):
        b - A

    with pytest.raises(ValueError):
        A - s


def test_ops_NotImplementedError():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    A = Tensor(u * v * dx)

    with pytest.raises(NotImplementedError):
        A + f

    with pytest.raises(NotImplementedError):
        f + A

    with pytest.raises(NotImplementedError):
        A - f

    with pytest.raises(NotImplementedError):
        f - A

    with pytest.raises(NotImplementedError):
        f * A


def test_illegal_mul():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    W = FunctionSpace(mesh, "CG", 2)
    w = TrialFunction(W)
    x = TestFunction(W)

    A = Tensor(u * v * dx)
    B = Tensor(w * x * dx)

    with pytest.raises(ValueError):
        B * A

    with pytest.raises(ValueError):
        A * B


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
    from firedrake.slate.slac import compile_expression as compile_slate
    V = FunctionSpace(UnitSquareMesh(1, 1), "CG", 1)
    v = TestFunction(V)
    form = v * dx
    with pytest.raises(ValueError):
        compile_slate(form)


def _is_equal_subdomain_data(a, b):
    """Compare subdomain data of a and b."""
    sd_a = {domain: {integral_type: [v for v in val if v is not None]}
            for domain, data in a.subdomain_data().items()
            for integral_type, val in data.items()}
    sd_b = {domain: {integral_type: [v for v in val if v is not None]}
            for domain, data in b.subdomain_data().items()
            for integral_type, val in data.items()}
    return sd_a == sd_b
