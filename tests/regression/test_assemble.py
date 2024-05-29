import pytest
import numpy as np
from firedrake import *
from firedrake.utils import ScalarType, IntType


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module', params=['cg1', 'vcg1', 'tcg1',
                                        'cg1cg1', 'cg1cg1[0]', 'cg1cg1[1]',
                                        'cg1vcg1[0]', 'cg1vcg1[1]',
                                        'cg1dg0', 'cg1dg0[0]', 'cg1dg0[1]',
                                        'cg2dg1', 'cg2dg1[0]', 'cg2dg1[1]'])
def fs(request, mesh):
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    vcg1 = VectorFunctionSpace(mesh, "CG", 1)
    tcg1 = TensorFunctionSpace(mesh, "CG", 1)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    return {'cg1': cg1,
            'vcg1': vcg1,
            'tcg1': tcg1,
            'cg1cg1': cg1*cg1,
            'cg1cg1[0]': (cg1*cg1)[0],
            'cg1cg1[1]': (cg1*cg1)[1],
            'cg1vcg1': cg1*vcg1,
            'cg1vcg1[0]': (cg1*vcg1)[0],
            'cg1vcg1[1]': (cg1*vcg1)[1],
            'cg1dg0': cg1*dg0,
            'cg1dg0[0]': (cg1*dg0)[0],
            'cg1dg0[1]': (cg1*dg0)[1],
            'cg2dg1': cg2*dg1,
            'cg2dg1[0]': (cg2*dg1)[0],
            'cg2dg1[1]': (cg2*dg1)[1]}[request.param]


@pytest.fixture
def f(fs):
    f = Function(fs, name="f")
    f_split = f.subfunctions
    x = SpatialCoordinate(fs.mesh())[0]

    # NOTE: interpolation of UFL expressions into mixed
    # function spaces is not yet implemented
    for fi in f_split:
        fs_i = fi.function_space()
        if fs_i.rank == 1:
            fi.interpolate(as_vector((x,) * fs_i.value_size))
        elif fs_i.rank == 2:
            fi.interpolate(as_tensor([[x for i in range(fs_i.mesh().geometric_dimension())]
                                      for j in range(fs_i.rank)]))
        else:
            fi.interpolate(x)
    return f


@pytest.fixture
def one(fs):
    one = Function(fs, name="one")
    ones = one.subfunctions

    # NOTE: interpolation of UFL expressions into mixed
    # function spaces is not yet implemented
    for fi in ones:
        fs_i = fi.function_space()
        if fs_i.rank == 1:
            fi.interpolate(Constant((1.0,) * fs_i.value_size))
        elif fs_i.rank == 2:
            fi.interpolate(Constant([[1.0 for i in range(fs_i.mesh().geometric_dimension())]
                                     for j in range(fs_i.rank)]))
        else:
            fi.interpolate(Constant(1.0))
    return one


@pytest.fixture
def M(fs):
    uhat = TrialFunction(fs)
    v = TestFunction(fs)
    return inner(uhat, v) * dx


def test_one_form(M, f):
    one_form = assemble(action(M, f))
    assert isinstance(one_form, Cofunction)
    for f in one_form.subfunctions:
        if f.function_space().rank == 2:
            assert abs(f.dat.data.sum() - 0.5*sum(f.function_space().shape)) < 1.0e-12
        else:
            assert abs(f.dat.data.sum() - 0.5*f.function_space().value_size) < 1.0e-12


def test_zero_form(M, f, one):
    zero_form = assemble(action(action(M, f), one))
    assert isinstance(zero_form, ScalarType.type)
    assert abs(zero_form - 0.5 * np.prod(f.ufl_shape)) < 1.0e-12


def test_assemble_with_tensor(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    L = conj(v) * dx
    f = Cofunction(V.dual())
    # Assemble a form into f
    f = assemble(L, tensor=f)
    # Assemble a different form into f
    f = assemble(Constant(2)*L, tensor=f)
    # Make sure we get the result of the last assembly
    assert np.allclose(f.dat.data, 2*assemble(L).dat.data, rtol=1e-14)


def test_assemble_mat_with_tensor(mesh):
    V = FunctionSpace(mesh, "DG", 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v) * dx
    M = assemble(a)
    # Assemble a different form into M
    M = assemble(Constant(2)*a, tensor=M)
    # Make sure we get the result of the last assembly
    assert np.allclose(M.M.values, 2*assemble(a).M.values, rtol=1e-14)


def test_assemble_diagonal(mesh):
    V = FunctionSpace(mesh, "P", 3)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u, v)*dx
    M = assemble(a, mat_type="aij")
    Mdiag = assemble(a, diagonal=True)
    assert np.allclose(M.petscmat.getDiagonal().array_r, Mdiag.dat.data_ro)


def test_assemble_diagonal_bcs(mesh):
    V = FunctionSpace(mesh, "P", 3)
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, 0, (1, 4))
    a = inner(grad(u), grad(v))*dx
    M = assemble(a, mat_type="aij", bcs=bc)
    Mdiag = assemble(a, bcs=bc, diagonal=True)
    assert np.allclose(M.petscmat.getDiagonal().array_r, Mdiag.dat.data_ro)


def test_zero_bc_nodes(mesh):
    V = FunctionSpace(mesh, "P", 3)
    x, y = SpatialCoordinate(mesh)
    f = Function(V).interpolate(sin(x + y))
    v = TestFunction(V)
    bcs = [DirichletBC(V, 1.0, 1), DirichletBC(V, 2.0, 3)]
    a = inner(grad(f), grad(v))*dx

    b1 = assemble(a)
    for bc in bcs:
        bc.zero(b1)
    b2 = assemble(a, bcs=bcs, zero_bc_nodes=True)
    assert np.allclose(b1.dat.data, b2.dat.data)


@pytest.mark.xfail(reason="Assembler caching not currently supported for zero forms")
def test_zero_form_assembler_cache(mesh):
    from firedrake.assemble import _FORM_CACHE_KEY

    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    f = Function(V)
    zero_form = action(conj(v)*dx, f)

    assert _FORM_CACHE_KEY not in zero_form._cache.keys()

    assemble(zero_form)
    assert len(zero_form._cache[_FORM_CACHE_KEY]) == 1

    # changing form_compiler_parameters should increase the cache size
    assemble(zero_form, form_compiler_parameters={"quadrature_degree": 2})
    assert len(zero_form._cache[_FORM_CACHE_KEY]) == 2


def test_one_form_assembler_cache(mesh):
    from firedrake.assemble import _FORM_CACHE_KEY

    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    L = conj(v) * dx

    assert _FORM_CACHE_KEY not in L._cache.keys()

    assemble(L)
    assert len(L._cache[_FORM_CACHE_KEY]) == 1

    # changing tensor should not increase the cache size
    tensor = Cofunction(V.dual())
    assemble(L, tensor=tensor)
    assert len(L._cache[_FORM_CACHE_KEY]) == 1

    # changing bcs should increase the cache size
    bc = DirichletBC(V, 0, (1, 4))
    assemble(L, bcs=bc)
    assert len(L._cache[_FORM_CACHE_KEY]) == 2

    # changing form_compiler_parameters should increase the cache size
    assemble(L, form_compiler_parameters={"quadrature_degree": 2})
    assert len(L._cache[_FORM_CACHE_KEY]) == 3

    # changing zero_bc_nodes should increase the cache size
    assemble(L, zero_bc_nodes=True)
    assert len(L._cache[_FORM_CACHE_KEY]) == 4


@pytest.mark.xfail(reason="Assembler caching not currently supported for two forms")
def test_two_form_assembler_cache(mesh):
    from firedrake.assemble import _FORM_CACHE_KEY

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx

    assert _FORM_CACHE_KEY not in a._cache.keys()

    M = assemble(a)
    assert len(a._cache[_FORM_CACHE_KEY]) == 1

    # changing tensor should not increase the cache size
    assemble(a, tensor=M.copy())
    assert len(a._cache[_FORM_CACHE_KEY]) == 1

    # changing mat_type should not increase the cache size
    assemble(a, mat_type="nest")
    assert len(a._cache[_FORM_CACHE_KEY]) == 1

    # changing bcs should increase the cache size
    bc = DirichletBC(V, 0, (1, 4))
    assemble(a, bcs=bc)
    assert len(a._cache[_FORM_CACHE_KEY]) == 2

    # specifying diagonal should increase the cache size
    assemble(a, diagonal=True)
    assert len(a._cache[_FORM_CACHE_KEY]) == 3

    # changing form_compiler_parameters should increase the cache size
    assemble(a, form_compiler_parameters={"quadrature_degree": 2})
    assert len(a._cache[_FORM_CACHE_KEY]) == 4


def test_assemble_only_valid_with_floats(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V, dtype=np.int32)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(f*u, v) * dx
    with pytest.raises(ValueError):
        assemble(a)


def test_assemble_mixed_function_sparse():
    mesh = UnitSquareMesh(1, 1)
    V0 = FunctionSpace(mesh, "CG", 1)
    V = V0 * V0 * V0 * V0 * V0 * V0 * V0 * V0
    f = Function(V)
    f.sub(1).interpolate(Constant(2.0))
    f.sub(4).interpolate(Constant(3.0))
    v = assemble((inner(f[1], f[1]) + inner(f[4], f[4])) * dx)
    assert np.allclose(v, 13.0)


def test_3125():
    # see https://github.com/firedrakeproject/firedrake/issues/3125
    mesh = UnitSquareMesh(3, 3)
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = MixedFunctionSpace([V, W])
    z = Function(Z)
    u, p = split(z)
    tst = TestFunction(Z)
    v, q = split(tst)
    d = Function(W)
    F = inner(z, tst)*dx + inner(u, v)/(d+p)*dx(2, degree=10)
    # should run without error
    solve(F == 0, z)


@pytest.mark.xfail(reason="Arguments on vector-valued R spaces are not supported")
def test_assemble_vector_rspace_one_form(mesh):
    V = VectorFunctionSpace(mesh, "Real", 0, dim=2)
    u = Function(V)
    U = inner(u, u)*dx
    L = derivative(U, u)
    assemble(L)


def test_assemble_sparsity_no_redundant_entries():
    mesh = UnitSquareMesh(2, 2, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V * V
    u = TrialFunction(W)
    v = TestFunction(W)
    A = assemble(inner(u, v) * dx, mat_type="nest")
    for i in range(len(W)):
        for j in range(len(W)):
            if i != j:
                assert np.all(A.M.sparsity[i][j].nnz == np.zeros(9, dtype=IntType))


def test_assemble_sparsity_diagonal_entries_for_bc():
    mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V
    u = TrialFunction(W)
    v = TestFunction(W)
    bc = DirichletBC(W.sub(1), 0, "on_boundary")
    A = assemble(inner(u[1], v[0]) * dx, bcs=[bc], mat_type="nest")
    # Make sure that diagonals are allocated.
    assert np.all(A.M.sparsity[1][1].nnz == np.ones(4, dtype=IntType))
