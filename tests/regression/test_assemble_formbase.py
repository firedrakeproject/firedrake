import pytest
import numpy as np
from firedrake import *
from firedrake.utils import ScalarType
import ufl


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
    f_split = f.split()
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
    ones = one.split()

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


@pytest.fixture
def a(fs, f):
    v = TestFunction(fs)
    return inner(f, v) * dx


def test_assemble_cofun(a):
    res = assemble(a)
    assert isinstance(res, Cofunction)


def test_assemble_matrix(M):
    res = assemble(M)
    assert isinstance(res, ufl.Matrix)


def test_assemble_adjoint(M):
    res = assemble(adjoint(M))
    assembledM = assemble(M)
    res2 = assemble(adjoint(assembledM))
    assert isinstance(res, ufl.Matrix)
    assert res.M.handle == res.petscmat
    assert np.allclose(res.M.handle[:, :], res2.M.handle[:, :], rtol=1e-14)


def test_assemble_action(M, f):
    res = assemble(action(M, f))
    assembledM = assemble(M)
    res2 = assemble(action(assembledM, f))
    assert isinstance(res2, Cofunction)
    assert isinstance(res, Cofunction)
    assert abs(res.dat.data.sum() - res2.dat.data.sum()) < 1.0e-12
    for f in res2.split():
        if f.function_space().rank == 2:
            assert abs(f.dat.data.sum() - 0.5*sum(f.function_space().shape)) < 1.0e-12
        else:
            assert abs(f.dat.data.sum() - 0.5*f.function_space().value_size) < 1.0e-12


def test_vector_formsum(a):
    res = assemble(a)
    preassemble = assemble(a + a)
    formsum = res + a
    res2 = assemble(formsum)

    assert isinstance(formsum, ufl.form.FormSum)
    assert isinstance(res2, Cofunction)
    assert isinstance(preassemble, Cofunction)
    assert abs(preassemble.dat.data.sum() - res2.dat.data.sum()) < 1.0e-12


def test_matrix_formsum(M):
    res = assemble(M)
    sumfirst = assemble(M+M)
    formsum = res + M
    assert isinstance(formsum, ufl.form.FormSum)
    res2 = assemble(formsum)
    assert isinstance(res2, ufl.Matrix)
    assert np.allclose(sumfirst.petscmat[:, :],
                       res2.petscmat[:, :], rtol=1e-14)


def test_zero_form(M, f, one):
    zero_form = assemble(action(action(M, f), one))
    assert isinstance(zero_form, ScalarType.type)
    assert abs(zero_form - 0.5 * np.prod(f.ufl_shape)) < 1.0e-12


def helmholtz(r, quadrilateral=False, degree=2, mesh=None):
    # Create mesh and define function space
    if mesh is None:
        mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    lmbda = 1
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2))
    a = (inner(grad(u), grad(v)) + lmbda * inner(u, v)) * dx

    assembled_matrix = assemble(a)
    preassemble_action = assemble(action(a, f))
    postassemble_action = assemble(action(assembled_matrix, f))

    assert np.allclose(preassemble_action.M.values, postassemble_action.M.values, rtol=1e-14)
