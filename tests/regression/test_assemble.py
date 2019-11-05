import pytest
import numpy as np
from firedrake import *


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


def test_one_form(M, f):
    one_form = assemble(action(M, f))
    assert isinstance(one_form, Function)
    for f in one_form.split():
        if f.function_space().rank == 2:
            assert abs(f.dat.data.sum() - 0.5*sum(f.function_space().shape)) < 1.0e-12
        else:
            assert abs(f.dat.data.sum() - 0.5*f.function_space().value_size) < 1.0e-12


def test_zero_form(M, f, one):
    zero_form = assemble(action(action(M, f), one))
    assert isinstance(zero_form, float)
    assert abs(zero_form - 0.5 * np.prod(f.ufl_shape)) < 1.0e-12


def test_assemble_with_tensor(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    L = v*dx
    f = Function(V)
    # Assemble a form into f
    f = assemble(L, f)
    # Assemble a different form into f
    f = assemble(Constant(2)*L, f)
    # Make sure we get the result of the last assembly
    assert np.allclose(f.dat.data, 2*assemble(L).dat.data, rtol=1e-14)


def test_assemble_mat_with_tensor(mesh):
    V = FunctionSpace(mesh, "DG", 0)
    u = TestFunction(V)
    v = TrialFunction(V)
    a = u*v*dx
    M = assemble(a)
    # Assemble a different form into M
    M = assemble(Constant(2)*a, M)
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
