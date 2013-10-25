import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module')
def cg1(mesh):
    return FunctionSpace(mesh, "Lagrange", 1)


@pytest.fixture(scope='module')
def cg1cg1(mesh):
    CG1 = FunctionSpace(mesh, "CG", 1)
    return CG1 * CG1


@pytest.fixture(scope='module')
def cg1dg0(mesh):
    CG1 = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)
    return CG1 * DG0


@pytest.fixture(scope='module')
def cg2dg1(mesh):
    CG2 = FunctionSpace(mesh, "CG", 2)
    DG1 = FunctionSpace(mesh, "DG", 1)
    return CG2 * DG1


@pytest.fixture(scope='module', params=['cg1', 'cg1cg1', 'cg1dg0', 'cg2dg1'])
def fs(request, cg1, cg1cg1, cg1dg0, cg2dg1):
    return {'cg1': cg1, 'cg1cg1': cg1cg1, 'cg1dg0': cg1dg0, 'cg2dg1': cg2dg1}[request.param]


@pytest.fixture
def f(fs):
    f = Function(fs, name="f")
    f.interpolate(Expression("x[0]"))
    return f


@pytest.fixture
def one(fs):
    one = Function(fs, name="one")
    one.interpolate(Expression("1"))
    return one


@pytest.fixture
def M(fs):
    uhat = TrialFunction(fs)
    v = TestFunction(fs)
    return inner(uhat, v) * dx


def test_one_form(M, f):
    one_form = assemble(action(M, f))
    assert isinstance(one_form, Function)
    for d in one_form.dat:
        assert abs(sum(d.data) - 0.5) < 1.0e-12


def test_zero_form(M, f, one):
    zero_form = assemble(action(action(M, f), one))
    assert isinstance(zero_form, float)
    assert abs(zero_form - 0.5 * np.prod(f.shape())) < 1.0e-12

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
