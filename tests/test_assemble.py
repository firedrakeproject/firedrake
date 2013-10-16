import pytest
import firedrake as fd


@pytest.fixture
def fs():
    mesh = fd.UnitSquareMesh(5, 5)
    return fd.FunctionSpace(mesh, "Lagrange", 1)


@pytest.fixture
def f(fs):
    f = fd.Function(fs, name="f")
    f.interpolate(fd.Expression("x[0]"))
    return f


@pytest.fixture
def one(fs):
    one = fd.Function(fs, name="one")
    one.interpolate(fd.Expression("1"))
    return one


@pytest.fixture
def M(fs):
    uhat = fd.TrialFunction(fs)
    v = fd.TestFunction(fs)
    return uhat * v * fd.dx


def test_one_form(M, f):
    one_form = fd.assemble(fd.action(M, f))
    assert isinstance(one_form, fd.Function)
    assert abs(sum(one_form.dat.data) - 0.5) < 1.0e-14


def test_zero_form(M, f, one):
    zero_form = fd.assemble(fd.action(fd.action(M, f), one))
    assert isinstance(zero_form, float)
    assert abs(zero_form - 0.5) < 1.0e-14

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
