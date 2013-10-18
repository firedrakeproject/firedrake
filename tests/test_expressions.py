import pytest

import numpy as np
import ufl

import firedrake as fd


@pytest.fixture(scope='module')
def mesh():
    return fd.UnitSquareMesh(5, 5)


@pytest.fixture(scope='module')
def fs(mesh):
    return fd.FunctionSpace(mesh, "Lagrange", 1)


@pytest.fixture(scope='module')
def vfs(mesh):
    return fd.VectorFunctionSpace(mesh, "Lagrange", 1)


@pytest.fixture
def f(fs):
    return fd.Function(fs, name="f")


@pytest.fixture
def one(fs):
    f = fd.Function(fs, name="one")
    f.interpolate(fd.Expression("1"))
    return f


@pytest.fixture
def two(fs):
    f = fd.Function(fs, name="two")
    f.interpolate(fd.Expression("2"))
    return f


@pytest.fixture
def minusthree(fs):
    f = fd.Function(fs, name="minusthree")
    f.interpolate(fd.Expression("-3"))
    return f


@pytest.fixture
def vf(vfs):
    return fd.Function(vfs, name="vf")


@pytest.fixture
def vone(vfs):
    vf = fd.Function(vfs, name="vone")
    vf.assign(1)
    return vf


@pytest.fixture
def vtwo(vfs):
    vf = fd.Function(vfs, name="vtwo")
    vf.assign(2)
    return vf


@pytest.fixture
def vminusthree(vfs):
    vf = fd.Function(vfs, name="vminusthree")
    vf.assign(-3)
    return vf

exprtest = lambda expr, x: (expr, x, np.all(fd.assemble(expr).dat.data == x))

assigntest = lambda f, expr, x: (str(f) + " = " + str(expr) + ", " + str(f), x,
                                 np.all(f.assign(expr).dat.data == x))


def iaddtest(f, expr, x):
    f += expr
    return (str(f) + " += " + str(expr) + ", " + str(f), x,
            np.all(f.dat.data == x))


def isubtest(f, expr, x):
    f -= expr
    return (str(f) + " -= " + str(expr) + ", " + str(f), x,
            np.all(f.dat.data == x))


def imultest(f, expr, x):
    f *= expr
    return (str(f) + " *= " + str(expr) + ", " + str(f), x,
            np.all(f.dat.data == x))


def idivtest(f, expr, x):
    f /= expr
    return (str(f) + " /= " + str(expr) + ", " + str(f), x,
            np.all(f.dat.data == x))


@pytest.fixture(params=range(1, 23))
def alltests(request, f, one, two, minusthree, vf, vone, vtwo, vminusthree):
    return {
        1: exprtest(one + two, 3),
        2: exprtest(ufl.ln(one), 0),
        3: exprtest(two ** minusthree, 0.125),
        4: exprtest(ufl.sign(minusthree), -1),
        5: exprtest(one + two / two ** minusthree, 17),
        6: assigntest(f, one + two, 3),
        7: iaddtest(f, two, 5),
        8: iaddtest(f, 2, 7),
        9: isubtest(f, 2, 5),
        10: imultest(f, 2, 10),
        11: idivtest(f, 2, 5),
        12: exprtest(vone + vtwo, 3),
        13: exprtest(ufl.ln(vone), 0),
        14: exprtest(vtwo ** vminusthree, 0.125),
        15: exprtest(ufl.sign(vminusthree), -1),
        16: exprtest(vone + vtwo / vtwo ** vminusthree, 17),
        17: assigntest(vf, vone + vtwo, 3),
        18: iaddtest(vf, vtwo, 5),
        19: iaddtest(vf, 2, 7),
        20: isubtest(vf, 2, 5),
        21: imultest(vf, 2, 10),
        22: idivtest(vf, 2, 5)
    }[request.param]


def test_expressions(alltests):
    assert alltests[2]


def test_vf_assign_f(f, vf):
    with pytest.raises(ValueError):
        vf.assign(f)


def test_f_assign_vf(f, vf):
    with pytest.raises(ValueError):
        f.assign(vf)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
