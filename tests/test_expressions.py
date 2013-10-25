import pytest

import numpy as np
import ufl

from firedrake import *
from common import *


@pytest.fixture(scope='module')
def vcg1(mesh):
    return VectorFunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope='module', params=['cg1', 'vcg1'])
def fs(request, cg1, vcg1):
    return {'cg1': cg1, 'vcg1': vcg1}[request.param]


@pytest.fixture(params=['assign', 'interpolate'])
def functions(request, fs):
    f = Function(fs, name="f")
    one = Function(fs, name="one")
    two = Function(fs, name="two")
    minusthree = Function(fs, name="minusthree")
    if request.param == 'assign':
        one.assign(1)
        two.assign(2)
        minusthree.assign(-3)
    elif isinstance(fs, VectorFunctionSpace):
        one.interpolate(Expression(("1",)*one.geometric_dimension()))
        two.interpolate(Expression(("2",)*two.geometric_dimension()))
        minusthree.interpolate(Expression(("-3",)*minusthree.geometric_dimension()))
    else:
        one.interpolate(Expression("1"))
        two.interpolate(Expression("2"))
        minusthree.interpolate(Expression("-3"))
    return f, one, two, minusthree


@pytest.fixture
def sf(cg1):
    return Function(cg1, name="sf")


@pytest.fixture
def vf(vcg1):
    return Function(vcg1, name="vf")

exprtest = lambda expr, x: (expr, x, np.all(assemble(expr).dat.data == x))

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


@pytest.fixture(params=range(1, 12))
def alltests(request, functions):
    f, one, two, minusthree = functions
    # Not all test cases work for vector function spaces
    if isinstance(one.function_space(), VectorFunctionSpace):
        return {
            1: exprtest(one + two, 3),
            2: (None, None, True),
            3: (None, None, True),
            4: (None, None, True),
            5: (None, None, True),
            6: assigntest(f, one + two, 3),
            7: iaddtest(f, two, 5),
            8: iaddtest(f, 2, 7),
            9: isubtest(f, 2, 5),
            10: imultest(f, 2, 10),
            11: idivtest(f, 2, 5),
        }[request.param]
    else:
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
        }[request.param]


def test_expressions(alltests):
    assert alltests[2]


def test_vf_assign_f(sf, vf):
    with pytest.raises(ValueError):
        vf.assign(sf)


def test_f_assign_vf(sf, vf):
    with pytest.raises(ValueError):
        sf.assign(vf)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
