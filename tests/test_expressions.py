import pytest

import numpy as np
import ufl

from firedrake import *
from common import *


@pytest.fixture(scope='module', params=['cg1', 'vcg1',
                                        'cg1cg1', 'cg1cg1[0]', 'cg1cg1[1]',
                                        'cg1vcg1', 'cg1vcg1[0]', 'cg1vcg1[1]',
                                        'cg1dg0', 'cg1dg0[0]', 'cg1dg0[1]',
                                        'cg2dg1', 'cg2dg1[0]', 'cg2dg1[1]'])
def fs(request, cg1, vcg1, cg1cg1, cg1vcg1, cg1dg0, cg2dg1):
    return {'cg1': cg1,
            'vcg1': vcg1,
            'cg1cg1': cg1cg1,
            'cg1cg1[0]': cg1cg1[0],
            'cg1cg1[1]': cg1cg1[1],
            'cg1vcg1': cg1vcg1,
            'cg1vcg1[0]': cg1vcg1[0],
            'cg1vcg1[1]': cg1vcg1[1],
            'cg1dg0': cg1dg0,
            'cg1dg0[0]': cg1dg0[0],
            'cg1dg0[1]': cg1dg0[1],
            'cg2dg1': cg2dg1,
            'cg2dg1[0]': cg2dg1[0],
            'cg2dg1[1]': cg2dg1[1]}[request.param]


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
    elif isinstance(fs, (MixedFunctionSpace, VectorFunctionSpace)):
        one.interpolate(Expression(("1",)*one.function_space().cdim))
        two.interpolate(Expression(("2",)*two.function_space().cdim))
        minusthree.interpolate(Expression(("-3",)*minusthree.function_space().cdim))
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


def to_bool(v):
    try:
        return np.all(v)
    except:
        return v

exprtest = lambda expr, x: (expr, x, to_bool(d == x for d in assemble(expr).dat.data))

assigntest = lambda f, expr, x: (str(f) + " = " + str(expr) + ", " + str(f), x,
                                 to_bool(d == x for d in f.assign(expr).dat.data))


def iaddtest(f, expr, x):
    f += expr
    return (str(f) + " += " + str(expr) + ", " + str(f), x,
            to_bool(d == x for d in f.dat.data))


def isubtest(f, expr, x):
    f -= expr
    return (str(f) + " -= " + str(expr) + ", " + str(f), x,
            to_bool(d == x for d in f.dat.data))


def imultest(f, expr, x):
    f *= expr
    return (str(f) + " *= " + str(expr) + ", " + str(f), x,
            to_bool(d == x for d in f.dat.data))


def idivtest(f, expr, x):
    f /= expr
    return (str(f) + " /= " + str(expr) + ", " + str(f), x,
            to_bool(d == x for d in f.dat.data))


@pytest.fixture(params=range(1, 12))
def alltests(request, functions):
    f, one, two, minusthree = functions
    tests = {
        1: exprtest(one + two, 3),
        2: assigntest(f, one + two, 3),
        3: iaddtest(f, two, 5),
        4: iaddtest(f, 2, 7),
        5: isubtest(f, 2, 5),
        6: imultest(f, 2, 10),
        7: idivtest(f, 2, 5),
    }
    # Not all test cases work for vector function spaces
    if isinstance(one.function_space(), FunctionSpace):
        tests[8] = exprtest(ufl.ln(one), 0)
        tests[9] = exprtest(two ** minusthree, 0.125)
        tests[10] = exprtest(ufl.sign(minusthree), -1)
        tests[11] = exprtest(one + two / two ** minusthree, 17)
    try:
        return tests[request.param]
    except KeyError:
        pytest.skip()


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
