import pytest

import numpy as np
import ufl  # noqa: used in eval'd expressions

from firedrake import *
from common import *


@pytest.fixture(scope='module', params=['cg1', 'cg1cg1[0]', 'cg1cg1[1]',
                                        'cg1vcg1[0]', 'cg1dg0[0]', 'cg1dg0[1]',
                                        'cg2dg1[0]', 'cg2dg1[1]'])
def sfs(request, cg1, vcg1, cg1cg1, cg1vcg1, cg1dg0, cg2dg1):
    return {'cg1': cg1,
            'cg1cg1[0]': cg1cg1[0],
            'cg1cg1[1]': cg1cg1[1],
            'cg1vcg1[0]': cg1vcg1[0],
            'cg1dg0[0]': cg1dg0[0],
            'cg1dg0[1]': cg1dg0[1],
            'cg2dg1[0]': cg2dg1[0],
            'cg2dg1[1]': cg2dg1[1]}[request.param]


@pytest.fixture(scope='module', params=['vcg1', 'cg1cg1', 'cg1vcg1',
                                        'cg1vcg1[1]', 'cg1dg0', 'cg2dg1'])
def mfs(request, cg1, vcg1, cg1cg1, cg1vcg1, cg1dg0, cg2dg1):
    return {'vcg1': vcg1,
            'cg1cg1': cg1cg1,
            'cg1vcg1': cg1vcg1,
            'cg1vcg1[1]': cg1vcg1[1],
            'cg1dg0': cg1dg0,
            'cg2dg1': cg2dg1}[request.param]


def func_factory(fs, method):
    f = Function(fs, name="f")
    one = Function(fs, name="one")
    two = Function(fs, name="two")
    minusthree = Function(fs, name="minusthree")
    if method == 'assign':
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


@pytest.fixture(params=['assign', 'interpolate'])
def functions(request, sfs):
    return func_factory(sfs, method=request.param)


@pytest.fixture(params=['assign', 'interpolate'])
def mfunctions(request, mfs):
    return func_factory(mfs, method=request.param)


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


common_tests = [
    'exprtest(one + one, 2)',
    'exprtest(3 * one, 3)',
    'exprtest(one + two, 3)',
    'assigntest(f, one + two, 3)',
    'iaddtest(f, f, 6)',
    'iaddtest(f, two, 8)',
    'iaddtest(f, 2, 10)',
    'isubtest(f, 5, 5)',
    'imultest(f, 2, 10)',
    'idivtest(f, 2, 5)',
    'isubtest(f, f, 0)']

scalar_tests = common_tests + [
    'exprtest(ufl.ln(one), 0)',
    'exprtest(two ** minusthree, 0.125)',
    'exprtest(ufl.sign(minusthree), -1)',
    'exprtest(one + two / two ** minusthree, 17)']


@pytest.mark.parametrize('expr', scalar_tests)
def test_scalar_expressions(expr, functions):
    f, one, two, minusthree = functions
    assert eval(expr)[2]


@pytest.mark.parametrize('expr', common_tests)
def test_mixed_expressions(expr, mfunctions):
    f, one, two, minusthree = mfunctions
    assert eval(expr)[2]


def test_vf_assign_f(sf, vf):
    with pytest.raises(ValueError):
        vf.assign(sf)


def test_f_assign_vf(sf, vf):
    with pytest.raises(ValueError):
        sf.assign(vf)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
