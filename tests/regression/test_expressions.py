from operator import iadd, isub, imul, idiv
from functools import partial
from itertools import permutations

import pytest

import numpy as np
import ufl  # noqa: used in eval'd expressions

from firedrake import *
from tests.common import *


@pytest.fixture(scope='module', params=['cg1', 'cg1cg1[0]', 'cg1cg1[1]',
                                        'cg1vcg1[0]', 'cg1dg0[0]', 'cg1dg0[1]',
                                        'cg2dg1[0]', 'cg2dg1[1]'])
def sfs(request, cg1, cg1cg1, cg1vcg1, cg1dg0, cg2dg1):
    """A parametrized fixture for scalar function spaces."""
    return {'cg1': cg1,
            'cg1cg1[0]': cg1cg1[0],
            'cg1cg1[1]': cg1cg1[1],
            'cg1vcg1[0]': cg1vcg1[0],
            'cg1dg0[0]': cg1dg0[0],
            'cg1dg0[1]': cg1dg0[1],
            'cg2dg1[0]': cg2dg1[0],
            'cg2dg1[1]': cg2dg1[1]}[request.param]


@pytest.fixture(scope='module', params=['vcg1', 'cg1vcg1[1]'])
def vfs(request, vcg1, cg1vcg1):
    """A parametrized fixture for vector function spaces."""
    return {'vcg1': vcg1,
            'cg1vcg1[1]': cg1vcg1[1]}[request.param]


@pytest.fixture(scope='module', params=['tcg1'])
def tfs(request, tcg1):
    """A parametrized fixture for tensor function spaces."""
    return {'tcg1': tcg1}[request.param]


@pytest.fixture(scope='module', params=['cg1cg1', 'cg1vcg1', 'cg1dg0', 'cg2dg1'])
def mfs(request, cg1cg1, cg1vcg1, cg1dg0, cg2dg1):
    """A parametrized fixture for mixed function spaces."""
    return {'cg1cg1': cg1cg1,
            'cg1vcg1': cg1vcg1,
            'cg1dg0': cg1dg0,
            'cg2dg1': cg2dg1}[request.param]


def func_factory(fs):
    f = Function(fs, name="f")
    one = Function(fs, name="one").assign(1)
    two = Function(fs, name="two").assign(2)
    minusthree = Function(fs, name="minusthree").assign(-3)
    return f, one, two, minusthree


@pytest.fixture()
def functions(request, sfs):
    return func_factory(sfs)


@pytest.fixture()
def vfunctions(request, vfs):
    return func_factory(vfs)


@pytest.fixture()
def tfunctions(request, tfs):
    return func_factory(tfs)


@pytest.fixture()
def mfunctions(request, mfs):
    return func_factory(mfs)


@pytest.fixture
def msfunctions(request, mfs):
    return Function(mfs), Function(mfs[0]).assign(1), Function(mfs[1]).assign(2)


@pytest.fixture
def sf(cg1):
    return Function(cg1, name="sf")


@pytest.fixture
def vf(vcg1):
    return Function(vcg1, name="vf")


@pytest.fixture
def tf(tcg1):
    return Function(tcg1, name="tf")


@pytest.fixture
def mf(cg1, vcg1):
    return Function(cg1 * vcg1, name="mf")


@pytest.fixture(params=permutations(['sf', 'vf', 'tf', 'mf'], 2))
def fs_combinations(sf, vf, tf, mf, request):
    funcs = {'sf': sf, 'vf': vf, 'tf': tf, 'mf': mf}
    return [funcs[p] for p in request.param]


def evaluate(v, x):
    try:
        assert len(v) == len(x)
    except TypeError:
        x = (x,) * len(v)
    try:
        return all(np.all(v_ == x_) for v_, x_ in zip(v, x))
    except:
        return v == x


def ioptest(f, expr, x, op):
    return evaluate(op(f, expr).dat.data, x)


def interpolatetest(f, expr, x):
    if f.function_space().dim > 1:
        expr = (expr,) * f.function_space().dim
    return evaluate(f.interpolate(Expression(expr)).dat.data, x)

exprtest = lambda expr, x: evaluate(assemble(expr).dat.data, x)
assigntest = lambda f, expr, x: evaluate(f.assign(expr).dat.data, x)
iaddtest = partial(ioptest, op=iadd)
isubtest = partial(ioptest, op=isub)
imultest = partial(ioptest, op=imul)
idivtest = partial(ioptest, op=idiv)


common_tests = [
    'assigntest(f, 1, 1)',
    'assigntest(f, 2.0*(one + one), 4)',
    'exprtest(one + one, 2)',
    'exprtest(3 * one, 3)',
    'exprtest(one + two, 3)',
    'assigntest(f, one + two, 3)',
    'iaddtest(one, one, 2)',
    'iaddtest(one, two, 3)',
    'iaddtest(f, 2, 2)',
    'isubtest(two, 1, 1)',
    'imultest(one, 2, 2)',
    'imultest(one, two, 2)',
    'idivtest(two, 2, 1)',
    'idivtest(one, two, 0.5)',
    'isubtest(one, one, 0)',
    'assigntest(f, 2 * one, 2)',
    'assigntest(f, one - one, 0)']

scalar_tests = common_tests + [
    'interpolatetest(f, 0.0, 0)',
    'interpolatetest(f, "sin(pi/2)", 1)',
    'assigntest(f, sqrt(one), 1)',
    'exprtest(ufl.ln(one), 0)',
    'exprtest(two ** minusthree, 0.125)',
    'exprtest(ufl.sign(minusthree), -1)',
    'exprtest(one + two / two ** minusthree, 17)']

mixed_tests = common_tests + [
    'interpolatetest(f, "sin(pi/2)", (1, 1))',
    'exprtest(one[0] + one[1], (1, 1))',
    'exprtest(one[1] + two[0], (2, 1))',
    'exprtest(one[0] - one[1], (1, -1))',
    'exprtest(one[1] - two[0], (-2, 1))',
    'assigntest(f, one[0], (1, 0))',
    'assigntest(f, one[1], (0, 1))',
    'assigntest(two, one[0], (1, 0))',
    'assigntest(two, one[1], (0, 1))',
    'assigntest(two, one[0] + two[0], (3, 0))',
    'assigntest(two, two[1] - one[1], (0, 1))',
    'assigntest(f, one[0] + two[1], (1, 2))',
    'iaddtest(one, one[0], (2, 1))',
    'iaddtest(one, one[1], (1, 2))',
    'assigntest(f, 2 * two[1] + 2 * minusthree[0], (-6, 4))']

indexed_fs_tests = [
    'assigntest(f, one, (1, 0))',
    'assigntest(f, two, (0, 2))',
    'iaddtest(f, one, (1, 0))',
    'iaddtest(f, two, (0, 2))',
    'isubtest(f, one, (-1, 0))',
    'isubtest(f, two, (0, -2))']


@pytest.mark.parametrize('expr', scalar_tests)
def test_scalar_expressions(expr, functions):
    f, one, two, minusthree = functions
    assert eval(expr)


@pytest.mark.parametrize('expr', common_tests)
def test_vector_expressions(expr, vfunctions):
    f, one, two, minusthree = vfunctions
    assert eval(expr)


@pytest.mark.parametrize('expr', common_tests)
def test_tensor_expressions(expr, tfunctions):
    f, one, two, minusthree = tfunctions
    assert eval(expr)


@pytest.mark.parametrize('expr', mixed_tests)
def test_mixed_expressions(expr, mfunctions):
    f, one, two, minusthree = mfunctions
    assert eval(expr)


@pytest.mark.parametrize('expr', indexed_fs_tests)
def test_mixed_expressions_indexed_fs(expr, msfunctions):
    f, one, two = msfunctions
    assert eval(expr)


def test_different_fs_asign_fails(fs_combinations):
    """Assigning to a Function on a different function space should raise
    ValueError."""
    f1, f2 = fs_combinations
    with pytest.raises(ValueError):
        f1.assign(f2)


def test_asign_to_nonindexed_subspace_fails(mfs):
    """Assigning a Function on a non-indexed sub space of a mixed function
    space to a function on the mixed function space should fail."""
    for fs in mfs:
        with pytest.raises(ValueError):
            Function(mfs).assign(Function(fs._fs))


def test_assign_mixed_no_nan(mfs):
    w = Function(mfs)
    vs = w.split()
    vs[0].assign(2)
    w /= vs[0]
    assert np.allclose(vs[0].dat.data_ro, 1.0)
    for v in vs[1:]:
        assert not np.isnan(v.dat.data_ro).any()


def test_assign_mixed_no_zero(mfs):
    w = Function(mfs)
    vs = w.split()
    w.assign(2)
    w *= vs[0]
    assert np.allclose(vs[0].dat.data_ro, 4.0)
    for v in vs[1:]:
        assert np.allclose(v.dat.data_ro, 2.0)


def test_assign_vector_const_to_mfs_scalar_vector(cg1, vcg1):
    W = cg1*vcg1

    w = Function(W)

    c = Constant(range(1, w.element().value_shape()[0]+1))

    w.assign(c)

    s, v = w.split()

    assert np.allclose(s.dat.data_ro, c.dat.data_ro[0])
    assert np.allclose(v.dat.data_ro, c.dat.data_ro[1:])


def test_assign_vector_const_to_mfs_scalar_vector_vector(cg1, vcg1):
    W = cg1*vcg1*vcg1

    w = Function(W)

    c = Constant(range(1, w.element().value_shape()[0]+1))

    w.assign(c)

    s, v, v2 = w.split()

    assert np.allclose(s.dat.data_ro, c.dat.data_ro[0])
    assert np.allclose(v.dat.data_ro, c.dat.data_ro[1:3])
    assert np.allclose(v2.dat.data_ro, c.dat.data_ro[3:])


def test_assign_vector_const_to_vfs(vcg1):
    f = Function(vcg1)

    c = Constant(range(1, f.element().value_shape()[0]+1))

    f.assign(c)
    assert np.allclose(f.dat.data_ro, c.dat.data_ro)


def test_assign_scalar_const_to_vfs(vcg1):
    f = Function(vcg1)

    c = Constant(10.0)

    f.assign(c)
    assert np.allclose(f.dat.data_ro, c.dat.data_ro)


def test_assign_vector_const_to_mfs_scalars(cg1):
    W = cg1*cg1*cg1

    w = Function(W)

    c = Constant(range(1, w.element().value_shape()[0]+1))

    w.assign(c)

    for i, w_ in enumerate(w.split()):
        assert np.allclose(w_.dat.data_ro, c.dat.data_ro[i])


def test_assign_to_mfs_sub(cg1, vcg1):
    W = cg1*vcg1

    w = Function(W)
    u = Function(cg1)
    v = Function(vcg1)
    u.assign(4)
    v.assign(10)

    w.sub(0).assign(u)

    assert np.allclose(w.sub(0).dat.data_ro, 4)
    assert np.allclose(w.sub(1).dat.data_ro, 0)

    w.sub(1).assign(v)
    assert np.allclose(w.sub(0).dat.data_ro, 4)
    assert np.allclose(w.sub(1).dat.data_ro, 10)

    Q = vcg1*cg1
    q = Function(Q)
    q.assign(11)
    w.sub(1).assign(q.sub(0))

    assert np.allclose(w.sub(1).dat.data_ro, 11)
    assert np.allclose(w.sub(0).dat.data_ro, 4)

    with pytest.raises(ValueError):
        w.sub(1).assign(q.sub(1))

    with pytest.raises(ValueError):
        w.sub(1).assign(w.sub(0))

    with pytest.raises(ValueError):
        w.sub(1).assign(u)

    with pytest.raises(ValueError):
        w.sub(0).assign(v)

    w.sub(0).assign(ufl.ln(q.sub(1)))
    assert np.allclose(w.sub(0).dat.data_ro, ufl.ln(11))

    with pytest.raises(ValueError):
        w.assign(q.sub(1))


def test_assign_from_mfs_sub(cg1, vcg1):
    W = cg1*vcg1

    w = Function(W)
    u = Function(cg1)
    v = Function(vcg1)

    w1, w2 = w.split()

    w1.assign(4)
    w2.assign(10)

    u.assign(w1)

    assert np.allclose(u.dat.data_ro, w1.dat.data_ro)

    v.assign(w2)
    assert np.allclose(v.dat.data_ro, w2.dat.data_ro)

    Q = vcg1*cg1
    q = Function(Q)

    q1, q2 = q.split()

    q1.assign(11)
    q2.assign(12)

    v.assign(q1)
    assert np.allclose(v.dat.data_ro, q1.dat.data_ro)

    u.assign(q2)
    assert np.allclose(u.dat.data_ro, q2.dat.data_ro)

    with pytest.raises(ValueError):
        u.assign(q1)

    with pytest.raises(ValueError):
        v.assign(q2)

    with pytest.raises(ValueError):
        u.assign(w2)

    with pytest.raises(ValueError):
        v.assign(w1)


@pytest.mark.parametrize("uservar", ["A", "X", "x_", "k", "d", "i"])
def test_scalar_user_defined_values(uservar):
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'CG', 1)
    f = Function(V)
    e = Expression(uservar, **{uservar: 1.0})
    f.interpolate(e)

    assert np.allclose(f.dat.data_ro, 1.0)

    setattr(e, uservar, 2.0)
    f.interpolate(e)

    assert np.allclose(f.dat.data_ro, 2.0)


def test_vector_user_defined_values():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'CG', 1)
    f = Function(V)
    e = Expression('n[0] + n[1]', n=[1.0, 2.0])

    f.interpolate(e)

    assert np.allclose(f.dat.data_ro, 3.0)

    e.n = [2.0, 4.0]
    f.interpolate(e)

    assert np.allclose(f.dat.data_ro, 6.0)


def test_scalar_increment_fails():
    e = Expression('n', n=1.0)

    # Some versions of numpy raise RuntimeError on access to read-only
    # array view, rather than ValueError.
    with pytest.raises((ValueError, RuntimeError)):
        e.n += 1

    with pytest.raises((ValueError, RuntimeError)):
        e.n[0] += 2

    assert np.allclose(e.n, 1.0)


def test_vector_increment_fails():
    e = Expression('n', n=[1.0, 1.0])

    with pytest.raises((ValueError, RuntimeError)):
        e.n += 1

    with pytest.raises((ValueError, RuntimeError)):
        e.n[0] += 2

    assert np.allclose(e.n, 1.0)


def test_tensor_increment_fails():
    e = Expression('n', n=[[1.0, 1.0], [1.0, 1.0]])

    with pytest.raises((ValueError, RuntimeError)):
        e.n += 1

    with pytest.raises((ValueError, RuntimeError)):
        e.n[0] += 2

    assert np.allclose(e.n, 1.0)


@pytest.mark.parametrize('value', [1, 10, 20, -1, -10, -20],
                         ids=lambda v: "(f = %d)" % v)
@pytest.mark.parametrize('expr', ['f',
                                  '2*f',
                                  'tanh(f)',
                                  '2 * tanh(f)',
                                  'f + tanh(f)',
                                  'cos(f) + sin(f)',
                                  'cos(f)*cos(f) + sin(f)*sin(f)',
                                  'tanh(f) + cos(f) + sin(f)',
                                  '1.0/tanh(f) + 1.0/f',
                                  'sqrt(f*f)',
                                  'sin(cos(f))',
                                  'sqrt(2 + cos(f))',
                                  'sin(sqrt(abs(f)))',
                                  '1.0/tanh(sqrt(f*f)) + 1.0/f + sqrt(f*f)'])
def test_math_functions(expr, value):
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    f.assign(value)

    actual = Function(V)

    actual.assign(eval(expr))
    from math import *
    f = value
    expect = eval(expr)
    assert np.allclose(actual.dat.data_ro, expect)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
