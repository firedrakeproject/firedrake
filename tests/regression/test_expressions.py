from operator import iadd, isub, imul, itruediv
from functools import partial
from itertools import permutations

import pytest

import numpy as np
import ufl  # noqa: F401

from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module')
def cg1(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope='module')
def cg2(mesh):
    return FunctionSpace(mesh, "CG", 2)


@pytest.fixture(scope='module')
def dg0(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.fixture(scope='module')
def vcg1(mesh):
    return VectorFunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope='module')
def tcg1(mesh):
    return TensorFunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope='module', params=['cg1cg1', 'cg1vcg1', 'cg2dg0'])
def mfs(request, cg1, cg2, vcg1, dg0):
    """A parametrized fixture for mixed function spaces."""
    return {'cg1cg1': cg1*cg1,
            'cg1vcg1': cg1*vcg1,
            'cg2dg0': cg2*dg0}[request.param]


@pytest.fixture(scope='module', params=["cg1", "dg0"])
def sfs(request, cg1, dg0):
    """A parametrized fixture for scalar function spaces."""
    return {"cg1": cg1,
            "dg0": dg0}[request.param]


def func_factory(fs):
    f = Function(fs, name="f")
    one = Function(fs, name="one").assign(1)
    two = Function(fs, name="two").assign(2)
    return f, one, two


@pytest.fixture()
def functions(request, sfs):
    return func_factory(sfs)


@pytest.fixture()
def vfunctions(request, vcg1):
    return func_factory(vcg1)


@pytest.fixture()
def tfunctions(request, tcg1):
    return func_factory(tcg1)


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
    return all(np.all(abs(v_ - x_) < 1.e-14) for v_, x_ in zip(v, x))


def ioptest(f, expr, x, op):
    return evaluate(op(f, expr).dat.data, x)


exprtest = lambda expr, x: evaluate(assemble(expr).dat.data, x)
assigntest = lambda f, expr, x: evaluate(f.assign(expr).dat.data, x)
iaddtest = partial(ioptest, op=iadd)
isubtest = partial(ioptest, op=isub)
imultest = partial(ioptest, op=imul)
itruedivtest = partial(ioptest, op=itruediv)


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
    'itruedivtest(two, 2, 1)',
    'isubtest(one, one, 0)',
    'assigntest(f, 2 * one, 2)',
    'assigntest(f, one - one, 0)']


@pytest.mark.parametrize('expr', common_tests)
def test_scalar_expressions(expr, functions):
    f, one, two = functions
    assert eval(expr)


@pytest.mark.parametrize('expr', common_tests)
def test_vector_expressions(expr, vfunctions):
    f, one, two = vfunctions
    assert eval(expr)


@pytest.mark.parametrize('expr', common_tests)
def test_tensor_expressions(expr, tfunctions):
    f, one, two = tfunctions
    assert eval(expr)


def test_mixed_expressions(mfunctions):
    f, one, two = mfunctions

    f.sub(0).assign(one.sub(0))
    assert evaluate(f.dat.data, (1, 0))
    f.assign(0)

    f.sub(1).assign(one.sub(1))
    assert evaluate(f.dat.data, (0, 1))
    f.assign(0)

    two.sub(0).assign(one.sub(0))
    assert evaluate(two.dat.data, (1, 2))
    two.assign(2)

    two.sub(1).assign(one.sub(1))
    assert evaluate(two.dat.data, (2, 1))
    two.assign(2)

    two.sub(0).assign(one.sub(0) + two.sub(0))
    assert evaluate(two.dat.data, (3, 2))
    two.assign(2)

    two.sub(1).assign(two.sub(1) - one.sub(1))
    assert evaluate(two.dat.data, (2, 1))
    two.assign(2)

    one0 = one.sub(0)
    one0 += one.sub(0)
    assert evaluate(one.dat.data, (2, 1))
    one.assign(1)

    one1 = one.sub(1)
    one1 -= one.sub(1)
    assert evaluate(one.dat.data, (1, 0))


def test_mixed_expressions_indexed_fs(msfunctions):
    f, one, two = msfunctions

    f.sub(0).assign(one)
    assert evaluate(f.dat.data, (1, 0))
    f.assign(0)

    f.sub(1).assign(two)
    assert evaluate(f.dat.data, (0, 2))
    f.sub(0).assign(one)
    assert evaluate(f.dat.data, (1, 2))

    one.assign(2*f.sub(0) + 1)
    assert evaluate(one.dat.data, 3)

    two += f.sub(1)
    assert evaluate(two.dat.data, 4)


def test_iadd_combination(sfs):
    f = Function(sfs)
    g = Function(sfs)
    t = Constant(2)
    g.assign(1)
    f.assign(2)
    f += t*g
    assert np.allclose(f.dat.data_ro, 2 + 2)


def test_iadd_vector(sfs):
    f = Function(sfs)
    g = Function(sfs)
    g.assign(1)
    f.assign(2)
    f += g.vector()
    assert np.allclose(f.dat.data_ro, 3)


def test_different_fs_assign_fails(fs_combinations):
    """Assigning to a Function on a different function space should raise
    ValueError."""
    f1, f2 = fs_combinations
    with pytest.raises(ValueError):
        f1.assign(f2)


def test_assign_mfs_lincomp(mfs):
    f = Function(mfs)
    f.assign(1)
    g = Function(mfs)
    g.assign(2)
    h = Function(mfs)
    h.assign(3)
    c = Constant(2)
    d = Constant(4)
    f.assign(f + c*g + d*h)
    for f_ in f.dat.data_ro:
        assert np.allclose(f_, 1 + 2*2 + 3 * 4)


def test_assign_to_nonindexed_subspace_fails(mfs):
    """Assigning a Function on a non-indexed sub space of a mixed function
    space to a function on the mixed function space should fail."""
    for fs in mfs:
        with pytest.raises(ValueError):
            f = FunctionSpace(fs.mesh(), fs.ufl_element())
            Function(mfs).assign(Function(f))


def test_assign_with_different_meshes_fails():
    m1 = UnitSquareMesh(5, 5)
    m2 = UnitSquareMesh(5, 5)

    V1 = FunctionSpace(m1, "CG", 3)
    V2 = FunctionSpace(m2, "CG", 3)

    u1 = Function(V1).assign(1)
    u2 = Function(V2).assign(2)

    with pytest.raises(ValueError):
        u2.assign(u1)

    with pytest.raises(ValueError):
        u1 += u2


def test_assign_vector_const_to_vfs(vcg1):
    f = Function(vcg1)

    c = Constant(range(1, f.function_space().value_shape[0]+1))

    f.assign(c)
    assert np.allclose(f.dat.data_ro, c.dat.data_ro)


def test_assign_scalar_const_to_vfs(vcg1):
    f = Function(vcg1)

    c = Constant(10.0)

    f.assign(c)
    assert np.allclose(f.dat.data_ro, c.dat.data_ro)


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

    with pytest.raises(ValueError):
        w.assign(q.sub(1))


def test_assign_to_vfs_sub(cg1, vcg1):
    v = Function(cg1).assign(2)
    w = Function(vcg1).assign(0)

    w.sub(0).assign(v)
    assert np.allclose(w.sub(0).dat.data_ro, 2)
    assert np.allclose(w.sub(1).dat.data_ro, 0)

    v.assign(w.sub(1))
    assert np.allclose(v.dat.data_ro, 0)

    v += w.sub(0)
    assert np.allclose(v.dat.data_ro, 2)


def test_assign_from_mfs_sub(cg1, vcg1):
    W = cg1*vcg1

    w = Function(W)
    u = Function(cg1)
    v = Function(vcg1)

    w1, w2 = w.subfunctions

    w1.assign(4)
    w2.assign(10)

    u.assign(w1)

    assert np.allclose(u.dat.data_ro, w1.dat.data_ro)

    v.assign(w2)
    assert np.allclose(v.dat.data_ro, w2.dat.data_ro)

    Q = vcg1*cg1
    q = Function(Q)

    q1, q2 = q.subfunctions

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


@pytest.mark.skipif(not utils.complex_mode, reason="Test specific to complex mode")
def test_assign_complex_value(cg1):
    f = Function(cg1)
    g = Function(cg1)

    f.assign(1+1j)
    assert np.allclose(f.dat.data_ro, 1+1j)

    f.assign(1j)
    assert np.allclose(f.dat.data_ro, 1j)

    g.assign(2.0)
    f.assign((1+1j)*g)
    assert np.allclose(f.dat.data_ro, 2+2j)


@pytest.mark.parametrize('value', [10, -10],
                         ids=lambda v: "(f = %d)" % v)
@pytest.mark.parametrize('expr', ['f', '2*f'])
def test_math_functions(expr, value):
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    f.assign(value)

    actual = Function(V)

    actual.assign(eval(expr))
    f = value
    expect = eval(expr)
    assert np.allclose(actual.dat.data_ro, expect)


def test_assign_mixed_multiple_shaped():
    mesh = UnitTriangleMesh()
    V = VectorFunctionSpace(mesh, "DG", 0)
    Q = FunctionSpace(mesh, "P", 1)
    P = FunctionSpace(mesh, "RT", 2)
    X = TensorFunctionSpace(mesh, "DG", 1)

    Z = V*Q*P*X

    z1 = Function(Z)
    z2 = Function(Z)

    z1.dat[0].data[:] = [1, 2]
    z1.dat[1].data[:] = 3
    z1.dat[2].data[:] = 4
    z1.dat[3].data[:] = [[6, 7], [8, 9]]

    z2.dat[0].data[:] = [10, 11]
    z2.dat[1].data[:] = 12
    z2.dat[2].data[:] = 13
    z2.dat[3].data[:] = [[15, 16], [17, 18]]

    q = assemble(z1 - z2)
    for q, p1, p2 in zip(q.subfunctions, z1.subfunctions, z2.subfunctions):
        assert np.allclose(q.dat.data_ro, p1.dat.data_ro - p2.dat.data_ro)


def test_augmented_assignment_broadcast():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "BDM", 1)
    u = Function(V)
    a = Constant(1)
    b = Constant(2)
    u.assign(a)

    assert np.allclose(u.dat.data_ro, 1)

    u *= -(a + b)
    assert np.allclose(u.dat.data_ro, -3)

    u += b*2
    assert np.allclose(u.dat.data_ro, 1)

    u /= -(b + a)
    assert np.allclose(u.dat.data_ro, -1/3)

    u -= 2 + a + b
    assert np.allclose(u.dat.data_ro, -16/3)


def make_subset(cg1):
    """Return a subset consisting of one owned and one ghost element.

    This function will only work in parallel.

    """
    # the second entry in node_set.sizes is the number of owned values, which
    # is also the index of the first ghost value
    indices = [0, cg1.node_set.sizes[1]]
    return op2.Subset(cg1.node_set, indices)


@pytest.mark.parallel(nprocs=2)
def test_assign_with_dirty_halo_and_no_subset_sets_halo_values(cg1):
    u = Function(cg1)
    assert u.dat.halo_valid

    u.dat.halo_valid = False
    u.assign(1)

    # use private attribute here to avoid triggering any halo exchanges
    assert u.dat.halo_valid
    assert np.allclose(u.dat._data, 1)


@pytest.mark.parallel(nprocs=2)
def test_assign_with_valid_halo_and_subset_sets_halo_values(cg1):
    u = Function(cg1)
    assert u.dat.halo_valid

    subset = make_subset(cg1)
    u.assign(1, subset=subset)

    expected = [0] * u.dat.dataset.total_size
    expected[0] = 1
    expected[u.dat.dataset.size] = 1

    # use private attribute here to avoid triggering any halo exchanges
    assert u.dat.halo_valid
    assert np.allclose(u.dat._data, expected)


@pytest.mark.parallel(nprocs=2)
def test_assign_with_dirty_halo_and_subset_skips_halo_values(cg1):
    u = Function(cg1)
    assert u.dat.halo_valid

    u.dat.halo_valid = False
    subset = make_subset(cg1)
    u.assign(1, subset=subset)

    expected = [0] * u.dat.dataset.total_size
    expected[0] = 1

    # use private attribute here to avoid triggering any halo exchanges
    assert not u.dat.halo_valid
    assert np.allclose(u.dat._data, expected)


@pytest.mark.parallel(nprocs=2)
def test_assign_with_dirty_expression_halo_skips_halo_values(cg1):
    u = Function(cg1)
    v = Function(cg1)
    assert u.dat.halo_valid
    assert v.dat.halo_valid

    v.assign(1)
    assert v.dat.halo_valid

    v.dat.halo_valid = False
    u.assign(v)

    # use private attribute here to avoid triggering any halo exchanges
    assert not u.dat.halo_valid
    assert np.allclose(u.dat._data[:u.dat.dataset.size], 1)
    assert np.allclose(u.dat._data[u.dat.dataset.size:], 0)
