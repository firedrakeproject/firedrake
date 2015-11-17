import pytest
import numpy as np
from firedrake import *
from tests.common import *


# FIXME: cg1vcg1 is not supported yet
@pytest.fixture(scope='module', params=['cg1', 'vcg1', 'tcg1',
                                        'cg1cg1', 'cg1cg1[0]', 'cg1cg1[1]',
                                        'cg1vcg1[0]', 'cg1vcg1[1]',
                                        'cg1dg0', 'cg1dg0[0]', 'cg1dg0[1]',
                                        'cg2dg1', 'cg2dg1[0]', 'cg2dg1[1]'])
def fs(request, cg1, vcg1, tcg1, cg1cg1, cg1vcg1, cg1dg0, cg2dg1):
    return {'cg1': cg1,
            'vcg1': vcg1,
            'tcg1': tcg1,
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


@pytest.fixture
def f(fs):
    f = Function(fs, name="f")
    if isinstance(fs, (MixedFunctionSpace, VectorFunctionSpace, TensorFunctionSpace)):
        f.interpolate(Expression(("x[0]",) * fs.dim))
    else:
        f.interpolate(Expression("x[0]"))
    return f


@pytest.fixture
def one(fs):
    one = Function(fs, name="one")
    if isinstance(fs, (MixedFunctionSpace, VectorFunctionSpace, TensorFunctionSpace)):
        one.interpolate(Expression(("1",) * fs.dim))
    else:
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
    for f in one_form.split():
        if(isinstance(f.function_space(), TensorFunctionSpace)):
            assert abs(f.dat.data.sum() - 0.5*sum(f.function_space().shape)) < 1.0e-12
        else:
            assert abs(f.dat.data.sum() - 0.5*f.function_space().dim) < 1.0e-12


def test_zero_form(M, f, one):
    zero_form = assemble(action(action(M, f), one))
    assert isinstance(zero_form, float)
    assert abs(zero_form - 0.5 * np.prod(f.shape())) < 1.0e-12


def test_assemble_with_tensor(cg1):
    v = TestFunction(cg1)
    L = v*dx
    f = Function(cg1)
    # Assemble a form into f
    f = assemble(L, f)
    # Assemble a different form into f
    f = assemble(Constant(2)*L, f)
    # Make sure we get the result of the last assembly
    assert np.allclose(f.dat.data, 2*assemble(L).dat.data, rtol=1e-14)


def test_assemble_mat_with_tensor(dg0):
    u = TestFunction(dg0)
    v = TrialFunction(dg0)
    a = u*v*dx
    M = assemble(a)
    # Assemble a different form into M
    M = assemble(Constant(2)*a, M)
    # Make sure we get the result of the last assembly
    assert np.allclose(M.M.values, 2*assemble(a).M.values, rtol=1e-14)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
