import pytest
from firedrake import *
from common import *


# FIXME: cg1vcg1 is not supported yet
@pytest.fixture(scope='module', params=['cg1', 'vcg1',
                                        'cg1cg1', 'cg1cg1[0]', 'cg1cg1[1]',
                                        'cg1vcg1[0]', 'cg1vcg1[1]',
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


@pytest.fixture
def f(fs):
    f = Function(fs, name="f")
    if isinstance(fs, (MixedFunctionSpace, VectorFunctionSpace)):
        f.interpolate(Expression(("x[0]",) * fs.cdim))
    else:
        f.interpolate(Expression("x[0]"))
    return f


@pytest.fixture
def one(fs):
    one = Function(fs, name="one")
    if isinstance(fs, (MixedFunctionSpace, VectorFunctionSpace)):
        one.interpolate(Expression(("1",) * fs.cdim))
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
        assert abs(f.dat.data.sum() - 0.5*f.function_space().dim) < 1.0e-12


def test_zero_form(M, f, one):
    zero_form = assemble(action(action(M, f), one))
    assert isinstance(zero_form, float)
    assert abs(zero_form - 0.5 * np.prod(f.shape())) < 1.0e-12

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
