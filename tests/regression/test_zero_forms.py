import pytest
import numpy as np
import itertools
from firedrake import *


class Bunch(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    quadrilateral = request.param
    return UnitSquareMesh(10, 10, quadrilateral=quadrilateral)


@pytest.fixture(scope='module')
def one(mesh):
    return Constant(1, domain=mesh)


domains = [(1, 2),
           (2, 3),
           (3, 4),
           (4, 1),
           (1, 2, 3, 4)]


def test_ds_dx(one):
    assert np.allclose(assemble(one*dx + one*ds), 5.0)


@pytest.mark.parametrize('domains', domains)
def test_dsn(one, domains):

    assert np.allclose(assemble(one*ds(domains)), len(domains))

    form = one*ds(domains[0])

    for d in domains[1:]:
        form += one*ds(d)
    assert np.allclose(assemble(form), len(domains))


@pytest.mark.parallel
def test_dsn_parallel():
    c = one(mesh(Bunch(param=False)))

    for d in domains:
        assert np.allclose(assemble(c*ds(d)), len(d))

    for domain in domains:
        form = c*ds(domain[0])
        for d in domain[1:]:
            form += c*ds(d)
        assert np.allclose(assemble(form), len(domain))


@pytest.mark.parallel
def test_dsn_parallel_on_quadrilaterals():
    c = one(mesh(Bunch(param=True)))

    for d in domains:
        assert np.allclose(assemble(c*ds(d)), len(d))

    for domain in domains:
        form = c*ds(domain[0])
        for d in domain[1:]:
            form += c*ds(d)
        assert np.allclose(assemble(form), len(domain))


@pytest.mark.parametrize(['expr', 'value', 'typ', 'vector'],
                         itertools.product(['f',
                                            '2*f',
                                            'tanh(f)',
                                            '2 * tanh(f)',
                                            'f + tanh(f)',
                                            'cos(f) + sin(f)',
                                            'cos(f)*cos(f) + sin(f)*sin(f)',
                                            'tanh(f) + cos(f) + sin(f)',
                                            '1.0/tanh(f) + 1.0/f',
                                            'sqrt(f*f)',
                                            '1.0/tanh(sqrt(f*f)) + 1.0/f + sqrt(f*f)'],
                                           [1, 10, 20, -1, -10, -20],
                                           ['function', 'constant'],
                                           [False, True]))
def test_math_functions(mesh, expr, value, typ, vector):
    if typ == 'function':
        if vector:
            V = VectorFunctionSpace(mesh, 'CG', 1)
        else:
            V = FunctionSpace(mesh, 'CG', 1)
        f = Function(V)
        f.assign(value)
        if vector:
            f = dot(f, f)
    elif typ == 'constant':
        if vector:
            f = Constant([value, value], domain=mesh)
            f = dot(f, f)
        else:
            f = Constant(value, domain=mesh)

    actual = assemble(eval(expr)*dx)

    from math import *
    if vector:
        f = 2*value**2
    else:
        f = value
    expect = eval(expr)
    assert np.allclose(actual, expect)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
