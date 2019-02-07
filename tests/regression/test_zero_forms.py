import pytest
import numpy as np
import itertools
from firedrake import *


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
def test_dsn_parallel(one):
    for d in domains:
        assert np.allclose(assemble(one*ds(d)), len(d))

    for domain in domains:
        form = one*ds(domain[0])
        for d in domain[1:]:
            form += one*ds(d)
        assert np.allclose(assemble(form), len(domain))


@pytest.mark.parametrize(['expr', 'value', 'typ', 'fs_type'],
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
                                           ['scalar', 'vector', 'tensor']))
def test_math_functions(mesh, expr, value, typ, fs_type):
    if typ == 'function':
        if fs_type == "vector":
            V = VectorFunctionSpace(mesh, 'CG', 1)
        elif fs_type == "tensor":
            V = TensorFunctionSpace(mesh, 'CG', 1)
        else:
            V = FunctionSpace(mesh, 'CG', 1)
        f = Function(V)
        f.assign(value)
        if fs_type == "vector":
            f = dot(f, f)
        elif fs_type == "tensor":
            f = inner(f, f)
    elif typ == 'constant':
        if fs_type == "vector":
            f = Constant([value, value], domain=mesh)
            f = dot(f, f)
        elif fs_type == "tensor":
            f = Constant([[value, value], [value, value]], domain=mesh)
            f = inner(f, f)
        else:
            f = Constant(value, domain=mesh)

    actual = assemble(eval(expr)*dx)

    if fs_type == "vector":
        f = 2*value**2
    elif fs_type == "tensor":
        f = 4*value**2
    else:
        f = value
    expect = eval(expr)
    assert np.allclose(actual, expect)
