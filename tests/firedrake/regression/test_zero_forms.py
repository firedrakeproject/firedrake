import pytest
import numpy as np
import itertools
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    quadrilateral = request.param
    return UnitSquareMesh(10, 10, quadrilateral=quadrilateral)


domains = [(1, 2),
           (2, 3),
           (3, 4),
           (4, 1),
           (1, 2, 3, 4)]


def test_ds_dx(mesh):
    assert np.allclose(assemble(1*dx(domain=mesh) + 1*ds(domain=mesh)), 5.0)


@pytest.mark.parametrize('domains', domains)
def test_dsn(mesh, domains):

    assert np.allclose(assemble(1*ds(domains, domain=mesh)), len(domains))

    form = 1*ds(domains[0], domain=mesh)

    for d in domains[1:]:
        form += 1*ds(d, domain=mesh)
    assert np.allclose(assemble(form), len(domains))


@pytest.mark.parallel
def test_dsn_parallel(mesh):
    for d in domains:
        assert np.allclose(assemble(1*ds(d, domain=mesh)), len(d))

    for domain in domains:
        form = 1*ds(domain[0], domain=mesh)
        for d in domain[1:]:
            form += 1*ds(d, domain=mesh)
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
        family, degree = 'CG', 1
    elif typ == 'constant':
        family, degree = 'Real', 0

    if fs_type == "vector":
        V = VectorFunctionSpace(mesh, family, degree)
    elif fs_type == "tensor":
        V = TensorFunctionSpace(mesh, family, degree)
    else:
        V = FunctionSpace(mesh, family, degree)
    f = Function(V)
    f.assign(value)
    if fs_type == "vector":
        f = dot(f, f)
    elif fs_type == "tensor":
        f = inner(f, f)

    actual = assemble(eval(expr)*dx)

    if fs_type == "vector":
        f = 2*value**2
    elif fs_type == "tensor":
        f = 4*value**2
    else:
        f = value
    expect = eval(expr)
    assert np.allclose(actual, expect)
