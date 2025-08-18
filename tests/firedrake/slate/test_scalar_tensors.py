import pytest
import itertools
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    quadrilateral = request.param
    return UnitSquareMesh(5, 5, quadrilateral=quadrilateral)


@pytest.mark.parametrize(['expr', 'value', 'typ', 'fs_type'],
                         itertools.product(['f',
                                            '2.0*tanh(f) + cos(f) + sin(f)',
                                            '1.0/tanh(f) + 1.0/f'],
                                           [1, 10, -1, -10],
                                           ['function', 'constant'],
                                           ['scalar', 'vector', 'tensor']))
def test_functions(mesh, expr, value, typ, fs_type):
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

    actual = assemble(Tensor(eval(expr)*dx))

    if fs_type == "vector":
        f = 2*value**2
    elif fs_type == "tensor":
        f = 4*value**2
    else:
        f = value
    expect = eval(expr)
    assert np.allclose(actual, expect)
