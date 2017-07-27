from __future__ import absolute_import, print_function, division
from six.moves import range

import numpy
import pytest

from coffee.visitors import EstimateFlops

from ufl import (Mesh, FunctionSpace, FiniteElement, VectorElement,
                 TestFunction, TrialFunction, TensorProductCell, dx,
                 action, interval, triangle, quadrilateral, dot, grad)

from tsfc import compile_form


def helmholtz(cell, degree):
    m = Mesh(VectorElement('CG', cell, 1))
    V = FunctionSpace(m, FiniteElement('CG', cell, degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    return (u*v + dot(grad(u), grad(v)))*dx


def count_flops(form):
    kernel, = compile_form(form, parameters=dict(mode='spectral'))
    return EstimateFlops().visit(kernel.ast)


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 5),
                          (TensorProductCell(interval, interval), 5),
                          (TensorProductCell(triangle, interval), 7),
                          (TensorProductCell(quadrilateral, interval), 7)])
def test_lhs(cell, order):
    degrees = list(range(3, 8))
    if cell == TensorProductCell(triangle, interval):
        degrees = list(range(3, 6))
    flops = [count_flops(helmholtz(cell, degree))
             for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 3),
                          (TensorProductCell(interval, interval), 3),
                          (TensorProductCell(triangle, interval), 5),
                          (TensorProductCell(quadrilateral, interval), 4)])
def test_rhs(cell, order):
    degrees = list(range(3, 8))
    if cell == TensorProductCell(triangle, interval):
        degrees = list(range(3, 6))
    flops = [count_flops(action(helmholtz(cell, degree)))
             for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees))
    assert (rates < order).all()


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
