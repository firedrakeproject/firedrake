from functools import reduce

import numpy
import pytest

from ufl import (Mesh, FunctionSpace, TestFunction, TrialFunction, TensorProductCell, dx,
                 action, interval, quadrilateral, hexahedron, dot, grad)
from finat.ufl import FiniteElement, VectorElement

from FIAT import ufc_cell
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule, GaussLegendreQuadratureLineRule

from finat.point_set import GaussLobattoLegendrePointSet, GaussLegendrePointSet
from finat.quadrature import QuadratureRule, TensorProductQuadratureRule

from tsfc import compile_form


def gll_quadrature_rule(cell, elem_deg):
    fiat_cell = ufc_cell("interval")
    fiat_rule = GaussLobattoLegendreQuadratureLineRule(fiat_cell, elem_deg + 1)
    line_rules = [QuadratureRule(GaussLobattoLegendrePointSet(fiat_rule.get_points()),
                                 fiat_rule.get_weights())
                  for _ in range(cell.topological_dimension)]
    finat_rule = reduce(lambda a, b: TensorProductQuadratureRule([a, b]), line_rules)
    return finat_rule


def gl_quadrature_rule(cell, elem_deg):
    fiat_cell = ufc_cell("interval")
    fiat_rule = GaussLegendreQuadratureLineRule(fiat_cell, elem_deg + 1)
    line_rules = [QuadratureRule(GaussLegendrePointSet(fiat_rule.get_points()),
                                 fiat_rule.get_weights())
                  for _ in range(cell.topological_dimension)]
    finat_rule = reduce(lambda a, b: TensorProductQuadratureRule([a, b]), line_rules)
    return finat_rule


def mass_cg(cell, degree):
    m = Mesh(VectorElement('Q', cell, 1))
    V = FunctionSpace(m, FiniteElement('Q', cell, degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    return u*v*dx(scheme=gll_quadrature_rule(cell, degree))


def mass_dg(cell, degree):
    m = Mesh(VectorElement('Q', cell, 1))
    V = FunctionSpace(m, FiniteElement('DQ', cell, degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    return u*v*dx(scheme=gl_quadrature_rule(cell, degree))


def laplace(cell, degree):
    m = Mesh(VectorElement('Q', cell, 1))
    V = FunctionSpace(m, FiniteElement('Q', cell, degree, variant='spectral'))
    u = TrialFunction(V)
    v = TestFunction(V)
    return dot(grad(u), grad(v))*dx(scheme=gll_quadrature_rule(cell, degree))


def count_flops(form):
    kernel, = compile_form(form, parameters=dict(mode='spectral'))
    return kernel.flop_count


@pytest.mark.parametrize('form', [mass_cg, mass_dg])
@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 2),
                          (TensorProductCell(interval, interval), 2),
                          (TensorProductCell(quadrilateral, interval), 3)])
def test_mass(form, cell, order):
    degrees = numpy.arange(4, 10)
    flops = [count_flops(form(cell, int(degree))) for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees + 1))
    assert (rates < order).all()


@pytest.mark.parametrize('form', [mass_cg, mass_dg])
@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 2),
                          (TensorProductCell(interval, interval), 2),
                          (TensorProductCell(quadrilateral, interval), 3)])
def test_mass_action(form, cell, order):
    degrees = numpy.arange(4, 10)
    flops = [count_flops(action(form(cell, int(degree)))) for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees + 1))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 4),
                          (TensorProductCell(interval, interval), 4),
                          (TensorProductCell(quadrilateral, interval), 5)])
def test_laplace(cell, order):
    degrees = numpy.arange(4, 10)
    flops = [count_flops(laplace(cell, int(degree))) for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees + 1))
    assert (rates < order).all()


@pytest.mark.parametrize(('cell', 'order'),
                         [(quadrilateral, 3),
                          (hexahedron, 4)])
def test_laplace_action(cell, order):
    # The matrix-free action of the collocated GLL Laplacian must sum factorise:
    # flops should grow no faster than O(p^{d+1}), i.e. rate < d + 1.  On a
    # genuine d-way tensor-product cell (e.g. hexahedron) the value tabulation
    # is the identity but was previously materialised as a dense tabulation
    # broadcast over the other quadrature directions, which defeated sum
    # factorisation and made the action scale like O(p^{2d}) (rate ~5.8 in 3D).
    degrees = numpy.arange(4, 10)
    flops = [count_flops(action(laplace(cell, int(degree)))) for degree in degrees]
    rates = numpy.diff(numpy.log(flops)) / numpy.diff(numpy.log(degrees + 1))
    assert (rates < order).all()


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
