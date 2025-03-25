import pytest
import numpy as np
from firedrake import *
from firedrake.__future__ import *
from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement
from FIAT.reference_element import UFCInterval
from FIAT.functional import PointEvaluation, IntegralMoment
from FIAT.quadrature import make_quadrature
from FIAT.polynomial_set import ONPolynomialSet
from finat.fiat_elements import ScalarFiatElement
from finat.element_factory import convert, as_fiat_cell
import finat.ufl

ufcint = UFCInterval()


# Purpose of these tests: test interpolation with a simple 1D element newly
# created in a single location
# New FIAT Element For P3 on an interval with moment nodes
class P3IntMomentsDualSet(DualSet):
    r"""
    Dual set for an element, only defined for order 3 on interval cells, with
    nodes:
    ..math::
     phi_0'(f) = f(0.)
     phi_1'(f) = f(1.)
     pfi_2'(f) = \int_{0}^{1} f(x) \,dx
     pfi_3'(f) = \int_{0}^{1} x f(x) \,dx
     """

    def __init__(self, cell, order):
        assert cell == UFCInterval() and order == 3
        entity_ids = {0: {0: [0], 1: [1]},
                      1: {0: [2, 3]}}
        vertnodes = [PointEvaluation(cell, xx)
                     for xx in cell.vertices]
        Q = make_quadrature(cell, 3)
        # 1st integral moment node is integral(1*f(x)*dx)
        ones = np.asarray([1.0 for x in Q.pts])
        # 2nd integral moment node is integral(x*f(x)*dx)
        xs = np.asarray([x for (x,) in Q.pts])
        intnodes = [IntegralMoment(cell, Q, ones),
                    IntegralMoment(cell, Q, xs)]
        nodes = vertnodes + intnodes
        super().__init__(nodes, cell, entity_ids)


class P3IntMoments(CiarletElement):
    r"""
    An element, only defined for order 3 on interval cells, with 4 nodes:
    ..math::
     phi_0'(f) = f(0.)
     phi_1'(f) = f(1.)
     pfi_2'(f) = \int_{0}^{1} f(x) \,dx
     pfi_3'(f) = \int_{0}^{1} x f(x) \,dx
     """

    def __init__(self, cell, order):
        assert cell == UFCInterval() and order == 3
        poly_set = ONPolynomialSet(cell, 3)
        dual_set = P3IntMomentsDualSet(cell, 3)
        super().__init__(poly_set, dual_set, 3, 0)


def test_fiat_p3intmoments():
    # can create
    el = P3IntMoments(UFCInterval(), 3)
    # have expected number of nodes
    assert len(el.dual_basis()) == 4
    # dual eval gives expected values from sum of point evaluations
    fns = (lambda x: x[0], lambda x: x[0]**2)
    expected = ([0, 1, 1/2, 1/3], [0, 1, 1/3, 1/4])
    for fn, expect in zip(fns, expected):
        node_vals = []
        for node in el.dual_basis():
            pt_dict = node.pt_dict
            node_val = 0.0
            for pt in pt_dict:
                for (w, _) in pt_dict[pt]:
                    node_val += w * fn(pt)
            node_vals.append(node_val)
        assert np.allclose(node_vals, expect)


# FInAT equivalent
class FInAT_P3IntMoments(ScalarFiatElement):
    def __init__(self, cell, degree):
        super().__init__(P3IntMoments(cell, degree))


# Replace the old finat.element_factory.convert dispatch with a new one that
# gives the the new FInAT element for P3 on an interval with variant
# "interior-moment"
old_convert = convert.dispatch(finat.ufl.FiniteElement)


def temp_convert(element, **kwargs):
    if element.family() == "Lagrange" and element.cell == interval \
       and element.degree() == 3 and element.variant() == "interior-moment":
        return FInAT_P3IntMoments(as_fiat_cell(element.cell), element.degree()), set()
    else:
        return old_convert(element, **kwargs)


# Register the new tsfc covert method - remove after tests have run (post yield)
@pytest.fixture
def add_p3intmoments_tsfc():
    convert.register(finat.ufl.FiniteElement, temp_convert)
    yield
    convert.register(finat.ufl.FiniteElement, old_convert)


# Test New Element Dual Evaluation
def test_basic_dual_eval_p3intmoments(add_p3intmoments_tsfc):
    mesh = UnitIntervalMesh(1)
    e = finat.ufl.FiniteElement("CG", "interval", 3, variant="interior-moment")
    V = FunctionSpace(mesh, e)
    x = SpatialCoordinate(mesh)
    expr = Constant(1.)
    f = assemble(interpolate(expr, V))
    dual_basis = f.function_space().finat_element.fiat_equivalent.dual_basis()
    assert np.allclose(f.dat.data_ro[f.cell_node_map().values],
                       [node(expr) for node in dual_basis])
    expr = x[0]
    # Account for cell and corresponding expression being flipped onto
    # reference cell before reaching FIAT
    expr_fiat = 1-x[0]
    f = assemble(interpolate(expr, V))
    dual_basis = f.function_space().finat_element.fiat_equivalent.dual_basis()
    assert np.allclose(f.dat.data_ro[f.cell_node_map().values],
                       [node(expr_fiat) for node in dual_basis])
    expr = x[0]**2
    expr_fiat = (1-x[0])**2
    f = assemble(interpolate(expr, V))
    dual_basis = f.function_space().finat_element.fiat_equivalent.dual_basis()
    assert np.allclose(f.dat.data_ro[f.cell_node_map().values],
                       [node(expr_fiat) for node in dual_basis])
