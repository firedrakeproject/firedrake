"""Functions to translate UFL reference geometric quantities into GEM
expressions."""

from __future__ import absolute_import

from numpy import array, nan, vstack
from singledispatch import singledispatch
import sympy

from ufl.classes import (CellCoordinate, CellEdgeVectors,
                         CellFacetJacobian, CellOrientation,
                         FacetCoordinate, ReferenceCellVolume,
                         ReferenceFacetVolume, ReferenceNormal)

import gem

from tsfc.constants import NUMPY_TYPE


@singledispatch
def translate(terminal, mt, params):
    """Translate geometric UFL quantity into GEM expression.

    :arg terminal: UFL geometric quantity terminal
    :arg mt: modified terminal data (e.g. for restriction)
    :arg params: miscellaneous
    """
    raise AssertionError("Cannot handle geometric quantity type: %s" % type(terminal))


@translate.register(CellOrientation)
def translate_cell_orientation(terminal, mt, params):
    cell_orientations = params.cell_orientations
    f = {None: 0, '+': 0, '-': 1}[mt.restriction]
    co_int = gem.Indexed(cell_orientations, (f, 0))
    return gem.Conditional(gem.Comparison("==", co_int, gem.Literal(1)),
                           gem.Literal(-1),
                           gem.Conditional(gem.Comparison("==", co_int, gem.Zero()),
                                           gem.Literal(1),
                                           gem.Literal(nan)))


@translate.register(ReferenceCellVolume)
def translate_reference_cell_volume(terminal, mt, params):
    return gem.Literal(params.fiat_cell.volume())


@translate.register(ReferenceFacetVolume)
def translate_reference_facet_volume(terminal, mt, params):
    # FIXME: simplex only code path
    dim = params.fiat_cell.get_spatial_dimension()
    facet_cell = params.fiat_cell.construct_subelement(dim - 1)
    return gem.Literal(facet_cell.volume())


@translate.register(CellFacetJacobian)
def translate_cell_facet_jacobian(terminal, mt, params):
    assert params.integration_dim != params.fiat_cell.get_dimension()
    dim = params.fiat_cell.construct_subelement(params.integration_dim).get_spatial_dimension()
    X = sympy.DeferredVector('X')
    point = [X[j] for j in range(dim)]
    result = []
    for entity_id in params.entity_ids:
        f = params.fiat_cell.get_entity_transform(params.integration_dim, entity_id)
        y = f(point)
        J = [[sympy.diff(y_i, X[j])
              for j in range(dim)]
             for y_i in y]
        J = array(J, dtype=float)
        result.append(J)
    return params.select_facet(gem.Literal(result), mt.restriction)


@translate.register(ReferenceNormal)
def translate_reference_normal(terminal, mt, params):
    result = []
    for facet_i in params.entity_ids:
        n = params.fiat_cell.compute_scaled_outward_normal(params.integration_dim, facet_i)
        result.append(n)
    return params.select_facet(gem.Literal(result), mt.restriction)


@translate.register(CellEdgeVectors)
def translate_cell_edge_vectors(terminal, mt, params):
    from FIAT.reference_element import TensorProductCell as fiat_TensorProductCell
    fiat_cell = params.fiat_cell
    if isinstance(fiat_cell, fiat_TensorProductCell):
        raise NotImplementedError("CellEdgeVectors not implemented on TensorProductElements yet")

    nedges = len(fiat_cell.get_topology()[1])
    vecs = vstack(map(fiat_cell.compute_edge_tangent, range(nedges))).astype(NUMPY_TYPE)
    assert vecs.shape == terminal.ufl_shape
    return gem.Literal(vecs)


@translate.register(CellCoordinate)
def translate_cell_coordinate(terminal, mt, params):
    return gem.partial_indexed(params.select_facet(gem.Literal(params.entity_points),
                                                   mt.restriction),
                               (params.point_index,))


@translate.register(FacetCoordinate)
def translate_facet_coordinate(terminal, mt, params):
    assert params.integral_type != 'cell'
    points = params.points
    return gem.partial_indexed(gem.Literal(points), (params.point_index,))
