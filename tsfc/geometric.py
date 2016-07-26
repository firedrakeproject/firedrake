"""Functions to translate UFL reference geometric quantities into GEM
expressions."""

from __future__ import absolute_import

from numpy import array, nan, vstack
from singledispatch import singledispatch
import sympy

from ufl import interval, triangle, quadrilateral, tetrahedron
from ufl import TensorProductCell
from ufl.classes import (CellCoordinate, CellEdgeVectors,
                         CellFacetJacobian, CellOrientation,
                         FacetCoordinate, ReferenceCellVolume,
                         ReferenceFacetVolume, ReferenceNormal)

import gem

from tsfc.constants import NUMPY_TYPE
from tsfc.fiatinterface import as_fiat_cell


interval_x_interval = TensorProductCell(interval, interval)
triangle_x_interval = TensorProductCell(triangle, interval)
quadrilateral_x_interval = TensorProductCell(quadrilateral, interval)


# Facet normals of the reference cells
reference_normal = {
    interval: array([[-1.0],
                     [1.0]], dtype=NUMPY_TYPE),
    triangle: array([[1.0, 1.0],
                     [-1.0, 0.0],
                     [0.0, -1.0]], dtype=NUMPY_TYPE),
    quadrilateral: array([[-1.0, 0.0],
                          [1.0, 0.0],
                          [0.0, -1.0],
                          [0.0, 1.0]], dtype=NUMPY_TYPE),
    tetrahedron: array([[1.0, 1.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0]], dtype=NUMPY_TYPE),
    # Product cells. Convention:
    # Bottom, top, then vertical facets in the order of the base cell
    interval_x_interval: array([[0.0, -1.0],
                                [0.0, 1.0],
                                [-1.0, 0.0],
                                [1.0, 0.0]], dtype=NUMPY_TYPE),
    triangle_x_interval: array([[0.0, 0.0, -1.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0]], dtype=NUMPY_TYPE),
    quadrilateral_x_interval: array([[0.0, 0.0, -1.0],
                                     [0.0, 0.0, 1.0],
                                     [-1.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [0.0, -1.0, 0.0],
                                     [0.0, 1.0, 0.0]], dtype=NUMPY_TYPE),
}


def reference_cell(terminal):
    """Reference cell of terminal"""
    cell = terminal.ufl_domain().ufl_cell()
    return cell.reconstruct(geometric_dimension=cell.topological_dimension())


def strip_table(table, integral_type):
    """Select horizontal or vertical parts for facet integrals on
    extruded cells. No-op in all other cases."""
    if integral_type == "exterior_facet_bottom":
        return table[0:1]
    elif integral_type == "exterior_facet_top":
        return table[1:2]
    elif integral_type == "interior_facet_horiz":
        # Bottom and top
        return table[:2]
    elif integral_type in ["exterior_facet_vert", "interior_facet_vert"]:
        # Vertical facets in the order of the base cell
        return table[2:]
    else:
        return table


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
    table = reference_normal[reference_cell(terminal)]
    table = strip_table(table, params.integral_type)
    table = table.reshape(table.shape[:1] + terminal.ufl_shape)
    return params.select_facet(gem.Literal(table), mt.restriction)


@translate.register(CellEdgeVectors)
def translate_cell_edge_vectors(terminal, mt, params):
    from FIAT.reference_element import TensorProductCell as fiat_TensorProductCell
    fiat_cell = as_fiat_cell(terminal.ufl_domain().ufl_cell())
    if isinstance(fiat_cell, fiat_TensorProductCell):
        raise NotImplementedError("CellEdgeVectors not implemented on TensorProductElements yet")

    nedges = len(fiat_cell.get_topology()[1])
    vecs = vstack(map(fiat_cell.compute_edge_tangent, range(nedges))).astype(NUMPY_TYPE)
    assert vecs.shape == terminal.ufl_shape
    return gem.Literal(vecs)


@translate.register(CellCoordinate)
def translate_cell_coordinate(terminal, mt, params):
    points = params.tabulation_manager.points
    if params.integral_type != 'cell':
        points = list(params.facet_manager.facet_transform(points))
    return gem.partial_indexed(params.select_facet(gem.Literal(points),
                                                   mt.restriction),
                               (params.point_index,))


@translate.register(FacetCoordinate)
def translate_facet_coordinate(terminal, mt, params):
    assert params.integral_type != 'cell'
    points = params.tabulation_manager.points
    return gem.partial_indexed(gem.Literal(points), (params.point_index,))
