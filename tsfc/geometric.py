"""Functions to translate UFL reference geometric quantities into GEM
expressions."""

from __future__ import absolute_import

from numpy import array, nan, vstack
from singledispatch import singledispatch

from ufl import interval, triangle, quadrilateral, tetrahedron
from ufl import TensorProductCell
from ufl.classes import (CellEdgeVectors, CellFacetJacobian,
                         CellOrientation, ReferenceCellVolume,
                         ReferenceNormal)

import gem

from tsfc.constants import NUMPY_TYPE
from tsfc.fiatinterface import as_fiat_cell


interval_x_interval = TensorProductCell(interval, interval)
triangle_x_interval = TensorProductCell(triangle, interval)
quadrilateral_x_interval = TensorProductCell(quadrilateral, interval)


# Volume of the reference cells
reference_cell_volume = {
    interval: 1.0,
    triangle: 1.0/2.0,
    quadrilateral: 1.0,
    tetrahedron: 1.0/6.0,
    interval_x_interval: 1.0,
    triangle_x_interval: 1.0/2.0,
    quadrilateral_x_interval: 1.0,
}


# Jacobian of the mapping from a facet to the cell on the reference cell
cell_facet_jacobian = {
    interval: array([[1.0],
                     [1.0]], dtype=NUMPY_TYPE),
    triangle: array([[-1.0, 1.0],
                     [0.0, 1.0],
                     [1.0, 0.0]], dtype=NUMPY_TYPE),
    quadrilateral: array([[0.0, 1.0],
                          [0.0, 1.0],
                          [1.0, 0.0],
                          [1.0, 0.0]], dtype=NUMPY_TYPE),
    tetrahedron: array([[-1.0, -1.0, 1.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=NUMPY_TYPE),
    # Product cells. Convention:
    # Bottom, top, then vertical facets in the order of the base cell
    interval_x_interval: array([[1.0, 0.0],
                                [1.0, 0.0],
                                [0.0, 1.0],
                                [0.0, 1.0]], dtype=NUMPY_TYPE),
    triangle_x_interval: array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                               dtype=NUMPY_TYPE),
    quadrilateral_x_interval: array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                     [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                     [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                                     [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                     [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                                    dtype=NUMPY_TYPE),
}


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
    if integral_type in ["exterior_facet_bottom", "exterior_facet_top", "interior_facet_horiz"]:
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


@translate.register(CellOrientation)  # noqa
def _(terminal, mt, params):
    cell_orientations = params.cell_orientations
    f = {None: 0, '+': 0, '-': 1}[mt.restriction]
    co_int = gem.Indexed(cell_orientations, (f, 0))
    return gem.Conditional(gem.Comparison("==", co_int, gem.Literal(1)),
                           gem.Literal(-1),
                           gem.Conditional(gem.Comparison("==", co_int, gem.Zero()),
                                           gem.Literal(1),
                                           gem.Literal(nan)))


@translate.register(ReferenceCellVolume)  # noqa
def _(terminal, mt, params):
    return gem.Literal(reference_cell_volume[reference_cell(terminal)])


@translate.register(CellFacetJacobian)  # noqa
def _(terminal, mt, params):
    table = cell_facet_jacobian[reference_cell(terminal)]
    table = strip_table(table, params.integral_type)
    table = table.reshape(table.shape[:1] + terminal.ufl_shape)
    return gem.partial_indexed(gem.Literal(table), (params.facet[mt.restriction],))


@translate.register(ReferenceNormal)  # noqa
def _(terminal, mt, params):
    table = reference_normal[reference_cell(terminal)]
    table = strip_table(table, params.integral_type)
    table = table.reshape(table.shape[:1] + terminal.ufl_shape)
    return gem.partial_indexed(gem.Literal(table), (params.facet[mt.restriction],))


@translate.register(CellEdgeVectors)  # noqa
def _(terminal, mt, params):
    from FIAT.reference_element import TensorProductCell as fiat_TensorProductCell
    fiat_cell = as_fiat_cell(terminal.ufl_domain().ufl_cell())
    if isinstance(fiat_cell, fiat_TensorProductCell):
        raise NotImplementedError("CellEdgeVectors not implemented on TensorProductElements yet")

    nedges = len(fiat_cell.get_topology()[1])
    vecs = vstack(map(fiat_cell.compute_edge_tangent, range(nedges))).astype(NUMPY_TYPE)
    assert vecs.shape == terminal.ufl_shape
    return gem.Literal(vecs)
