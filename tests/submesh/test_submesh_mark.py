# Simple Poisson equation
# =========================

import numpy as np
import math
import pytest

from firedrake import *
from firedrake.cython import dmplex
from firedrake.petsc import PETSc
from pyop2.datatypes import IntType
import ufl


@pytest.mark.parallel
def test_submesh_facet_extraction():

    # manually mark facets 1, 2, 3, 4 and compare
    # with the default label

    n = 100
    msh = RectangleMesh(2 * n, n, 2., 1., quadrilateral=True)
    msh.init()

    x, y = SpatialCoordinate(msh)
    RTCF = FunctionSpace(msh, 'RTCF', 1)
    fltr = Function(RTCF).project(as_vector([ufl.conditional(real(x) < 1, n * (x - 1), 0), 0]))

    # mark facet only using coordinates
    msh.markSubdomain("custom_facet", 111, "facet", fltr, filterName="exterior_facets", filterValue=1)
    msh.markSubdomain("custom_facet", 222, "facet", None, geometric_expr = lambda x: x[0] > 1.9999)
    msh.markSubdomain("custom_facet", 333, "facet", None, geometric_expr = lambda x: x[1] < 0.0001)
    msh.markSubdomain("custom_facet", 444, "facet", None, geometric_expr = lambda x: x[1] > 0.9999)

    plex = msh._plex
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 1) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 111).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 1).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 2) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 222).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 2).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 3) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 333).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 3).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 4) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 444).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 4).getIndices())))


@pytest.mark.parallel
def test_submesh_facet_extraction1():

    # manually mark facets 1, 2, 3, 4 and compare
    # with the default label

    msh = RectangleMesh(200, 100, 2., 1., quadrilateral=True)
    msh.init()

    # mark facet only using coordinates
    msh.markSubdomain("custom_facet", 111, "facet", None, geometric_expr = lambda x: x[0] < 0.0001)
    msh.markSubdomain("custom_facet", 222, "facet", None, geometric_expr = lambda x: x[0] > 1.9999)
    msh.markSubdomain("custom_facet", 333, "facet", None, geometric_expr = lambda x: x[1] < 0.0001)
    msh.markSubdomain("custom_facet", 444, "facet", None, geometric_expr = lambda x: x[1] > 0.9999)

    plex = msh._plex
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 1) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 111).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 1).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 2) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 222).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 2).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 3) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 333).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 3).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 4) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 444).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 4).getIndices())))


@pytest.mark.parallel
def test_submesh_edge_extraction():

    # Currently we can not define a FunctionSpace
    # at a 0-dimensional point, so just check if
    # we can appropriately mark a point

    # mark edge at [0., 1.]

    msh = RectangleMesh(4, 2, 2., 1., quadrilateral=True)
    msh.init()

    # mark edge only using coordinates
    msh.markSubdomain("custom_edge", 123, "edge", None, geometric_expr = lambda x: x[0] < 0.0001 and x[1] > 0.9999)

    plex = msh._plex
    coords = plex.getCoordinatesLocal()
    coord_sec = plex.getCoordinateSection()

    if plex.getStratumSize("custom_edge", 123) > 0:
        edges = plex.getStratumIS("custom_edge", 123).getIndices()
        for edge in edges:
            v = plex.vecGetClosure(coord_sec, coords, edge)
            assert(np.allclose(v, np.array([0., 1.])))
