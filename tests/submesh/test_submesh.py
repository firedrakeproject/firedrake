# Simple Poisson equation
# =========================

import numpy as np
import math
import pytest

from firedrake import *
from firedrake import dmplex
from firedrake.petsc import PETSc
from pyop2.datatypes import IntType


@pytest.mark.parallel
def test_submesh_facet_extraction():

    # manually mark facets 1, 2, 3, 4 and compare
    # with the default label

    msh = RectangleMesh(200, 100, 2., 1., quadrilateral=True)
    msh.init()

    # mark facet only using coordinates
    msh.markSubdomain("custom_facet", 111, "facet", lambda x: x[0] < 0.0001)
    msh.markSubdomain("custom_facet", 222, "facet", lambda x: x[0] > 1.9999)
    msh.markSubdomain("custom_facet", 333, "facet", lambda x: x[1] < 0.0001)
    msh.markSubdomain("custom_facet", 444, "facet", lambda x: x[1] > 0.9999)

    plex = msh._plex
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 1) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 111).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 1).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 2) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 222).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 2).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 3) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 333).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 3).getIndices())))
    if plex.getStratumSize(dmplex.FACE_SETS_LABEL, 4) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 444).getIndices(), plex.getStratumIS(dmplex.FACE_SETS_LABEL, 4).getIndices())))

    # make submesh
    _ = SubMesh(msh, "custom_facet", 111, "facet")


@pytest.mark.parallel
def test_submesh_edge_extraction():

    # Currently we can not define a FunctionSpace
    # at a 0-dimensional point, so just check if
    # we can appropriately mark a point

    # mark edge at [0., 1.]

    msh = RectangleMesh(4, 2, 2., 1., quadrilateral=True)
    msh.init()

    # mark facet only using coordinates
    msh.markSubdomain("custom_facet", 111, "facet", lambda x: x[0] < 0.0001)
    # mark facet using coordinate and existing labels
    msh.markSubdomain("custom_facet", 222, "facet", lambda x: x[1] > 0.9999, filterName="exterior_facets", filterValue=1)
    # mark edge as an intersection of two labeled facets
    msh.markSubdomainIntersection("custom_edge", 333, "edge", "custom_facet", (111, 222))

    # mark edge only using coordinates
    msh.markSubdomain("custom_edge2", 444, "edge", lambda x: x[0] < 0.0001 and x[1] > 0.9999)

    plex = msh._plex
    coords = plex.getCoordinatesLocal()
    coord_sec = plex.getCoordinateSection()

    if plex.getStratumSize("custom_edge", 333) > 0:
        edges = plex.getStratumIS("custom_edge", 333).getIndices()
        for edge in edges:
            v = plex.vecGetClosure(coord_sec, coords, edge)
            assert(np.allclose(v, np.array([0., 1.])))

    if plex.getStratumSize("custom_edge2", 444) > 0:
        edges = plex.getStratumIS("custom_edge2", 444).getIndices()
        for edge in edges:
            v = plex.vecGetClosure(coord_sec, coords, edge)
            assert(np.allclose(v, np.array([0., 1.])))

    # make submesh
    _ = SubMesh(msh, "custom_edge", 333, "edge")


def test_submesh_poisson_cell():

    msh = UnitSquareMesh(20, 20)

    V = FunctionSpace(msh, "CG", 3)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate(-8.0*pi*pi*cos(x*pi*2)*cos(y*pi*2))

    a = - dot(grad(v), grad(u)) * dx
    L = f * v * dx

    g = Function(V)
    g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    n = FacetNormal(mesh)
    e2 = as_vector([0., 1.])
    bc1 = EquationBC((-dot(grad(v), e2) * dot(grad(u), e2) + 4 * pi * pi * v * u ) * ds(1) == 0, u, 1, bcs=[bbc])

    parameters = {
        "mat_type": "aij",
        "snes_max_it": 1,
        "snes_monitor": True,
        "ksp_type": "gmres",
        "ksp_rtol": 1.e-12,
        "ksp_atol": 1.e-12,
        "ksp_max_it": 500000,
        "pc_type": "asm"
    }

    solve(a - L == 0, u, bcs = [bc1], solver_parameters=parameters)

    f.interpolate(cos(x*pi*2)*cos(y*pi*2))
    print(math.log2(sqrt(assemble(dot(u - f, u - f) * dx))))
