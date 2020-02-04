# Simple Poisson equation
# =========================

import numpy as np
import math
import pytest

from firedrake import *
from firedrake.cython import dmplex
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


@pytest.mark.parallel
def test_submesh_edge_extraction():

    # Currently we can not define a FunctionSpace
    # at a 0-dimensional point, so just check if
    # we can appropriately mark a point

    # mark edge at [0., 1.]

    msh = RectangleMesh(4, 2, 2., 1., quadrilateral=True)
    msh.init()

    # mark edge only using coordinates
    msh.markSubdomain("custom_edge", 123, "edge", lambda x: x[0] < 0.0001 and x[1] > 0.9999)

    plex = msh._plex
    coords = plex.getCoordinatesLocal()
    coord_sec = plex.getCoordinateSection()

    if plex.getStratumSize("custom_edge", 123) > 0:
        edges = plex.getStratumIS("custom_edge", 123).getIndices()
        for edge in edges:
            v = plex.vecGetClosure(coord_sec, coords, edge)
            assert(np.allclose(v, np.array([0., 1.])))


@pytest.mark.parallel
@pytest.mark.parametrize("f_lambda", [lambda x: x[0] < 1.0001, lambda x: x[0] > 0.9999])
@pytest.mark.parametrize("b_lambda", [lambda x: x[0] > 0.9999, lambda x: x[0] < 1.0001, lambda x: x[1] < 0.0001, lambda x: x[1] > 0.9999])
def test_submesh_poisson_cell(f_lambda, b_lambda):

    # This test is for checking an edge case
    # where we have few elements.

    msh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    msh.init()

    msh.markSubdomain("half_domain", 111, "cell", f_lambda)

    submsh = SubMesh(msh, "half_domain", 111, "cell")
    submsh.markSubdomain(dmplex.FACE_SETS_LABEL, 1, "facet", b_lambda, filterName="exterior_facets", filterValue=1)

    V = FunctionSpace(submsh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(submsh)
    f.interpolate(-8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2))

    dx = Measure("cell", submsh)

    a = - dot(grad(v), grad(u)) * dx
    L = f * v * dx

    g = Function(V)
    g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    bc1 = DirichletBC(V, g, 1)

    parameters = {"mat_type": "aij",
                  "snes_type": "ksponly",
                  "ksp_type": "preonly",
                  "pc_type": "lu"}

    solve(a - L == 0, u, bcs = [bc1], solver_parameters=parameters)


@pytest.mark.parametrize("f_lambda", [lambda x: x[0] < 1.0001, lambda x: x[0] > 0.9999])
@pytest.mark.parametrize("b_lambda", [lambda x: x[0] > 0.9999, lambda x: x[0] < 1.0001, lambda x: x[1] < 0.0001, lambda x: x[1] > 0.9999])
def test_submesh_poisson_cell_error(f_lambda, b_lambda):

    msh = RectangleMesh(200, 100, 2., 1., quadrilateral=True)
    msh.init()

    msh.markSubdomain("half_domain", 111, "cell", f_lambda)

    submsh = SubMesh(msh, "half_domain", 111, "cell")
    submsh.markSubdomain(dmplex.FACE_SETS_LABEL, 1, "facet", b_lambda, filterName="exterior_facets", filterValue=1)

    V = FunctionSpace(submsh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(submsh)
    f.interpolate(-8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2))

    dx = Measure("cell", submsh)

    a = - dot(grad(v), grad(u)) * dx
    L = f * v * dx

    g = Function(V)
    g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    bc1 = DirichletBC(V, g, 1)

    parameters = {"mat_type": "aij",
                  "snes_type": "ksponly",
                  "ksp_type": "preonly",
                  "pc_type": "lu"}

    solve(a - L == 0, u, bcs = [bc1], solver_parameters=parameters)

    assert(sqrt(assemble(dot(u - g, u - g) * dx)) < 0.00016)
