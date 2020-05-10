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


#@pytest.mark.parallel
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


@pytest.mark.parallel
@pytest.mark.parametrize("f_lambda", [lambda x: x[0] < 1.0001, lambda x: x[0] > 0.9999])
@pytest.mark.parametrize("b_lambda", [lambda x: x[0] > 0.9999, lambda x: x[0] < 1.0001, lambda x: x[1] < 0.0001, lambda x: x[1] > 0.9999])
def test_submesh_poisson_cell(f_lambda, b_lambda):

    # This test is for checking an edge case
    # where we have few elements.

    msh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    msh.init()

    msh.markSubdomain("half_domain", 111, "cell", None, geometric_expr = f_lambda)

    submsh = SubMesh(msh, "half_domain", 111, "cell")
    submsh.markSubdomain(dmplex.FACE_SETS_LABEL, 1, "facet", None, geometric_expr = b_lambda, filterName="exterior_facets", filterValue=1)

    V = FunctionSpace(submsh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(submsh)
    f.interpolate(-8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2))

    dx = Measure("cell", submsh)

    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

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

    x, y = SpatialCoordinate(msh)
    DP = FunctionSpace(msh, 'DP', 0)
    fltr = Function(DP)
    fltr = Function(DP).interpolate(ufl.conditional(real(x) < 1, 1, 0))

    msh.markSubdomain("half_domain", 111, "cell", fltr, geometric_expr = f_lambda)

    submsh = SubMesh(msh, "half_domain", 111, "cell")
    # mesh_topology._facets (exterior_facets, interior_facets) is cached,
    # so dmplex.FACE_SETS_LABEL must be set before any call of _facets.
    # This makes it difficult to create a FunctionSpace here.
    # So for now we only allow lambda expression to set dmplex.FACE_SETS_LABEL.
    #RTCF = FunctionSpace(submsh, 'RTCF', 1)
    #fltr = Function(RTCF).project(as_vector([ufl.conditional(real(x) < 1, n * (x - 1), 0), 0]))
    submsh.markSubdomain(dmplex.FACE_SETS_LABEL, 1, "facet", None, geometric_expr = b_lambda, filterName="exterior_facets", filterValue=1)

    V = FunctionSpace(submsh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V)
    x, y = SpatialCoordinate(submsh)
    f.interpolate(-8.0 * pi * pi * cos(x * pi * 2) * cos(y * pi * 2))

    dx = Measure("cell", submsh)

    a = - inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    g = Function(V)
    g.interpolate(cos(2 * pi * x) * cos(2 * pi * y))

    bc1 = DirichletBC(V, g, 1)

    parameters = {"mat_type": "aij",
                  "snes_type": "ksponly",
                  "ksp_type": "preonly",
                  "pc_type": "lu"}

    solve(a - L == 0, u, bcs = [bc1], solver_parameters=parameters)

    assert(sqrt(assemble(inner(u - g, u - g) * dx)) < 0.00016)


def test_submesh_poisson_cell_error2():

    msh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    msh.init()

    msh.markSubdomain("half_domain", 222, "cell", None, geometric_expr = lambda x: x[0] > 0.9999)

    submsh = SubMesh(msh, "half_domain", 222, "cell")

    V0 = FunctionSpace(msh, "CG", 1)
    V1 = FunctionSpace(submsh, "CG", 1)

    W = V0 * V1

    w = Function(W)
    u0, u1 = TrialFunctions(W)
    v0, v1 = TestFunctions(W)

    f0 = Function(V0)
    x0, y0 = SpatialCoordinate(msh)
    f0.interpolate(-8.0 * pi * pi * cos(x0 * pi * 2) * cos(y0 * pi * 2))

    dx0 = Measure("cell", domain=msh)
    dx1 = Measure("cell", domain=submsh)


    a = inner(u0, v0) * dx0 + inner(u0 - u1, v1) * dx1
    L = inner(f0, v0) * dx0

    mat = assemble(a)
    print(mat.M[1][0].values)

    m00 = np.array([[2./9. , 1./9. , 1./36., 1./18., 1./18., 1./36.],
                    [1./9. , 2./9. , 1./18., 1./36., 1./36., 1./18.],
                    [1./36., 1./18., 1./9. , 1./18., 0.    , 0.    ],
                    [1./18., 1./36., 1./18., 1./9. , 0.    , 0.    ],
                    [1./18., 1./36., 0.    , 0.    , 1./9. , 1./18.],
                    [1./36., 1./18., 0.    , 0.    , 1./18., 1./9. ]])

    m01 = np.array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0., 0., 0., 0.]])

    m10 = np.array([[1./18., 1./9. , 1./18., 1./36., 0., 0.],
                    [1./9. , 1./18., 1./36., 1./18., 0., 0.],
                    [1./18., 1./36., 1./18., 1./9. , 0., 0.],
                    [1./36., 1./18., 1./9. , 1./18., 0., 0.]])

    m11 = np.array([[-1./9. , -1./18., -1./36., -1./18.],
                    [-1./18., -1./9. , -1./18., -1./36.],
                    [-1./36., -1./18., -1./9. , -1./18.],
                    [-1./18., -1./36., -1./18., -1./9. ]])

    assert(np.allclose(mat.M[0][0].values, m00))
    assert(np.allclose(mat.M[0][1].values, m01))
    assert(np.allclose(mat.M[1][0].values, m10))
    assert(np.allclose(mat.M[1][1].values, m11))


def test_submesh_helmholtz():

    msh = RectangleMesh(2, 1, 2., 1., quadrilateral=True)
    msh.init()

    msh.markSubdomain("half_domain", 222, "cell", None, geometric_expr = lambda x: x[0] > 0.9999)

    submsh = SubMesh(msh, "half_domain", 222, "cell")

    V0 = FunctionSpace(msh, "CG", 1)
    V1 = FunctionSpace(submsh, "CG", 1)

    W = V0 * V1

    w = Function(W)
    u0, u1 = TrialFunctions(W)
    v0, v1 = TestFunctions(W)

    f0 = Function(V0)
    x0, y0 = SpatialCoordinate(msh)
    f0.interpolate(-8.0 * pi * pi * cos(x0 * pi * 2) * cos(y0 * pi * 2))

    dx0 = Measure("cell", domain=msh)
    dx1 = Measure("cell", domain=submsh)


    a = inner(grad(u0), grad(v0)) * dx0 + inner(u0 - u1, v1) * dx1
    L = inner(f0, v0) * dx0

    mat = assemble(a)
