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
def test_submesh_submesh():

    # manually mark facets 1, 2, 3, 4 and compare
    # with the default label

    n = 100
    msh = RectangleMesh(2 * n, n, 2., 1., quadrilateral=True)
    msh.init()

    x, y = SpatialCoordinate(msh)
    DP = FunctionSpace(msh, 'DP', 0)
    fltr = Function(DP)
    fltr = Function(DP).interpolate(ufl.conditional(real(x) < 1, 1, 0))

    #msh.markSubdomain("half_domain", 111, "cell", fltr, geometric_expr = f_lambda)
    msh.markSubdomain("half_domain", 111, "cell", fltr)

    submsh = SubMesh(msh, "half_domain", 111, "cell")
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


