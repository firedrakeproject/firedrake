# Simple Poisson equation
# =========================

import numpy as np
import math
import pytest

from firedrake import *
from firedrake.cython import dmcommon
from firedrake.petsc import PETSc
from pyop2.datatypes import IntType
import ufl


@pytest.mark.parallel
def test_submesh_submesh_mark_subdomain():
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

    plex = msh._topology_dm
    if plex.getStratumSize(dmcommon.FACE_SETS_LABEL, 1) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 111).getIndices(), plex.getStratumIS(dmcommon.FACE_SETS_LABEL, 1).getIndices())))
    if plex.getStratumSize(dmcommon.FACE_SETS_LABEL, 2) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 222).getIndices(), plex.getStratumIS(dmcommon.FACE_SETS_LABEL, 2).getIndices())))
    if plex.getStratumSize(dmcommon.FACE_SETS_LABEL, 3) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 333).getIndices(), plex.getStratumIS(dmcommon.FACE_SETS_LABEL, 3).getIndices())))
    if plex.getStratumSize(dmcommon.FACE_SETS_LABEL, 4) > 0:
        assert(np.all(np.equal(plex.getStratumIS("custom_facet", 444).getIndices(), plex.getStratumIS(dmcommon.FACE_SETS_LABEL, 4).getIndices())))


@pytest.mark.parallel
@pytest.mark.parametrize("quadrilateral", [True, False])
def test_submesh_submesh_cell_closure_order(quadrilateral):
    # Check that the following diagram commutes:
    # 
    #                          cell_numbering
    #            plex   c   ------------------->  c_      cell_closure[c_, :]
    #                   ^                                      ^
    #       subpointMap |                                      | subpointMap
    #                   |     subcell_numbering                |
    #          subplex subc -------------------> subc_ subcell_closure[subc_, :]
    #
    # Confirmed to work in parallel with comm size = 1,2,3,5,7,11,13,17.

    msh = RectangleMesh(4, 2, 2., 1., quadrilateral=quadrilateral)
    msh.init()

    x0, y0 = SpatialCoordinate(msh)
    DP = FunctionSpace(msh, 'DP', 0)
    fltr = Function(DP)
    fltr = Function(DP).interpolate(ufl.conditional(real(x0) > 1, 1, 0))
    msh.markSubdomain("half_domain", 222, "cell", fltr)

    submsh = SubMesh(msh, "half_domain", 222, "cell")
    submsh.init()
    plex = msh.topology._topology_dm
    subplex = submsh.topology._topology_dm


    cell_numbering = msh._cell_numbering
    cell_closure = msh.cell_closure
    subcell_numbering = submsh._cell_numbering
    subcell_closure = submsh.cell_closure
    subcStart, subcEnd = subplex.getHeightStratum(0)
    subpoint_map = subplex.getSubpointIS().getIndices()
    for subc in range(subcStart, subcEnd):
        # parent mesh cell closure
        c = subpoint_map[subc]
        c_ = cell_numbering.getOffset(c)
        cc = cell_closure[c_]
        # submesh cell closure
        subc_ = subcell_numbering.getOffset(subc)
        subcc = subcell_closure[subc_]
        print("cc::::::", cc)
        print("subcc:::", subpoint_map[subcc])
        assert(np.all(cc == subpoint_map[subcc]))
