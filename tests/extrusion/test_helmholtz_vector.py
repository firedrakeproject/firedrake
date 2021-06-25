"""Tests for mixed Helmholtz convergence on extruded meshes"""
import numpy as np
import pytest

from firedrake import *


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [(("RT", 1, "DG", 0, "h", "DG", 0, (2, 5)), 0.9),
                          (("RT", 2, "DG", 0, "h", "DG", 1, (2, 4)), 1.55),
                          (("RT", 3, "DG", 0, "h", "DG", 2, (2, 4)), 2.55),
                          (("BDM", 1, "DG", 0, "h", "DG", 0, (2, 5)), 0.9),
                          (("BDM", 2, "DG", 0, "h", "DG", 1, (2, 4)), 1.59),
                          (("BDFM", 2, "DG", 0, "h", "DG", 1, (2, 4)), 1.55),
                          (("DG", 0, "CG", 1, "v", "DG", 0, (2, 5)), 0.9),
                          (("DG", 0, "CG", 2, "v", "DG", 1, (2, 5)), 1.9)])
def test_scalar_convergence(extmesh, testcase, convrate):
    hfamily, hdegree, vfamily, vdegree, ori, altfamily, altdegree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        mesh = extmesh(2**ii, 2**ii, 2**ii)

        horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
        vert_elt = FiniteElement(vfamily, "interval", vdegree)
        product_elt = HDiv(TensorProductElement(horiz_elt, vert_elt))
        V1 = FunctionSpace(mesh, product_elt)

        if ori == "h":
            # use same vertical variation, but different horizontal
            # (think about product of complexes...)
            horiz_elt = FiniteElement(altfamily, "triangle", altdegree)
            vert_elt = FiniteElement(vfamily, "interval", vdegree)
        elif ori == "v":
            # opposite
            horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
            vert_elt = FiniteElement(altfamily, "interval", altdegree)
        product_elt = TensorProductElement(horiz_elt, vert_elt)
        V2 = FunctionSpace(mesh, product_elt)

        x, y, z = SpatialCoordinate(mesh)

        f = Function(V2)
        exact = Function(V2)
        if ori == "h":
            f.interpolate((1+8*np.pi*np.pi)*sin(2*np.pi*x)*sin(2*np.pi*y))
            exact.interpolate(sin(2*np.pi*x)*sin(2*np.pi*y))
        elif ori == "v":
            f.interpolate((1+4*np.pi*np.pi)*sin(2*np.pi*z))
            exact.interpolate(sin(2*np.pi*z))

        W = V1 * V2
        u, p = TrialFunctions(W)
        v, q = TestFunctions(W)
        a = (inner(p, q) - inner(div(u), q) + inner(u, v) + inner(p, div(v)))*dx
        L = inner(f, q)*dx

        out = Function(W)
        solve(a == L, out, solver_parameters={'pc_type': 'fieldsplit',
                                              'pc_fieldsplit_type': 'schur',
                                              'ksp_type': 'cg',
                                              'pc_fieldsplit_schur_fact_type': 'FULL',
                                              'fieldsplit_0_ksp_type': 'cg',
                                              'fieldsplit_1_ksp_type': 'cg'})
        l2err[ii - start] = sqrt(assemble(inner((out[3]-exact), (out[3]-exact))*dx))
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()
