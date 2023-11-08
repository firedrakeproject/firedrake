"""Tests for Galerkin projection convergence on extruded meshes"""
import numpy as np
import pytest

from firedrake import *


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [(("CG", 1), 1.5), (("CG", 2), 2.6),
                          (("DG", 0), 0.9), (("DG", 1), 1.7)])
def test_scalar_convergence(extmesh, testcase, convrate):
    family, degree = testcase
    l2err = np.zeros(2)
    for ii in range(len(l2err)):
        mesh = extmesh(2**(ii+1), 2**(ii+1), 2**ii)

        fspace = FunctionSpace(mesh, family, degree, vfamily=family, vdegree=degree)
        exactfspace = FunctionSpace(mesh, "Lagrange", 3)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        x, y, z = SpatialCoordinate(mesh)
        expr = x*x*y*z
        exact = project(expr, exactfspace)

        out = Function(fspace)
        solve(inner(u, v)*dx == inner(exact, v)*dx, out)
        l2err[ii] = sqrt(assemble(inner((out-exact), (out-exact))*dx))
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [(("RT", 1, "DG", 0, "h"), 0.9),
                          (("RT", 2, "DG", 1, "h"), 1.94),
                          (("RT", 3, "DG", 2, "h"), 2.9),
                          (("BDM", 1, "DG", 1, "h"), 1.8),
                          (("BDM", 2, "DG", 2, "h"), 2.8),
                          (("BDFM", 2, "DG", 1, "h"), 1.95),
                          (("N1curl", 1, "DG", 0, "h"), 0.9),
                          (("N1curl", 2, "DG", 1, "h"), 1.9),
                          (("N2curl", 1, "DG", 1, "h"), 1.8),
                          (("N2curl", 2, "DG", 2, "h"), 2.85),
                          (("DG", 1, "CG", 1, "v"), 1.84),
                          (("DG", 2, "CG", 2, "v"), 2.98),
                          (("RT", 1, "Real", 0, "h"), 0.9)])
def test_hdiv_convergence(extmesh, testcase, convrate):
    hfamily, hdegree, vfamily, vdegree, orientation = testcase
    l2err = np.zeros(2)
    for ii in range(len(l2err)):
        mesh = extmesh(2**(ii+1), 2**(ii+1), 2**(ii+1))

        exactfspace = VectorFunctionSpace(mesh, "Lagrange", 3)

        horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
        vert_elt = FiniteElement(vfamily, "interval", vdegree)
        product_elt = HDiv(TensorProductElement(horiz_elt, vert_elt))
        fspace = FunctionSpace(mesh, product_elt)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        x, y, z = SpatialCoordinate(mesh)

        if orientation == "h":
            expr = as_vector([x*y*z*z, x*x*y*z, Constant(0.0)])
        elif orientation == "v":
            expr = as_vector([Constant(0.0), Constant(0.0), x*y*y*z])
        exact = Function(exactfspace)
        exact.interpolate(expr)

        out = Function(fspace)
        solve(inner(u, v)*dx == inner(exact, v)*dx, out)
        l2err[ii] = sqrt(assemble(inner((out-exact), (out-exact))*dx))
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [(("BDM", 1, "CG", 1, "h"), 1.82),
                          (("BDM", 2, "CG", 2, "h"), 2.9),
                          (("RT", 2, "CG", 1, "h"), 1.87),
                          (("RT", 3, "CG", 2, "h"), 2.95),
                          (("BDFM", 2, "CG", 1, "h"), 1.77),
                          (("N1curl", 2, "CG", 1, "h"), 1.87),
                          (("N2curl", 1, "CG", 1, "h"), 1.82),
                          (("N2curl", 2, "CG", 2, "h"), 2.9),
                          (("CG", 1, "DG", 1, "v"), 1.6),
                          (("CG", 2, "DG", 2, "v"), 2.7)])
def test_hcurl_convergence(extmesh, testcase, convrate):
    hfamily, hdegree, vfamily, vdegree, orientation = testcase
    l2err = np.zeros(2)
    for ii in range(len(l2err)):
        mesh = extmesh(2**(ii+1), 2**(ii+1), 2**(ii+1))

        exactfspace = VectorFunctionSpace(mesh, "Lagrange", 3)

        horiz_elt = FiniteElement(hfamily, "triangle", hdegree)
        vert_elt = FiniteElement(vfamily, "interval", vdegree)
        product_elt = HCurl(TensorProductElement(horiz_elt, vert_elt))
        fspace = FunctionSpace(mesh, product_elt)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        x, y, z = SpatialCoordinate(mesh)

        if orientation == "h":
            expr = as_vector([x*y*z*z, x*x*y*z, Constant(0.0)])
        elif orientation == "v":
            expr = as_vector([Constant(0.0), Constant(0.0), x*y*y*z])
        exact = Function(exactfspace)
        exact.interpolate(expr)

        out = Function(fspace)
        solve(inner(u, v)*dx == inner(exact, v)*dx, out)
        l2err[ii] = sqrt(assemble(inner((out-exact), (out-exact))*dx))
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()
