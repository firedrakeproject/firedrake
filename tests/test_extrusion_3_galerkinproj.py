"""Tests for Galerkin projection convergence on extruded meshes"""
import numpy as np
import pytest

from firedrake import *


def convergence_test_scalar(testcase):
    family, degree = testcase
    l2err = np.zeros(3)
    for ii in range(len(l2err)):
        m = UnitSquareMesh(2**(ii+1), 2**(ii+1))
        mesh = ExtrudedMesh(m, 2**ii + 1, layer_height=1.0/(2**ii))

        fspace = FunctionSpace(mesh, family, degree, vfamily=family, vdegree=degree)
        exactfspace = FunctionSpace(mesh, "Lagrange", 3)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        expr = Expression("x[0]*x[0]*x[1]*x[2]")
        exact = project(expr, exactfspace)

        out = Function(fspace)
        solve(u*v*dx == exact*v*dx, out)
        l2err[ii] = sqrt(assemble((out-exact)*(out-exact)*dx))
    return np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)])


def convergence_test_hdiv(testcase):
    family, degree, vfamily, vdegree, orientation = testcase
    l2err = np.zeros(3)
    for ii in range(len(l2err)):
        m = UnitSquareMesh(2**(ii+1), 2**(ii+1))
        mesh = ExtrudedMesh(m, 2**(ii+1) + 1, layer_height=1.0/(2**(ii+1)))

        exactfspace = VectorFunctionSpace(mesh, "Lagrange", 3)

        horiz_elt = FiniteElement(family, "triangle", degree)
        vert_elt = FiniteElement(vfamily, "interval", vdegree)
        product_elt = HDiv(OuterProductElement(horiz_elt, vert_elt))
        fspace = FunctionSpace(mesh, product_elt)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        if orientation == "h":
            expr = Expression(("x[0]*x[1]*x[2]*x[2]", "x[0]*x[0]*x[1]*x[2]", "0.0"))
        elif orientation == "v":
            expr = Expression(("0.0", "0.0", "x[0]*x[1]*x[1]*x[2]"))
        exact = Function(exactfspace)
        exact.interpolate(expr)

        out = Function(fspace)
        solve(dot(u, v)*dx == dot(exact, v)*dx, out)
        l2err[ii] = sqrt(assemble(dot((out-exact), (out-exact))*dx))
    return np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)])


def convergence_test_hcurl(testcase):
    family, degree, vfamily, vdegree, orientation = testcase
    l2err = np.zeros(3)
    for ii in range(len(l2err)):
        m = UnitSquareMesh(2**(ii+1), 2**(ii+1))
        mesh = ExtrudedMesh(m, 2**(ii+1) + 1, layer_height=1.0/(2**(ii+1)))

        exactfspace = VectorFunctionSpace(mesh, "Lagrange", 3)

        horiz_elt = FiniteElement(family, "triangle", degree)
        vert_elt = FiniteElement(vfamily, "interval", vdegree)
        product_elt = HCurl(OuterProductElement(horiz_elt, vert_elt))
        fspace = FunctionSpace(mesh, product_elt)

        u = TrialFunction(fspace)
        v = TestFunction(fspace)

        if orientation == "h":
            expr = Expression(("x[0]*x[1]*x[2]*x[2]", "x[0]*x[0]*x[1]*x[2]", "0.0"))
        elif orientation == "v":
            expr = Expression(("0.0", "0.0", "x[0]*x[1]*x[1]*x[2]"))
        exact = Function(exactfspace)
        exact.interpolate(expr)

        out = Function(fspace)
        solve(dot(u, v)*dx == dot(exact, v)*dx, out)
        l2err[ii] = sqrt(assemble(dot((out-exact), (out-exact))*dx))
    return np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)])


def test_scalar_convergence_P1():
    testcase = ("CG", 1)
    conv = convergence_test_scalar(testcase)
    assert (conv > 1.95).all()


def test_scalar_convergence_P2():
    testcase = ("CG", 2)
    conv = convergence_test_scalar(testcase)
    assert (conv > 2.7).all()


def test_scalar_convergence_P0():
    testcase = ("DG", 0)
    conv = convergence_test_scalar(testcase)
    assert (conv > 0.9).all()


def test_scalar_convergence_P1DG():
    testcase = ("DG", 1)
    conv = convergence_test_scalar(testcase)
    assert (conv > 1.98).all()


def test_hdiv_convergence_RT1_P0():
    testcase = ("RT", 1, "DG", 0, "h")
    conv = convergence_test_hdiv(testcase)
    assert (conv > 0.9).all()


def test_hdiv_convergence_RT2_P1DG():
    testcase = ("RT", 2, "DG", 1, "h")
    conv = convergence_test_hdiv(testcase)
    assert (conv > 1.95).all()


def test_hdiv_convergence_RT3_P2DG():
    testcase = ("RT", 3, "DG", 2, "h")
    conv = convergence_test_hdiv(testcase)
    assert (conv > 2.9).all()


def test_hdiv_convergence_BDM1_P1DG():
    testcase = ("BDM", 1, "DG", 1, "h")
    conv = convergence_test_hdiv(testcase)
    assert (conv > 1.9).all()


def test_hdiv_convergence_P1DG_P1():
    testcase = ("DG", 1, "CG", 1, "v")
    conv = convergence_test_hdiv(testcase)
    assert (conv > 1.98).all()


def test_hdiv_convergence_P2DG_P2():
    testcase = ("DG", 2, "CG", 2, "v")
    conv = convergence_test_hdiv(testcase)
    assert (conv > 2.98).all()


def test_hcurl_convergence_BDM1_P1():
    testcase = ("BDM", 1, "CG", 1, "h")
    conv = convergence_test_hcurl(testcase)
    assert (conv > 1.9).all()


def test_hcurl_convergence_RT2_P1():
    testcase = ("RT", 2, "CG", 1, "h")
    conv = convergence_test_hcurl(testcase)
    assert (conv > 1.95).all()


def test_hcurl_convergence_RT3_P2():
    testcase = ("RT", 3, "CG", 2, "h")
    conv = convergence_test_hcurl(testcase)
    assert (conv > 2.95).all()


def test_hcurl_convergence_P1_P1DG():
    testcase = ("CG", 1, "DG", 1, "v")
    conv = convergence_test_hcurl(testcase)
    assert (conv > 1.95).all()


def test_hcurl_convergence_P2_P2DG():
    testcase = ("CG", 2, "DG", 2, "v")
    conv = convergence_test_hcurl(testcase)
    assert (conv > 2.7).all()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
