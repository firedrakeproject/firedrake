"""Tests for successful identity solve on extruded meshes"""
import numpy as np
import pytest

from firedrake import *


def identity_test_scalar(family, degree, vfamily, vdegree):
    m = UnitSquareMesh(4, 4)
    layers = 3
    mesh = ExtrudedMesh(m, layers, layer_height=0.5)

    fspace = FunctionSpace(mesh, family, degree, vfamily=vfamily, vdegree=vdegree)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    #f = Function(fspace)
    f = project(Expression("x[2]-x[0]"), fspace)

    out = Function(fspace)
    solve(u*v*dx == f*v*dx, out)
    return np.max(np.abs(out.dat.data - f.dat.data))


def identity_test_hdiv(family, degree, vfamily, vdegree):
    m = UnitSquareMesh(4, 4)
    layers = 3
    mesh = ExtrudedMesh(m, layers, layer_height=0.5)

    horiz_elt = FiniteElement(family, "triangle", degree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HDiv(OuterProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    f = project(Expression(("x[1]", "-x[0]", "x[2]")), fspace)

    out = Function(fspace)
    solve(dot(u, v)*dx == dot(f, v)*dx, out)
    return np.max(np.abs(out.dat.data - f.dat.data))


def identity_test_hcurl(family, degree, vfamily, vdegree):
    m = UnitSquareMesh(4, 4)
    layers = 3
    mesh = ExtrudedMesh(m, layers, layer_height=0.5)

    horiz_elt = FiniteElement(family, "triangle", degree)
    vert_elt = FiniteElement(vfamily, "interval", vdegree)
    product_elt = HCurl(OuterProductElement(horiz_elt, vert_elt))
    fspace = FunctionSpace(mesh, product_elt)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    f = project(Expression(("x[1]", "-x[0]", "x[2]")), fspace)

    out = Function(fspace)
    solve(dot(u, v)*dx == dot(f, v)*dx, out)
    return np.max(np.abs(out.dat.data - f.dat.data))


def test_scalar_assembly():
    testcases = [("CG", 1), ("CG", 2), ("DG", 0), ("DG", 1)]
    error = [identity_test_scalar(f, d, vfam, fdeg) for (vfam, fdeg) in testcases for (f, d) in testcases]
    assert (np.array(error) < 1.0e-7).all()


def test_hdiv_assembly():
    CG = [("CG", 1), ("CG", 2)]
    DG = [("DG", 0), ("DG", 1)]
    hdiv = [("RT", 1), ("RT", 2), ("RT", 3), ("BDM", 1)]
    # two valid combinations for hdiv
    # 1) BDM/RT x DG
    # 2) DG x CG
    error = [identity_test_hdiv(f, d, vfam, fdeg) for (vfam, fdeg) in DG for (f, d) in hdiv]
    error += [identity_test_hdiv(f, d, vfam, fdeg) for (vfam, fdeg) in CG for (f, d) in DG]
    assert (np.array(error) < 1.0e-7).all()


def test_hcurl_assembly():
    CG = [("CG", 1), ("CG", 2)]
    DG = [("DG", 0), ("DG", 1)]
    hdiv = [("RT", 1), ("RT", 2), ("RT", 3), ("BDM", 1)]
    # two valid combinations for hcurl
    # 1) BDM/RT x CG
    # 2) CG x DG
    error = [identity_test_hcurl(f, d, vfam, fdeg) for (vfam, fdeg) in CG for (f, d) in hdiv]
    error += [identity_test_hcurl(f, d, vfam, fdeg) for (vfam, fdeg) in DG for (f, d) in CG]
    assert (np.array(error) < 1.0e-7).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
