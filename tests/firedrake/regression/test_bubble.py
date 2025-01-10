"""Test bubble function space"""

from firedrake import *


def test_simple():
    mesh = UnitTriangleMesh()
    V = FunctionSpace(mesh, "B", 3)
    x = SpatialCoordinate(mesh)
    u = project(27*x[0]*x[1]*(1-x[0]-x[1]), V)
    assert (u.dat.data[0] - 1.0) < 1e-14


def test_enrichment():
    mesh = UnitTriangleMesh()
    x = SpatialCoordinate(mesh)
    P2 = FiniteElement("CG", "triangle", 2)
    Bubble = FiniteElement("Bubble", "triangle", 3)
    P2B3 = P2 + Bubble
    V = FunctionSpace(mesh, P2B3)
    W = FunctionSpace(mesh, "CG", 3)
    u = project(27*x[0]*x[1]*(1-x[0]-x[1]), V)
    exact = Function(W)
    exact.interpolate(27*x[0]*x[1]*(1-x[0]-x[1]))
    # make sure that these are the same
    assert sqrt(assemble((u-exact)*(u-exact)*dx)) < 1e-14


def test_BDFM():
    mesh = UnitTriangleMesh()
    x = SpatialCoordinate(mesh)
    P2 = FiniteElement("CG", "triangle", 2)
    Bubble = FiniteElement("Bubble", "triangle", 3)
    P2B3 = P2 + Bubble
    V0 = FunctionSpace(mesh, P2B3)
    V1 = FunctionSpace(mesh, "BDFM", 2)
    u = project(27*x[0]*x[1]*(1-x[0]-x[1]), V0)

    v = TrialFunction(V1)
    w = TestFunction(V1)
    out = Function(V1)
    solve(inner(v, w)*dx == inner(curl(u), w)*dx, out)
    # testing against known result where the interior DOFS of BDFM are excited
    a = out.dat.data
    a.sort()
    assert (abs(a[1:7]) < 1e-12).all()
    assert abs(a[0] + 6.75) < 1e-12
    assert abs(a[7] - 6.75) < 1e-12
    assert abs(a[8] - 13.5) < 1e-12
