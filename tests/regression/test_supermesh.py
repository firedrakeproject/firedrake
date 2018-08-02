from firedrake import *
from numpy.testing import assert_almost_equal


def test_two_triangles():
    parameters['pyop2_options']['debug'] = True
    mesh1 = UnitSquareMesh(3, 3)
    mesh2 = UnitSquareMesh(10, 10)
    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    dx1 = dx(domain=mesh1)
    dx2 = dx(domain=mesh2)

    # test assembly of mixed 0-forms:
    V1 = FunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)
    u1 = interpolate(x1, V1)
    u2 = interpolate(y2, V2)
    assert_almost_equal(assemble(u1*u2*dx1), assemble(x1*y1*dx1))
    assert_almost_equal(assemble(u1*u2*u2*dx1), assemble(x1*y1*y1*dx1))

    # some other function spaces
    V1 = FunctionSpace(mesh1, "CG", 2)
    V2 = FunctionSpace(mesh2, "DG", 3)
    u1 = interpolate(x1*y1, V1)
    u2 = interpolate(y2+x2**3, V2)
    assert_almost_equal(assemble(u1*u2*dx1), assemble((x1*y1)*(y1+x1**3)*dx))
    # test a straight projection
    u1 = project(u2, V1)
    v1 = project(y1+x1**3, V1)
    assert_almost_equal(u1.dat.data, v1.dat.data)

    # test a solve with mixed rhs
    # (lhs can't be mixed yet)
    # TODO: it appears the argument (tst2) currently needs to be on the same mesh as the measure (dx2)
    V1 = VectorFunctionSpace(mesh1, "DG", 1)
    V2 = VectorFunctionSpace(mesh2, "CG", 2)
    u1 = interpolate(as_vector((x1, y1)), V1)
    tst2 = TestFunction(V2)
    u2 = Function(V2)
    F = dot(tst2, u2) * dx2 - dot(tst2, x2*u1) * dx2
    solve(F == 0, u2)
    v2 = interpolate(x2*as_vector((x2, y2)), V2)
    assert_almost_equal(u2.dat.data, v2.dat.data)
