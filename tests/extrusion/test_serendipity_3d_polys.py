import pytest

from firedrake import *


@pytest.mark.parametrize(('testcase'),
                         [(5), (6)])
def test_scalar_convergence(testcase):
    mesh = UnitSquareMesh(2**2, 2**2, quadrilateral=True)
    mesh = ExtrudedMesh(mesh, 2**2)

    fspace = FunctionSpace(mesh, "S", testcase)

    u = TrialFunction(fspace)
    v = TestFunction(fspace)

    x, y, z = SpatialCoordinate(mesh)

    uex = x**testcase + y**testcase + x**2*y**3
    f = uex

    a = inner(u, v)*dx(degree=12)
    L = inner(f, v)*dx(degree=12)

    sol = Function(fspace)
    solve(a == L, sol)

    l2err = sqrt(assemble((sol-uex)*(sol-uex)*dx))
    assert l2err < 1e-6
