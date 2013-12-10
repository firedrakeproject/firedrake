"""This demo program sets the top, bottom and side boundaries
of an extruded unit square. We then check against the actual solution
of the equation.
"""
import pytest
from firedrake import *


def run_test(x, degree, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitSquareMesh(2, 2)
    layers = 11
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (layers - 1))

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    bcs = [DirichletBC(V, Expression("x[0]*x[0] - x[1]*x[1]"), "bottom"),
           DirichletBC(V, Expression("x[0]*x[0] - x[1]*x[1] - 1"), "top"),
           DirichletBC(V, Expression("-x[1]*x[1] - x[2]*x[2]"), 1),
           DirichletBC(V, Expression("1 - x[1]*x[1] - x[2]*x[2]"), 2),
           DirichletBC(V, Expression("x[0]*x[0] - x[2]*x[2]"), 3),
           DirichletBC(V, Expression("x[0]*x[0] - 1 - x[2]*x[2]"), 4)]

    v = TestFunction(V)
    u = TrialFunction(V)
    a = dot(grad(u), grad(v)) * dx

    f = Function(V)
    f.assign(2)

    L = v * f * dx

    out = Function(V)

    exact = Function(V)
    exact.interpolate(Expression('x[0]*x[0] - x[1]*x[1] - x[2]*x[2]'))

    solve(a == L, out, bcs=bcs)

    res = sqrt(assemble(dot(out - exact, out - exact) * dx))

    if not test_mode:
        print "The error is ", res
        file = File("side-bcs-computed.pvd")
        file << out
        file = File("side-bcs-expected.pvd")
        file << exact
    print res
    return res


@pytest.mark.xfail
def test_extrusion_side_strong_bcs():
    assert (run_test(1, 1, test_mode=True) < 1.e-13)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
