"""This demo program sets opposite boundary sides to 10 and 42 and
then checks that the exact result has bee achieved.
"""
import pytest
from firedrake import *


def run_test(x, degree, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitSquareMesh(3, 3)
    layers = 11
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (layers - 1))

    # Define variational problem
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    u = Function(V)
    bcs = [DirichletBC(V, 10, 1),
           DirichletBC(V, 42, 2)]
    for bc in bcs:
        bc.apply(u)
    v = Function(V)
    v.interpolate(Expression("x[0] < 0.05 ? 10.0 : x[0] > 0.95 ? 42.0 : 0.0"))
    res = sqrt(assemble(dot(u - v, u - v) * dx))

    u1 = Function(V)
    bcs1 = [DirichletBC(V, 10, 3),
            DirichletBC(V, 42, 4)]
    for bc in bcs1:
        bc.apply(u1)
    v1 = Function(V)
    v1.interpolate(Expression("x[1] < 0.05 ? 10.0 : x[1] > 0.95 ? 42.0 : 0.0"))
    res1 = sqrt(assemble(dot(u1 - v1, u1 - v1) * dx))

    if not test_mode:
        print "The error is ", res1
        file = File("side-bcs-computed.pvd")
        file << u1
        file = File("side-bcs-expected.pvd")
        file << v1

    return (res, res1)


def test_extrusion_rhs_bcs():
    res1, res2 = run_test(1, 1, test_mode=True)
    assert (res1 < 1.e-13 and res2 < 1.e-13)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
