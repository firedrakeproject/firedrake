"""This demo program sets the top and bottom boundaries
of an extruded unit square to 42.
"""
import pytest
from firedrake import *


def run_test(x, degree, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitSquareMesh(1, 1)
    layers = 11
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / (layers - 1))
    V = FunctionSpace(mesh, "CG", degree)

    boundary = 42

    # Define variational problem
    u = Function(V)
    bcs = [DirichletBC(V, boundary, "bottom"),
           DirichletBC(V, boundary, "top")]

    for bc in bcs:
        bc.apply(u)

    v = TestFunction(V)

    res = abs(sum(assemble(u * v * dx).dat.data)
              - (boundary * 1.0 / (layers - 1)))

    if not test_mode:
        print "The error is ", res
        file = File("bt-bcs.pvd")
        file << u

    return res


def test_extrusion_rhs_bcs():
    assert (run_test(1, 1, test_mode=True) < 1.e-13)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
