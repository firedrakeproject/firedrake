"""This demo program sets the top and bottom boundaries
of an extruded unit square to 42.
"""
from firedrake import *


def run_test(x, degree, quadrilateral, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitSquareMesh(1, 1, quadrilateral=quadrilateral)
    layers = 10
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / layers)
    V = FunctionSpace(mesh, "CG", degree)

    boundary = 42

    # Define variational problem
    u = Function(V)
    bcs = [DirichletBC(V, boundary, "bottom"),
           DirichletBC(V, boundary, "top")]

    for bc in bcs:
        bc.apply(u)

    v = TestFunction(V)

    res = abs(sum(assemble(inner(u, v) * dx).dat.data)
              - (boundary * 1.0 / layers))

    if not test_mode:
        from firedrake.output import VTKFile
        print("The error is ", res)
        file = VTKFile("bt-bcs.pvd")
        file.write(u)

    return res


def test_extrusion_rhs_bcs():
    assert (run_test(1, 1, quadrilateral=False, test_mode=True) < 1.e-13)


def test_extrusion_rhs_bcs_quadrilateral():
    assert (run_test(1, 1, quadrilateral=True, test_mode=True) < 1.e-13)
