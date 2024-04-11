"""This demo program sets opposite boundary sides to 10 and 42 and
then checks that the exact result has bee achieved.
"""
from firedrake import *


def run_test(x, degree, quadrilateral, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitSquareMesh(3, 3, quadrilateral=quadrilateral)
    layers = 10
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / layers)

    # Define variational problem
    V = FunctionSpace(mesh, "CG", degree)

    # Define variational problem
    u = Function(V)
    bcs = [DirichletBC(V, 10, 1),
           DirichletBC(V, 42, 2)]
    for bc in bcs:
        bc.apply(u)

    v = Function(V)
    xs = SpatialCoordinate(mesh)
    v.interpolate(conditional(real(xs[0]) < 0.05,
                              10,
                              conditional(real(xs[0]) > 0.95, 42.0, 0.0)))
    res = abs(sqrt(assemble(inner(u - v, u - v) * dx)))

    u1 = Function(V, name="computed")
    bcs1 = [DirichletBC(V, 10, 3),
            DirichletBC(V, 42, 4)]
    for bc in bcs1:
        bc.apply(u1)
    v1 = Function(V, name="expected")
    v1.interpolate(conditional(real(xs[1]) < 0.05,
                               10.0,
                               conditional(real(xs[1]) > 0.95, 42.0, 0.0)))
    res1 = abs(sqrt(assemble(inner(u1 - v1, u1 - v1) * dx)))

    if not test_mode:
        from firedrake.output import VTKFile
        print("The error is ", res1)
        file = VTKFile("side-bcs.pvd")
        file.write(u1, v1)

    return (res, res1)


def test_extrusion_rhs_bcs():
    res1, res2 = run_test(1, 1, quadrilateral=False, test_mode=True)
    assert (res1 < 1.e-13 and res2 < 1.e-13)


def test_extrusion_rhs_bcs_quadrilateral():
    res1, res2 = run_test(1, 1, quadrilateral=True, test_mode=True)
    assert (res1 < 1.e-13 and res2 < 1.e-13)
