"""This demo program sets the top, bottom and side boundaries
of an extruded unit square. We then check against the actual solution
of the equation.
"""
from firedrake import *


def run_test_3D(size, quadrilateral, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitSquareMesh(size, size, quadrilateral=quadrilateral)
    layers = size
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / layers)

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    x, y, z = SpatialCoordinate(mesh)
    exp = x*x - y*y - z*z
    bcs = [DirichletBC(V, exp, "bottom"),
           DirichletBC(V, exp, "top"),
           DirichletBC(V, exp, 1),
           DirichletBC(V, exp, 2),
           DirichletBC(V, exp, 3),
           DirichletBC(V, exp, 4)]

    v = TestFunction(V)
    u = TrialFunction(V)
    a = inner(grad(u), grad(v)) * dx

    f = Function(V)
    f.assign(2)

    L = inner(f, v) * dx

    out = Function(V, name="computed")

    exact = Function(V, name="exact")
    exact.interpolate(exp)

    solve(a == L, out, bcs=bcs)

    res = sqrt(assemble(inner(out - exact, out - exact) * dx))

    if not test_mode:
        print("The error is ", res)
        file = VTKFile("side-bcs.pvd")
        file.write(out, exact)
    return res


def run_test_2D(intervals, parameters={}, test_mode=False):
    # Create mesh and define function space
    m = UnitIntervalMesh(intervals)
    layers = intervals
    mesh = ExtrudedMesh(m, layers, layer_height=1.0 / layers)

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    exp = x*x - 2*y*y
    bcs = [DirichletBC(V, exp, "bottom"),
           DirichletBC(V, exp, "top"),
           DirichletBC(V, exp, 1),
           DirichletBC(V, exp, 2)]

    v = TestFunction(V)
    u = TrialFunction(V)
    a = inner(grad(u), grad(v)) * dx

    f = Function(V)
    f.assign(2)

    L = inner(f, v) * dx

    out = Function(V, name="computed")

    exact = Function(V, name="exact")
    exact.interpolate(exp)

    solve(a == L, out, bcs=bcs)

    res = sqrt(assemble(inner(out - exact, out - exact) * dx))

    if not test_mode:
        print("The error is ", res)
        file = VTKFile("side-bcs.pvd")
        file.write(out, exact)
    return res


def test_extrusion_side_strong_bcs():
    assert (run_test_3D(3, quadrilateral=False, test_mode=True) < 1.e-13)


def test_extrusion_side_strong_bcs_large():
    assert (run_test_3D(6, quadrilateral=False, test_mode=True) < 1.3e-08)


def test_extrusion_side_strong_bcs_quadrilateral():
    assert (run_test_3D(3, quadrilateral=True, test_mode=True) < 1.e-13)


def test_extrusion_side_strong_bcs_quadrilateral_large():
    assert (run_test_3D(6, quadrilateral=True, test_mode=True) < 1.3e-08)


def test_extrusion_side_strong_bcs_2D():
    assert (run_test_2D(2, test_mode=True) < 1.e-13)


def test_extrusion_side_strong_bcs_2D_large():
    assert (run_test_2D(4, test_mode=True) < 1.e-12)


def test_get_all_bc_nodes():
    m = UnitSquareMesh(1, 1)
    m = ExtrudedMesh(m, layers=2)

    V = FunctionSpace(m, 'CG', 2)

    bc = DirichletBC(V, 0, 1)

    # Exterior facet nodes on a single column are:
    #  o--o--o
    #  |     |
    #  o  o  o
    #  |     |
    #  o--o--o
    #  |     |
    #  o  o  o
    #  |     |
    #  o--o--o
    #
    # And there is 1 base facet with the "1" marker.  So we expect to
    # see 15 dofs in the bc object.
    assert len(bc.nodes) == 15
