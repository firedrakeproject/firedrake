import pytest

from firedrake import *
from firedrake.adjoint import *
import numpy as np


@pytest.fixture(autouse=True)
def test_taping(set_test_tape):
    pass


@pytest.fixture(autouse=True, scope="module")
def module_annotation(set_module_annotation):
    pass


@pytest.mark.skipcomplex
def test_dynamic_meshes_2D():
    mesh = UnitSquareMesh(10, 10)
    S = mesh.coordinates.function_space()
    s = [Function(S), Function(S), Function(S)]
    mesh.coordinates.assign(mesh.coordinates + s[0])

    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u0 = project(cos(pi*x[0])*sin(pi*x[1]), V)

    mesh.coordinates.assign(mesh.coordinates + s[1])

    u, v = TrialFunction(V), TestFunction(V)
    f = cos(x[0]) + x[1] * sin(2 * pi * x[1])

    u, v = TrialFunction(V), TestFunction(V)
    dt = Constant(0.1)
    k = Constant(1/dt)
    F = k*inner(u-u0, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u1 = Function(V)
    solve(lhs(F) == rhs(F), u1)
    J = float(dt)*assemble(u1**2*dx)

    mesh.coordinates.assign(mesh.coordinates + s[2])
    F = k*inner(u-u1, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u2 = Function(V)
    solve(lhs(F) == rhs(F), u2)
    J += float(dt)*assemble(u2**2*dx)

    ctrls = [Control(c) for c in s]
    Jhat = ReducedFunctional(J, ctrls)

    from pyadjoint.tape import stop_annotating
    from pyadjoint.verification import taylor_to_dict
    with stop_annotating():
        A, B, C = 2, 1, 3
        taylor = [project(A*as_vector((cos(2*pi*x[1]), x[0])), S),
                  project(B*as_vector((cos(x[0]), cos(x[1]))), S),
                  project(C*as_vector((-x[0]**2, x[1])), S)]
        zero = [Function(S), Function(S), Function(S)]
        results = taylor_to_dict(Jhat, zero, taylor)
    print(results)
    assert (np.mean(results["R0"]["Rate"]) > 0.9)
    assert (np.mean(results["R1"]["Rate"]) > 1.9)
    assert (np.mean(results["R2"]["Rate"]) > 2.9)


@pytest.mark.skipcomplex
@pytest.mark.parametrize("mesh_type", ["UnitCubeMesh",
                                       "UnitOctahedralSphereMesh",
                                       "UnitIcosahedralSphereMesh",
                                       "UnitCubedSphereMesh",
                                       "TorusMesh",
                                       "CylinderMesh"])
def test_dynamic_meshes_3D(mesh_type):
    if mesh_type == "UnitCubeMesh":
        mesh = UnitCubeMesh(4, 4, 5)
    if mesh_type == "UnitOctahedralSphereMesh":
        mesh = UnitOctahedralSphereMesh(3)
    if mesh_type == "UnitIcosahedralSphereMesh":
        mesh = UnitIcosahedralSphereMesh(3)
    if mesh_type == "UnitCubedSphereMesh":
        mesh = UnitCubedSphereMesh(3)
    if mesh_type == "TorusMesh":
        mesh = TorusMesh(25, 10, 1, 0.5)
    if mesh_type == "CylinderMesh":
        mesh = CylinderMesh(10, 25, radius=0.5, depth=0.8)

    S = mesh.coordinates.function_space()
    s = [Function(S), Function(S), Function(S)]
    mesh.coordinates.assign(mesh.coordinates + s[0])

    x = SpatialCoordinate(mesh)
    if mesh.cell_dimension() != mesh.geometric_dimension:
        mesh.init_cell_orientations(x)

    V = FunctionSpace(mesh, "CG", 1)
    u0 = project(cos(pi*x[0])*sin(pi*x[1])*x[2]**2, V)

    mesh.coordinates.assign(mesh.coordinates + s[1])

    u, v = TrialFunction(V), TestFunction(V)
    f = x[2]*cos(x[0]) + x[1] * sin(2 * pi * x[1])

    u, v = TrialFunction(V), TestFunction(V)
    dt = Constant(0.1)
    k = Constant(1/dt)
    F = k*inner(u-u0, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u1 = Function(V)
    solve(lhs(F) == rhs(F), u1)
    J = float(dt)*assemble(u1**2*dx)

    mesh.coordinates.assign(mesh.coordinates + s[2])

    F = k*inner(u-u1, v)*dx + inner(grad(u), grad(v))*dx - f*v*dx
    u2 = Function(V)
    solve(lhs(F) == rhs(F), u2)
    J += float(dt)*assemble(u2**2*dx)

    ctrls = [Control(c) for c in s]
    Jhat = ReducedFunctional(J, ctrls)

    from pyadjoint.tape import stop_annotating
    from pyadjoint.verification import taylor_to_dict
    with stop_annotating():
        A = 0.1
        B = 2
        C = 1.6
        taylor = [project(A*as_vector((
            sin(2*pi*x[2]), cos(2*pi*x[1]), cos(2*pi*x[0]*x[1]))), S),
            project(B*as_vector((1, 0.2, 3*cos(x[1]))), S),
            project(C*as_vector((cos(-x[0]**2), cos(x[2]), x[1])), S)]
        zero = [Function(S), Function(S), Function(S)]
        results = taylor_to_dict(Jhat, zero, taylor)
    print(results)
    assert (np.mean(results["R0"]["Rate"]) > 0.9)
    assert (np.mean(results["R1"]["Rate"]) > 1.9)
    assert (np.mean(results["R2"]["Rate"]) > 2.9)
