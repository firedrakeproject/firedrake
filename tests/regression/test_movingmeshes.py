from firedrake import *
import pytest


def test_move_mesh():
    mesh = IcosahedralSphereMesh(1., 2)
    xnew = Function(mesh.coordinates)
    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    # make another mesh so that we can store the mesh position before movement
    mesh_old = Mesh(Function(mesh.coordinates))

    # mesh rotation
    theta = pi/15.
    costh = cos(theta)
    sinth = sin(theta)
    rotation_expr = as_vector([x[0], costh*x[1]-sinth*x[2], sinth*x[1]+costh*x[2]])
    xnew.interpolate(rotation_expr)
    mesh.coordinates.dat.data[:] = xnew.dat.data[:]

    V = FunctionSpace(mesh, "BDM", 1)
    u = TestFunction(V)
    v = TrialFunction(V)
    f = Function(V)
    a = inner(u, v)*dx
    L = inner(u, f)*dx(domain=mesh_old)
    prob = LinearVariationalProblem(a, L, f)
    solver = LinearVariationalSolver(prob)
    solver.solve()
