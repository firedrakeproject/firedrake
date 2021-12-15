from firedrake import *
import numpy


def test_linear_solver_change_bc():
    mesh = UnitSquareMesh(4, 4, quadrilateral=False)
    V = FunctionSpace(mesh, "P", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx

    bcval = Function(V)
    x, y = SpatialCoordinate(mesh)
    bcval.interpolate(1 + 2*y)
    bc = DirichletBC(V, bcval, "on_boundary")

    A = assemble(a, bcs=bc)
    b = Cofunction(V.dual())

    solver = LinearSolver(A)

    uh = Function(V)

    solver.solve(uh, b)

    assert numpy.allclose(uh.dat.data_ro, bc.function_arg.dat.data_ro)

    bcval.interpolate(-(1 + 2*y))

    solver.solve(uh, b)
    assert numpy.allclose(uh.dat.data_ro, bc.function_arg.dat.data_ro)
