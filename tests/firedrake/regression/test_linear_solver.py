from firedrake import *
from firedrake.petsc import PETSc
import numpy


def test_linear_solver_update_after_error():
    mesh = UnitIntervalMesh(10)
    space = FunctionSpace(mesh, "CG", 1)
    test = TestFunction(space)
    trial = TrialFunction(space)

    solver = LinearSolver(assemble(inner(trial, test) * dx),
                          solver_parameters={"ksp_type": "cg",
                                             "pc_type": "none",
                                             "ksp_max_it": 1,
                                             "ksp_atol": 1.0e-2})
    b = assemble(inner(Constant(1), test) * dx)

    u = Function(space, name="u")
    u.assign(-1)
    uinit = Function(u, name="uinit")
    try:
        solver.solve(u, b)
    except firedrake.exceptions.ConvergenceError:
        assert solver.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT

    assert not numpy.allclose(u.dat.data_ro, uinit.dat.data_ro)


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
