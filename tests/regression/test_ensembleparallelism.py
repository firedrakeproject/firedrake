from firedrake import *
import pytest


@pytest.mark.parallel(nprocs=6)
def test_ensemble_allreduce():
    manager = Ensemble(COMM_WORLD, 2)

    mesh = UnitSquareMesh(20, 20, comm=manager.comm)

    x, y = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    u_correct = Function(V)
    u = Function(V)
    usum = Function(V)

    u_correct.interpolate(sin(pi*x)*cos(pi*y) + sin(2*pi*x)*cos(2*pi*y) + sin(3*pi*x)*cos(3*pi*y))
    q = Constant(manager.ensemble_comm.rank + 1)
    u.interpolate(sin(q*pi*x)*cos(q*pi*y))
    manager.allreduce(u, usum)

    assert assemble((u_correct - usum)**2*dx) < 1e-4


@pytest.mark.parallel(nprocs=6)
def test_ensemble_solvers():
    # this test uses linearity of the equation to solve two problems
    # with different RHS on different subcommunicators,
    # and compare the reduction with a problem solved with the sum
    # of the two RHS
    manager = Ensemble(COMM_WORLD, 2)

    mesh = UnitSquareMesh(20, 20, comm=manager.comm)

    x, y = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    f_combined = Function(V)
    f_separate = Function(V)

    f_combined.interpolate(sin(pi*x)*cos(pi*y) + sin(2*pi*x)*cos(2*pi*y) + sin(3*pi*x)*cos(3*pi*y))
    q = Constant(manager.ensemble_comm.rank + 1)
    f_separate.interpolate(sin(q*pi*x)*cos(q*pi*y))

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (u*v + inner(grad(u), grad(v)))*dx
    Lcombined = v*f_combined*dx
    Lseparate = v*f_separate*dx

    u_combined = Function(V)
    u_separate = Function(V)
    usum = Function(V)

    params = {'ksp_type': 'preonly',
              'pc_type': 'redundant',
              "redundant_pc_type": "lu",
              "redundant_pc_factor_mat_solver_type": "mumps",
              "redundant_mat_mumps_icntl_14": 200}

    combinedProblem = LinearVariationalProblem(a, Lcombined, u_combined)
    combinedSolver = LinearVariationalSolver(combinedProblem,
                                             solver_parameters=params)

    separateProblem = LinearVariationalProblem(a, Lseparate, u_separate)
    separateSolver = LinearVariationalSolver(separateProblem,
                                             solver_parameters=params)

    combinedSolver.solve()
    separateSolver.solve()
    manager.allreduce(u_separate, usum)

    assert assemble((u_combined - usum)**2*dx) < 1e-4


def test_comm_manager():
    with pytest.raises(ValueError):
        Ensemble(COMM_WORLD, 2)


@pytest.mark.parallel(nprocs=3)
def test_comm_manager_parallel():
    with pytest.raises(ValueError):
        Ensemble(COMM_WORLD, 2)


@pytest.mark.parallel(nprocs=2)
def test_comm_manager_allreduce():
    manager = Ensemble(COMM_WORLD, 1)

    mesh = UnitSquareMesh(1, 1, comm=manager.global_comm)

    mesh2 = UnitSquareMesh(2, 2, comm=manager.ensemble_comm)

    V = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh2, "CG", 1)

    f = Function(V)
    f2 = Function(V2)

    with pytest.raises(ValueError):
        manager.allreduce(f, f2)

    f3 = Function(V2)

    with pytest.raises(ValueError):
        manager.allreduce(f3, f2)

    V3 = FunctionSpace(mesh, "DG", 0)
    g = Function(V3)
    with pytest.raises(ValueError):
        manager.allreduce(f, g)
