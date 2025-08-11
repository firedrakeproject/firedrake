from firedrake import *
import pytest


@pytest.mark.parallel(4)
def test_ensemble_manual_example():
    # [test_ensemble_manual_example 1 >]
    my_ensemble = Ensemble(COMM_WORLD, 2)
    # [test_ensemble_manual_example 1 <]

    # [test_ensemble_manual_example 2 >]
    mesh = UnitSquareMesh(20, 20, comm=my_ensemble.comm)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    # [test_ensemble_manual_example 2 <]

    # [test_ensemble_manual_example 3 >]
    q = Constant(my_ensemble.ensemble_comm.rank + 1)
    u.interpolate(sin(q*pi*x)*cos(q*pi*y))
    # [test_ensemble_manual_example 3 <]

    # [test_ensemble_manual_example 4 >]
    ensemble_rank = my_ensemble.ensemble_rank
    ensemble_size = my_ensemble.ensemble_size

    dest = (ensemble_rank + 1) % ensemble_size
    source = (ensemble_rank - 1) % ensemble_size
    root = 0
    usum = Function(V)

    my_ensemble.send(u, dest)
    my_ensemble.recv(u, source)

    my_ensemble.reduce(u, usum, root=root)
    my_ensemble.allreduce(u, usum)

    my_ensemble.bcast(u, root)
    # [test_ensemble_manual_example 4 <]

    # [test_ensemble_manual_example 5 >]
    V = FunctionSpace(mesh, "CG", 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = U*V

    if ensemble_rank == 0:
        local_spaces = [V, U]
    else:
        local_spaces = [V, U, W]

    efs = EnsembleFunctionSpace(local_spaces, my_ensemble)
    eds = efs.dual()
    # [test_ensemble_manual_example 5 <]

    # [test_ensemble_manual_example 6 >]
    efunc = EnsembleFunction(efs)
    ecofunc = EnsembleCofunction(eds)

    v = Function(V).assign(6)
    efunc.subfunctions[0].project(v)

    ustar = Cofunction(eds.local_spaces[1])
    efunc.subfunctions[1].assign(ustar.riesz_representation())
    # [test_ensemble_manual_example 6 <]

    # [test_ensemble_manual_example 7 >]
    u = TrialFunction(efs.local_spaces[0])
    v = TestFunction(efs.local_spaces[0])

    a = inner(u, v)*dx + inner(grad(u), grad(v))*dx
    L = ecofunc.subfunctions[0]

    prefix = f"lvs_{ensemble_rank}_0"
    lvp = LinearVariationalProblem(a, L, efunc.subfunctions[0])
    lvs = LinearVariationalSolver(lvp, options_prefix=prefix)

    ecofunc.subfunctions[0].assign(1)
    lvs.solve()
    # [test_ensemble_manual_example 7 <]

    # [test_ensemble_manual_example 8 >]
    with efunc.vec_ro() as vec:
        PETSc.Sys.Print(f"{vec.norm() = }")
    # [test_ensemble_manual_example 8 <]
