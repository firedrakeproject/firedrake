from firedrake import *
from firedrake.adjoint import *
import pytest
from numpy.testing import assert_allclose



@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()

@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    from firedrake.adjoint import annotate_tape, continue_annotation
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


@pytest.mark.parallel(nprocs=2)
def test_verification():
    ensemble = Ensemble(COMM_WORLD, 1)
    size = ensemble.ensemble_comm.size
    mesh = UnitSquareMesh(4, 4, comm=ensemble.comm)
    R = FunctionSpace(mesh, "R", 0)
    x = function.Function(R, val=1.0)
    J = assemble(x * x * dx(domain=mesh))
    rf = EnsembleReducedFunctional(J, Control(x), ensemble)
    ensemble_J = rf(x)
    dJdm = rf.derivative()
    assert_allclose(ensemble_J, size, rtol=1e-12)
    assert_allclose(dJdm.dat.data_ro, 2.0 * size, rtol=1e-12)
    assert taylor_test(rf, x, Function(R, val=0.1))


@pytest.mark.parallel(nprocs=3)
def test_minimise():
    # Optimisation test using a list of controls.
    # This test is equivalent to the one found at:
    # https://github.com/dolfin-adjoint/pyadjoint/blob/master/tests/firedrake_adjoint/test_optimisation.py#L9.
    # In this test, the functional is the result of an ensemble allreduce operation.
    ensemble = Ensemble(COMM_WORLD, 1)
    mesh = UnitSquareMesh(4, 4, comm=ensemble.comm)
    R = FunctionSpace(mesh, "R", 0)
    n = 2
    x = [Function(R) for i in range(n)]
    c = [Control(xi) for xi in x]
    # Rosenbrock function https://en.wikipedia.org/wiki/Rosenbrock_function
    # with minimum at x = (1, 1, 1, ...)
    f = 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    J = assemble(f * dx(domain=mesh))
    rf = EnsembleReducedFunctional(J, c, ensemble)
    result = minimize(rf)
    assert_allclose([float(xi) for xi in result], 1., rtol=1e-8)
