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
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    if annotate_tape():
        pause_annotation()


@pytest.mark.parallel(nprocs=4)
@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_verification():
    ensemble = Ensemble(COMM_WORLD, 2)
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
    assert taylor_test(rf, x, Function(R, val=0.1)) > 1.9


@pytest.mark.parallel(nprocs=4)
@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.xfail(reason="Taylor's test fails because the inner product \
                   between the perturbation and gradient is not allreduced \
                   for `scatter_control=False`.")
def test_verification_gather_functional_adjfloat():
    ensemble = Ensemble(COMM_WORLD, 2)
    rank = ensemble.ensemble_comm.rank
    mesh = UnitSquareMesh(4, 4, comm=ensemble.comm)
    R = FunctionSpace(mesh, "R", 0)
    x = function.Function(R, val=rank+1)
    J = assemble(x * x * dx(domain=mesh))
    a = AdjFloat(1.0)
    b = AdjFloat(1.0)
    Jg_m = [Control(a), Control(b)]
    Jg = ReducedFunctional(a**2 + b**2, Jg_m)
    rf = EnsembleReducedFunctional(J, Control(x), ensemble,
                                   scatter_control=False,
                                   gather_functional=Jg)
    ensemble_J = rf(x)
    dJdm = rf.derivative()
    assert_allclose(ensemble_J, 1.0**4+2.0**4, rtol=1e-12)
    assert_allclose(dJdm.dat.data_ro, 4*(rank+1)**3, rtol=1e-12)
    assert taylor_test(rf, x, Function(R, val=0.1)) > 1.9


@pytest.mark.parallel(nprocs=4)
@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.xfail(reason="Taylor's test fails because the inner product \
                   between the perturbation and gradient is not allreduced \
                   for `scatter_control=False`.")
def test_verification_gather_functional_Function():
    ensemble = Ensemble(COMM_WORLD, 2)
    rank = ensemble.ensemble_comm.rank
    mesh = UnitSquareMesh(4, 4, comm=ensemble.comm)
    R = FunctionSpace(mesh, "R", 0)
    x = function.Function(R, val=rank+1)
    J = Function(R).assign(x**2)
    a = Function(R).assign(1.0)
    b = Function(R).assign(1.0)
    Jg_m = [Control(a), Control(b)]
    Jg = assemble((a**2 + b**2)*dx)
    Jghat = ReducedFunctional(Jg, Jg_m)
    rf = EnsembleReducedFunctional(J, Control(x), ensemble,
                                   scatter_control=False,
                                   gather_functional=Jghat)
    ensemble_J = rf(x)
    dJdm = rf.derivative()
    assert_allclose(ensemble_J, 1.0**4+2.0**4, rtol=1e-12)
    assert_allclose(dJdm.dat.data_ro, 4*(rank+1)**3, rtol=1e-12)
    assert taylor_test(rf, x, Function(R, val=0.1)) > 1.9


@pytest.mark.parallel(nprocs=6)
@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_minimise():
    # Optimisation test using a list of controls.
    # This test is equivalent to the one found at:
    # https://github.com/dolfin-adjoint/pyadjoint/blob/master/tests/firedrake_adjoint/test_optimisation.py#L9.
    # In this test, the functional is the result of an ensemble allreduce operation.
    ensemble = Ensemble(COMM_WORLD, 2)
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
