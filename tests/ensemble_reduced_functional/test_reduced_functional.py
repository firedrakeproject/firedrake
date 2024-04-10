from firedrake import *
from firedrake.adjoint import *
import pytest
from numpy.testing import assert_allclose


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
def test_ensemble_functional():
    my_ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
    mesh = UnitSquareMesh(1, 1)
    R = FunctionSpace(mesh, "R", 0)
    x = [Function(R) for i in range(2)]
    c = [Control(xi) for xi in x]

    # Rosenbrock function https://en.wikipedia.org/wiki/Rosenbrock_function
    # with minimum at x = (1, 1, 1, ...)
    f = 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

    J = assemble(f * dx(domain=mesh))
    rf = EnsembleReducedFunctional(J, c, my_ensemble)
    taylor_test(rf, x, Function(R, val=1.0))
    result = minimize(rf)
    print([float(xi) for xi in result])
    assert_allclose([float(xi) for xi in result], 1., rtol=1e-4)