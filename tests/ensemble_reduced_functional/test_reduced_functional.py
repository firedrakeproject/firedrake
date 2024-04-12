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
    ensemble = Ensemble(COMM_WORLD, 1)
    size = COMM_WORLD.size
    mesh = UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = FunctionSpace(mesh, "R", 0)
    x = function.Function(V).assign(1)
    J = assemble(x * x * dx(domain=mesh))
    rf = EnsembleReducedFunctional(J, Control(x), ensemble)
    ensemble_J = rf(x)
    dJdm = rf.derivative()
    assert_allclose(ensemble_J, size, rtol=1e-12)
    assert_allclose(dJdm.dat.data_ro, 2.0 * size, rtol=1e-12)
    assert taylor_test(rf, x, Function(V).assign(1.0))
