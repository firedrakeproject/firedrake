import pytest
from firedrake import *
from firedrake.adjoint import *


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.mark.skipcomplex
@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("m", (0, 2, 4))
@pytest.mark.parametrize("family", ("CG", "DG"))
def test_covariance_adjoint_norm(m, family):
    """Test that covariance operators are properly taped.
    """
    nx = 20
    L = 0.2
    sigma = 0.1

    mesh = UnitIntervalMesh(nx)
    x, = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, family, 1)

    u = Function(V).project(sin(2*pi*x))
    v = Function(V).project(2 - 0.5*sin(6*pi*x))

    form = 'IP' if family == 'DG' else 'CG'
    B = AutoregressiveCovariance(V, L, sigma, m, form=form)

    continue_annotation()
    with set_working_tape() as tape:
        w = Function(V).project(u**4 + v)
        J = B.norm(w)
        Jhat = ReducedFunctional(J, Control(u), tape=tape)
    pause_annotation()

    m = Function(V).project(sin(2*pi*(x+0.2)))
    h = Function(V).project(sin(4*pi*(x-0.2)))

    taylor = taylor_to_dict(Jhat, m, h)

    assert min(taylor['R0']['Rate']) > 0.95, taylor['R0']
    assert min(taylor['R1']['Rate']) > 1.95, taylor['R1']
    assert min(taylor['R2']['Rate']) > 2.95, taylor['R2']
