import pytest
from firedrake import *
from firedrake.adjoint import *


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.mark.skipcomplex
def test_rf_external_operator():

    def interpolation_(u_, Vdst_):
        return interpolate(3 * u_ + u_ * u_, Vdst_)

    mesh = UnitIntervalMesh(16)
    (x,) = SpatialCoordinate(mesh)

    Vsrc = FunctionSpace(mesh, "CG", 1)
    Vdst = FunctionSpace(mesh, "DG", 2)

    continue_annotation()
    with set_working_tape() as tape:
        m = Function(Vsrc)
        J = assemble(interpolation_(m, Vdst))
        Jhat = ReducedFunctional(J, Control(m), tape=tape)
    pause_annotation()

    N = ReducedFunctionalOperator(Jhat)

    rsrc = Function(Vsrc).interpolate(1 + sin(4 * pi * x))
    rdst = Function(Vdst).interpolate(2 + sin(4 * pi * x))

    usrc = Function(Vsrc)
    M = interpolation_(usrc, Vdst)

    usrc.interpolate(cos(2 * pi * x))
    wdst = assemble(M)

    m.assign(usrc)
    udst = assemble(N)

    assert errornorm(wdst, udst) < 1e-14

    usrc.assign(1 + usrc)
    dMdu = derivative(M, usrc)

    m.assign(usrc)
    dNdu = derivative(N, m)

    wact = assemble(action(dMdu, rsrc))
    uact = assemble(action(dNdu, rsrc))

    assert errornorm(wact, uact) < 1e-14

    usrc.assign(2 + usrc)
    dMdu_adj = adjoint(dMdu)

    m.assign(usrc)
    dNdu_adj = adjoint(dNdu)

    wadj = assemble(action(dMdu_adj, rdst.riesz_representation())).riesz_representation()
    uadj = assemble(action(dNdu_adj, rdst.riesz_representation())).riesz_representation()

    assert errornorm(wadj, uadj) < 1e-14
