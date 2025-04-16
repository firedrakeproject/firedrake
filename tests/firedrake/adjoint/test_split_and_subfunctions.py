import pytest

from firedrake import *
from firedrake.adjoint import *
import numpy as np


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


mesh = UnitSquareMesh(2, 2)
cg2 = FiniteElement("CG", triangle, 2)
cg1 = FiniteElement("CG", triangle, 1)
ele = MixedElement([cg2, cg1])
ZZ = FunctionSpace(mesh, ele)
V2 = FunctionSpace(mesh, cg2)


# the tests are run on functions from the MixedFunctionSpace ZZ
# and on a normal (non-mixed) FunctionSpace. Calling split() on
# a non-mixed function is trivial, but was previously broken
@pytest.fixture(params=[ZZ, V2], ids=('mixed', 'non-mixed'))
def Z(request):
    return request.param


rg = RandomGenerator()


def main(ic, fnsplit=True):
    u = Function(V2)
    w = TrialFunction(V2)
    v = TestFunction(V2)

    if fnsplit:
        ic_u = ic.sub(0)
    else:
        ic_u = split(ic)[0]

    mass = inner(w, v)*dx
    rhs = inner(ic_u, v)*dx

    solve(mass == rhs, u)

    return u


@pytest.mark.skipcomplex
def test_split(Z):
    ic = Function(Z)

    u = main(ic, fnsplit=False)
    j = assemble(u**2*dx)
    rf = ReducedFunctional(j, Control(ic))

    h0 = Function(Z).assign(1.)
    assert taylor_test(rf, ic.copy(deepcopy=True), h=h0) > 1.9


@pytest.mark.skipcomplex
def test_fn_split(Z):
    set_working_tape(Tape())
    ic = Function(Z)

    u = main(ic, fnsplit=True)
    j = assemble(u**2*dx)
    rf = ReducedFunctional(j, Control(ic))

    h = rg.uniform(Z)

    assert taylor_test(rf, ic, h) > 1.9


@pytest.mark.skipcomplex
def test_fn_split_hessian(Z):
    set_working_tape(Tape())
    ic = Function(Z)

    u = main(ic, fnsplit=True)
    j = assemble(u ** 4 * dx)
    rf = ReducedFunctional(j, Control(ic))

    h = rg.uniform(Z)
    dJdm = rf.derivative()._ad_dot(h)
    Hm = rf.hessian(h)._ad_dot(h)
    assert taylor_test(rf, ic, h, dJdm=dJdm, Hm=Hm) > 2.9


@pytest.mark.skipcomplex
def test_fn_split_no_annotate(Z):
    set_working_tape(Tape())
    ic = Function(Z)

    u = Function(V2)
    w = TrialFunction(V2)
    v = TestFunction(V2)

    ic_u = ic.sub(0)
    with stop_annotating():
        ic_uv = ic.sub(0)

    mass = inner(w, v) * dx
    rhs = inner(ic_u, v) * dx

    solve(mass == rhs, u)
    j = assemble(u ** 4 * dx + ic_uv * dx)
    rf = ReducedFunctional(j, Control(ic))

    h = rg.uniform(Z)
    r = taylor_to_dict(rf, ic, h)

    assert min(r["R0"]["Rate"]) > 0.9
    assert min(r["R1"]["Rate"]) > 1.9
    assert min(r["R2"]["Rate"]) > 2.9


@pytest.mark.skipcomplex
def test_split_subvariables_update(Z):
    z = Function(Z)
    u = z.sub(0)
    u.project(Constant(1.))
    assert np.allclose(z.sub(0).vector().dat.data, u.vector().dat.data)


@pytest.mark.skipcomplex
def test_merge_blocks():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, 'CG', 1)
    R = FunctionSpace(mesh, 'R', 0)
    W = V * V
    w = Function(W)
    w1, w2 = w.subfunctions
    w1_const = Function(R, val=0.1)
    w2_const = Function(R, val=0.2)
    w1.project(w1_const)
    w2.project(w2_const)
    J = assemble(w1*w1*dx)
    c = Control(w1_const)
    rf = ReducedFunctional(J, c)
    assert taylor_test(rf, Function(R, val=0.3), Function(R, val=0.01)) > 1.95


@pytest.mark.skipcomplex
def test_subfunctions_always_create_blocks():
    """
    Function.subfunctions use pyadjoint.FloatingType to create Blocks on the fly when
    they are accessed that record the connection between the full and sub spaces.
    However, because subfunctions is a cached property, they may be created _before_
    annotations are enabled, but used _after_ annotations are enabled. In this case,
    we still need the subfunctions to know how to create the relevant Blocks, even
    though usually we don't do any adjoint-y things unless annotations are currently
    switched on.

    This test checks that this is done correctly by creating a Function and forcing
    subfunctions to be instantiated before calling annotations are switched on. If the
    FloatingType isn't created correctly then using _both_ the full Function and the
    subfunction when recording the tape will create a disconnected tape.
    We check this isn't happening by evaluating ReducedFunctional.derivative and making
    sure that the functional is dependent on the control.
    """
    with stop_annotating():
        mesh = UnitIntervalMesh(1)
        R = FunctionSpace(mesh, "R", 0)

        u = Function(R).assign(4)
        kappa = Function(R).assign(2.0)
        control = Control(kappa)

        # force the subfunctions to be created
        _ = kappa.subfunctions

    continue_annotation()
    with set_working_tape() as tape:
        u.assign(kappa.subfunctions[0])
        J = assemble(inner(u, u)*dx)
        rf = ReducedFunctional(J, control, tape=tape)
    pause_annotation()

    rf.derivative()
    assert control.block_variable.adj_value is not None, "Functional should depend on Control"


@pytest.mark.skipcomplex
def test_writing_to_subfunctions():
    with stop_annotating():
        mesh = UnitIntervalMesh(1)
        R = FunctionSpace(mesh, "R", 0)

        kappa = Function(R).assign(2.0)
        u = Function(R)
        usub = u.subfunctions[0]

    continue_annotation()
    with set_working_tape() as tape:
        u.assign(kappa)
        usub *= 2
        J = assemble(inner(u, u) * dx)
        print(f"{type(J)=}")
        rf = ReducedFunctional(J, Control(kappa), tape=tape)
    pause_annotation()

    assert taylor_test(rf, kappa, Constant(0.1)) > 1.9
