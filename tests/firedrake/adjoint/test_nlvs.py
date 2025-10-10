import pytest

from firedrake import *
from firedrake.adjoint import *


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


def forward(ic, dt, nt, bc_arg=None):
    """Burgers equation solver."""
    V = ic.function_space()

    if bc_arg:
        bc_val = bc_arg.copy(deepcopy=True)
        bcs = [DirichletBC(V, bc_val, 1),
               DirichletBC(V, 0, 2)]
    else:
        bcs = None

    nu = Function(dt.function_space()).assign(0.1)

    u0 = Function(V)
    u1 = Function(V)
    v = TestFunction(V)

    F = ((u1 - u0)*v
         + dt*u1*u1.dx(0)*v
         + dt*nu*u1.dx(0)*v.dx(0))*dx

    problem = NonlinearVariationalProblem(F, u1, bcs=bcs)
    solver = NonlinearVariationalSolver(problem)

    u1.assign(ic)

    for i in range(nt):
        u0.assign(u1)
        solver.solve()
        nu += dt
        # if bc_arg:
        #     bc_val.assign(bc_val + dt)

    J = assemble(u1*u1*dx)
    return J


@pytest.mark.skipcomplex
@pytest.mark.parametrize("control_type", ["ic_control",
                                          "dt_control",
                                          "bc_control"])
@pytest.mark.parametrize("bc_type", ["neumann_bc",
                                     "dirichlet_bc"])
def test_nlvs_adjoint(control_type, bc_type):
    if control_type == 'bc_control' and bc_type == 'neumann_bc':
        pytest.skip("Cannot use Neumann BCs as control")

    mesh = UnitIntervalMesh(6)
    x, = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    nt = 2
    dt = Function(R).assign(0.1)
    ic = Function(V).interpolate(cos(2*pi*x))

    dt0 = dt.copy(deepcopy=True)
    ic0 = ic.copy(deepcopy=True)

    if bc_type == 'neumann_bc':
        bc_arg = None
        bc_arg0 = None
    elif bc_type == 'dirichlet_bc':
        bc_arg = Function(R).assign(1.)
        bc_arg0 = bc_arg.copy(deepcopy=True)
    else:
        raise ValueError(f"Unrecognised {bc_type=}")

    if control_type == 'ic_control':
        control = ic0
    elif control_type == 'dt_control':
        control = dt0
    elif control_type == 'bc_control':
        control = bc_arg0
    else:
        raise ValueError(f"Unrecognised {control_type=}")

    print("record tape")
    continue_annotation()
    with set_working_tape() as tape:
        J = forward(ic0, dt0, nt, bc_arg=bc_arg0)
        Jhat = ReducedFunctional(J, Control(control), tape=tape)
    pause_annotation()

    if control_type == 'ic_control':
        m = Function(V).assign(0.5*ic)
        h = Function(V).interpolate(-0.5*cos(4*pi*x))

        ic2 = m.copy(deepcopy=True)
        dt2 = dt
        bc_arg2 = bc_arg

    elif control_type == 'dt_control':
        m = Function(R).assign(0.05)
        h = Function(R).assign(0.01)

        ic2 = ic
        dt2 = m.copy(deepcopy=True)
        bc_arg2 = bc_arg

    elif control_type == 'bc_control':
        m = Function(R).assign(0.5)
        h = Function(R).assign(-0.1)

        ic2 = ic
        dt2 = dt
        bc_arg2 = m.copy(deepcopy=True)

    # recompute component
    print("recompute test")
    assert abs(Jhat(m) - forward(ic2, dt2, nt, bc_arg=bc_arg2)) < 1e-14

    # tlm
    print("tlm test")
    Jhat(m)
    assert taylor_test(Jhat, m, h, dJdm=Jhat.tlm(h)) > 1.95

    # adjoint
    print("adjoint test")
    Jhat(m)
    assert taylor_test(Jhat, m, h) > 1.95

    # hessian
    print("hessian test")
    Jhat(m)
    taylor = taylor_to_dict(Jhat, m, h)
    from pprint import pprint
    pprint(taylor)

    assert min(taylor['R2']['Rate']) > 2.95


if __name__ == "__main__":
    control_type = "ic_control"
    bc_type = "neumann_bc"
    print(f"{control_type=} | {bc_type=}")
    test_nlvs_adjoint(control_type, bc_type)
