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


def forward(ic, dt, nt):
    """Burgers equation solver."""
    V = ic.function_space()

    if isinstance(dt, Constant):
        nu = Constant(0.1)
    else:
        nu = Function(dt.function_space()).assign(0.1)

    u0 = Function(V)
    u1 = Function(V)
    v = TestFunction(V)

    F = ((u1 - u0)*v
         + dt*u1*u1.dx(0)*v
         + dt*nu*u1.dx(0)*v.dx(0))*dx

    problem = NonlinearVariationalProblem(F, u1)
    solver = NonlinearVariationalSolver(problem)

    u1.assign(ic)

    for i in range(nt):
        u0.assign(u1)
        solver.solve()
        if not isinstance(nu, Constant):
            nu += dt

    J = assemble(u1*u1*dx)
    return J


@pytest.mark.skipcomplex
@pytest.mark.parametrize("control_type", ["ic_control", "dt_control"])
def test_nlvs_adjoint(control_type):
    mesh = UnitIntervalMesh(8)
    x, = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    nt = 4
    dt = Function(R).assign(0.1)
    # dt = Constant(0.1)
    ic = Function(V).interpolate(cos(2*pi*x))

    if control_type == 'ic_control':
        control = ic
    elif control_type == 'dt_control':
        control = dt
    else:
        raise ValueError

    continue_annotation()
    with set_working_tape() as tape:
        J = forward(ic, dt, nt)
        Jhat = ReducedFunctional(J, Control(control), tape=tape)
    pause_annotation()

    if control_type == 'ic_control':
        m = Function(V).assign(0.5*ic)
        h = Function(V).interpolate(-0.5*cos(4*pi*x))

        # recompute component
        assert abs(Jhat(m) - forward(m, dt, nt)) < 1e-14

        # tlm
        assert taylor_test(Jhat, m, h, dJdm=Jhat.tlm(h)) > 1.95

    elif control_type == 'dt_control':
        m = Function(R).assign(0.05)
        h = Function(R).assign(0.01)

        # recompute component
        assert abs(Jhat(m) - forward(ic, m, nt)) < 1e-14

        # tlm
        assert taylor_test(Jhat, m, h, dJdm=Jhat.tlm(h)) > 1.95


if __name__ == "__main__":
    ctype = "ic"
    print(f"Control type: {ctype}")
    test_nlvs_adjoint(f"{ctype}_control")
