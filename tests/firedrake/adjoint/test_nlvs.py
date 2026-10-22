import pytest

from firedrake import *
from firedrake.adjoint import *


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


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
        #     bc_val.assign(bc_val + dt/nt)

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

    nx = 100
    nt = 5

    mesh = UnitIntervalMesh(nx)
    x, = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(mesh, "R", 0)

    dt = Function(R).assign(2/nx)
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

    PETSc.Sys.Print("record tape")
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

    from mpi4py import MPI

    # recompute component
    PETSc.Sys.Print("recompute test")
    assert abs(Jhat(m) - forward(ic2, dt2, nt, bc_arg=bc_arg2)) < 1e-14

    # tlm
    PETSc.Sys.Print("tlm test")
    Jhat(m)
    assert taylor_test(Jhat, m, h, dJdm=Jhat.tlm(h)) > 1.95

    # adjoint
    PETSc.Sys.Print("adjoint test")
    Jhat(m)
    assert taylor_test(Jhat, m, h) > 1.95

    # hessian
    PETSc.Sys.Print("hessian test")
    Jhat(m)
    taylor = taylor_to_dict(Jhat, m, h)
    from pprint import pprint
    pprint(taylor)

    assert min(taylor['R0']['Rate']) > 0.95
    assert min(taylor['R1']['Rate']) > 1.95
    assert min(taylor['R2']['Rate']) > 2.95

    return

    for _ in range(3):
        stime = MPI.Wtime()
        Jhat(m)
        etime = MPI.Wtime()
        PETSc.Sys.Print(f"Recompute time: {etime - stime:.4f}")

    for _ in range(3):
        stime = MPI.Wtime()
        Jhat.derivative()
        etime = MPI.Wtime()
        PETSc.Sys.Print(f"Derivative time: {etime - stime:.4f}")

    for _ in range(3):
        stime = MPI.Wtime()
        Jhat.tlm(h)
        etime = MPI.Wtime()
        PETSc.Sys.Print(f"TLM time: {etime - stime:.4f}")

    for _ in range(3):
        stime = MPI.Wtime()
        Jhat.hessian(h, evaluate_tlm=False)
        etime = MPI.Wtime()
        PETSc.Sys.Print(f"Hessian time: {etime - stime:.4f}")


def _checkpointed_adjoint_growth(nt):
    """Return how many Functions in V a checkpointed gradient leaves alive.

    The helper counts the live Functions in V before it tapes ``nt`` NLVS
    steps of the viscous Burgers equation on a new tape with checkpointing.
    It then computes the gradient twice and counts again. The return value
    is the second count minus the first count. The first count comes before
    the taping, so that data that the blocks allocate at taping time is
    also counted.
    """
    import gc
    from checkpoint_schedules import SingleMemoryStorageSchedule

    def count_functions(V):
        # Count the live Functions whose function space is V.
        gc.collect()
        return sum(1 for obj in gc.get_objects()
                   if isinstance(obj, Function) and obj.function_space() == V)

    with set_working_tape() as tape:
        # This schedule keeps all forward data in memory, so the adjoint
        # can be evaluated more than once.
        tape.enable_checkpointing(SingleMemoryStorageSchedule())

        mesh = UnitIntervalMesh(20)
        x, = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "CG", 1)
        ic = Function(V).interpolate(cos(2*pi*x))

        # Implicit Euler step of the viscous Burgers equation.
        dt = Constant(0.05)
        nu = Constant(0.1)
        u0 = Function(V)
        u1 = Function(V)
        v = TestFunction(V)
        F = ((u1 - u0)*v
             + dt*u1*u1.dx(0)*v
             + dt*nu*u1.dx(0)*v.dx(0))*dx
        solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F, u1))
        m = ic.copy(deepcopy=True)
        n_before = count_functions(V)

        u1.assign(ic)
        for _ in tape.timestepper(iter(range(nt))):
            u0.assign(u1)
            solver.solve()

        J = assemble(u1*u1*dx)
        Jhat = ReducedFunctional(J, Control(ic))

        # The checkpoint manager needs a functional evaluation before each
        # gradient.
        for _ in range(2):
            Jhat(m)
            Jhat.derivative()
        n_adjoint = count_functions(V)

    return n_adjoint - n_before


@pytest.mark.skipcomplex
def test_nlvs_checkpointed_adjoint_memory():
    """Check that a checkpointed gradient through NLVS keeps no per-step data.

    The number of Functions that the gradient leaves alive does not depend
    on the number of taped steps.
    """
    delta_short = _checkpointed_adjoint_growth(2)
    delta_long = _checkpointed_adjoint_growth(8)
    assert delta_long == delta_short


if __name__ == "__main__":
    control_type = "ic_control"
    bc_type = "neumann_bc"
    PETSc.Sys.Print(f"{control_type=} | {bc_type=}")
    test_nlvs_adjoint(control_type, bc_type)
