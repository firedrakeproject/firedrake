import pytest

from numpy.random import rand
from numpy.testing import assert_allclose

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


@pytest.mark.skipcomplex
def test_assemble_0_forms():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V)
    u.assign(4.0)
    a1 = assemble(u * dx)
    a2 = assemble(u**2 * dx)
    a3 = assemble(u**3 * dx)

    # this previously failed when in Firedrake "vectorial" adjoint values
    # where stored as a Function instead of Vector()
    s = a1 + a2 + 2.0 * a3
    rf = ReducedFunctional(s, Control(u))
    dJdm = rf.derivative()
    assert_allclose(dJdm.dat.data_ro, 1. + 2. * 4. + 6. * 16.)


@pytest.mark.skipcomplex
def test_assemble_0_forms_mixed():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = Function(V * V)
    u.assign(7.0)
    a1 = assemble(u[0] * dx)
    a2 = assemble(u[0]**2 * dx)
    a3 = assemble(u[0]**3 * dx)

    # this previously failed when in Firedrake "vectorial" adjoint values
    # where stored as a Function instead of Vector()
    s = a1 + 2. * a2 + a3
    s -= a3  # this is done deliberately to end up with an adj_input of 0.0 for the a3 AssembleBlock
    rf = ReducedFunctional(s, Control(u))
    # derivative is: (1+4*u)*dx - summing is equivalent to testing with 1
    dJdm = rf.derivative()
    assert_allclose(dJdm.dat.data_ro[0], 1. + 4. * 7)
    assert_allclose(dJdm.dat.data_ro[1], 0.0)


@pytest.mark.skipcomplex
def test_assemble_1_forms_adjoint():
    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    v = TestFunction(V)
    x, = SpatialCoordinate(mesh)
    f = Function(V).interpolate(cos(x))

    def J(f):
        w1 = assemble(inner(f, v) * dx)
        w2 = assemble(inner(f**2, v) * dx)
        w3 = assemble(inner(f**3, v) * dx)
        inner_dual = lambda x: assemble(action(x, x.riesz_representation()))
        return sum(inner_dual(c) for c in (w1, w2, w3))

    _test_adjoint(J, f)


@pytest.mark.skipcomplex
def test_assemble_1_forms_tlm():
    tape = Tape()
    set_working_tape(tape)

    mesh = IntervalMesh(10, 0, 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    v = TestFunction(V)
    f = Function(V).assign(1)

    w1 = assemble(inner(f, v) * dx)
    w2 = assemble(inner(f**2, v) * dx)
    w3 = assemble(inner(f**3, v) * dx)
    inner_dual = lambda x: assemble(action(x, x.riesz_representation()))
    J = sum(inner_dual(c) for c in (w1, w2, w3))

    Jhat = ReducedFunctional(J, Control(f))
    h = Function(V)
    h.vector()[:] = rand(h.dof_dset.size)
    g = f.copy(deepcopy=True)
    f.block_variable.tlm_value = h
    tape.evaluate_tlm()
    assert (taylor_test(Jhat, g, h, dJdm=J.block_variable.tlm_value) > 1.9)


@pytest.mark.skipcomplex
def _test_adjoint(J, f):
    import numpy.random
    tape = Tape()
    set_working_tape(tape)

    V = f.function_space()
    h = Function(V)
    h.vector()[:] = numpy.random.rand(V.dim())

    eps_ = [0.01 / 2.0**i for i in range(5)]
    residuals = []
    for eps in eps_:

        Jp = J(f + eps * h)
        tape.clear_tape()
        Jm = J(f)
        Jm.block_variable.adj_value = 1.0
        tape.evaluate_adj()

        dJdf = f.block_variable.adj_value.vector()

        residual = abs(Jp - Jm - eps * dJdf.inner(h.vector()))
        residuals.append(residual)

    r = convergence_rates(residuals, eps_)
    print(r)
    print(residuals)

    tol = 1E-1
    assert (r[-1] > 2 - tol)


@pytest.mark.skipcomplex
def convergence_rates(E_values, eps_values):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i] / E_values[i - 1]) / log(eps_values[i] / eps_values[i - 1]))

    return r
