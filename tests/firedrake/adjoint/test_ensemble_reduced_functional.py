from firedrake import *
from firedrake.adjoint import *
from firedrake.adjoint.ensemble_adjvec import EnsembleAdjVec
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
import pytest
from numpy.testing import assert_allclose
from numpy import mean
from pytest_mpi.parallel_assert import parallel_assert


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


@pytest.mark.parallel(nprocs=[1, 2, 3, 6])
@pytest.mark.skipcomplex
def test_ensemble_bcast_float():
    ensemble = Ensemble(COMM_WORLD, 1)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_floats = 6
    nlocal_floats = nglobal_floats // size

    c = AdjFloat(0.0)
    J = EnsembleAdjVec(
        [AdjFloat(0.0) for _ in range(nlocal_floats)], ensemble)

    Jhat = EnsembleBcastReducedFunctional(J, Control(c), ensemble=ensemble)

    # check the control is broadcast to all ranks
    eps = 1e-12

    x = AdjFloat(3.0)
    Jx = Jhat(x)

    expect = x
    match_local = all((Ji - expect) < eps for Ji in Jx.subvec)

    parallel_assert(
        lambda: match_local,
        msg=f"Broadcast AdjFloats {Jx.subvec} do not match expected value {expect}."
    )

    # Check the adjoint is reduced back to all ranks.
    # Because the functional is an array we need to
    # pass an adj_input of an array.

    offset = rank*nlocal_floats
    adj_input = EnsembleAdjVec(
        [AdjFloat(offset + i + 1.0) for i in range(nlocal_floats)],
        ensemble=ensemble)

    expect = AdjFloat(sum(i+1.0 for i in range(nglobal_floats)))

    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local = dJ - expect < eps

    parallel_assert(
        lambda: match_local,
        msg=f"Broadcast derivative {dJ} does not match"
            f" expected value {expect}."
    )


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
@pytest.mark.skipcomplex
def test_ensemble_bcast_function():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_funcs = 6
    nlocal_funcs = nglobal_funcs // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_funcs)], ensemble)

    c = Function(R).assign(1.0)
    J = EnsembleFunction(Re)

    Jhat = EnsembleBcastReducedFunctional(J, Control(c))

    # check the control is broadcast to all ranks
    eps = 1e-12

    x = Function(R).assign(3.0)
    Jx = Jhat(x)

    expect = x
    match_local = all(errornorm(Ji, expect) < eps
                      for Ji in Jx.subfunctions)

    parallel_assert(
        lambda: match_local,
        msg=f"Broadcast Functions do not match on rank {rank}"
    )

    # Check the adjoint is reduced back to all ranks.
    # Because the functional is an EnsembleFunction we
    # need to pass an adj_input of an EnsembleCofunction.

    adj_input = EnsembleFunction(Re)
    offset = rank*nlocal_funcs
    for i, adji in enumerate(adj_input.subfunctions):
        adji.assign(offset + i + 1.0)
    adj_input = adj_input.riesz_representation()

    expect = Function(R).assign(sum(i+1.0 for i in range(nglobal_funcs)))

    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local = errornorm(dJ, expect) < eps

    parallel_assert(
        lambda: match_local,
        msg=f"Broadcast derivative {dJ.dat.data[:]} does not match"
            f" expected value {expect.dat.data[:]}."
    )


@pytest.mark.parallel(nprocs=[1, 2, 3, 6])
@pytest.mark.skipcomplex
def test_ensemble_reduction_float():
    ensemble = Ensemble(COMM_WORLD, 1)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_floats = 6
    nlocal_floats = nglobal_floats // size

    control = EnsembleAdjVec(
        [AdjFloat(0.0) for _ in range(nlocal_floats)],
        ensemble=ensemble)
    J = AdjFloat(0.0)

    Jhat = EnsembleReduceReducedFunctional(
        J, Control(control), ensemble=ensemble)

    # check the control is reduced to all ranks
    eps = 1e-12

    offset = rank*nlocal_floats
    x = EnsembleAdjVec(
        [AdjFloat(offset + i + 1.0) for i in range(nlocal_floats)],
        ensemble=ensemble)

    Jx = Jhat(x)

    expect = AdjFloat(sum(i+1.0 for i in range(nglobal_floats)))
    match_local = Jx - expect < eps

    parallel_assert(
        lambda: match_local,
        msg=f"Reduced AdjFloat {Jx} does not match"
            f" expected value {expect}"
    )

    # Check the adjoint is broadcast back to all ranks.
    # Because the functional is a Function we need to
    # pass an adj_input of an Cofunction.

    adj_value = 20.0
    adj_input = AdjFloat(adj_value)

    expect = AdjFloat(adj_value)
    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local = all((dJi - expect) < eps for dJi in dJ.subvec)

    parallel_assert(
        lambda: match_local,
        msg=f"Reduced derivatives {dJ} do not match expected value {expect}."
    )


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
@pytest.mark.skipcomplex
def test_ensemble_reduction_function():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_funcs = 6
    nlocal_funcs = nglobal_funcs // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_funcs)], ensemble)

    c = EnsembleFunction(Re)
    J = Function(R)

    Jhat = EnsembleReduceReducedFunctional(J, Control(c))

    # check the control is reduced to all ranks
    eps = 1e-12

    x = EnsembleFunction(Re)

    offset = rank*nlocal_funcs
    for i, xi in enumerate(x.subfunctions):
        xi.assign(offset + i + 1.0)

    Jx = Jhat(x)

    expect = Function(R).assign(sum(i+1.0 for i in range(nglobal_funcs)))
    match_local = errornorm(Jx, expect) < eps

    parallel_assert(
        lambda: match_local,
        msg=f"Reduced Function {Jx.dat.data[:]} does not match"
            f" expected value {expect.dat.data[:]}"
    )

    # Check the adjoint is broadcast back to all ranks.
    # Because the functional is a Function we need to
    # pass an adj_input of an Cofunction.

    adj_value = 20.0
    adj_input = (Function(R).assign(adj_value)).riesz_representation()

    expect = Function(R).assign(adj_value)
    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local = all(errornorm(dJi, expect) < eps
                      for dJi in dJ.subfunctions)

    parallel_assert(
        lambda: match_local,
        msg=f"Reduced derivatives {[dJi.dat.data[:] for dJi in dJ.subfunctions]}"
            f" do not match expected value {expect.dat.data[:]}."
    )


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
@pytest.mark.skipcomplex
def test_ensemble_transform_float():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_funcs = 6
    nlocal_funcs = nglobal_funcs // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_funcs)], ensemble)

    c = EnsembleFunction(Re)

    rfs = []
    J = []
    offset = rank*nlocal_funcs
    for ci in c.subfunctions:
        with set_working_tape() as tape:
            Ji = assemble(ci*ci*dx)
            J.append(Ji)
            rfs.append(ReducedFunctional(Ji, Control(ci), tape=tape))

    J = EnsembleAdjVec(J, ensemble)

    Jhat = EnsembleTransformReducedFunctional(
        rfs, J, Control(c), ensemble=ensemble)

    # check the control is reduced to all ranks
    eps = 1e-12

    x = EnsembleFunction(Re)

    for i, xi in enumerate(x.subfunctions):
        xi.assign(offset + i + 1.0)

    # check
    Jx = Jhat(x)

    expect = [rf(xi) for rf, xi in zip(rfs, x.subfunctions)]

    match_local = all((Ji - ei) < eps for Ji, ei in zip(Jx.subvec, expect))

    parallel_assert(
        lambda: match_local,
        msg=f"Transformed results {Jx} do not match expected values {expect}"
    )

    # Check the adjoint matches on all slots.
    # Because the functional is a list[AdjFloat] we need to
    # pass an adj_input of a list[AdjFloat].

    adj_input = EnsembleAdjVec(
        [AdjFloat(offset + i + 1.0)
         for i in range(nlocal_funcs)],
        ensemble=ensemble)

    expect = EnsembleFunction(Re)
    for rf, adji, ei in zip(rfs, adj_input.subvec, expect.subfunctions):
        ei.assign(rf.derivative(adj_input=adji, apply_riesz=True))

    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ.subfunctions, expect.subfunctions))

    parallel_assert(
        lambda: match_local,
        msg=f"Reduced derivatives {[dJi.dat.data[:] for dJi in dJ.subfunctions]}"
            f" do not match expected value {ei.dat.data[:] for ei in expect.subfunctions}."
    )

    _ = Jhat.tlm(x)
    _ = Jhat.hessian(x, hessian_input=adj_input)


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
@pytest.mark.skipcomplex
def test_ensemble_transform_float_two_controls():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_funcs = 6
    nlocal_funcs = nglobal_funcs // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_funcs)], ensemble)

    c0 = EnsembleFunction(Re)
    c1 = EnsembleFunction(Re)

    rfs = []
    J = []
    offset = rank*nlocal_funcs
    for c0i, c1i in zip(c0.subfunctions, c1.subfunctions):
        with set_working_tape() as tape:
            Ji = assemble((c0i*c0i + c1i*c1i)*dx)
            J.append(Ji)
            rfs.append(ReducedFunctional(
                Ji, [Control(c0i), Control(c1i)], tape=tape))

    J = EnsembleAdjVec(J, ensemble)

    Jhat = EnsembleTransformReducedFunctional(
        rfs, J, [Control(c0), Control(c1)], ensemble=ensemble)

    # check the control is reduced to all ranks
    eps = 1e-12

    x0 = EnsembleFunction(Re)
    x1 = EnsembleFunction(Re)

    for i, (x0i, x1i) in enumerate(zip(x0.subfunctions, x1.subfunctions)):
        x0i.assign(offset + i + 1.0)
        x1i.assign(2*(offset + i + 1.0))

    # check
    Jx = Jhat([x0, x1])

    expect = [rf([x0i, x1i])
              for rf, x0i, x1i in zip(rfs, x0.subfunctions, x1.subfunctions)]

    match_local = all((Ji - ei) < eps for Ji, ei in zip(Jx.subvec, expect))

    parallel_assert(
        lambda: match_local,
        msg=f"Transformed results {Jx} do not match expected values {expect}"
    )

    # Check the adjoint matches on all slots.
    # Because the functional is a AdjFloat we need to
    # pass an adj_input of a list[AdjFloat].

    adj_input = EnsembleAdjVec(
        [AdjFloat(offset + i + 1.0)
         for i in range(nlocal_funcs)],
        ensemble=ensemble)

    expect0 = EnsembleFunction(Re)
    expect1 = EnsembleFunction(Re)
    for rf, adji, e0i, e1i in zip(rfs, adj_input.subvec,
                                  expect0.subfunctions,
                                  expect1.subfunctions):
        e0, e1 = rf.derivative(adj_input=adji, apply_riesz=True)
        e0i.assign(e0)
        e1i.assign(e1)

    dJ0, dJ1 = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local0 = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ0.subfunctions, expect0.subfunctions))

    match_local1 = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ1.subfunctions, expect1.subfunctions))

    parallel_assert(
        lambda: match_local0,
        msg=f"Reduced derivatives {[dJ0i.dat.data[:] for dJ0i in dJ0.subfunctions]}"
            f" do not match expected value {e0i.dat.data[:] for e0i in expect0.subfunctions}."
    )

    parallel_assert(
        lambda: match_local1,
        msg=f"Reduced derivatives {[dJ1i.dat.data[:] for dJ1i in dJ1.subfunctions]}"
            f" do not match expected value {e1i.dat.data[:] for e1i in expect1.subfunctions}."
    )

    _ = Jhat.tlm([x0, x1])
    _ = Jhat.hessian([x0, x1], hessian_input=adj_input)


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
@pytest.mark.skipcomplex
def test_ensemble_transform_function():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_funcs = 6
    nlocal_funcs = nglobal_funcs // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_funcs)], ensemble)

    c = EnsembleFunction(Re)
    J = EnsembleFunction(Re)

    rfs = []
    offset = rank*nlocal_funcs
    for i, (Ji, ci) in enumerate(zip(J.subfunctions, c.subfunctions)):
        with set_working_tape() as tape:
            Ji.assign(ci)
            Ji += 2*(offset + i + 1.0)
            rfs.append(ReducedFunctional(Ji, Control(ci), tape=tape))

    Jhat = EnsembleTransformReducedFunctional(rfs, J, Control(c))

    # check the control is reduced to all ranks
    eps = 1e-12

    x = EnsembleFunction(Re)

    for i, xi in enumerate(x.subfunctions):
        xi.assign(offset + i + 1.0)

    # check
    Jx = Jhat(x)

    expect = EnsembleFunction(Re)
    for rf, xi, ei in zip(rfs, x.subfunctions,
                          expect.subfunctions):
        ei.assign(rf(xi))

    match_local = all(
        errornorm(Ji, ei) < eps
        for Ji, ei in zip(Jx.subfunctions, expect.subfunctions))

    parallel_assert(
        lambda: match_local,
        msg=f"Transformed Functions {[Ji.dat.data[:] for Ji in Jx.subfunctions]}"
            f" do not match expected value {[ei.dat.data[:] for ei in expect.subfunctions]}"
    )

    # Check the adjoint matches on all slots.
    # Because the functional is a Function we need to
    # pass an adj_input of an Cofunction.

    adj_input = EnsembleFunction(Re)
    for i, adj in enumerate(adj_input.subfunctions):
        adj.assign(offset + i + 1.0)

    adj_input = adj_input.riesz_representation()

    expect = EnsembleFunction(Re)
    for rf, adji, ei in zip(rfs, adj_input.subfunctions,
                            expect.subfunctions):
        ei.assign(rf.derivative(adj_input=adji, apply_riesz=True))

    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ.subfunctions, expect.subfunctions))

    parallel_assert(
        lambda: match_local,
        msg=f"Reduced derivatives {[dJi.dat.data[:] for dJi in dJ.subfunctions]}"
            f" do not match expected value {ei.dat.data[:] for ei in expect.subfunctions}."
    )

    _ = Jhat.tlm(x)
    _ = Jhat.hessian(x, hessian_input=adj_input)


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
@pytest.mark.skipcomplex
def test_ensemble_transform_function_two_controls():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_funcs = 6
    nlocal_funcs = nglobal_funcs // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)

    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_funcs)], ensemble)

    c0 = EnsembleFunction(Re)
    c1 = EnsembleFunction(Re)
    J = EnsembleFunction(Re)

    rfs = []
    offset = rank*nlocal_funcs
    for i, (Ji, c0i, c1i) in enumerate(zip(J.subfunctions,
                                           c0.subfunctions,
                                           c1.subfunctions)):
        with set_working_tape() as tape:
            Ji.assign(c0i + c1i)
            rfs.append(ReducedFunctional(
                Ji, [Control(c0i), Control(c1i)], tape=tape))

    Jhat = EnsembleTransformReducedFunctional(rfs, J, [Control(c0), Control(c1)])

    # check the control is reduced to all ranks
    eps = 1e-12

    x0 = EnsembleFunction(Re)
    x1 = EnsembleFunction(Re)

    for i, (x0i, x1i) in enumerate(zip(x0.subfunctions,
                                       x1.subfunctions)):
        x0i.assign(offset + i + 1.0)
        x1i.assign(2*(offset + i + 1.0))

    Jx = Jhat([x0, x1])

    expect = EnsembleFunction(Re)
    for rf, x0i, x1i, ei in zip(rfs, x0.subfunctions,
                                x1.subfunctions,
                                expect.subfunctions):
        ei.assign(rf([x0i, x1i]))

    match_local = all(
        errornorm(Ji, ei) < eps
        for Ji, ei in zip(Jx.subfunctions,
                          expect.subfunctions))

    parallel_assert(
        lambda: match_local,
        msg=f"Transformed Functions {[Ji.dat.data[:] for Ji in Jx.subfunctions]}"
            f" do not match expected value {[ei.dat.data[:] for ei in expect.subfunctions]}"
    )

    # Check the adjoint matches on all slots.
    # Because the functional is a Function we need to
    # pass an adj_input of an Cofunction.

    adj_input = EnsembleFunction(Re)
    for i, adj in enumerate(adj_input.subfunctions):
        adj.assign(offset + i + 1.0)

    adj_input = adj_input.riesz_representation()

    expect0 = EnsembleFunction(Re)
    expect1 = EnsembleFunction(Re)
    for rf, adji, e0i, e1i in zip(rfs, adj_input.subfunctions,
                                  expect0.subfunctions,
                                  expect1.subfunctions):
        e0, e1 = rf.derivative(adj_input=adji, apply_riesz=True)
        e0i.assign(e0)
        e1i.assign(e1)

    dJ0, dJ1 = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    match_local0 = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ0.subfunctions, expect0.subfunctions))

    match_local1 = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ1.subfunctions, expect1.subfunctions))

    parallel_assert(
        lambda: match_local0,
        msg=f"Reduced derivatives {[dJ0i.dat.data[:] for dJ0i in dJ0.subfunctions]}"
            f" do not match expected value {e0i.dat.data[:] for e0i in expect0.subfunctions}."
    )

    parallel_assert(
        lambda: match_local1,
        msg=f"Reduced derivatives {[dJ1i.dat.data[:] for dJ1i in dJ1.subfunctions]}"
            f" do not match expected value {e1i.dat.data[:] for e1i in expect1.subfunctions}."
    )

    _ = Jhat.tlm([x0, x1])
    _ = Jhat.hessian([x0, x1], hessian_input=adj_input)


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_ensemble_rf_function_to_float():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_rfs = 6
    nlocal_rfs = nglobal_rfs // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)
    R = FunctionSpace(mesh, "R", 0)

    control = Control(Function(R))
    J = AdjFloat(0.)

    rfs = []
    offset = rank*nlocal_rfs
    for i in range(nlocal_rfs):
        c = Function(R)
        weight = (offset + i + 1.0)
        with set_working_tape() as tape:
            Ji = weight*assemble((c**4)*dx)
            rfs.append(
                ReducedFunctional(Ji, Control(c), tape=tape))

    Jhat = EnsembleReducedFunctional(
        rfs, J, control, ensemble=ensemble)

    sum_weights = sum((i + 1.0) for i in range(nglobal_rfs))

    xval = 3.0
    Jexpect = (xval**4)*sum_weights

    x = Function(R).assign(xval)
    J = Jhat(x)
    assert_allclose(J, Jexpect, rtol=1e-12)

    adj_input = AdjFloat(4.0)
    edJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    assert_allclose(edJ.dat.data_ro, adj_input*(4*xval**3)*sum_weights, rtol=1e-12)

    dy = Function(R, val=0.1)
    assert taylor_test(Jhat, x, dy) > 1.95

    _ = Jhat.tlm(x)
    _ = Jhat.hessian(x)

    taylor = taylor_to_dict(Jhat, x, dy)

    R0 = mean(taylor['R0']['Rate'])
    R1 = mean(taylor['R1']['Rate'])
    R2 = mean(taylor['R2']['Rate'])

    assert R0 > 0.95
    assert R1 > 1.95
    assert R2 > 2.95


@pytest.mark.parallel(nprocs=[1, 2, 3, 4, 6])
@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_ensemble_rf_efunction_to_float():
    nspatial_ranks = 2 if COMM_WORLD.size == 4 else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    rank = ensemble.ensemble_rank
    size = ensemble.ensemble_size

    nglobal_funcs = 6
    nlocal_funcs = nglobal_funcs // size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)
    R = FunctionSpace(mesh, "R", 0)
    Re = EnsembleFunctionSpace(
        [R for _ in range(nlocal_funcs)], ensemble)

    control = Control(EnsembleFunction(Re))
    J = AdjFloat(0.)

    rfs = []
    for i in range(nlocal_funcs):
        ci = Function(R)
        with set_working_tape() as tape:
            Ji = assemble((ci**4)*dx)
            rfs.append(
                ReducedFunctional(Ji, Control(ci), tape=tape))

    Jhat = EnsembleReducedFunctional(
        rfs, J, control, bcast_control=False, ensemble=ensemble)

    x = EnsembleFunction(Re)

    offset = rank*nlocal_funcs
    for i, xi in enumerate(x.subfunctions):
        xi.assign(offset + i + 1.0)

    eps = 1e-12

    J = Jhat(x)

    expect = sum(w**4 for w in range(1, nglobal_funcs+1))

    parallel_assert(
        lambda: (J - expect) < eps,
        msg=f"Functional {J} does not match expected value {expect}."
    )

    adj_input = 3.0
    dJ = Jhat.derivative(adj_input=adj_input, apply_riesz=True)

    expect = EnsembleFunction(Re)
    for i, ei in enumerate(expect.subfunctions):
        ei.assign(adj_input*4*(offset + i + 1.0)**3)

    match_local = all(
        errornorm(dJi, ei) < eps
        for dJi, ei in zip(dJ.subfunctions, expect.subfunctions))

    parallel_assert(
        lambda: match_local,
        msg=f"Derivatives {[dJi.dat.data[:] for dJi in dJ.subfunctions]}"
            f" do not match expected values {ei.dat.data[:] for ei in expect.subfunctions}."
    )

    dy = EnsembleFunction(Re)

    for i, dyi in enumerate(dy.subfunctions):
        dyi.assign(0.1*(-0.5*offset - (i + 1.0)))

    assert taylor_test(Jhat, x, dy) > 1.95

    _ = Jhat.tlm(x)
    _ = Jhat.hessian(x)

    taylor = taylor_to_dict(Jhat, x, dy)

    R0 = mean(taylor['R0']['Rate'])
    R1 = mean(taylor['R1']['Rate'])
    R2 = mean(taylor['R2']['Rate'])

    assert R0 > 0.95
    assert R1 > 1.95
    assert R2 > 2.95


@pytest.mark.parallel(nprocs=[1, 2, 3, 6])
@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_ensemble_rf_minimise():
    """
    Optimisation test using a list of controls.
    This test is equivalent to the one found at:
    https://github.com/firedrakeproject/firedrake/blob/master/tests/firedrake/adjoint/test_optimisation.py#L92
    In this test, the functional is the result of an ensemble allreduce operation.
    """
    nspatial_ranks = 2 if COMM_WORLD.size in (2, 6) else 1
    ensemble = Ensemble(COMM_WORLD, nspatial_ranks)

    size = ensemble.ensemble_size

    mesh = UnitIntervalMesh(12, comm=ensemble.comm)
    R = FunctionSpace(mesh, "R", 0)

    nglobal_rfs = 3
    nlocal_rfs = nglobal_rfs//size
    n = 2

    rfs = []
    for _ in range(nlocal_rfs):
        x = [Function(R) for _ in range(n)]
        controls = [Control(xi) for xi in x]
        with set_working_tape() as tape:
            # Rosenbrock function https://en.wikipedia.org/wiki/Rosenbrock_function
            # with minimum at x = (1, 1, 1, ...)
            f = 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
            J = assemble(f * dx(domain=mesh))
            rfs.append(
                ReducedFunctional(J, controls, tape=tape))

    J = AdjFloat(0.)
    controls = [Control(Function(R)) for _ in range(n)]

    rf = EnsembleReducedFunctional(rfs, J, controls, ensemble=ensemble)
    rf_np = ReducedFunctionalNumPy(rf)

    result = minimize(rf_np)
    assert_allclose([float(xi) for xi in result], 1., rtol=1e-8)
