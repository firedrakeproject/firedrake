from firedrake import *
import pytest


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("fs_type", ["additive", "lower", "upper"])
def test_fieldsplit_snes(fs_type):
    max_it = 3

    mesh = PeriodicUnitIntervalMesh(50)
    x, = SpatialCoordinate(mesh)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)
    W = V2 * V1

    w0 = Function(W)
    u0, T0 = w0.subfunctions
    u0.interpolate(1 + 0.4*cos(2*pi*x))
    T0.interpolate(1 + 0.3*cos(4*pi*(x + 0.25)))

    dt = Constant(0.1)
    re = Constant(30)

    def nu(T):
        return (1 - 0.5*(T - 1)**2)/re

    def FT(u, T, dT):
        return (
            inner(T - T0, dT)*dx
            + dt*inner(as_vector([u]), grad(T))*dT*dx
            + dt*inner(nu(T)*grad(T), grad(dT))*dx
        )

    def Fu(u, T, du):
        return (
            inner(u - u0, du)*dx
            + dt*inner(as_vector([u]), grad(u))*du*dx
            + dt*inner(nu(T)*grad(u), grad(du))*dx
            - dt*(0.1*sin(2*pi*x))*du*dx
        )

    # Manual component-wise fieldsplit iteration
    wk1 = Function(W).zero()
    wk = Function(W).zero()

    # use subfunctions not split because we
    # want a separate solver on each component
    uk1, Tk1 = wk1.subfunctions
    uk, Tk = wk.subfunctions

    du = TestFunction(V2)
    dT = TestFunction(V1)

    if fs_type == "additive":
        Gu = Fu(uk1, Tk, du)
        GT = FT(uk, Tk1, dT)
    elif fs_type == "lower":
        Gu = Fu(uk1, Tk, du)
        GT = FT(uk1, Tk1, dT)
    elif fs_type == "upper":
        Gu = Fu(uk1, Tk1, du)
        GT = FT(uk, Tk1, dT)

    sub_params = {
        'snes_rtol': 1e-8,
        'snes_type': 'newtonls',
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    Gu_solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(Gu, uk1),
        solver_parameters=sub_params)

    GT_solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(GT, Tk1),
        solver_parameters=sub_params)

    for i in range(max_it):
        wk.assign(wk1)
        if fs_type == "upper":
            GT_solver.solve()
            Gu_solver.solve()
        else:
            Gu_solver.solve()
            GT_solver.solve()

    # SNES fieldsplit solve

    w = Function(W).zero()
    u, T = split(w)
    du, dT = TestFunctions(W)

    F = (
        Fu(u, T, du)
        + FT(u, T, dT)
    )

    if fs_type == "additive":
        fs_options = {
            'snes_fieldsplit_type': 'additive',
            'snes_fieldsplit_0_fields': '0',
            'snes_fieldsplit_1_fields': '1',
        }
    elif fs_type == "lower":
        fs_options = {
            'snes_fieldsplit_type': 'multiplicative',
            'snes_fieldsplit_0_fields': '0',
            'snes_fieldsplit_1_fields': '1',
        }
    elif fs_type == "upper":
        fs_options = {
            'snes_fieldsplit_type': 'multiplicative',
            'snes_fieldsplit_0_fields': '1',
            'snes_fieldsplit_1_fields': '0',
        }

    fs_params = {
        'snes_view': ':snes_fs_view.log',
        'snes_convergence_test': 'skip',
        'snes_max_it': max_it,
        'snes_type': 'python',
        'snes_python_type': 'firedrake.FieldsplitSNES',
        **fs_options,
        'fieldsplit_0': sub_params,
        'fieldsplit_1': sub_params,
    }

    fs_solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, w),
        solver_parameters=fs_params)

    fs_solver.solve()
    u, T = w.subfunctions
    uref, Tref = wk1.subfunctions
    assert errornorm(uref, u) < 1e-14
    assert errornorm(Tref, T) < 1e-14


@pytest.mark.parametrize("fs_type", ["additive", "lower", "upper"])
def test_fieldsplit_snes_forcing(fs_type):
    """Check that FieldsplitSNES correctly applies a nonzero forcing
    term (G^{k} - F^{k}). This is zero when the outer SNES solves
    F(u) = 0, but is nonzero when the SNES is given a right hand
    side, e.g. as the coarse solver in FAS. Here we call the step
    method directly with a residual f = F(x) - r, so the forcing is r.
    """
    mesh = UnitIntervalMesh(20)
    x, = SpatialCoordinate(mesh)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 2)
    W = V2 * V1

    def Fu(u, T, du):
        return (inner(grad(u), grad(du))*dx
                + inner(u + u*u*u + T*T, du)*dx)

    def FT(u, T, dT):
        return (inner(grad(T), grad(dT))*dx
                + inner(T + T*T*T + u*T, dT)*dx)

    # The current iterate and the nonzero right hand side.
    wk = Function(W)
    wk.subfunctions[0].interpolate(0.5 + 0.2*sin(2*pi*x))
    wk.subfunctions[1].interpolate(0.3 + 0.1*cos(2*pi*x))

    du, dT = TestFunctions(W)
    r = assemble(inner(1 + x, du)*dx + inner(2 - x, dT)*dx)

    # Manual component-wise fieldsplit iteration solving G(w^{k+1}) = r
    wk1 = Function(W).assign(wk)
    uk1, Tk1 = wk1.subfunctions
    uk, Tk = wk.subfunctions
    ru, rT = r.subfunctions
    vu = TestFunction(V2)
    vT = TestFunction(V1)

    if fs_type == "additive":
        Gu = Fu(uk1, Tk, vu)
        GT = FT(uk, Tk1, vT)
    elif fs_type == "lower":
        Gu = Fu(uk1, Tk, vu)
        GT = FT(uk1, Tk1, vT)
    elif fs_type == "upper":
        Gu = Fu(uk1, Tk1, vu)
        GT = FT(uk, Tk1, vT)

    sub_params = {
        'snes_rtol': 1e-12,
        'snes_atol': 1e-14,
        'snes_type': 'newtonls',
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    Gu_solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(Gu - ru, uk1),
        solver_parameters=sub_params)
    GT_solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(GT - rT, Tk1),
        solver_parameters=sub_params)

    if fs_type == "upper":
        GT_solver.solve()
        Gu_solver.solve()
    else:
        Gu_solver.solve()
        GT_solver.solve()

    # FieldsplitSNES step with residual f = F(x) - r
    w = Function(W)
    u, T = split(w)
    F = Fu(u, T, du) + FT(u, T, dT)

    if fs_type == "additive":
        fs_options = {'snes_fieldsplit_type': 'additive'}
    elif fs_type == "lower":
        fs_options = {'snes_fieldsplit_type': 'multiplicative'}
    elif fs_type == "upper":
        fs_options = {
            'snes_fieldsplit_type': 'multiplicative',
            'snes_fieldsplit_0_fields': '1',
            'snes_fieldsplit_1_fields': '0',
        }

    fs_params = {
        'snes_convergence_test': 'skip',
        'snes_max_it': 1,
        'snes_type': 'python',
        'snes_python_type': 'firedrake.FieldsplitSNES',
        **fs_options,
        'fieldsplit_0': sub_params,
        'fieldsplit_1': sub_params,
    }

    fs_solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, w),
        solver_parameters=fs_params)

    # Solve once so that the python SNES is set up.
    fs_solver.solve()

    snes = fs_solver.snes
    fs_snes = snes.getPythonContext()

    w.assign(wk)
    f = assemble(F)
    f -= r

    y = Function(W)
    with wk.dat.vec_ro as xvec, f.dat.vec_ro as fvec, y.dat.vec_wo as yvec:
        fs_snes.step(snes, xvec, fvec, yvec)

    # The step returns the update y = x^{k} - x^{k+1}
    expected = Function(W).assign(wk - wk1)
    for ys, es in zip(y.subfunctions, expected.subfunctions):
        assert errornorm(es, ys) < 1e-10


def test_fieldsplit_snes_update():
    """Check that FieldsplitSNES rebuilds itself if the context on
    the DM is replaced (e.g. by mesh adaptivity), and keeps its
    existing state otherwise.
    """
    from firedrake import dmhooks

    sub_params = {
        'snes_type': 'newtonls',
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }
    fs_params = {
        'snes_convergence_test': 'skip',
        'snes_max_it': 1,
        'snes_type': 'python',
        'snes_python_type': 'firedrake.FieldsplitSNES',
        'snes_fieldsplit_type': 'multiplicative',
        'fieldsplit_0': sub_params,
        'fieldsplit_1': sub_params,
    }

    def make_solver(n):
        mesh = UnitIntervalMesh(n)
        W = FunctionSpace(mesh, "CG", 1) * FunctionSpace(mesh, "CG", 1)
        w = Function(W)
        u, T = split(w)
        du, dT = TestFunctions(W)
        F = (inner(grad(u), grad(du))*dx + inner(u + 0.1*T*T - 1, du)*dx
             + inner(grad(T), grad(dT))*dx + inner(T + 0.1*u*u - 2, dT)*dx)
        solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(F, w),
            solver_parameters=fs_params)
        # Solve once so that the python SNES is set up.
        solver.solve()
        return solver, w, F

    solver, _, _ = make_solver(10)
    snes = solver.snes
    fs_snes = snes.getPythonContext()

    dm = snes.getDM()
    split_solvers = fs_snes.split_solvers

    # No context on the DM (outside of a solve): nothing is rebuilt.
    fs_snes.update(snes)
    assert fs_snes.split_solvers is split_solvers

    # Unchanged context (as during a solve): nothing is rebuilt.
    dmhooks.push_appctx(dm, solver._ctx)
    try:
        fs_snes.update(snes)
    finally:
        dmhooks.pop_appctx(dm, solver._ctx)
    assert fs_snes.split_solvers is split_solvers

    # Replace the context on the DM with one on a different mesh,
    # as mesh adaptivity would.
    new_solver, new_w, new_F = make_solver(17)
    new_ctx = new_solver._ctx
    dmhooks.push_appctx(dm, new_ctx)
    try:
        # A real re-setup happens during solve, when the outer solver's
        # options are in the PETSc options database. They are needed
        # because initialize reads e.g. snes_fieldsplit_type from them.
        with solver.inserted_options():
            fs_snes.update(snes)
        assert fs_snes.fieldsplit_type == 'multiplicative'
        assert fs_snes.split_solvers is not split_solvers
        assert fs_snes.u is new_ctx._x

        # The rebuilt FieldsplitSNES should take the same step as
        # one that was originally built on the new context.
        # Both are built from the same (cached) splits of new_ctx, so
        # use a nonzero forcing r to check that they do not interfere.
        # Use a separate x because step writes into new_w.
        new_w.zero()
        x = Function(new_w.function_space())
        du, dT = TestFunctions(new_w.function_space())
        r = assemble(inner(1, du)*dx + inner(2, dT)*dx)
        f = assemble(new_F)
        f -= r
        y = Function(new_w.function_space())
        y_ref = Function(new_w.function_space())
        # Step the reference first so that its forcing is nonzero
        # when the rebuilt FieldsplitSNES steps.
        with x.dat.vec_ro as xvec, f.dat.vec_ro as fvec:
            with y_ref.dat.vec_wo as yvec:
                new_solver.snes.getPythonContext().step(
                    new_solver.snes, xvec, fvec, yvec)
            with y.dat.vec_wo as yvec:
                fs_snes.step(snes, xvec, fvec, yvec)
    finally:
        dmhooks.pop_appctx(dm, new_ctx)

    assert norm(y_ref) > 0
    assert errornorm(y_ref, y) < 1e-12


@pytest.mark.parametrize("options,nfields,expected", [
    # no splits specified: one split per field
    ({}, 3, [[0], [1], [2]]),
    ({'default': None}, 3, [[0], [1], [2]]),
    # all fields specified
    ({'0_fields': '2', '1_fields': '0,1'}, 3, [[2], [0, 1]]),
    # missing fields appended in field order with -snes_fieldsplit_default
    ({'0_fields': '2', 'default': None}, 4, [[2], [0], [1], [3]]),
    ({'0_fields': '1,3', 'default': None}, 4, [[1, 3], [0], [2]]),
    ({'0_fields': '3', '1_fields': '1', 'default': None}, 4,
     [[3], [1], [0], [2]]),
    # default does nothing if all fields are already specified
    ({'0_fields': '1', '1_fields': '0', 'default': None}, 2, [[1], [0]]),
])
def test_fieldsplit_snes_get_splits(options, nfields, expected):
    prefix = "test_get_splits_snes_fieldsplit_"
    opts = PETSc.Options(prefix)
    for k, v in options.items():
        opts[k] = v
    try:
        splits = FieldsplitSNES()._get_splits(opts, nfields)
    finally:
        for k in options:
            opts.delValue(k)
    assert [list(s) for s in splits] == expected


@pytest.mark.parametrize("options,nfields,error", [
    # gap in the split numbering
    ({'0_fields': '0', '2_fields': '1'}, 2, "consecutively"),
    # split numbering does not start from 0
    ({'1_fields': '0,1'}, 2, "consecutively"),
    # field index out of range
    ({'0_fields': '0', '1_fields': '1,2'}, 2, "only 2 fields"),
    # field in more than one split
    ({'0_fields': '0,1', '1_fields': '1'}, 2, "more than one split"),
    ({'0_fields': '0,1', '1_fields': '1', 'default': None}, 3,
     "more than one split"),
    # missing field without -snes_fieldsplit_default
    ({'0_fields': '0', '1_fields': '2'}, 3, "not in any split"),
])
def test_fieldsplit_snes_get_splits_errors(options, nfields, error):
    prefix = "test_get_splits_errors_snes_fieldsplit_"
    opts = PETSc.Options(prefix)
    for k, v in options.items():
        opts[k] = v
    try:
        with pytest.raises(ValueError, match=error):
            FieldsplitSNES()._get_splits(opts, nfields)
    finally:
        for k in options:
            opts.delValue(k)


def test_fieldsplit_snes_default_fields():
    """Check that -snes_fieldsplit_default is respected in a solve.
    The splits with default fields should match explicitly specifying
    the same splits.
    """
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V * V

    def solve_with(fs_options):
        w = Function(W)
        u0, u1, u2 = split(w)
        v0, v1, v2 = TestFunctions(W)
        F = (inner(grad(u0), grad(v0))*dx + inner(u0 + 0.1*u1*u2 - 1, v0)*dx
             + inner(grad(u1), grad(v1))*dx + inner(u1 + 0.1*u0*u2 - 2, v1)*dx
             + inner(grad(u2), grad(v2))*dx + inner(u2 + 0.1*u0*u1 - 3, v2)*dx)
        sub_params = {
            'snes_type': 'newtonls',
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu',
        }
        params = {
            'snes_convergence_test': 'skip',
            'snes_max_it': 2,
            'snes_type': 'python',
            'snes_python_type': 'firedrake.FieldsplitSNES',
            'snes_fieldsplit_type': 'multiplicative',
            **fs_options,
            **{f'fieldsplit_{i}': sub_params for i in range(3)},
        }
        solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(F, w),
            solver_parameters=params)
        solver.solve()
        return w, solver.snes.getPythonContext().splits

    w_default, splits_default = solve_with({
        'snes_fieldsplit_0_fields': '2',
        'snes_fieldsplit_default': None,
    })
    w_explicit, splits_explicit = solve_with({
        'snes_fieldsplit_0_fields': '2',
        'snes_fieldsplit_1_fields': '0',
        'snes_fieldsplit_2_fields': '1',
    })

    assert [list(s) for s in splits_default] == [[2], [0], [1]]
    assert [list(s) for s in splits_explicit] == [[2], [0], [1]]
    assert errornorm(w_explicit, w_default) < 1e-14


@pytest.mark.parametrize("named", [False, True])
def test_fieldsplit_snes_default_options(named):
    """Check that options with the fieldsplit_ prefix are used as
    defaults for every split, that options for a specific split take
    priority over them, and that they take priority over Firedrake's
    default solver parameters.
    """
    mesh = UnitIntervalMesh(10)
    if named:
        V0 = FunctionSpace(mesh, "CG", 1, name="first")
        V1 = FunctionSpace(mesh, "CG", 1, name="second")
        names = ["first", "second"]
    else:
        V0 = FunctionSpace(mesh, "CG", 1)
        V1 = FunctionSpace(mesh, "CG", 1)
        names = ["0", "1"]
    W = V0 * V1

    w = Function(W)
    u0, u1 = split(w)
    v0, v1 = TestFunctions(W)
    F = (inner(grad(u0), grad(v0))*dx + inner(u0 + 0.1*u1*u1 - 1, v0)*dx
         + inner(grad(u1), grad(v1))*dx + inner(u1 + 0.1*u0*u0 - 2, v1)*dx)

    params = {
        'snes_type': 'python',
        'snes_python_type': 'firedrake.FieldsplitSNES',
        'snes_fieldsplit_type': 'multiplicative',
        'snes_max_it': 2,
        'snes_convergence_test': 'skip',
        # defaults for all splits
        'fieldsplit': {
            'snes_rtol': 1e-3,
            'mat_type': 'aij',
            # Firedrake's default would be preonly
            'ksp_type': 'gmres',
            'pc_type': 'lu',
        },
        # overrides for the second split
        f'fieldsplit_{names[1]}': {
            'snes_rtol': 1e-7,
        },
    }

    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, w),
        solver_parameters=params)
    solver.solve()

    split_solvers = solver.snes.getPythonContext().split_solvers
    assert [s.snes.getOptionsPrefix() for s in split_solvers] == [
        f"{solver.snes.getOptionsPrefix()}fieldsplit_{name}_" for name in names]

    expected_rtols = [1e-3, 1e-7]
    for split_solver, rtol in zip(split_solvers, expected_rtols):
        snes = split_solver.snes
        assert snes.getTolerances()[0] == rtol
        assert snes.ksp.getType() == "gmres"
        assert snes.ksp.pc.getType() == "lu"


def test_fieldsplit_snes_nested():
    """Check that a FieldsplitSNES can be used as the solver for a split
    of another FieldsplitSNES. One multiplicative sweep over fields 0
    and 1 inside a multiplicative sweep over [0, 1] and [2] is the same
    as a multiplicative sweep over [0], [1] and [2]. This is checked
    for a full solve, and for a single step with a nonzero forcing.
    """
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V * V

    def make_solver(params):
        w = Function(W)
        u0, u1, u2 = split(w)
        v0, v1, v2 = TestFunctions(W)
        F = (inner(grad(u0), grad(v0))*dx + inner(u0 + 0.1*u1*u2 - 1, v0)*dx
             + inner(grad(u1), grad(v1))*dx + inner(u1 + 0.1*u0*u2 - 2, v1)*dx
             + inner(grad(u2), grad(v2))*dx + inner(u2 + 0.1*u0*u1 - 3, v2)*dx)
        return NonlinearVariationalSolver(
            NonlinearVariationalProblem(F, w), solver_parameters=params)

    newton = {
        'snes_type': 'newtonls',
        'snes_rtol': 1e-12,
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    def fieldsplit(max_it):
        return {
            'snes_type': 'python',
            'snes_python_type': 'firedrake.FieldsplitSNES',
            'snes_fieldsplit_type': 'multiplicative',
            'snes_max_it': max_it,
            'snes_convergence_test': 'skip',
            'fieldsplit': newton,
        }

    flat_solver = make_solver(fieldsplit(3))

    # The defaults for the outer splits are not used by the inner
    # splits, so the inner FieldsplitSNES needs its own defaults.
    nested_solver = make_solver({
        **fieldsplit(3),
        'snes_fieldsplit_0_fields': '0,1',
        'snes_fieldsplit_1_fields': '2',
        'fieldsplit_0': fieldsplit(1),
    })

    flat_solver.solve()
    nested_solver.solve()

    outer_prefix = nested_solver.snes.getOptionsPrefix()
    inner_fs = nested_solver.snes.getPythonContext().split_solvers[0].snes.getPythonContext()
    assert isinstance(inner_fs, FieldsplitSNES)
    assert inner_fs.splits == [[0], [1]]
    assert [s.snes.getOptionsPrefix() for s in inner_fs.split_solvers] == [
        f"{outer_prefix}fieldsplit_0_fieldsplit_{i}_" for i in range(2)]

    w_flat = flat_solver._problem.u
    w_nested = nested_solver._problem.u
    assert errornorm(w_flat, w_nested) < 1e-12

    # Take one step from the same x with a residual f = F(x) - r,
    # so the inner FieldsplitSNES must split the outer forcing.
    x = Function(W)
    for i, xi in enumerate(x.subfunctions):
        xi.assign(0.5 + 0.1*i)
    v0, v1, v2 = TestFunctions(W)
    r = assemble(inner(1, v0)*dx + inner(2, v1)*dx + inner(3, v2)*dx)

    def step(solver):
        solver._problem.u.assign(x)
        f = assemble(solver._problem.F)
        f -= r
        y = Function(W)
        # The step is taken outside a solve, so the options of the outer
        # solver must be inserted for the inner solvers to find them.
        with solver.inserted_options():
            with x.dat.vec_ro as xvec, f.dat.vec_ro as fvec, y.dat.vec_wo as yvec:
                solver.snes.getPythonContext().step(solver.snes, xvec, fvec, yvec)
        return y

    y_flat = step(flat_solver)
    y_nested = step(nested_solver)
    assert norm(y_flat) > 0
    assert errornorm(y_flat, y_nested) < 1e-12
