from firedrake import *
import pytest


def _get_split_solvers(solver):
    """Create the field-split children of a solver with their equations."""
    from firedrake import dmhooks

    with solver.inserted_options(), \
            dmhooks.add_hooks(solver.snes.dm, solver, appctx=solver._ctx):
        return solver.snes.getFieldSplitSubSNES()


def _solve_native(solver, rhs=None):
    """Solve on a copy of the solution, as the solver does."""
    from firedrake import dmhooks
    from contextlib import ExitStack

    u = solver._problem.u
    with u.dat.vec_ro as v:
        x = v.copy()
    with ExitStack() as stack:
        stack.enter_context(solver.inserted_options())
        stack.enter_context(dmhooks.add_hooks(solver.snes.dm, solver, appctx=solver._ctx))
        b = None if rhs is None else stack.enter_context(rhs.dat.vec_ro)
        solver.snes.solve(b, x)
    with u.dat.vec_wo as v:
        x.copy(v)


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

    Gu_problem = NonlinearVariationalProblem(Gu, uk1)
    Gu_solver = NonlinearVariationalSolver(Gu_problem, solver_parameters=sub_params)

    GT_problem = NonlinearVariationalProblem(GT, Tk1)
    GT_solver = NonlinearVariationalSolver(GT_problem, solver_parameters=sub_params)

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
        'snes_type': 'fieldsplit',
        **fs_options,
        **{f'fieldsplit_{i}_{key}': value
           for i in range(2) for key, value in sub_params.items()},
    }

    fs_problem = NonlinearVariationalProblem(F, w)
    fs_solver = NonlinearVariationalSolver(fs_problem, solver_parameters=fs_params)

    fs_solver.solve()
    u, T = w.subfunctions
    uref, Tref = wk1.subfunctions
    assert errornorm(uref, u) < 1e-14
    assert errornorm(Tref, T) < 1e-14


@pytest.mark.parametrize("fs_type", ["additive", "lower", "upper"])
def test_fieldsplit_snes_forcing(fs_type):
    """Compare a sweep with nonzero RHS against component solves."""
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

    Gu_problem = NonlinearVariationalProblem(Gu - ru, uk1)
    Gu_solver = NonlinearVariationalSolver(Gu_problem, solver_parameters=sub_params)
    GT_problem = NonlinearVariationalProblem(GT - rT, Tk1)
    GT_solver = NonlinearVariationalSolver(GT_problem, solver_parameters=sub_params)

    if fs_type == "upper":
        GT_solver.solve()
        Gu_solver.solve()
    else:
        Gu_solver.solve()
        GT_solver.solve()

    # Native fieldsplit sweep with nonzero right hand side r.
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
        'snes_type': 'fieldsplit',
        **fs_options,
        **{f'fieldsplit_{i}_{key}': value
           for i in range(2) for key, value in sub_params.items()},
    }

    fs_problem = NonlinearVariationalProblem(F, w)
    fs_solver = NonlinearVariationalSolver(fs_problem, solver_parameters=fs_params)

    w.assign(wk)
    _solve_native(fs_solver, r)
    assert errornorm(wk1, w) < 1e-10


def test_fieldsplit_snes_independent_solvers():
    """Keep restricted equations independent on a shared FunctionSpace."""
    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V

    v, q = TestFunctions(W)
    solutions = [Function(W), Function(W)]
    params = {
        'snes_type': 'fieldsplit',
        'snes_fieldsplit_type': 'multiplicative',
        'snes_max_it': 2,
        'snes_convergence_test': 'skip',
        'fieldsplit_ksp_type': 'preonly',
        'fieldsplit_pc_type': 'lu',
    }
    solvers = []
    for i, w in enumerate(solutions):
        u, p = split(w)
        F = ((u + 0.1*p - (i + 1))*v + (p + 0.1*u - 2)*q)*dx
        problem = NonlinearVariationalProblem(F, w)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)
        solvers.append(solver)

    for solver in solvers:
        solver.set_transfer_manager(TransferManager())
    solvers[0].solve()
    saved = solutions[0].copy(deepcopy=True)
    solvers[1].solve()
    assert errornorm(saved, solutions[0]) == 0
    assert errornorm(solutions[0], solutions[1]) > 0.1
    solutions[0].zero()
    solvers[0].solve()
    assert errornorm(saved, solutions[0]) < 1e-13


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
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    W = MixedFunctionSpace([V]*nfields)

    w = Function(W)
    v = TestFunction(W)
    F = inner(w, v)*dx
    params = {
        'snes_type': 'fieldsplit',
        **{f'snes_fieldsplit_{k}': v for k, v in options.items()},
    }
    problem = NonlinearVariationalProblem(F, w)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)

    snes = solver.snes
    splits = [snes.getFieldSplitFields(i) for i in range(len(_get_split_solvers(solver)))]
    assert [list(s) for s in splits] == expected


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
            'snes_type': 'fieldsplit',
            'snes_fieldsplit_type': 'multiplicative',
            **fs_options,
            **{f'fieldsplit_{i}_{key}': value
               for i in range(3) for key, value in sub_params.items()},
        }
        problem = NonlinearVariationalProblem(F, w)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)
        solver.solve()
        return w, [solver.snes.getFieldSplitFields(i)
                   for i in range(len(solver.snes.getFieldSplitSubSNES()))]

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


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("npc", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_fieldsplit_snes_split_options(named, npc):
    """Check that each split reads its options under its own prefix, which
    is named after its field. A nonlinear preconditioner's splits use its
    own options prefix.
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
        'snes_type': 'fieldsplit',
        'snes_fieldsplit_type': 'multiplicative',
        'snes_max_it': 2,
        'snes_convergence_test': 'skip',
    }
    for name, rtol in zip(names, [1e-3, 1e-7]):
        params.update({
            f'fieldsplit_{name}_snes_rtol': rtol,
            f'fieldsplit_{name}_ksp_type': 'gmres',
            f'fieldsplit_{name}_pc_type': 'lu',
        })

    if npc:
        params = {
            'snes_type': 'ngmres',
            'snes_max_it': 2,
            'snes_convergence_test': 'skip',
            **{f'npc_{key}': value for key, value in params.items()},
            'npc_snes_max_it': 1,
        }

    problem = NonlinearVariationalProblem(F, w)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    fs_snes = solver.snes.getNPC() if npc else solver.snes
    split_solvers = fs_snes.getFieldSplitSubSNES()
    assert [s.getOptionsPrefix() for s in split_solvers] == [
        f"{fs_snes.getOptionsPrefix()}fieldsplit_{name}_" for name in names]

    expected_rtols = [1e-3, 1e-7]
    for split_solver, rtol in zip(split_solvers, expected_rtols):
        snes = split_solver
        assert snes.getTolerances()[0] == rtol
        assert snes.ksp.getType() == "gmres"
        assert snes.ksp.pc.getType() == "lu"


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize(('auxiliary', 'group', 'late_type'),
                         [(False, '0,1', False), (False, '1,0', False),
                          (True, '0,1', False), (False, '1,0', True)])
def test_fieldsplit_snes_nested(auxiliary, group, late_type):
    """Check that a native fieldsplit can be used as the solver for a split
    of another native fieldsplit. One multiplicative sweep over fields 0
    and 1 inside a multiplicative sweep over [0, 1] or [1, 0] and [2]
    is the same as a flat sweep in the corresponding order. This is checked
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
        problem = NonlinearVariationalProblem(F, w)
        solver = NonlinearVariationalSolver(problem, solver_parameters=params)
        return solver

    newton = {
        'snes_type': 'newtonls',
        'snes_rtol': 1e-12,
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
    }

    def fieldsplit(max_it):
        return {
            'snes_type': 'fieldsplit',
            'snes_fieldsplit_type': 'multiplicative',
            'snes_max_it': max_it,
            'snes_convergence_test': 'skip',
            **{f'fieldsplit_{key}': value for key, value in newton.items()},
        }

    flat_params = fieldsplit(3)
    if group == '1,0':
        flat_params.update({
            'snes_fieldsplit_0_fields': '1',
            'snes_fieldsplit_1_fields': '0',
            'snes_fieldsplit_2_fields': '2',
        })
    flat_solver = make_solver(flat_params)

    # The defaults for the outer splits are not used by the inner
    # splits, so the inner native fieldsplit needs its own defaults.
    inner_options = fieldsplit(1)
    if auxiliary:
        inner_options = {
            'snes_type': 'python',
            'snes_python_type': 'firedrake.AuxiliaryOperatorSNES',
            'snes_max_it': 1,
            'snes_convergence_test': 'skip',
            **{f'aux_{key}': value for key, value in inner_options.items()},
        }
    if late_type:
        inner_options.pop('snes_type')
    nested_params = {
        **fieldsplit(3),
        'snes_fieldsplit_0_fields': group,
        'snes_fieldsplit_1_fields': '2',
        **{f'fieldsplit_0_{key}': value for key, value in inner_options.items()},
    }
    nested_solver = make_solver(nested_params)
    if late_type:
        from firedrake import dmhooks

        with nested_solver.inserted_options(), \
                dmhooks.add_hooks(nested_solver.snes.dm, nested_solver, appctx=nested_solver._ctx):
            nested_solver.snes.setUp()
            inner_fs = nested_solver.snes.getFieldSplitSubSNES()[0]
            assert inner_fs.getType() != 'fieldsplit'
            inner_fs.setType('fieldsplit')
            inner_fs.setFieldSplitType(PETSc.SNES.CompositeType.MULTIPLICATIVE)

    flat_solver.solve()
    nested_solver.solve()

    outer_prefix = nested_solver.snes.getOptionsPrefix()
    inner_fs = nested_solver.snes.getFieldSplitSubSNES()[0]
    if auxiliary:
        inner_fs = inner_fs.getPythonContext().solver.snes
        outer_prefix += "fieldsplit_0_aux_"
    else:
        outer_prefix += "fieldsplit_0_"
    assert inner_fs.getType() == "fieldsplit"
    assert [inner_fs.getFieldSplitFields(i) for i in range(2)] == [(0,), (1,)]
    assert [s.getOptionsPrefix() for s in inner_fs.getFieldSplitSubSNES()] == [
        f"{outer_prefix}fieldsplit_{i}_" for i in range(2)]

    w_flat = flat_solver._problem.u
    w_nested = nested_solver._problem.u
    assert errornorm(w_flat, w_nested) < 1e-12

    # Take one step from the same x with a residual f = F(x) - r,
    # so the inner native fieldsplit must split the outer forcing.
    x = Function(W)
    for i, xi in enumerate(x.subfunctions):
        xi.assign(0.5 + 0.1*i)
    v0, v1, v2 = TestFunctions(W)
    r = assemble(inner(1, v0)*dx + inner(2, v1)*dx + inner(3, v2)*dx)

    def step(solver):
        solver._problem.u.assign(x)
        solver.snes.setTolerances(max_it=1)
        _solve_native(solver, r)
        return solver._problem.u.copy(deepcopy=True)

    y_flat = step(flat_solver)
    y_nested = step(nested_solver)
    assert norm(y_flat) > 0
    assert errornorm(y_flat, y_nested) < 1e-12


def test_fieldsplit_snes_nullspaces():
    """Check that the nullspace, transpose nullspace and near nullspace
    of the full problem are split and given to the solver for each split.
    """
    from firedrake import dmhooks

    mesh = UnitIntervalMesh(10)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V * V

    w = Function(W)
    u0, u1, u2 = split(w)
    v0, v1, v2 = TestFunctions(W)
    F = (inner(grad(u0), grad(v0))*dx + inner(u0 - 1, v0)*dx
         + inner(grad(u1), grad(v1))*dx + inner(u1 - 2, v1)*dx
         + inner(grad(u2), grad(v2))*dx + inner(u2 - 3, v2)*dx)

    # Each nullspace has a basis on a different field, so that we
    # can check that each one is split separately.
    def make_nullspace(field):
        return MixedVectorSpaceBasis(
            W, [VectorSpaceBasis(constant=True, comm=mesh.comm) if i == field else W.sub(i)
                for i in range(len(W))])

    nullspace = make_nullspace(0)
    transpose_nullspace = make_nullspace(1)
    near_nullspace = make_nullspace(2)

    params = {
        'snes_type': 'fieldsplit',
        'snes_fieldsplit_0_fields': '0',
        'snes_fieldsplit_1_fields': '1,2',
    }
    problem = NonlinearVariationalProblem(F, w)
    solver = NonlinearVariationalSolver(
        problem,
        nullspace=nullspace,
        transpose_nullspace=transpose_nullspace,
        near_nullspace=near_nullspace,
        solver_parameters=params)

    # Set up the native fieldsplit in the same way as a solve would.
    snes = solver.snes
    with solver.inserted_options(), \
            dmhooks.add_hooks(snes.getDM(), solver, appctx=solver._ctx):
        snes.setUp()
        split_ctxs = [dmhooks.get_appctx(child.dm) for child in snes.getFieldSplitSubSNES()]

    # The first split is field 0, which only the nullspace has a basis on.
    assert split_ctxs[0]._nullspace is nullspace._bases[0]
    assert split_ctxs[0]._nullspace_T is None
    assert split_ctxs[0]._near_nullspace is None

    # The second split is fields 1 and 2, so each of its nullspaces is
    # mixed, with a basis only where the full nullspace has one.
    assert all(not isinstance(base, VectorSpaceBasis)
               for base in split_ctxs[1]._nullspace._bases)
    for split_nullspace, basis, field in [
            (split_ctxs[1]._nullspace_T, transpose_nullspace, 1),
            (split_ctxs[1]._near_nullspace, near_nullspace, 2)]:
        assert isinstance(split_nullspace, MixedVectorSpaceBasis)
        bases = split_nullspace._bases
        assert bases[field - 1] is basis._bases[field]
        assert not isinstance(bases[2 - field], VectorSpaceBasis)


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("pre_apply_bcs", [False, True])
@pytest.mark.parametrize("mat_type", ["aij", "matfree"])
def test_fieldsplit_snes_bcs_jp(pre_apply_bcs, mat_type):
    """Preserve nonzero BCs and distinct assembled preconditioning forms."""
    from firedrake import dmhooks

    mesh = UnitIntervalMesh(5)
    x, = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V

    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)
    du, dp = TrialFunctions(W)
    u_exact, p_exact = 1 + x, 2 - x
    F = ((u + u**3 + 0.1*p - u_exact - u_exact**3 - 0.1*p_exact)*v
         + (p + p**3 + 0.1*u - p_exact - p_exact**3 - 0.1*u_exact)*q)*dx
    Jp = ((1 + 3*u**2)*du*v + (1 + 3*p**2)*dp*q)*dx
    bcs = [DirichletBC(W.sub(0), u_exact, "on_boundary"),
           DirichletBC(W.sub(1), p_exact, "on_boundary")]

    params = {
        'snes_type': 'fieldsplit',
        'snes_fieldsplit_type': 'multiplicative',
        'snes_rtol': 1e-10,
        'mat_type': mat_type,
        'pmat_type': 'aij',
        'fieldsplit_ksp_type': 'gmres',
        'fieldsplit_ksp_rtol': 1e-12,
        'fieldsplit_pc_type': 'lu',
    }
    problem = NonlinearVariationalProblem(F, w, bcs=bcs, Jp=Jp)
    solver = NonlinearVariationalSolver(
        problem, pre_apply_bcs=pre_apply_bcs, solver_parameters=params)

    solver.solve()
    solver.snes.reset()
    w.zero()
    solver.solve()
    u, p = w.subfunctions
    assert errornorm(u_exact, u) < 1e-9
    assert errornorm(p_exact, p) < 1e-9
    for child in solver.snes.getFieldSplitSubSNES():
        with solver.inserted_options(), \
                dmhooks.add_hooks(solver.snes.dm, solver, appctx=solver._ctx):
            ctx = dmhooks.get_appctx(child.dm)
        assert ctx.Jp is not None
        assert ctx.mat_type == mat_type
        assert ctx.pmat_type == 'aij'

        J, P, _ = child.getJacobian()
        assert J == ctx._jac.petscmat
        assert P == ctx._pjac.petscmat


@pytest.mark.parametrize("grouped", [False, True])
def test_fieldsplit_snes_reset_options_invalidation(grouped):
    """Reuse child settings and invalidate their constant Jacobian caches."""
    mesh = UnitIntervalMesh(3)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    w = Function(W)
    forcing = Constant(1)
    coefficient = Constant(1)
    a = coefficient*(u*v + p*q)*dx
    L = forcing*(v + 2*q)*dx
    parameters = {
        'snes_type': 'fieldsplit',
        'fieldsplit_ksp_type': 'preonly',
        'fieldsplit_pc_type': 'lu',
    }
    if grouped:
        parameters.update({
            'snes_fieldsplit_0_fields': '0,1',
            'fieldsplit_ksp_type': 'gmres',
            'fieldsplit_pc_type': 'fieldsplit',
            'fieldsplit_fieldsplit_ksp_type': 'preonly',
            'fieldsplit_fieldsplit_pc_type': 'lu',
        })
    problem = LinearVariationalProblem(a, L, w, constant_jacobian=True)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters)

    original_ctx = solver._ctx
    children = _get_split_solvers(solver)
    children[0].setTolerances(rtol=1e-9)
    solver.solve()
    assert solver._ctx is original_ctx
    assert children[0].getTolerances()[0] == 1e-9
    solver.invalidate_jacobian()
    solver.snes.reset()
    forcing.assign(3)
    coefficient.assign(2)
    solver.solve()
    assert children[0].getTolerances()[0] == 1e-9
    u, p = w.subfunctions
    assert errornorm(Constant(1.5), u) < 1e-12
    assert errornorm(Constant(3), p) < 1e-12


def test_fieldsplit_snes_grouped_linear_fieldsplit():
    """Use a matrix-free grouped equation with a linear field decomposition."""
    mesh = UnitIntervalMesh(3)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V

    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)
    F = ((u + 0.1*p - 1)*v + (p + 0.1*u - 2)*q)*dx
    params = {
        'snes_type': 'fieldsplit',
        'snes_fieldsplit_0_fields': '0,1',
        'mat_type': 'matfree',
        'pmat_type': 'aij',
        'fieldsplit_ksp_type': 'gmres',
        'fieldsplit_pc_type': 'fieldsplit',
        'fieldsplit_fieldsplit_ksp_type': 'preonly',
        'fieldsplit_fieldsplit_pc_type': 'lu',
    }
    problem = NonlinearVariationalProblem(F, w)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)

    solver.solve()
    u, p = w.subfunctions
    assert errornorm(Constant(0.8/0.99), u) < 1e-9
    assert errornorm(Constant(1.9/0.99), p) < 1e-9


def test_fieldsplit_snes_child_fd():
    """Keep a child's finite-difference Jacobian local to its owned DM."""
    from firedrake import dmhooks

    mesh = UnitIntervalMesh(3)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V

    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)
    F = ((u + 0.1*p - 1)*v + (p + 0.1*u - 2)*q)*dx
    params = {
        'snes_type': 'fieldsplit',
        'fieldsplit_ksp_type': 'preonly',
        'fieldsplit_pc_type': 'lu',
        'fieldsplit_0_snes_fd': None,
    }
    problem = NonlinearVariationalProblem(F, w)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)

    solver.solve()
    solver.snes.reset()
    w.zero()
    solver.solve()
    u, p = w.subfunctions
    assert errornorm(Constant(0.8/0.99), u) < 1e-8
    assert solver.snes.dm != solver.snes.getFieldSplitSubSNES()[0].dm
    with solver.inserted_options(), dmhooks.add_hooks(solver.snes.dm, solver, appctx=solver._ctx):
        J, P, _ = solver.snes.getJacobian()
        solver.snes.computeJacobian(solver.snes.getSolution(), J, P)
    expected = assemble(derivative(F, w))
    difference = solver.snes.getJacobian()[0].copy()
    difference.axpy(-1, expected.petscmat)
    assert difference.norm() < 1e-12


def test_fieldsplit_snes_scalar_nullspace():
    """Bind a scalar equation and retain its full-space nullspace."""
    mesh = UnitIntervalMesh(8)
    x, = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    exact = Function(V).interpolate(cos(2*pi*x))
    v = TestFunction(V)
    F = inner(grad(u - exact), grad(v))*dx
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    params = {
        'snes_type': 'fieldsplit',
        'fieldsplit_ksp_type': 'cg',
        'fieldsplit_pc_type': 'none',
    }
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, nullspace=nullspace,
                                        solver_parameters=params)

    solver.solve()
    assert assemble(inner(grad(u - exact), grad(u - exact))*dx) < 1e-20


@pytest.mark.parallel([1, 2])
def test_fieldsplit_snes_single_field():
    """Solve a nonlinear equation on an unmixed space with one split."""
    mesh = UnitIntervalMesh(8)
    x, = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    exact = Function(V).interpolate(1 + x)
    u = Function(V)
    v = TestFunction(V)
    F = inner(u**3 - exact**3, v)*dx
    params = {
        'snes_type': 'fieldsplit',
        'snes_fieldsplit_0_fields': '0',
        'snes_rtol': 1e-12,
        'fieldsplit_0_snes_type': 'ksponly',
        'fieldsplit_0_ksp_type': 'preonly',
        'fieldsplit_0_pc_type': 'lu',
    }
    problem = NonlinearVariationalProblem(F, u)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)

    u.assign(1)
    solver.solve()
    assert errornorm(exact, u) < 1e-10


def test_fieldsplit_snes_objective():
    """Evaluate child objectives with the other field fixed at its base value."""
    mesh = UnitIntervalMesh(3)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V

    w = Function(W)
    u, p = w.subfunctions
    u.assign(1)
    p.assign(2)
    u, p = split(w)
    v, q = TestFunctions(W)
    F = ((u + 0.1*p - 1)*v + (p + 0.1*u - 2)*q)*dx
    E = (0.5*(u**2 + p**2) + 0.1*u*p - u - 2*p)*dx
    params = {
        'snes_type': 'fieldsplit',
        'snes_max_it': 1,
        'snes_convergence_test': 'skip',
        'fieldsplit_ksp_type': 'preonly',
        'fieldsplit_pc_type': 'lu',
    }
    problem = NonlinearVariationalProblem(F, w, objective=E)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)

    values = []

    def monitor(snes, iteration, rnorm):
        if iteration == 0:
            values.append(snes.computeObjective(snes.getSolution()))

    for child in _get_split_solvers(solver):
        child.setMonitor(monitor)
    solver.solve()
    assert values == pytest.approx([-2.3, -2.3])
    u, p = w.subfunctions
    assert errornorm(Constant(0.8), u) < 1e-12
    assert errornorm(Constant(1.9), p) < 1e-12


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("fs_type", ["additive", "multiplicative"])
def test_fieldsplit_snes_independent_jacobian(fs_type):
    """Evaluate child Jacobians at new trials with refreshed frozen fields."""
    mesh = UnitIntervalMesh(3)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V
    w = Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)
    F = (u**3*(1 + p**2) - 20)*v*dx + (p**3*(1 + u**2) - 50)*q*dx
    problem = NonlinearVariationalProblem(F, w)
    solver = NonlinearVariationalSolver(problem, solver_parameters={
        'snes_type': 'fieldsplit',
        'snes_fieldsplit_type': fs_type,
        'snes_max_it': 1,
        'snes_convergence_test': 'skip',
        'fieldsplit_snes_rtol': 1e-12,
        'fieldsplit_ksp_type': 'preonly',
        'fieldsplit_pc_type': 'lu',
    })
    children = _get_split_solvers(solver)

    for initial in [(1, 2), (2, 3)]:
        first = (20/(1 + initial[1]**2))**(1 / 3)
        frozen = [initial[1], initial[0] if fs_type == 'additive' else first]
        checked = []

        def monitor(child, iteration, rnorm):
            if iteration != 0:
                return
            field = children.index(child)
            J, P, _ = child.getJacobian()
            trial = child.getSolution().duplicate()
            for value in [1.5, 2.5]:
                trial.set(value)
                child.computeJacobian(trial, J, P)
                expected = assemble(3*value**2*(1 + frozen[field]**2)
                                    * TrialFunction(V)*TestFunction(V)*dx)
                difference = J.copy()
                difference.axpy(-1, expected.petscmat)
                assert difference.norm() < 1e-10
            checked.append(field)

        for child in children:
            child.monitorCancel()
            child.setMonitor(monitor)
        for wi, value in zip(w.subfunctions, initial):
            wi.assign(value)
        solver.solve()
        assert checked == [0, 1]
        assert errornorm(Constant(first), w.subfunctions[0]) < 1e-10
        second = (50/(1 + frozen[1]**2))**(1 / 3)
        assert errornorm(Constant(second), w.subfunctions[1]) < 1e-10
