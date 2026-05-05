from firedrake import *
import pytest


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
