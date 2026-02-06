import firedrake as fd


def f(u, v, w=0.):
    if not isinstance(w, fd.Constant):
        w = fd.Constant(w)
    return (
        fd.inner(u, v)*fd.dx
        + w*fd.inner(u*u*u, v)*fd.dx
        - fd.inner(fd.Constant(1), v)*fd.dx
    )


class AuxiliaryPolySNES(fd.AuxiliaryOperatorSNES):
    def form(self, snes, state, func, test):
        prefix = (snes.getOptionsPrefix() or "") + f"snes_{self._prefix}"
        w = fd.PETSc.Options().getScalar(prefix + "w")
        return f(func, test, w=w), None


def test_auxiliary_snes():

    mesh = fd.UnitIntervalMesh(1)
    V = fd.FunctionSpace(mesh, "R", 0)

    uk1 = fd.Function(V).zero()
    uk = fd.Function(V).zero()
    v = fd.TestFunction(V)

    w = 1.0
    wg = 0.75

    Fk = f(uk, v, w=w)
    Gk = f(uk, v, w=wg)
    Gk1 = f(uk1, v, w=wg)

    Fp = Gk1 - (Gk - Fk)

    inner_params = {
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_type": "newtonls",
        "snes_rtol": 1e-2,
        "ksp_type": "gmres",
        "pc_type": "none",
    }

    aux_params = {
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_type": "nrichardson",
        "snes_atol": 1e-4,
        "npc_snes_max_its": 1,
        "npc_snes_type": "python",
        "npc_snes_python_type": f"{__name__}.AuxiliaryPolySNES",
        "npc_snes_aux_w": wg,
        "npc_aux": inner_params
    }

    solver_inner = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(Fp, uk1),
        solver_parameters=inner_params,
        options_prefix="Finner")

    solver_aux = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(Fk, uk),
        solver_parameters=aux_params,
        options_prefix="Faux")

    uk1.zero()
    uk.assign(uk1)
    res = []
    with fd.assemble(Fk).dat.vec as rvec:
        res.append(rvec.norm())
    for k in range(5):
        solver_inner.solve()
        uk.assign(uk1)
        with fd.assemble(Fk).dat.vec as rvec:
            res.append(rvec.norm())

    def comparison_monitor(snes, its, rnorm):
        assert abs(rnorm - res[its]) < 1e-14

    solver_aux.snes.setMonitor(comparison_monitor)

    uk.zero()
    solver_aux.solve()
