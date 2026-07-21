import firedrake as fd
from firedrake.utils import single_mode


def f(u, v, w=0.):
    if not isinstance(w, fd.Constant):
        w = fd.Constant(w)
    return (
        fd.inner(u, v)*fd.dx
        + w*fd.inner(u*u*u, v)*fd.dx
        - fd.inner(fd.Constant(1), v)*fd.dx
    )


class AuxiliaryPolynomialSNES(fd.AuxiliaryOperatorSNES):
    def form(self, snes, state, func, test):
        F, bcs = super().form(snes, state, func, test)
        prefix = (snes.getOptionsPrefix() or "") + f"snes_{self._prefix}"
        w = fd.PETSc.Options(prefix).getScalar("w")
        return f(func, test, w=w), bcs


def test_auxiliary_snes():
    """Check that AuxiliaryOperatorSNES is equivalent to a
    hand-rolled preconditioned nonlinear Richardson iteration.
    """

    mesh = fd.UnitIntervalMesh(1)
    V = fd.FunctionSpace(mesh, "R", 0)

    u = fd.Function(V).zero()
    u_k = fd.Function(V).zero()
    v = fd.TestFunction(V)

    w = 1.0
    wg = 0.75

    F_k = f(u_k, v, w=w)
    G_k = f(u_k, v, w=wg)
    G = f(u, v, w=wg)

    Fp = G - (G_k - F_k)

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
        "npc_snes_python_type": f"{__name__}.AuxiliaryPolynomialSNES",
        "npc_snes_aux_w": wg,
        "npc_aux": inner_params
    }

    solver_inner = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(Fp, u),
        solver_parameters=inner_params,
        options_prefix="Finner")

    solver_aux = fd.NonlinearVariationalSolver(
        fd.NonlinearVariationalProblem(F_k, u_k),
        solver_parameters=aux_params,
        options_prefix="Faux")

    # Record the residual at each handrolled richardson iteration
    u.zero()
    u_k.assign(u)
    res = []
    with fd.assemble(F_k).dat.vec as rvec:
        res.append(rvec.norm())
    for k in range(5):
        solver_inner.solve()
        u_k.assign(u)
        with fd.assemble(F_k).dat.vec as rvec:
            res.append(rvec.norm())

    # Custom monitor to compare residuals at each aux operator iteration
    def comparison_monitor(snes, its, rnorm):
        assert abs(rnorm - res[its]) < (1e-6 if single_mode else 1e-14)

    solver_aux.snes.setMonitor(comparison_monitor)

    u_k.zero()
    solver_aux.solve()
