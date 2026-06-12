"""Tests that appctx is correctly cloned for forward and adjoint
solver replays during tape evaluation.

When a NonlinearVariationalSolver is created with an appctx dict,
the adjoint machinery builds cached solvers for tape replay. The
cached solvers' form coefficients (F, J) are deep-copied so they can
be independently updated from the tape. The appctx dict must undergo
the same replacement so that preconditioners reading from appctx
(e.g. MassInvPC) see the cloned values that
``CachedSolverBlock.update_dependencies`` keeps in sync with the
tape.
"""
import pytest

from firedrake import *
from firedrake.adjoint import *
from firedrake.adjoint_utils.blocks import CachedSolverBlock
from firedrake.preconditioners.base import PCBase


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


class MuRecorderPC(PCBase):
    """Identity preconditioner that records the appctx 'mu' value at
    every initialize/update. Used to verify appctx liveness during
    replay."""
    mu_log = []

    def initialize(self, pc):
        appctx = self.get_appctx(pc)
        self._mu_ref = appctx["mu"]
        self._record()

    def update(self, pc):
        self._record()

    def _record(self):
        mu = self._mu_ref
        if isinstance(mu, Function):
            val = float(mu.dat.data_ro[0])
        elif isinstance(mu, Constant):
            val = float(mu)
        else:
            val = None
        MuRecorderPC.mu_log.append(val)

    def apply(self, pc, x, y):
        x.copy(y)

    def applyTranspose(self, pc, x, y):
        x.copy(y)


def _setup_problem_with_appctx(solver_params=None):
    """Build a Poisson-like problem where mu appears in both the form
    and the appctx.  Two solves are performed with mu=1 and mu=10 so
    that stale-appctx bugs are detectable.  The functional accumulates
    a contribution after each solve so that both solve blocks carry an
    adjoint input and neither adjoint solve is skipped."""
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    mu = Function(V, name="mu")
    mu.assign(1.0)

    u = Function(V, name="u")
    v = TestFunction(V)

    F = mu * inner(grad(u), grad(v)) * dx - Constant(1.0) * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    if solver_params is None:
        solver_params = {"snes_type": "ksponly",
                         "ksp_type": "cg", "pc_type": "sor"}

    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=solver_params, appctx={"mu": mu})

    control = Function(V, name="control")
    control.assign(1.0)

    mu.assign(control)
    solver.solve()
    J = assemble(u * u * dx)
    mu.assign(10.0 * control)
    solver.solve()
    J += assemble(u * u * dx)
    return J, control, mu


def _get_solve_blocks():
    tape = get_working_tape()
    return [b for b in tape.get_blocks()
            if isinstance(b, CachedSolverBlock)]


@pytest.mark.skipcomplex
def test_appctx_forward_clone_identity():
    """The cached forward solver's appctx['mu'] must be the same object
    as the mu coefficient inside the cached F form, so that
    update_dependencies keeps them in sync during replay."""
    J, control, mu = _setup_problem_with_appctx()

    solve_blocks = _get_solve_blocks()
    assert len(solve_blocks) >= 2

    fwd_solver = solve_blocks[0].forward_cache.solver
    appctx_mu = fwd_solver._ctx.appctx["mu"]
    assert appctx_mu is not mu

    fwd_F_coeffs = fwd_solver._problem.F.coefficients()
    fwd_mu_candidates = [c for c in fwd_F_coeffs if c.name() == "mu"]
    assert len(fwd_mu_candidates) == 1
    fwd_mu = fwd_mu_candidates[0]
    assert fwd_mu is not mu
    assert appctx_mu is fwd_mu


@pytest.mark.skipcomplex
def test_appctx_derived_solvers_identity():
    """The cached adjoint and TLM solvers' appctx['mu'] must be the
    same object as the mu coefficient inside their Jacobians. Both
    forms are derived from the cached forward problem, so this is the
    same cloned coefficient the forward solver uses, and
    update_dependencies keeps it in sync."""
    J, control, mu = _setup_problem_with_appctx()

    Jhat = ReducedFunctional(J, Control(control))
    Jhat.derivative()

    solve_blocks = _get_solve_blocks()
    assert len(solve_blocks) >= 2

    fwd_solver = solve_blocks[0].forward_cache.solver
    fwd_appctx_mu = fwd_solver._ctx.appctx["mu"]

    adj_lvs = solve_blocks[0].adjoint_cache.solver
    adj_appctx = adj_lvs._ctx.appctx
    assert "mu" in adj_appctx
    adj_appctx_mu = adj_appctx["mu"]

    assert adj_appctx_mu is not mu
    # There is a single replace map shared by all the cached solvers,
    # so the adjoint appctx references the forward clone's coefficient.
    assert adj_appctx_mu is fwd_appctx_mu

    adj_J_coeffs = adj_lvs._problem.J.coefficients()
    adj_mu_candidates = [c for c in adj_J_coeffs if c.name() == "mu"]
    assert len(adj_mu_candidates) == 1
    assert adj_appctx_mu is adj_mu_candidates[0]

    # The TLM solver shares the same forward clone.
    tlm_lvs = solve_blocks[0].tangent_cache.solver
    tlm_appctx = tlm_lvs._ctx.appctx
    assert "mu" in tlm_appctx
    assert tlm_appctx["mu"] is fwd_appctx_mu


@pytest.mark.skipcomplex
def test_appctx_forward_replay():
    """During forward tape replay the preconditioner must see the same
    mu values as the original forward run, not the stale end-of-run
    value."""
    MuRecorderPC.mu_log.clear()

    solver_params = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": __name__ + ".MuRecorderPC",
    }
    J, control, mu = _setup_problem_with_appctx(solver_params)
    fwd_log = list(MuRecorderPC.mu_log)

    MuRecorderPC.mu_log.clear()
    Jhat = ReducedFunctional(J, Control(control))
    Jhat(control)

    replay_log = list(MuRecorderPC.mu_log)
    assert fwd_log == replay_log


@pytest.mark.skipcomplex
def test_appctx_adjoint_replay():
    """During the adjoint pass the preconditioner must see the correct
    per-step mu from the forward trajectory, not the stale end-of-run
    value. The adjoint traverses the tape backward, so both mu=1 and
    mu=10 must appear in the log."""
    MuRecorderPC.mu_log.clear()

    solver_params = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": __name__ + ".MuRecorderPC",
    }
    J, control, mu = _setup_problem_with_appctx(solver_params)

    MuRecorderPC.mu_log.clear()
    Jhat = ReducedFunctional(J, Control(control))
    Jhat.derivative()

    adj_log = list(MuRecorderPC.mu_log)
    adj_values = set(round(v, 1) for v in adj_log if v is not None)
    assert 1.0 in adj_values, (
        f"adjoint PC never saw mu=1.0 (log: {adj_log})")
    assert 10.0 in adj_values, (
        f"adjoint PC never saw mu=10.0 (log: {adj_log})")


@pytest.mark.skipcomplex
def test_appctx_taylor_test():
    """Taylor test with a mu-weighted Poisson problem and appctx.
    A converged Krylov solve does not depend on the preconditioner, so
    this cannot detect a stale appctx; it is a sanity check that the
    appctx cloning machinery does not break the gradient."""
    J, control, mu = _setup_problem_with_appctx()

    Jhat = ReducedFunctional(J, Control(control))
    h = Function(control.function_space())
    h.assign(1.0)
    assert taylor_test(Jhat, control, h) > 1.9


@pytest.mark.skipcomplex
def test_appctx_legacy_solve_path():
    """An annotated ``solve(F == 0, u, ..., appctx=...)`` goes through
    GenericSolveBlock, whose adjoint solve uses an assembled matrix and
    cannot consume appctx. It must be dropped there (not passed
    through), otherwise computing the derivative raises."""
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    mu = Function(V, name="mu")
    mu.assign(1.0)

    u = Function(V, name="u")
    v = TestFunction(V)

    F = mu * inner(grad(u), grad(v)) * dx - Constant(1.0) * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    solve(F == 0, u, bcs=bc,
          solver_parameters={"snes_type": "ksponly",
                             "ksp_type": "cg", "pc_type": "sor"},
          appctx={"mu": mu})

    J = assemble(u * u * dx)
    Jhat = ReducedFunctional(J, Control(mu))
    dJ = Jhat.derivative()
    assert dJ.dat.data_ro.size > 0
