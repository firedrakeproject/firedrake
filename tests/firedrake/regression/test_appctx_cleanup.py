import numpy
import pytest
from firedrake import *


class NonePC(PCBase):
    def initialize(self, pc):
        V = dmhooks.get_function_space(pc.getDM())
        self.uh = Function(V)
        u = TrialFunction(V)
        v = TestFunction(V)
        problem = LinearVariationalProblem(inner(u, v) * dx, 2*conj(v)*dx, self.uh)
        self.solver = LinearVariationalSolver(problem)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        x.copy(y)
        self.solver.solve()
        assert numpy.allclose(self.uh.dat.data_ro, 2.0)

    def applyTranspose(self, pc, x, y):
        x.copy(y)


def test_appctx_cleanup():
    mesh = UnitSquareMesh(1, 1)
    mh = MeshHierarchy(mesh, 2)
    mesh = mh[-1]
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v) * dx
    L = conj(v) * dx

    uh = Function(V)

    solve(a == L, uh, solver_parameters={
        "mat_type": "matfree",
        "ksp_type": "cg",
        "pc_type": "mg",
        "mg_levels": {
            "pc_type": "python",
            "pc_python_type": "test_appctx_cleanup.NonePC",
        },
        "mg_coarse_mat_type": "aij",
        "mg_coarse_pc_type": "lu",
    })

    while hasattr(V, "_coarse"):
        assert dmhooks.get_appctx(V.dm) is None
        V = V._coarse

    assert numpy.allclose(uh.dat.data_ro, 1.0)


class Solver(object):
    """Stand-in for the solver object that :class:`~.add_hooks` saves hooks on."""


def test_broken_hook_stack_does_not_mask_error():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    dm = V.dm

    # An empty hook stack under a live context manager is what a solve looks
    # like when the appctx goes missing. The error raised inside the block is
    # the one worth reporting, not the state of the stack.
    with pytest.warns(RuntimeWarning, match="Setup hooks"):
        with pytest.raises(ZeroDivisionError):
            with dmhooks.add_hooks(dm, Solver(), appctx=None):
                dmhooks.pop_attr("__setup_hooks__", dm)
                raise ZeroDivisionError


def test_broken_hook_stack_is_reported():
    mesh = UnitSquareMesh(1, 1)
    V = FunctionSpace(mesh, "CG", 1)
    dm = V.dm

    with pytest.raises(RuntimeError, match="Setup hooks"):
        with dmhooks.add_hooks(dm, Solver(), appctx=None):
            dmhooks.pop_attr("__setup_hooks__", dm)
