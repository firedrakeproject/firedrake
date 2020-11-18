import numpy
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
        "mg_coarse_pc_type": "python",
        "mg_coarse_pc_python_type": "firedrake.AssembledPC",
        "mg_coarse_assembled_pc_type": "lu",
    })

    while hasattr(V, "_coarse"):
        assert dmhooks.get_appctx(V.dm) is None
        V = V._coarse

    assert numpy.allclose(uh.dat.data_ro, 1.0)
