from firedrake import *
from firedrake.mg.ufl_utils import coarsen as symbolic_coarsen
from functools import singledispatch


def test_coarsen_callback():
    mesh = UnitSquareMesh(4, 4)
    mh = MeshHierarchy(mesh, 1)
    mesh = mh[-1]

    V = FunctionSpace(mesh, "CG", 3)

    u = TrialFunction(V)

    v = TestFunction(V)

    a = dot(grad(u), grad(v))*dx + u*v*dx

    L = Constant(1)*v*dx

    @singledispatch
    def coarsen(expr, self, coefficient_mapping=None):
        return symbolic_coarsen(expr, self, coefficient_mapping=coefficient_mapping)

    @coarsen.register(functionspaceimpl.FunctionSpace)
    @coarsen.register(functionspaceimpl.WithGeometry)
    def coarsen_fs(V, self, coefficient_mapping=None):
        mesh = self(V.ufl_domain(), self)
        return FunctionSpace(mesh, "CG", 1)

    uh = Function(V)
    lvp = LinearVariationalProblem(a, L, uh)
    lvs = LinearVariationalSolver(lvp, solver_parameters={"ksp_type": "cg",
                                                          "pc_type": "mg"})
    with dmhooks.ctx_coarsener(V, coarsen):
        lvs.solve()

    Ac, _ = lvs.snes.ksp.pc.getMGCoarseSolve().getOperators()

    assert Ac.getSize() == (25, 25)
