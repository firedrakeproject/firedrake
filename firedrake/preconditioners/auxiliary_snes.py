from firedrake.preconditioners.base import SNESBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx, get_function_space
from firedrake.function import Function, TestFunction

__all__ = ("AuxiliaryOperatorSNES",)


class AuxiliaryOperatorSNES(SNESBase):
    prefix = "aux_"

    @PETSc.Log.EventDecorator()
    def initialize(self, snes):
        from firedrake.variational_solver import (  # ImportError if this is at file level
            NonlinearVariationalSolver, NonlinearVariationalProblem)

        appctx = get_appctx(snes.dm).appctx
        V = get_function_space(snes.dm).collapse()

        fcp = appctx.get("form_compiler_parameters")

        self.u = Function(V)
        v = TestFunction(V)

        F, bcs, J, Jp = self.form(snes, self.u, v)

        prefix = snes.getOptionsPrefix() + self.prefix

        self.solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(
                F, self.u, bcs=bcs, J=J, Jp=Jp,
                form_compiler_parameters=fcp),
            appctx=appctx, options_prefix=prefix)

    def update(self, snes):
        pass

    @PETSc.Log.EventDecorator()
    def step(self, snes, x, f, y):
        with self.u.dat.vec_wo as vec:
            x.copy(vec)
        self.solver.solve()
        with self.u.dat.vec_ro as vec:
            vec.copy(y)
            y.aypx(-1, x)

    def form(self, snes, u, v):
        raise NotImplementedError
