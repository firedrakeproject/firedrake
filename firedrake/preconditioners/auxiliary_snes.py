from firedrake.preconditioners.base import SNESBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx, get_function_space

__all__ = ("AuxiliaryOperatorSNES",)


class AuxiliaryOperatorSNES(SNESBase):
    prefix = "aux_"

    @PETSc.Log.EventDecorator()
    def initialize(self, snes):
        from firedrake import (  # ImportError if this is at file level
            NonlinearVariationalSolver,
            NonlinearVariationalProblem,
            Function, TestFunction, Cofunction)

        ctx = get_appctx(snes.dm)
        V = get_function_space(snes.dm).collapse()

        appctx = ctx.appctx
        fcp = appctx.get("form_compiler_parameters")

        u = Function(V)
        v = TestFunction(V)

        F, bcs, self.u = self.form(snes, u, v)

        self.b = Cofunction(V.dual())
        F += self.b

        prefix = snes.getOptionsPrefix() + self.prefix

        self.solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(
                F, self.u, bcs=bcs,
                form_compiler_parameters=fcp),
            appctx=appctx, options_prefix=prefix)
        outer_snes = snes
        inner_snes = self.solver.snes
        inner_snes.incrementTabLevel(1, parent=outer_snes)
        inner_snes.ksp.incrementTabLevel(1, parent=outer_snes)
        inner_snes.ksp.pc.incrementTabLevel(1, parent=outer_snes)

    def update(self, snes):
        pass

    @PETSc.Log.EventDecorator()
    def step(self, snes, x, f, y):
        from firedrake import errornorm
        with self.u.dat.vec_wo as vec:
            x.copy(vec)
            # PETSc.Sys.Print(f"{x.norm() = }")
        if f is not None:
            with self.b.dat.vec_wo as vec:
                f.copy(vec)
        else:
            self.b.zero()
        # self.b.zero()

        # PETSc.Sys.Print(f"Before: {errornorm(self.un, self.u) =  :.5e}")
        PETSc.Sys.Print(f"Before: {errornorm(self.un1, self.u) = :.5e}")
        self.solver.solve()
        # PETSc.Sys.Print(f"After: {errornorm(self.un, self.u) =  :.5e}")
        PETSc.Sys.Print(f"After: {errornorm(self.un1, self.u) = :.5e}")
        with self.u.dat.vec_ro as vec:
            # PETSc.Sys.Print(f"{vec.norm() = }")
            vec.copy(y)
            y.aypx(-1, x)
            # PETSc.Sys.Print(f"{y.norm() = }")

    def form(self, snes, u, v):
        raise NotImplementedError

    def view(self, snes, viewer=None):
        super().view(snes, viewer)
        if hasattr(self, "solver"):
            viewer.printfASCII("SNES to apply auxiliary inverse\n")
            self.solver.snes.view(viewer)
