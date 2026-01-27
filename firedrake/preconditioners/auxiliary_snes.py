from firedrake.preconditioners.base import SNESBase
from firedrake.petsc import PETSc
from ufl import replace
from firedrake.dmhooks import get_appctx as get_dm_appctx

__all__ = ("AuxiliaryOperatorSNES",)


class AuxiliaryOperatorSNES(SNESBase):
    _prefix = "aux_"

    @PETSc.Log.EventDecorator()
    def initialize(self, snes):
        from firedrake import (  # ImportError if this is at file level
            NonlinearVariationalSolver,
            NonlinearVariationalProblem,
            Function, TestFunction,
            Cofunction)
        from firedrake.assemble import get_assembler

        ctx = get_dm_appctx(snes.dm)
        appctx = self.get_appctx(snes)

        parent_prefix = snes.getOptionsPrefix() or ""
        prefix = parent_prefix + self._prefix

        V = self.get_function_space(snes).collapse()

        # auxiliary form G(k+1)
        test = TestFunction(V)
        uk1 = Function(V)
        uk = Function(V)
        self.uk1 = uk1
        self.uk = uk

        Gk1, bcs = self.form(snes, uk, uk1, test)

        # Solve G(k+1) - b = 0
        # with forcing b = G(k) - F(k)
        Gk = replace(Gk1, {uk1: uk})
        b = Cofunction(V.dual())
        Gk1 -= b

        self.assemble_gk = get_assembler(
            Gk, bcs=bcs,
            form_compiler_parameters=ctx.fcp,
            options_prefix=prefix
        ).assemble

        self.Gk = Gk
        self.Gk1 = Gk1
        self.b = b

        self.solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(
                Gk1, uk1, bcs=bcs,
                form_compiler_parameters=ctx.fcp),
            nullspace=ctx._nullspace,
            transpose_nullspace=ctx._nullspace_T,
            near_nullspace=ctx._near_nullspace,
            appctx=appctx, options_prefix=prefix)

        # indent monitor outputs
        outer_snes = snes
        inner_snes = self.solver.snes
        inner_snes.incrementTabLevel(1, parent=outer_snes)
        inner_snes.ksp.incrementTabLevel(1, parent=outer_snes)
        inner_snes.ksp.pc.incrementTabLevel(1, parent=outer_snes)

    def update(self, snes):
        pass

    @PETSc.Log.EventDecorator()
    def step(self, snes, x, f, y):
        with self.uk.dat.vec_wo as vec:
            x.copy(vec)

        self.assemble_gk(tensor=self.b)

        if f is not None:
            with self.b.dat.vec as vec:
                vec -= f

        self.uk1.assign(self.uk)
        self.solver.solve()

        with self.uk1.dat.vec_ro as vec:
            vec.copy(y)
            y.aypx(-1, x)

    def form(self, snes, state, func, test):
        """Return the preconditioning nonlinear form and boundary conditions.

        Parameters
        ----------
        snes : PETSc.SNES
            The PETSc nonlinear solver object.
        state : firedrake.Function
            The current state of the outer SNES
        func : firedrake.Function
            The solution function.
        test : ufl.TestFunction
            The test function.

        Returns
        -------
        F : ufl.Form
            The preconditioning nonlinear form.
        bcs : DirichletBC[] or None
            The boundary conditions.

        Notes
        -----
        Subclasses may override this function to provide an auxiliary nonlinear
        form. Use `self.get_appctx(obj)` to get the user-supplied
        application-context, if desired.
        """
        ctx = get_dm_appctx(snes.dm)
        u = ctx._x
        form = replace(ctx._problem.F, {u: func})
        bcs = tuple(ctx._problem.bcs)
        return form, bcs

    def view(self, snes, viewer=None):
        super().view(snes, viewer)
        if hasattr(self, "solver"):
            viewer.printfASCII("SNES to apply auxiliary inverse\n")
            self.solver.snes.view(viewer)
