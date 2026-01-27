from firedrake.preconditioners.base import SNESBase
from firedrake.function import Function
from firedrake.ufl_expr import Argument
from firedrake.petsc import PETSc
from ufl import replace
from firedrake.dmhooks import get_appctx as get_dm_appctx

__all__ = ("AuxiliaryOperatorSNES",)


class AuxiliaryOperatorSNES(SNESBase):
    """
    Solve a residual form :math:`F(u) = 0` using a nonlinear Richardson
    iteration preconditioned with an auxiliary form :math:`G(u)`.
    This is usually used to create nonlinear preconditioners for
    iterative methods such as Anderson acceleration or NGMRES.

    The `k`-th nonlinearly preconditioned Richardson iteration is:

    .. math ::

        G(u^{k+1}) = G(u^{k}) - F(u^{k})

    The solution :math:`u^{*}` of :math:`F(u^{*}) = 0` is a fixed point
    of the Richardson iteration.

    .. math ::

        G(u^{k+1}) = G(u^{*}) - F(u^{*})

        G(u^{k+1}) = G(u^{*})

        \\implies u^{k+1} = u^{*}

    Options for the inner solve for :math:`G(u^{k+1})` are specified
    using the ``"aux_"`` prefix.

    The following solver parameters will specify the above Richardson
    iteration, assuming that the user has defined a class
    ``UserAuxiliarySNES`` which inherits from
    :class:`~.AuxiliaryOperatorSNES` and implements the
    :meth:`~.AuxiliaryOperatorSNES.form` method. In this example, the
    inner solve uses a Newton method with a relative tolerance of 1e-4.

    .. code-block:: python3

        solver_parameters = {
            "snes_rtol": 1e-8,
            "snes_type": "python",
            "snes_python_type": f"{__name__}.UserAuxiliarySNES",
            "aux": {
                "snes_rtol": 1e-4,
                "snes_type": "newtonls",
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu"
            }
        }

    The following parameters describe the same Richardson iteration
    as the parameters above, but explicitly specifying the auxiliary
    form as a nonlinear preconditioner using the ``npc_`` prefix.

    .. code-block:: python3

        solver_parameters = {
            "snes_rtol": 1e-8,
            "snes_type": "nrichardson",
            "npc_snes_max_it": 1,
            "npc_snes_type": "python",
            "npc_snes_python_type": f"{__name__}.UserAuxiliarySNES",
            "npc_aux": {
                "snes_rtol": 1e-4,
                "snes_type": "newtonls",
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu"
            }
        }

    Although using ``"npc_"`` to specifying the parameters is more
    verbose than the original, it allows for a wider variety of methods.
    For example, by changing the outer ``"snes_type"`` to ``"anderson"``,
    we can use preconditioned Anderson acceleration
    (`<https://petsc.org/release/manualpages/SNES/SNESANDERSON/>`_)
    """

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

    @PETSc.Log.EventDecorator()
    def form(self, snes, uk: Function, uk1: Function, test: Argument):
        """Return the auxiliary residual form and boundary conditions.
        Subclasses should override this method.

        The returned form is :math:`G(u^{k+1})` in the Richardson
        iteration. The forcing term :math:`G(u^{k})` is generated from
        the user provided :math:`G(u^{k})` using UFL manipulation, and
        the forcing term :math:`F(u^{k})` is provided by the outer SNES.

        Parameters
        ----------
        snes : PETSc.SNES
            The PETSc nonlinear solver object.
        uk :
            The current iterate :math:`u^{k}`.
        uk1 :
            The next iterate :math:`u^{k+1}` that will be solved for.
        test :
            The test function.

        Returns
        -------
        F : :class:`ufl.Form`
            The preconditioning residual form.
        bcs : Iterable[:class:`~.firedrake.bcs.DirichletBC`] | None
            The boundary conditions.

        Notes
        -----
        :math:`G(u^{k+1})` can optionally be parameterised by the current
        iterate :math:`u^{k}`, for example in Picard iterations for an
        advection term:

        .. math::

            F(u) = u + u\\cdot\\nabla u

            G(u^{k+1}) = u^{k+1} + u^{k}\\cdot\\nabla u^{k+1}
        """
        ctx = get_dm_appctx(snes.dm)
        u = ctx._x
        form = replace(ctx._problem.F, {u: uk1})
        bcs = tuple(ctx._problem.bcs)
        return form, bcs

    def view(self, snes, viewer=None):
        super().view(snes, viewer)
        if hasattr(self, "solver"):
            viewer.printfASCII("SNES to apply auxiliary inverse\n")
            self.solver.snes.view(viewer)
