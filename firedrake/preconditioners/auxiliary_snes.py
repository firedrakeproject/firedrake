from firedrake.preconditioners.base import SNESBase
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.ufl_expr import Argument, TestFunction
from firedrake.petsc import PETSc
from ufl import replace
from firedrake.dmhooks import get_function_space, get_appctx as get_dm_appctx

__all__ = ("AuxiliaryOperatorSNES",)


class AuxiliaryOperatorSNES(SNESBase):
    """
    Solve a residual form :math:`F(u) = 0` using a nonlinear Richardson
    iteration preconditioned with an auxiliary form :math:`G`.
    This may be used to create nonlinear preconditioners for
    iterative methods such as Anderson acceleration or NGMRES.

    The `k`-th nonlinearly preconditioned Richardson iteration is:

    .. math ::

        G(u_{k+1}; u_{k}) = G(u_{k}; u_{k}) - F(u_{k})

    where :math:`u_{k}` is the current guess and :math:`u_{k+1}` is the
    next guess to be computed. A solution :math:`u_{*}` of :math:`F(u_{*}) = 0`
    is a fixed point of the Richardson iteration.

    Options for the inner solve for :math:`G(u_{k+1}; u_{k})` are specified
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
                ...
            }
        }

    More details, including how to use this class as a nonlinear
    preconditioner with methods like Anderson acceleration, can
    be found in the manual page on :doc:`preconditioning`.

    Notes
    -----
    If the auxiliary form is linear, i.e. :math:`G(u)=Au`, with
    :math:`A\\approx\\nabla F` approximating the Jacobian of the
    outer residual, then the iteration is an inexact Newton method:

    .. math ::

        u_{k+1} = u_{k} - A^{-1}F(u_{k}).
    """

    _prefix = "aux_"

    @PETSc.Log.EventDecorator()
    def initialize(self, snes):
        from firedrake.variational_solver import (  # circular import at file level
            NonlinearVariationalSolver, NonlinearVariationalProblem)

        ctx = get_dm_appctx(snes.dm)

        parent_prefix = snes.getOptionsPrefix() or ""
        prefix = parent_prefix + self._prefix

        V = get_function_space(snes.getDM()).collapse()

        # buffers for current and next iterates
        u = Function(V)
        u_k = Function(V)
        self.u = u
        self.u_k = u_k

        # auxiliary form G(k+1)
        test = TestFunction(V)
        G, bcs = self.form(snes, u_k, u, test)

        b = Cofunction(V.dual())
        # This is the form we will solve:
        # G(u_{k+1}; u_{k}) - b = 0
        # and we will assemble G(u_{k}; u_{k}) - F(u_{k}) into b.
        Gb = G - b

        self.b = b
        # a buffer for intermediate values when assembling b = Gk - Fk
        self._b_wrk = Cofunction(V.dual())

        problem = NonlinearVariationalProblem(
            Gb, u, bcs=bcs, form_compiler_parameters=ctx.fcp
        )
        self.solver = NonlinearVariationalSolver(
            problem,
            nullspace=ctx._nullspace,
            transpose_nullspace=ctx._nullspace_T,
            near_nullspace=ctx._near_nullspace,
            appctx=ctx.appctx,
            options_prefix=prefix,
            pre_apply_bcs=ctx.pre_apply_bcs,
        )

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
        # x = u_{k} is state at current iteration
        with self.u_k.dat.vec_wo as vec:
            x.copy(vec)
        # initial guess u_{k+1} = u_{k}
        with self.u.dat.vec_wo as vec:
            x.copy(vec)

        # b = F(u_{k})
        with self.b.dat.vec as vec:
            f.copy(vec)

        # At this point we have:
        # u = x = u_{k}, and b = F(u_{k}),
        # so the solver's assembler computes:
        # b_wrk = G(u) - b = G(u_{k}) - F(u_{k})
        # which is exactly the forcing we need.
        self.solver._ctx._assemble_residual(tensor=self._b_wrk)

        # we assign b = G(u_{k}) - F(u_{k}) so now the
        # form in the solver has the correct forcing to
        # calculate u_{k+1} using G(u) - b = 0
        self.b.assign(self._b_wrk)

        self.solver.solve()

        # y = d = u_{k+1} - u_{k}
        with self.u.dat.vec_ro as vec:
            vec.copy(y)
            y.aypx(-1, x)

    @PETSc.Log.EventDecorator()
    def form(self, snes, u_k: Function, u: Function, test: Argument):
        """Return the auxiliary residual form :math:`G(u_{k+1}; u_k)` in
        the Richardson iteration and boundary conditions. Subclasses should
        override this method.

        Defaults to returning a copy of :math:`F(u)`, i.e.
        :math:`G(u_{k+1}; u_k)=F(u_{k+1})`.
        This means that ``AuxiliaryOperatorSNES`` can be used similarly
        to :class:`.AssembledPC`, in that it can be used to specify
        an alternative ``snes_type`` for solving the same residual form
        as the outer ``SNES``.

        Parameters
        ----------
        snes : petsc4py.PETSc.SNES
            The PETSc nonlinear solver object.
        u_k :
            The current iterate :math:`u_{k}`.
        u :
            The next iterate :math:`u_{k+1}` that will be solved for.
        test :
            The test function.

        Returns
        -------
        G : :class:`ufl.BaseForm`
            The preconditioning residual form.
        bcs : Iterable[:class:`~.firedrake.bcs.DirichletBC`] | None
            The boundary conditions.

        Notes
        -----
        :math:`G(u_{k+1}; u_{k})` is parameterised by the current
        iterate :math:`u_{k}`. For example, to use Picard iterations for the
        advection term:

        .. math::

            F(u) = \\ldots + u\\cdot\\nabla u

        in the Navier-Stokes equations, we would use

        .. math::

            G(u_{k+1}; u_{k}) = \\ldots + u_{k}\\cdot\\nabla u_{k+1}.
        """
        ctx = get_dm_appctx(snes.dm)
        form = replace(ctx._problem.F, {ctx._x: u})
        bcs = tuple(ctx._problem.bcs)
        return form, bcs

    def view(self, snes, viewer=None):
        super().view(snes, viewer)
        if hasattr(self, "solver"):
            viewer.printfASCII("SNES to apply a preconditioned Richardson iteration.\n")
            self.solver.snes.view(viewer)
