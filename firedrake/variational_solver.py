import ufl
from itertools import chain
from contextlib import ExitStack
from types import MappingProxyType

from firedrake import dmhooks, slate, solving, solving_utils, ufl_expr, utils
from firedrake import function
from firedrake.petsc import (
    PETSc, OptionsManager, flatten_parameters, DEFAULT_KSP_PARAMETERS,
    DEFAULT_SNES_PARAMETERS
)
from firedrake.function import Function
from firedrake.functionspace import RestrictedFunctionSpace
from firedrake.ufl_expr import TrialFunction, TestFunction
from firedrake.bcs import DirichletBC, EquationBC
from firedrake.adjoint_utils import NonlinearVariationalProblemMixin, NonlinearVariationalSolverMixin
from ufl import replace

__all__ = ["LinearVariationalProblem",
           "LinearVariationalSolver",
           "NonlinearVariationalProblem",
           "NonlinearVariationalSolver"]


def check_pde_args(F, J, Jp):
    if not isinstance(F, (ufl.BaseForm, slate.slate.TensorBase)):
        raise TypeError("Provided residual is a '%s', not a BaseForm or Slate Tensor" % type(F).__name__)
    if len(F.arguments()) != 1:
        raise ValueError("Provided residual is not a linear form")
    if not isinstance(J, (ufl.BaseForm, slate.slate.TensorBase)):
        raise TypeError("Provided Jacobian is a '%s', not a BaseForm or Slate Tensor" % type(J).__name__)
    if len(J.arguments()) != 2:
        raise ValueError("Provided Jacobian is not a bilinear form")
    if Jp is not None and not isinstance(Jp, (ufl.BaseForm, slate.slate.TensorBase)):
        raise TypeError("Provided preconditioner is a '%s', not a BaseForm or Slate Tensor" % type(Jp).__name__)
    if Jp is not None and len(Jp.arguments()) != 2:
        raise ValueError("Provided preconditioner is not a bilinear form")


def is_form_consistent(is_linear, bcs):
    # Check form style consistency
    if not (is_linear == all(bc.is_linear for bc in bcs if not isinstance(bc, DirichletBC))
            or not is_linear == all(not bc.is_linear for bc in bcs if not isinstance(bc, DirichletBC))):
        raise TypeError("Form style mismatch: some forms are given in 'F == 0' style, but others are given in 'A == b' style.")


class NonlinearVariationalProblem(NonlinearVariationalProblemMixin):
    r"""Nonlinear variational problem F(u; v) = 0."""

    @PETSc.Log.EventDecorator()
    @NonlinearVariationalProblemMixin._ad_annotate_init
    def __init__(self, F, u, bcs=None, J=None,
                 Jp=None,
                 form_compiler_parameters=None,
                 is_linear=False, restrict=False):
        r"""
        :param F: the nonlinear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = dF/du (optional)
        :param Jp: a form used for preconditioning the linear system,
                 optional, if not supplied then the Jacobian itself
                 will be used.
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        :is_linear: internally used to check if all domain/bc forms
            are given either in 'A == b' style or in 'F == 0' style.
        :param restrict: (optional) If `True`, use restricted function spaces,
            that exclude Dirichlet boundary condition nodes,  internally for
            the test and trial spaces.
        """
        V = u.function_space()
        self.output_space = V
        self.u = u

        if not isinstance(self.u, function.Function):
            raise TypeError("Provided solution is a '%s', not a Function" % type(self.u).__name__)

        # Use the user-provided Jacobian. If none is provided, derive
        # the Jacobian from the residual.
        self.J = J or ufl_expr.derivative(F, u)
        self.F = F
        self.Jp = Jp
        if bcs:
            for bc in bcs:
                if isinstance(bc, EquationBC):
                    restrict = False
        self.restrict = restrict

        if restrict and bcs:
            V_res = RestrictedFunctionSpace(V, boundary_set=set([bc.sub_domain for bc in bcs]))
            bcs = [DirichletBC(V_res, bc.function_arg, bc.sub_domain) for bc in bcs]
            self.u_restrict = Function(V_res).interpolate(u)
            v_res, u_res = TestFunction(V_res), TrialFunction(V_res)
            F_arg, = F.arguments()
            replace_dict = {F_arg: v_res}
            replace_dict[self.u] = self.u_restrict
            self.F = replace(F, replace_dict)
            v_arg, u_arg = self.J.arguments()
            self.J = replace(self.J, {v_arg: v_res, u_arg: u_res})
            if self.Jp:
                v_arg, u_arg = self.Jp.arguments()
                self.Jp = replace(self.Jp, {v_arg: v_res, u_arg: u_res})
            self.restricted_space = V_res
        else:
            self.u_restrict = u

        self.bcs = solving._extract_bcs(bcs)
        # Check form style consistency
        self.is_linear = is_linear
        is_form_consistent(self.is_linear, self.bcs)
        self.Jp_eq_J = Jp is None

        # Argument checking
        check_pde_args(self.F, self.J, self.Jp)

        # Store form compiler parameters
        self.form_compiler_parameters = form_compiler_parameters
        self._constant_jacobian = False

    def dirichlet_bcs(self):
        for bc in self.bcs:
            yield from bc.dirichlet_bcs()

    @utils.cached_property
    def dm(self):
        return self.u_restrict.function_space().dm


class NonlinearVariationalSolver(OptionsManager, NonlinearVariationalSolverMixin):
    r"""Solves a :class:`NonlinearVariationalProblem`."""

    DEFAULT_SNES_PARAMETERS = DEFAULT_SNES_PARAMETERS

    # Looser default tolerance for KSP inside SNES.
    # TODO: When we drop Python 3.8 replace this mess with
    # DEFAULT_KSP_PARAMETERS = MappingProxyType(DEFAULT_KSP_PARAMETERS | {'ksp_rtol': 1e-5})
    DEFAULT_KSP_PARAMETERS = MappingProxyType({
        k: v
        if k != 'ksp_rtol' else 1e-5
        for k, v in DEFAULT_KSP_PARAMETERS.items()
    })

    @PETSc.Log.EventDecorator()
    @NonlinearVariationalSolverMixin._ad_annotate_init
    def __init__(self, problem, *, solver_parameters=None,
                 options_prefix=None,
                 nullspace=None,
                 transpose_nullspace=None,
                 near_nullspace=None,
                 appctx=None,
                 pre_jacobian_callback=None,
                 post_jacobian_callback=None,
                 pre_function_callback=None,
                 post_function_callback=None):
        r"""
        :arg problem: A :class:`NonlinearVariationalProblem` to solve.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        :kwarg transpose_nullspace: as for the nullspace, but used to
               make the right hand side consistent.
        :kwarg near_nullspace: as for the nullspace, but used to
               specify the near nullspace (for multigrid solvers).
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
               This should be a dict mapping PETSc options to values.
        :kwarg appctx: A dictionary containing application context that
               is passed to the preconditioner if matrix-free.
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.
        :kwarg pre_jacobian_callback: A user-defined function that will
               be called immediately before Jacobian assembly. This can
               be used, for example, to update a coefficient function
               that has a complicated dependence on the unknown solution.
        :kwarg post_jacobian_callback: As above, but called after the
               Jacobian has been assembled.
        :kwarg pre_function_callback: As above, but called immediately
               before residual assembly.
        :kwarg post_function_callback: As above, but called immediately
               after residual assembly.

        Example usage of the ``solver_parameters`` option: to set the
        nonlinear solver type to just use a linear solver, use

        .. code-block:: python3

            {'snes_type': 'ksponly'}

        PETSc flag options (where the presence of the option means something) should
        be specified with ``None``.
        For example:

        .. code-block:: python3

            {'snes_monitor': None}

        To use the ``pre_jacobian_callback`` or ``pre_function_callback``
        functionality, the user-defined function must accept the current
        solution as a petsc4py Vec. Example usage is given below:

        .. code-block:: python3

            def update_diffusivity(current_solution):
                with cursol.dat.vec_wo as v:
                    current_solution.copy(v)
                solve(trial*test*dx == dot(grad(cursol), grad(test))*dx, diffusivity)

            solver = NonlinearVariationalSolver(problem,
                                                pre_jacobian_callback=update_diffusivity)

        """
        assert isinstance(problem, NonlinearVariationalProblem)

        solver_parameters = flatten_parameters(solver_parameters or {})
        solver_parameters = solving_utils.set_defaults(solver_parameters,
                                                       problem.J.arguments(),
                                                       ksp_defaults=self.DEFAULT_KSP_PARAMETERS,
                                                       snes_defaults=self.DEFAULT_SNES_PARAMETERS)
        super().__init__(solver_parameters, options_prefix)
        # Now the correct parameters live in self.parameters (via the
        # OptionsManager mixin)
        mat_type = self.parameters.get("mat_type")
        pmat_type = self.parameters.get("pmat_type")
        ctx = solving_utils._SNESContext(problem,
                                         mat_type=mat_type,
                                         pmat_type=pmat_type,
                                         appctx=appctx,
                                         pre_jacobian_callback=pre_jacobian_callback,
                                         pre_function_callback=pre_function_callback,
                                         post_jacobian_callback=post_jacobian_callback,
                                         post_function_callback=post_function_callback,
                                         options_prefix=self.options_prefix)

        self.snes = PETSc.SNES().create(comm=problem.dm.comm)

        self._problem = problem

        self._ctx = ctx
        self._work = problem.u_restrict.dof_dset.layout_vec.duplicate()
        self.snes.setDM(problem.dm)

        ctx.set_function(self.snes)
        ctx.set_jacobian(self.snes)
        ctx.set_nullspace(nullspace, problem.J.arguments()[0].function_space()._ises,
                          transpose=False, near=False)
        ctx.set_nullspace(transpose_nullspace, problem.J.arguments()[1].function_space()._ises,
                          transpose=True, near=False)
        ctx.set_nullspace(near_nullspace, problem.J.arguments()[0].function_space()._ises,
                          transpose=False, near=True)
        ctx._nullspace = nullspace
        ctx._nullspace_T = transpose_nullspace
        ctx._near_nullspace = near_nullspace

        # Set from options now, so that people who want to noodle with
        # the snes object directly (mostly Patrick), can.  We need the
        # DM with an app context in place so that if the DM is active
        # on a subKSP the context is available.
        dm = self.snes.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx, save=False):
            self.set_from_options(self.snes)

        # Used for custom grid transfer.
        self._transfer_operators = ()
        self._setup = False

    def set_transfer_manager(self, manager):
        r"""Set the object that manages transfer between grid levels.
        Typically a :class:`~.TransferManager` object.

        :arg manager: Transfer manager, should conform to the
            TransferManager interface.
        :raises ValueError: if called after the transfer manager is setup.
        """
        self._ctx.transfer_manager = manager

    @PETSc.Log.EventDecorator()
    @NonlinearVariationalSolverMixin._ad_annotate_solve
    def solve(self, bounds=None):
        r"""Solve the variational problem.

        :arg bounds: Optional bounds on the solution (lower, upper).
            ``lower`` and ``upper`` must both be
            :class:`~.Function`\s. or :class:`~.Vector`\s.

        .. note::

           If bounds are provided the ``snes_type`` must be set to
           ``vinewtonssls`` or ``vinewtonrsls``.
        """
        # Make sure the DM has this solver's callback functions
        self._ctx.set_function(self.snes)
        self._ctx.set_jacobian(self.snes)

        # Make sure appcontext is attached to every coefficient DM before we solve.
        problem = self._problem
        forms = (problem.F, problem.J, problem.Jp)
        coefficients = utils.unique(chain.from_iterable(form.coefficients() for form in forms if form is not None))
        # Make sure the solution dm is visited last
        solution_dm = self.snes.getDM()
        problem_dms = [V.dm for V in utils.unique(c.function_space() for c in coefficients) if V.dm != solution_dm]
        problem_dms.append(solution_dm)

        for dbc in problem.dirichlet_bcs():
            dbc.apply(problem.u_restrict)

        if bounds is not None:
            lower, upper = bounds
            with lower.dat.vec_ro as lb, upper.dat.vec_ro as ub:
                self.snes.setVariableBounds(lb, ub)

        work = self._work
        with problem.u_restrict.dat.vec as u:
            u.copy(work)
            with ExitStack() as stack:
                # Ensure options database has full set of options (so monitors
                # work right)
                for ctx in chain([self.inserted_options()],
                                 [dmhooks.add_hooks(dm, self, appctx=self._ctx) for dm in problem_dms],
                                 self._transfer_operators):
                    stack.enter_context(ctx)
                self.snes.solve(None, work)
            work.copy(u)
        self._setup = True
        if problem.restrict:
            problem.u.interpolate(problem.u_restrict)
        solving_utils.check_snes_convergence(self.snes)

        # Grab the comm associated with the `_problem` and call PETSc's garbage cleanup routine
        comm = self._problem.u_restrict.function_space().mesh()._comm
        PETSc.garbage_cleanup(comm=comm)


class LinearVariationalProblem(NonlinearVariationalProblem):
    r"""Linear variational problem a(u, v) = L(v)."""

    @PETSc.Log.EventDecorator()
    def __init__(self, a, L, u, bcs=None, aP=None,
                 form_compiler_parameters=None,
                 constant_jacobian=False, restrict=False):
        r"""
        :param a: the bilinear form
        :param L: the linear form
        :param u: the :class:`.Function` to which the solution will be assigned
        :param bcs: the boundary conditions (optional)
        :param aP: an optional operator to assemble to precondition
                 the system (if not provided a preconditioner may be
                 computed from ``a``)
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        :param constant_jacobian: (optional) flag indicating that the
                 Jacobian is constant (i.e. does not depend on
                 varying fields).  If your Jacobian does not change, set
                 this flag to ``True``.
        :param restrict: (optional) If `True`, use restricted function spaces,
            that exclude Dirichlet boundary condition nodes,  internally for
            the test and trial spaces.
        """
        # In the linear case, the Jacobian is the equation LHS.
        J = a
        # Jacobian is checked in superclass, but let's check L here.
        if not isinstance(L, (ufl.BaseForm, slate.slate.TensorBase)) and L == 0:
            F = ufl_expr.action(J, u)
        else:
            if not isinstance(L, (ufl.BaseForm, slate.slate.TensorBase)):
                raise TypeError("Provided RHS is a '%s', not a Form or Slate Tensor" % type(L).__name__)
            if len(L.arguments()) != 1:
                raise ValueError("Provided RHS is not a linear form")
            F = ufl_expr.action(J, u) - L

        super(LinearVariationalProblem, self).__init__(F, u, bcs, J, aP,
                                                       form_compiler_parameters=form_compiler_parameters,
                                                       is_linear=True, restrict=restrict)
        self._constant_jacobian = constant_jacobian


class LinearVariationalSolver(NonlinearVariationalSolver):
    r"""Solves a :class:`LinearVariationalProblem`.

    :arg problem: A :class:`LinearVariationalProblem` to solve.
    :kwarg solver_parameters: Solver parameters to pass to PETSc.
        This should be a dict mapping PETSc options to values.
    :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
        :class:`.MixedVectorSpaceBasis`) spanning the null
        space of the operator.
    :kwarg transpose_nullspace: as for the nullspace, but used to
        make the right hand side consistent.
    :kwarg options_prefix: an optional prefix used to distinguish
        PETSc options.  If not provided a unique prefix will be
        created.  Use this option if you want to pass options
        to the solver from the command line in addition to
        through the ``solver_parameters`` dict.
    :kwarg appctx: A dictionary containing application context that
        is passed to the preconditioner if matrix-free.
    :kwarg pre_jacobian_callback: A user-defined function that will
           be called immediately before Jacobian assembly. This can
           be used, for example, to update a coefficient function
           that has a complicated dependence on the unknown solution.
    :kwarg post_jacobian_callback: As above, but called after the
           Jacobian has been assembled.
    :kwarg pre_function_callback: As above, but called immediately
           before residual assembly.
    :kwarg post_function_callback: As above, but called immediately
           after residual assembly.

    See also :class:`NonlinearVariationalSolver` for nonlinear problems.
    """

    DEFAULT_SNES_PARAMETERS = {"snes_type": "ksponly"}

    # Tighter default tolerance for KSP only.
    DEFAULT_KSP_PARAMETERS = DEFAULT_KSP_PARAMETERS

    def invalidate_jacobian(self):
        r"""
        Forces the matrix to be reassembled next time it is required.
        """
        self._ctx._jacobian_assembled = False
