import ufl
from itertools import chain
from contextlib import ExitStack

from firedrake import dmhooks
from firedrake import slate
from firedrake import solving_utils
from firedrake import ufl_expr
from firedrake import utils
from firedrake.petsc import PETSc, OptionsManager

__all__ = ["LinearVariationalProblem",
           "LinearVariationalSolver",
           "NonlinearVariationalProblem",
           "NonlinearVariationalSolver"]


class NonlinearVariationalProblem(object):
    r"""Nonlinear variational problem F(u; v) = 0."""

    def __init__(self, F, u, bcs=None, J=None,
                 Jp=None,
                 form_compiler_parameters=None):
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
        """
        from firedrake import solving
        from firedrake import function

        # Store input UFL forms and solution Function
        self.F = F
        self.Jp = Jp
        self.u = u
        self.bcs = solving._extract_bcs(bcs)

        # Argument checking
        if not isinstance(self.F, (ufl.Form, slate.slate.TensorBase)):
            raise TypeError("Provided residual is a '%s', not a Form or Slate Tensor" % type(self.F).__name__)
        if len(self.F.arguments()) != 1:
            raise ValueError("Provided residual is not a linear form")
        if not isinstance(self.u, function.Function):
            raise TypeError("Provided solution is a '%s', not a Function" % type(self.u).__name__)

        # Use the user-provided Jacobian. If none is provided, derive
        # the Jacobian from the residual.
        self.J = J or ufl_expr.derivative(F, u)

        if not isinstance(self.J, (ufl.Form, slate.slate.TensorBase)):
            raise TypeError("Provided Jacobian is a '%s', not a Form or Slate Tensor" % type(self.J).__name__)
        if len(self.J.arguments()) != 2:
            raise ValueError("Provided Jacobian is not a bilinear form")
        if self.Jp is not None and not isinstance(self.Jp, (ufl.Form, slate.slate.TensorBase)):
            raise TypeError("Provided preconditioner is a '%s', not a Form or Slate Tensor" % type(self.Jp).__name__)
        if self.Jp is not None and len(self.Jp.arguments()) != 2:
            raise ValueError("Provided preconditioner is not a bilinear form")

        # Store form compiler parameters
        self.form_compiler_parameters = form_compiler_parameters
        self._constant_jacobian = False

    @utils.cached_property
    def dm(self):
        return self.u.function_space().dm


class NonlinearVariationalSolver(OptionsManager):
    r"""Solves a :class:`NonlinearVariationalProblem`."""

    def __init__(self, problem, **kwargs):
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
        :kwarg pre_function_callback: As above, but called immediately
               before residual assembly

        Example usage of the ``solver_parameters`` option: to set the
        nonlinear solver type to just use a linear solver, use

        .. code-block:: python

            {'snes_type': 'ksponly'}

        PETSc flag options should be specified with `bool` values.
        For example:

        .. code-block:: python

            {'snes_monitor': True}

        To use the ``pre_jacobian_callback`` or ``pre_function_callback``
        functionality, the user-defined function must accept the current
        solution as a petsc4py Vec. Example usage is given below:

        .. code-block:: python

            def update_diffusivity(current_solution):
                with cursol.dat.vec_wo as v:
                    current_solution.copy(v)
                solve(trial*test*dx == dot(grad(cursol), grad(test))*dx, diffusivity)

            solver = NonlinearVariationalSolver(problem,
                                                pre_jacobian_callback=update_diffusivity)

        """
        assert isinstance(problem, NonlinearVariationalProblem)

        parameters = kwargs.get("solver_parameters")
        if "parameters" in kwargs:
            raise TypeError("Use solver_parameters, not parameters")
        nullspace = kwargs.get("nullspace")
        nullspace_T = kwargs.get("transpose_nullspace")
        near_nullspace = kwargs.get("near_nullspace")
        options_prefix = kwargs.get("options_prefix")
        pre_j_callback = kwargs.get("pre_jacobian_callback")
        pre_f_callback = kwargs.get("pre_function_callback")

        super(NonlinearVariationalSolver, self).__init__(parameters, options_prefix)

        # Allow anything, interpret "matfree" as matrix_free.
        mat_type = self.parameters.get("mat_type")
        pmat_type = self.parameters.get("pmat_type")
        matfree = mat_type == "matfree"
        pmatfree = pmat_type == "matfree"

        appctx = kwargs.get("appctx")

        ctx = solving_utils._SNESContext(problem,
                                         mat_type=mat_type,
                                         pmat_type=pmat_type,
                                         appctx=appctx,
                                         pre_jacobian_callback=pre_j_callback,
                                         pre_function_callback=pre_f_callback,
                                         options_prefix=self.options_prefix)

        # No preconditioner by default for matrix-free
        if (problem.Jp is not None and pmatfree) or matfree:
            self.set_default_parameter("pc_type", "none")
        elif ctx.is_mixed:
            # Mixed problem, use jacobi pc if user has not supplied
            # one.
            self.set_default_parameter("pc_type", "jacobi")

        self.snes = PETSc.SNES().create(comm=problem.dm.comm)

        self._problem = problem

        self._ctx = ctx
        self._work = problem.u.dof_dset.layout_vec.duplicate()
        self.snes.setDM(problem.dm)

        ctx.set_function(self.snes)
        ctx.set_jacobian(self.snes)
        ctx.set_nullspace(nullspace, problem.J.arguments()[0].function_space()._ises,
                          transpose=False, near=False)
        ctx.set_nullspace(nullspace_T, problem.J.arguments()[1].function_space()._ises,
                          transpose=True, near=False)
        ctx.set_nullspace(near_nullspace, problem.J.arguments()[0].function_space()._ises,
                          transpose=False, near=True)

        # Set from options now, so that people who want to noodle with
        # the snes object directly (mostly Patrick), can.  We need the
        # DM with an app context in place so that if the DM is active
        # on a subKSP the context is available.
        dm = self.snes.getDM()
        with dmhooks.appctx(dm, self._ctx):
            self.set_from_options(self.snes)

        # Used for custom grid transfer.
        self._transfer_operators = ()
        self._setup = False

    def set_transfer_operators(self, *contextmanagers):
        r"""Set context managers which manages which grid transfer operators should be used.

        :arg contextmanagers: instances of :class:`~.dmhooks.transfer_operators`.
        :raises RuntimeError: if called after calling solve.
        """
        if self._setup:
            raise RuntimeError("Cannot set transfer operators after solve")
        self._transfer_operators = tuple(contextmanagers)

    def solve(self, bounds=None):
        r"""Solve the variational problem.

        :arg bounds: Optional bounds on the solution (lower, upper).
            ``lower`` and ``upper`` must both be
            :class:`~.Function`\s. or :class:`~.Vector`\s.

        .. note::

           If bounds are provided the ``snes_type`` must be set to
           ``vinewtonssls`` or ``vinewtonrsls``.
        """
        # Make sure appcontext is attached to the DM before we solve.
        dm = self.snes.getDM()
        # Apply the boundary conditions to the initial guess.
        for bc in self._problem.bcs:
            bc.apply(self._problem.u)

        if bounds is not None:
            lower, upper = bounds
            with lower.dat.vec_ro as lb, upper.dat.vec_ro as ub:
                self.snes.setVariableBounds(lb, ub)
        work = self._work
        with self._problem.u.dat.vec as u:
            u.copy(work)
            with ExitStack() as stack:
                # Ensure options database has full set of options (so monitors
                # work right)
                for ctx in chain((self.inserted_options(), dmhooks.appctx(dm, self._ctx)),
                                 self._transfer_operators):
                    stack.enter_context(ctx)
                self.snes.solve(None, work)
            work.copy(u)
        self._setup = True
        solving_utils.check_snes_convergence(self.snes)


class LinearVariationalProblem(NonlinearVariationalProblem):
    r"""Linear variational problem a(u, v) = L(v)."""

    def __init__(self, a, L, u, bcs=None, aP=None,
                 form_compiler_parameters=None,
                 constant_jacobian=True):
        r"""
        :param a: the bilinear form
        :param L: the linear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param aP: an optional operator to assemble to precondition
                 the system (if not provided a preconditioner may be
                 computed from ``a``)
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        :param constant_jacobian: (optional) flag indicating that the
                 Jacobian is constant (i.e. does not depend on
                 varying fields).  If your Jacobian can change, set
                 this flag to ``False``.
        """
        # In the linear case, the Jacobian is the equation LHS.
        J = a
        # Jacobian is checked in superclass, but let's check L here.
        if L is 0:
            F = ufl_expr.action(J, u)
        else:
            if not isinstance(L, (ufl.Form, slate.slate.TensorBase)):
                raise TypeError("Provided RHS is a '%s', not a Form or Slate Tensor" % type(L).__name__)
            if len(L.arguments()) != 1:
                raise ValueError("Provided RHS is not a linear form")
            F = ufl_expr.action(J, u) - L

        super(LinearVariationalProblem, self).__init__(F, u, bcs, J, aP,
                                                       form_compiler_parameters=form_compiler_parameters)
        self._constant_jacobian = constant_jacobian


class LinearVariationalSolver(NonlinearVariationalSolver):
    r"""Solves a :class:`LinearVariationalProblem`."""

    def __init__(self, *args, **kwargs):
        r"""
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
        """
        parameters = {}
        parameters.update(kwargs.get("solver_parameters", {}))
        parameters.setdefault('snes_type', 'ksponly')
        parameters.setdefault('ksp_rtol', 1.0e-7)
        kwargs["solver_parameters"] = parameters
        super(LinearVariationalSolver, self).__init__(*args, **kwargs)

    def invalidate_jacobian(self):
        r"""
        Forces the matrix to be reassembled next time it is required.
        """
        self._ctx._jacobian_assembled = False
