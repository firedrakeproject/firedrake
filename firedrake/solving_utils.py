from __future__ import absolute_import
import ufl

from firedrake import function
from firedrake.petsc import PETSc


class ParametersMixin(object):
    count = 0

    """Mixin class that helps with managing setting petsc options on solvers.

    :arg parameters: The dictionary of parameters to use."""
    def __init__(self, parameters, options_prefix):
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters.copy()
        self.options_prefix = options_prefix
        # Remember the user prefix, so we can see what's happening
        prefix = "firedrake_%d_" % ParametersMixin.count
        if options_prefix is not None:
            prefix = options_prefix + prefix
        self._prefix = prefix
        ParametersMixin.count += 1
        self._setfromoptions = False
        self.update_parameters_from_options()
        super(ParametersMixin, self).__init__()

    def update_parameters_from_options(self):
        """Update the parameters with any matching values in the petsc
        options database.

        :arg options_prefix: The prefix to use to look up values.  If
           ``None``, no values will match.

        This is used for pull options from the commandline."""
        if self.options_prefix is None:
            return
        opts = PETSc.Options(self.options_prefix)
        for k, v in opts.getAll().iteritems():
            # Copy appropriately prefixed options to parameters
            self.parameters[k] = v

    def set_from_options(self, petsc_obj):
        """Set up petsc_obj from the options database.

        :arg petsc_obj: The PETSc object to call setFromOptions on.

        Matt says: "Only ever call setFromOptions once".  This
        function ensures we do so.
        """
        if not self._setfromoptions:
            petsc_obj.setOptionsPrefix(self._prefix)
            # Call setfromoptions inserting appropriate options into
            # the options database.
            opts = PETSc.Options(self._prefix)
            for k, v in self.parameters.iteritems():
                if type(v) is bool:
                    if v:
                        opts[k] = None
                else:
                    opts[k] = v
            petsc_obj.setFromOptions()
            self._setfromoptions = True

    def clear_options(self):
        """Clear the auto-generated options from the options database.

        This is necessary to ensure the options database doesn't overflow."""
        if hasattr(self, "_prefix") and hasattr(self, "parameters"):
            prefix = self._prefix
            opts = PETSc.Options()
            for k in self.parameters.iterkeys():
                opts.delValue(prefix + k)
            delattr(self, "_prefix")
            delattr(self, "parameters")

    def __del__(self):
        self.clear_options()


def _make_reasons(reasons):
    return dict([(getattr(reasons, r), r)
                 for r in dir(reasons) if not r.startswith('_')])


KSPReasons = _make_reasons(PETSc.KSP.ConvergedReason())


SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())


def check_snes_convergence(snes):
    r = snes.getConvergedReason()
    try:
        reason = SNESReasons[r]
        inner = False
    except KeyError:
        r = snes.getKSP().getConvergedReason()
        try:
            inner = True
            reason = KSPReasons[r]
        except KeyError:
            reason = "unknown reason (petsc4py enum incomplete?), try with -snes_convered_reason and -ksp_converged_reason"
    if r < 0:
        if inner:
            msg = "Inner linear solve failed to converge after %d iterations with reason: %s" % \
                  (snes.getKSP().getIterationNumber(), reason)
        else:
            msg = reason
        raise RuntimeError("""Nonlinear solve failed to converge after %d nonlinear iterations.
Reason:
   %s""" % (snes.getIterationNumber(), msg))


class _SNESContext(object):
    """
    Context holding information for SNES callbacks.

    :arg problem: a :class:`NonlinearVariationalProblem`.
    :arg mat_type: Indicates whether the Jacobian is assembled
        monolithically ('aij'), as a block sparse matrix ('nest') or
        matrix-free (as :class:`~.ImplicitMatrix`\es, 'matfree').
    :arg pmat_type: Indicates whether the preconditioner (if present) is assembled
        monolithically ('aij'), as a block sparse matrix ('nest') or
        matrix-free (as :class:`~.ImplicitMatrix`\es, 'matfree').
    :arg appctx: Any extra information used in the assembler.  For the
        matrix-free case this will contain the Newton state in
        ``"state"``.

    The idea here is that the SNES holds a shell DM which contains
    this object as "user context".  When the SNES calls back to the
    user form_function code, we pull the DM out of the SNES and then
    get the context (which is one of these objects) to find the
    Firedrake level information.
    """
    def __init__(self, problem, mat_type, pmat_type, appctx=None):
        if pmat_type is None:
            pmat_type = mat_type
        self.mat_type = mat_type
        self.pmat_type = pmat_type

        matfree = mat_type == 'matfree'
        pmatfree = pmat_type == 'matfree'

        self._problem = problem
        # Build the jacobian with the correct sparsity pattern.  Note
        # that since matrix assembly is lazy this doesn't actually
        # force an additional assembly of the matrix since in
        # form_jacobian we call assemble again which drops this
        # computation on the floor.
        from firedrake.assemble import assemble
        # Function to hold current guess
        self._x = function.Function(problem.u.function_space())

        if appctx is None:
            appctx = {}

        if matfree or pmatfree:
            appctx["state"] = self._x

        self.appctx = appctx
        self.matfree = matfree
        self.pmatfree = pmatfree
        self.F = ufl.replace(problem.F, {problem.u: self._x})
        self.J = ufl.replace(problem.J, {problem.u: self._x})
        self._jac = assemble(self.J, bcs=problem.bcs,
                             form_compiler_parameters=problem.form_compiler_parameters,
                             mat_type=mat_type,
                             appctx=appctx)
        self.is_mixed = self._jac.block_shape != (1, 1)

        if mat_type != pmat_type or problem.Jp is not None:
            # Need separate pmat if either Jp is different or we want
            # a different pmat type to the mat type.
            if problem.Jp is None:
                self.Jp = self.J
            else:
                self.Jp = ufl.replace(problem.Jp, {problem.u: self._x})
            self._pjac = assemble(self.Jp, bcs=problem.bcs,
                                  form_compiler_parameters=problem.form_compiler_parameters,
                                  mat_type=pmat_type,
                                  appctx=appctx)
        else:
            # pmat_type == mat_type and Jp is None
            self.Jp = None
            self._pjac = self._jac

        self._F = function.Function(self.F.arguments()[0].function_space())
        self._jacobian_assembled = False

    def set_function(self, snes):
        """Set the residual evaluation function"""
        with self._F.dat.vec as v:
            snes.setFunction(self.form_function, v)

    def set_jacobian(self, snes):
        snes.setJacobian(self.form_jacobian, J=self._jac.petscmat,
                         P=self._pjac.petscmat)

    def set_nullspace(self, nullspace, ises=None, transpose=False):
        if nullspace is None:
            return
        nullspace._apply(self._jac, transpose=transpose)
        if self.Jp is not None:
            nullspace._apply(self._pjac, transpose=transpose)
        if ises is not None:
            nullspace._apply(ises, transpose=transpose)

    @staticmethod
    def create_matrix(dm):
        ctx = dm.getAppCtx()
        return ctx._jac.petscmat

    @staticmethod
    def form_function(snes, X, F):
        """Form the residual for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg F: the residual at X (a Vec)
        """
        from firedrake.assemble import assemble

        dm = snes.getDM()
        ctx = dm.getAppCtx()
        problem = ctx._problem
        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec as v:
            X.copy(v)

        assemble(ctx.F, tensor=ctx._F,
                 form_compiler_parameters=problem.form_compiler_parameters)
        # no mat_type -- it's a vector!
        for bc in problem.bcs:
            bc.zero(ctx._F)

        # F may not be the same vector as self._F, so copy
        # residual out to F.
        with ctx._F.dat.vec_ro as v:
            v.copy(F)

    @staticmethod
    def form_jacobian(snes, X, J, P):
        """Form the Jacobian for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake.assemble import assemble

        dm = snes.getDM()
        ctx = dm.getAppCtx()
        problem = ctx._problem
        if problem._constant_jacobian and ctx._jacobian_assembled:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        ctx._jacobian_assembled = True

        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec as v:
            X.copy(v)
        assemble(ctx.J,
                 tensor=ctx._jac,
                 bcs=problem.bcs,
                 form_compiler_parameters=problem.form_compiler_parameters,
                 mat_type=ctx.mat_type)
        ctx._jac.force_evaluation()
        if ctx.Jp is not None:
            assemble(ctx.Jp,
                     tensor=ctx._pjac,
                     bcs=problem.bcs,
                     form_compiler_parameters=problem.form_compiler_parameters,
                     mat_type=ctx.pmat_type)
            ctx._pjac.force_evaluation()
