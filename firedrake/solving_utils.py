from __future__ import absolute_import
import ufl

from pyop2.utils import as_tuple

from firedrake.mg import utils
from firedrake import function
from firedrake.logging import warning, RED
from firedrake.petsc import PETSc


def update_parameters(obj, petsc_obj):
    """Update parameters on a petsc object

    :arg obj: An object with a parameters dict (mapping to petsc options).
    :arg petsc_obj: The PETSc object to set parameters on."""
    # Skip if parameters haven't changed
    if hasattr(obj, '_set_parameters') and obj.parameters == obj._set_parameters:
        return
    opts = PETSc.Options(obj._opt_prefix)
    for k, v in obj.parameters.iteritems():
        if type(v) is bool:
            if v:
                opts[k] = None
        else:
            opts[k] = v
    petsc_obj.setFromOptions()
    obj._set_parameters = obj.parameters.copy()


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
            reason = 'unknown reason (petsc4py enum incomplete?)'
    if r < 0:
        if inner:
            msg = "Inner linear solve failed to converge after %d iterations with reason: %s" % \
                  (snes.getKSP().getIterationNumber(), reason)
        else:
            msg = reason
        raise RuntimeError("""Nonlinear solve failed to converge after %d nonlinear iterations.
Reason:
   %s""" % (snes.getIterationNumber(), msg))


def _extract_kwargs(**kwargs):
    parameters = kwargs.get('solver_parameters', None)
    if 'parameters' in kwargs:
        warning(RED % "The 'parameters' keyword is deprecated, use 'solver_parameters' instead.")
        parameters = kwargs['parameters']
        if 'solver_parameters' in kwargs:
            warning(RED % "'parameters' and 'solver_parameters' passed, using the latter")
            parameters = kwargs['solver_parameters']

    # Make sure we don't stomp on a dict the user has passed in.
    parameters = parameters.copy() if parameters is not None else {}
    nullspace = kwargs.get('nullspace', None)
    tnullspace = kwargs.get('transpose_nullspace', None)
    options_prefix = kwargs.get('options_prefix', None)

    return parameters, nullspace, tnullspace, options_prefix


class _SNESContext(object):
    """
    Context holding information for SNES callbacks.

    :arg problems: a :class:`NonlinearVariationalProblem` or iterable thereof.

    The idea here is that the SNES holds a shell DM which contains
    this object as "user context".  When the SNES calls back to the
    user form_function code, we pull the DM out of the SNES and then
    get the context (which is one of these objects) to find the
    Firedrake level information.
    """
    def __init__(self, problems, matfree=False, extra_ctx={}):
        problems = as_tuple(problems)
        self._problems = problems
        # Build the jacobian with the correct sparsity pattern.  Note
        # that since matrix assembly is lazy this doesn't actually
        # force an additional assembly of the matrix since in
        # form_jacobian we call assemble again which drops this
        # computation on the floor.
        # Function to hold current guess
        self._xs = tuple(function.Function(problem.u) for problem in problems)
        self.Fs = tuple(ufl.replace(problem.F, {problem.u: x}) for problem, x in zip(problems,
                                                                                     self._xs))
        self.Js = tuple(ufl.replace(problem.J, {problem.u: x}) for problem, x in zip(problems,
                                                                                     self._xs))
        if problems[-1].Jp is not None:
            self.Jps = tuple(ufl.replace(problem.Jp, {problem.u: x}) for problem, x in zip(problems,
                                                                                           self._xs))
        else:
            self.Jps = tuple(None for _ in problems)
        self._Fs = tuple(function.Function(F.arguments()[0].function_space())
                         for F in self.Fs)

        if matfree:
            extra_ctx["state"] = self._xs[-1]

        from firedrake.assemble import assemble
        self._jacs = tuple(assemble(J, bcs=problem.bcs,
                                    form_compiler_parameters=problem.form_compiler_parameters,
                                    nest=problem._nest, matfree=matfree, extra_ctx=extra_ctx)
                           for J, problem in zip(self.Js, problems))
        if problems[-1].Jp is not None:
            self._pjacs = tuple(assemble(Jp, bcs=problem.bcs,
                                         form_compiler_parameters=problem.form_compiler_parameters,
                                         nest=problem._nest, matfree=matfree, extra_ctx=extra_ctx)
                                for Jp, problem in zip(self.Jps, problems))
        else:
            self._pjacs = self._jacs

        self.matfree = matfree
        self._jacobians_assembled = [False for _ in problems]

    def set_function(self, snes):
        """Set the residual evaluation function"""
        with self._Fs[-1].dat.vec as v:
            snes.setFunction(self.form_function, v)

    def set_jacobian(self, snes):
        snes.setJacobian(self.form_jacobian,
                         J=self._jacs[-1].PETScMatHandle,
                         P=self._pjacs[-1].PETScMatHandle)

    def set_nullspace(self, nullspc, ises=None, transpose=False):
        if nullspc is None:
            return
        nullspc._apply(self._jacs[-1], transpose=transpose)
        if self.Jps[-1] is not None:
            nullspc._apply(self._pjacs[-1], transpose=transpose)
        if ises is not None:
            nullspc._apply(ises, transpose=transpose)

    def __len__(self):
        return len(self._problems)

    @classmethod
    def create_matrix(cls, dm):
        ctx = dm.getAppCtx()
        _, lvl = utils.get_level(dm)
        return ctx._jacs[lvl].PETScMatHandle

    @classmethod
    def form_function(cls, snes, X, F):
        """Form the residual for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg F: the residual at X (a Vec)
        """
        from firedrake.assemble import assemble

        dm = snes.getDM()
        ctx = dm.getAppCtx()
        _, lvl = utils.get_level(dm)

        # FIXME: Think about case where DM is refined but we don't
        # have a hierarchy of problems better.
        if len(ctx._problems) == 1:
            lvl = -1
        problem = ctx._problems[lvl]
        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._xs[lvl].dat.vec as v:
            if v != X:
                X.copy(v)

        assemble(ctx.Fs[lvl], tensor=ctx._Fs[lvl],
                 form_compiler_parameters=problem.form_compiler_parameters,
                 nest=problem._nest)
        for bc in problem.bcs:
            bc.zero(ctx._Fs[lvl])

        # F may not be the same vector as self._F, so copy
        # residual out to F.
        with ctx._Fs[lvl].dat.vec_ro as v:
            v.copy(F)

    @classmethod
    def form_jacobian(cls, snes, X, J, P):
        """Form the Jacobian for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake.assemble import assemble

        dm = snes.getDM()
        ctx = dm.getAppCtx()
        _, lvl = utils.get_level(dm)

        # FIXME: Think about case where DM is refined but we don't
        # have a hierarchy of problems better.
        if len(ctx._problems) == 1:
            lvl = -1
        problem = ctx._problems[lvl]
        if problem._constant_jacobian and ctx._jacobians_assembled[lvl]:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        ctx._jacobians_assembled[lvl] = True

        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._xs[lvl].dat.vec as v:
            X.copy(v)

        assemble(ctx.Js[lvl],
                 tensor=ctx._jacs[lvl],
                 bcs=problem.bcs,
                 form_compiler_parameters=problem.form_compiler_parameters,
                 nest=problem._nest,
                 matfree=ctx.matfree)
        ctx._jacs[lvl].force_evaluation()
        if ctx.Jps[lvl] is not None:
            assemble(ctx.Jps[lvl],
                     tensor=ctx._pjacs[lvl],
                     bcs=problem.bcs,
                     form_compiler_parameters=problem.form_compiler_parameters,
                     nest=problem._nest,
                     matfree=ctx.matfree)
            ctx._pjacs[lvl].force_evaluation()
