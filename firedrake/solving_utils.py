from __future__ import absolute_import, print_function, division
from six import iteritems
import numpy
import itertools
from contextlib import contextmanager

from firedrake import function, dmhooks
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import PETSc
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.logging import warning


def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in iteritems(parameters):
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            warning("Ignoring duplicate option: %s (existing value %s, new value %s)",
                    option, new[option], value)
        new[option] = value
    return new


class ParametersMixin(object):

    # What appeared on the commandline, we should never clear these.
    # They will override options passed in as a dict if an
    # options_prefix was supplied.
    commandline_options = frozenset(PETSc.Options().getAll())

    options_object = PETSc.Options()

    count = itertools.count()

    """Mixin class that helps with managing setting petsc options on solvers.

    :arg parameters: The dictionary of parameters to use.
    :arg options_prefix: The prefix to look up items in the global
        options database (may be ``None``, in which case only entries
        from ``parameters`` will be considered.  If no trailing
        underscore is provided, one is appended.  Hence ``foo_`` and
        ``foo`` are treated equivalently.  As an exception, if the
        prefix is the empty string, no underscore is appended.

    To use this, you must call its constructor to with the parameters
    you want in the options database.

    You then call :meth:`set_from_options`, passing the PETSc object
    you'd like to call ``setFromOptions`` on.  Note that this will
    actually only call ``setFromOptions`` the first time (so really
    this parameters object is a once-per-PETSc-object thing).

    So that the runtime monitors which look in the options database
    actually see options, you need to ensure that the options database
    is populated at the time of a ``SNESSolve`` or ``KSPSolve`` call.
    Do that using the :meth:`inserted_options` context manager.

    .. code-block:: python

       with self.inserted_options():
           self.snes.solve(...)

    This ensures that the options database has the relevant entries
    for the duration of the ``with`` block, before removing them
    afterwards.  This is a much more robust way of dealing with the
    fixed-size options database than trying to clear it out using
    destructors.
    """
    def __init__(self, parameters, options_prefix):
        if parameters is None:
            parameters = {}
        else:
            # Convert nested dicts
            parameters = flatten_parameters(parameters)
        if options_prefix is None:
            self.options_prefix = "firedrake_%d_" % next(self.count)
            self.parameters = parameters
            self.to_delete = set(parameters)
        else:
            if len(options_prefix) and not options_prefix.endswith("_"):
                options_prefix += "_"
            self.options_prefix = options_prefix
            # Remove those options from the dict that were passed on
            # the commandline.
            self.parameters = {k: v for k, v in iteritems(parameters)
                               if options_prefix + k not in self.commandline_options}
            self.to_delete = set(self.parameters)
            # Now update parameters from options, so that they're
            # available to solver setup (for, e.g., matrix-free).
            # Can't ask for the prefixed guy in the options object,
            # since that does not DTRT for flag options.
            for k, v in iteritems(self.options_object.getAll()):
                if k.startswith(self.options_prefix):
                    self.parameters[k[len(self.options_prefix):]] = v
        self._setfromoptions = False
        super(ParametersMixin, self).__init__()

    def set_default_parameter(self, key, val):
        """Set a default parameter value.

        :arg key: The parameter name
        :arg val: The parameter value.

        Ensures that the right thing happens cleaning up the options
        database.
        """
        k = self.options_prefix + key
        if k not in self.options_object and key not in self.parameters:
            self.parameters[key] = val
            self.to_delete.add(key)

    def set_from_options(self, petsc_obj):
        """Set up petsc_obj from the options database.

        :arg petsc_obj: The PETSc object to call setFromOptions on.

        Matt says: "Only ever call setFromOptions once".  This
        function ensures we do so.
        """
        if not self._setfromoptions:
            with self.inserted_options():
                petsc_obj.setOptionsPrefix(self.options_prefix)
                # Call setfromoptions inserting appropriate options into
                # the options database.
                petsc_obj.setFromOptions()
                self._setfromoptions = True

    @contextmanager
    def inserted_options(self):
        """Context manager inside which the petsc options database
    contains the parameters from this object."""
        try:
            for k, v in iteritems(self.parameters):
                key = self.options_prefix + k
                if type(v) is bool:
                    if v:
                        self.options_object[key] = None
                else:
                    self.options_object[key] = v
            yield
        finally:
            for k in self.to_delete:
                del self.options_object[self.options_prefix + k]


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
            reason = "unknown reason (petsc4py enum incomplete?), try with -snes_converged_reason and -ksp_converged_reason"
    if r < 0:
        if inner:
            msg = "Inner linear solve failed to converge after %d iterations with reason: %s" % \
                  (snes.getKSP().getIterationNumber(), reason)
        else:
            msg = reason
        raise ConvergenceError("""Nonlinear solve failed to converge after %d nonlinear iterations.
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
    :arg pre_jacobian_callback: User-defined function called immediately
        before Jacobian assembly
    :arg pre_function_callback: User-defined function called immediately
        before residual assembly

    The idea here is that the SNES holds a shell DM which contains
    this object as "user context".  When the SNES calls back to the
    user form_function code, we pull the DM out of the SNES and then
    get the context (which is one of these objects) to find the
    Firedrake level information.
    """
    def __init__(self, problem, mat_type, pmat_type, appctx=None, pre_jacobian_callback=None, pre_function_callback=None):
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        if pmat_type is None:
            pmat_type = mat_type
        self.mat_type = mat_type
        self.pmat_type = pmat_type

        matfree = mat_type == 'matfree'
        pmatfree = pmat_type == 'matfree'

        self._problem = problem
        self._pre_jacobian_callback = pre_jacobian_callback
        self._pre_function_callback = pre_function_callback

        fcp = problem.form_compiler_parameters
        # Function to hold current guess
        self._x = problem.u

        if appctx is None:
            appctx = {}

        if matfree or pmatfree:
            # A split context will already get the full state.
            # TODO, a better way of doing this.
            # Now we don't have a temporary state inside the snes
            # context we could just require the user to pass in the
            # full state on the outside.
            appctx.setdefault("state", self._x)

        self.appctx = appctx
        self.matfree = matfree
        self.pmatfree = pmatfree
        self.F = problem.F
        self.J = problem.J

        self._jac = allocate_matrix(self.J, bcs=problem.bcs,
                                    form_compiler_parameters=fcp,
                                    mat_type=mat_type,
                                    appctx=appctx)
        self._assemble_jac = create_assembly_callable(self.J,
                                                      tensor=self._jac,
                                                      bcs=problem.bcs,
                                                      form_compiler_parameters=fcp,
                                                      mat_type=mat_type)

        self.is_mixed = self._jac.block_shape != (1, 1)

        if mat_type != pmat_type or problem.Jp is not None:
            # Need separate pmat if either Jp is different or we want
            # a different pmat type to the mat type.
            if problem.Jp is None:
                self.Jp = self.J
            else:
                self.Jp = problem.Jp
            self._pjac = allocate_matrix(self.Jp, bcs=problem.bcs,
                                         form_compiler_parameters=fcp,
                                         mat_type=pmat_type,
                                         appctx=appctx)

            self._assemble_pjac = create_assembly_callable(self.Jp,
                                                           tensor=self._pjac,
                                                           bcs=problem.bcs,
                                                           form_compiler_parameters=fcp,
                                                           mat_type=pmat_type)
        else:
            # pmat_type == mat_type and Jp is None
            self.Jp = None
            self._pjac = self._jac

        self._F = function.Function(self.F.arguments()[0].function_space())
        self._assemble_residual = create_assembly_callable(self.F,
                                                           tensor=self._F,
                                                           form_compiler_parameters=fcp)

        self._jacobian_assembled = False
        self._splits = {}
        self._coarse = None
        self._fine = None

    def set_function(self, snes):
        """Set the residual evaluation function"""
        with self._F.dat.vec_wo as v:
            snes.setFunction(self.form_function, v)

    def set_jacobian(self, snes):
        snes.setJacobian(self.form_jacobian, J=self._jac.petscmat,
                         P=self._pjac.petscmat)

    def set_nullspace(self, nullspace, ises=None, transpose=False, near=False):
        if nullspace is None:
            return
        nullspace._apply(self._jac, transpose=transpose, near=near)
        if self.Jp is not None:
            nullspace._apply(self._pjac, transpose=transpose, near=near)
        if ises is not None:
            nullspace._apply(ises, transpose=transpose, near=near)

    def split(self, fields):
        from ufl import as_vector, replace
        from firedrake import NonlinearVariationalProblem as NLVP, FunctionSpace
        splits = self._splits.get(tuple(fields))
        if splits is not None:
            return splits

        splits = []
        problem = self._problem
        splitter = ExtractSubBlock()
        for field in fields:
            try:
                if len(field) > 1:
                    raise NotImplementedError("Can't split into subblock")
            except TypeError:
                # Just a single field, we can handle that
                pass
            F = splitter.split(problem.F, argument_indices=(field, ))
            J = splitter.split(problem.J, argument_indices=(field, field))
            us = problem.u.split()
            subu = us[field]
            vec = []
            for i, u in enumerate(us):
                for idx in numpy.ndindex(u.ufl_shape):
                    vec.append(u[idx])
            u = as_vector(vec)
            F = replace(F, {problem.u: u})
            J = replace(J, {problem.u: u})
            if problem.Jp is not None:
                Jp = splitter.split(problem.Jp, argument_indices=(field, field))
                Jp = replace(Jp, {problem.u: u})
            else:
                Jp = None
            bcs = []
            for bc in problem.bcs:
                if bc.function_space().index == field:
                    V = FunctionSpace(subu.ufl_domain(), subu.ufl_element())
                    bcs.append(type(bc)(V,
                                        bc.function_arg,
                                        bc.sub_domain,
                                        method=bc.method))
            new_problem = NLVP(F, subu, bcs=bcs, J=J, Jp=None,
                               form_compiler_parameters=problem.form_compiler_parameters)
            new_problem._constant_jacobian = problem._constant_jacobian
            splits.append(type(self)(new_problem, mat_type=self.mat_type, pmat_type=self.pmat_type,
                                     appctx=self.appctx))
        return self._splits.setdefault(tuple(fields), splits)

    @staticmethod
    def form_function(snes, X, F):
        """Form the residual for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg F: the residual at X (a Vec)
        """
        dm = snes.getDM()
        ctx = dmhooks.get_appctx(dm)
        problem = ctx._problem
        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec_wo as v:
            X.copy(v)

        if ctx._pre_function_callback is not None:
            ctx._pre_function_callback(X)

        ctx._assemble_residual()

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
        dm = snes.getDM()
        ctx = dmhooks.get_appctx(dm)
        problem = ctx._problem

        assert J.handle == ctx._jac.petscmat.handle
        if problem._constant_jacobian and ctx._jacobian_assembled:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        ctx._jacobian_assembled = True

        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec_wo as v:
            X.copy(v)

        if ctx._pre_jacobian_callback is not None:
            ctx._pre_jacobian_callback(X)

        ctx._assemble_jac()
        ctx._jac.force_evaluation()
        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac()
            ctx._pjac.force_evaluation()

    @staticmethod
    def compute_operators(ksp, J, P):
        """Form the Jacobian for this problem

        :arg ksp: a PETSc KSP object
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake import inject
        dm = ksp.getDM()
        ctx = dmhooks.get_appctx(dm)
        problem = ctx._problem

        assert J.handle == ctx._jac.petscmat.handle
        if problem._constant_jacobian and ctx._jacobian_assembled:
            # Don't need to do any work with a constant jacobian
            # that's already assembled
            return
        ctx._jacobian_assembled = True

        fine = ctx._fine
        if fine is not None:
            inject(fine._x, ctx._x)
            for bc in ctx._problem.bcs:
                bc.apply(ctx._x)

        ctx._assemble_jac()
        ctx._jac.force_evaluation()
        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac()
            ctx._pjac.force_evaluation()
