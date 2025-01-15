from itertools import chain

import numpy

from pyop2 import op2
from firedrake import function, cofunction, dmhooks
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import PETSc, DEFAULT_KSP_PARAMETERS
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.utils import cached_property
from firedrake.logging import warning


def _make_reasons(reasons):
    return {getattr(reasons, r): r
            for r in dir(reasons) if not r.startswith('_')}


KSPReasons = _make_reasons(PETSc.KSP.ConvergedReason())
SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())


def set_defaults(solver_parameters, arguments, *, ksp_defaults=None, snes_defaults=None):
    """Set defaults for solver parameters.

    :arg solver_parameters: dict of user solver parameters to override/extend defaults
    :arg arguments: arguments for the bilinear form (need to know if we have a Real block).
    :arg ksp_defaults: Default KSP parameters.
    :arg snes_defaults: Default SNES parameters."""
    if ksp_defaults is None:
        ksp_defaults = {}
    if snes_defaults is None:
        snes_defaults = {}
    if solver_parameters:
        # User configured something, try and set sensible direct solve
        # defaults for missing bits.
        parameters = solver_parameters.copy()
        for k, v in snes_defaults.items():
            parameters.setdefault(k, v)
        keys = frozenset(parameters.keys())
        ksp_defaults = ksp_defaults.copy()
        skip = set()
        if "pc_type" in keys:
            # Might reasonably expect to get petsc defaults
            skip.update({"pc_factor_mat_solver_type", "ksp_type"})
        if parameters.get("mat_type") in {"matfree", "nest"}:
            # Non-LU defaults.
            ksp_defaults["ksp_type"] = "gmres"
            ksp_defaults["pc_type"] = "jacobi"
        for k, v in ksp_defaults.items():
            if k not in skip:
                parameters.setdefault(k, v)
        return parameters

    # OK, we're in full defaults mode now.
    parameters = dict(chain.from_iterable(
        d.items() for d in (ksp_defaults, snes_defaults)))

    if any(V.ufl_element().family() == "Real"
           for a in arguments for V in a.function_space()):
        test, trial = arguments
        if test.function_space() != trial.function_space():
            # Don't know what to do here. How did it happen?
            raise ValueError("Can't generate defaults for non-square problems with real blocks")

        fields = []
        reals = []
        for i, V_ in enumerate(test.function_space()):
            if V_.ufl_element().family() == "Real":
                reals.append(i)
            else:
                fields.append(i)
        if len(fields) == 0:
            # Just reals, GMRES
            opts = {"ksp_type": "gmres",
                    "pc_type": "none"}
            parameters.update(opts)
        else:
            warning("Real block detected, generating Schur complement elimination PC")
            # Full Schur complement eliminating onto Real blocks.
            opts = {"mat_type": "matfree",
                    "ksp_type": "fgmres",
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_fact_type": "full",
                    "pc_fieldsplit_0_fields": ",".join(map(str, fields)),
                    "pc_fieldsplit_1_fields": ",".join(map(str, reals)),
                    "fieldsplit_0": {
                        "ksp_type": "preonly",
                        "pc_type": "python",
                        "pc_python_type": "firedrake.AssembledPC",
                        "assembled": DEFAULT_KSP_PARAMETERS,
                    },
                    "fieldsplit_1": {
                        "ksp_type": "gmres",
                        "pc_type": "none",
                    }}
            parameters.update(opts)
        return parameters
    else:
        # We can assemble an AIJ matrix, factor it with a sparse
        # direct method.
        return parameters


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
        raise ConvergenceError(r"""Nonlinear solve failed to converge after %d nonlinear iterations.
Reason:
   %s""" % (snes.getIterationNumber(), msg))


class _SNESContext(object):
    r"""
    Context holding information for SNES callbacks.

    :arg problem: a :class:`NonlinearVariationalProblem`.
    :arg mat_type: Indicates whether the Jacobian is assembled
        monolithically ('aij'), as a block sparse matrix ('nest') or
        matrix-free (as :class:`~.ImplicitMatrix`, 'matfree').
    :arg pmat_type: Indicates whether the preconditioner (if present) is assembled
        monolithically ('aij'), as a block sparse matrix ('nest') or
        matrix-free (as :class:`~.ImplicitMatrix`, 'matfree').
    :arg appctx: Any extra information used in the assembler.  For the
        matrix-free case this will contain the Newton state in
        ``"state"``.
    :arg pre_jacobian_callback: User-defined function called immediately
        before Jacobian assembly
    :arg post_jacobian_callback: User-defined function called immediately
        after Jacobian assembly
    :arg pre_function_callback: User-defined function called immediately
        before residual assembly
    :arg post_function_callback: User-defined function called immediately
        after residual assembly
    :arg options_prefix: The options prefix of the SNES.
    :arg transfer_manager: Object that can transfer functions between
        levels, typically a :class:`~.TransferManager`

    The idea here is that the SNES holds a shell DM which contains
    this object as "user context".  When the SNES calls back to the
    user form_function code, we pull the DM out of the SNES and then
    get the context (which is one of these objects) to find the
    Firedrake level information.
    """
    @PETSc.Log.EventDecorator()
    def __init__(self, problem, mat_type, pmat_type, appctx=None,
                 pre_jacobian_callback=None, pre_function_callback=None,
                 post_jacobian_callback=None, post_function_callback=None,
                 options_prefix=None,
                 transfer_manager=None):
        from firedrake.assemble import get_assembler

        if pmat_type is None:
            pmat_type = mat_type
        self.mat_type = mat_type
        self.pmat_type = pmat_type
        self.options_prefix = options_prefix

        matfree = mat_type == 'matfree'
        pmatfree = pmat_type == 'matfree'

        self._problem = problem
        self._pre_jacobian_callback = pre_jacobian_callback
        self._pre_function_callback = pre_function_callback
        self._post_jacobian_callback = post_jacobian_callback
        self._post_function_callback = post_function_callback

        self.fcp = problem.form_compiler_parameters
        # Function to hold current guess
        self._x = problem.u_restrict

        if appctx is None:
            appctx = {}
        # A split context will already get the full state.
        # TODO, a better way of doing this.
        # Now we don't have a temporary state inside the snes
        # context we could just require the user to pass in the
        # full state on the outside.
        appctx.setdefault("state", self._x)
        appctx.setdefault("form_compiler_parameters", self.fcp)

        self.appctx = appctx
        self.matfree = matfree
        self.pmatfree = pmatfree
        self.F = problem.F
        self.J = problem.J

        # For Jp to equal J, bc.Jp must equal bc.J for all EquationBC objects.
        Jp_eq_J = problem.Jp is None and all(bc.Jp_eq_J for bc in problem.bcs)

        if mat_type != pmat_type or not Jp_eq_J:
            # Need separate pmat if either Jp is different or we want
            # a different pmat type to the mat type.
            if problem.Jp is None:
                self.Jp = self.J
            else:
                self.Jp = problem.Jp
        else:
            # pmat_type == mat_type and Jp_eq_J
            self.Jp = None

        self.bcs_F = tuple(bc.extract_form('F') for bc in problem.bcs)
        self.bcs_J = tuple(bc.extract_form('J') for bc in problem.bcs)
        self.bcs_Jp = tuple(bc.extract_form('Jp') for bc in problem.bcs)

        self._assemble_residual = get_assembler(self.F, bcs=self.bcs_F,
                                                form_compiler_parameters=self.fcp,
                                                ).assemble

        self._jacobian_assembled = False
        self._splits = {}
        self._coarse = None
        self._fine = None

        self._nullspace = None
        self._nullspace_T = None
        self._near_nullspace = None
        self._transfer_manager = transfer_manager

    @property
    def transfer_manager(self):
        """This allows the transfer manager to be set from options, e.g.

        solver_parameters = {"ksp_type": "cg",
                             "pc_type": "mg",
                             "mg_transfer_manager": __name__ + ".manager"}

        The value for "mg_transfer_manager" can either be a specific instantiated
        object, or a function or class name. In the latter case it will be invoked
        with no arguments to instantiate the object.

        If "snes_type": "fas" is used, the relevant option is "fas_transfer_manager",
        with the same semantics.
        """
        if self._transfer_manager is None:
            opts = PETSc.Options()
            prefix = self.options_prefix or ""
            if opts.hasName(prefix + "mg_transfer_manager"):
                managername = opts[prefix + "mg_transfer_manager"]
            elif opts.hasName(prefix + "fas_transfer_manager"):
                managername = opts[prefix + "fas_transfer_manager"]
            else:
                managername = None

            if managername is None:
                from firedrake import TransferManager
                transfer = TransferManager(use_averaging=True)
            else:
                (modname, objname) = managername.rsplit('.', 1)
                mod = __import__(modname)
                obj = getattr(mod, objname)
                if isinstance(obj, type):
                    transfer = obj()
                else:
                    transfer = obj

            self._transfer_manager = transfer
        return self._transfer_manager

    @transfer_manager.setter
    def transfer_manager(self, manager):
        if self._transfer_manager is not None:
            raise ValueError("Must set transfer manager before first use.")
        self._transfer_manager = manager

    def set_function(self, snes):
        r"""Set the residual evaluation function"""
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

    @PETSc.Log.EventDecorator()
    def split(self, fields):
        from firedrake import replace, as_vector, split
        from firedrake import NonlinearVariationalProblem as NLVP
        from firedrake.bcs import DirichletBC, EquationBC
        fields = tuple(tuple(f) for f in fields)
        splits = self._splits.get(tuple(fields))
        if splits is not None:
            return splits

        splits = []
        problem = self._problem
        splitter = ExtractSubBlock()
        for field in fields:
            F = splitter.split(problem.F, argument_indices=(field, ))
            J = splitter.split(problem.J, argument_indices=(field, field))
            us = problem.u_restrict.subfunctions
            V = F.arguments()[0].function_space()
            # Exposition:
            # We are going to make a new solution Function on the sub
            # mixed space defined by the relevant fields.
            # But the form may refer to the rest of the solution
            # anyway.
            # So we pull it apart and will make a new function on the
            # subspace that shares data.
            pieces = [us[i].dat for i in field]
            if len(pieces) == 1:
                val, = pieces
                subu = function.Function(V, val=val)
                subsplit = (subu, )
            else:
                val = op2.MixedDat(pieces)
                subu = function.Function(V, val=val)
                # Split it apart to shove in the form.
                subsplit = split(subu)
            vec = []
            for i, u in enumerate(us):
                if i in field:
                    # If this is a field we're keeping, get it from
                    # the new function. Otherwise just point to the
                    # old data.
                    u = subsplit[field.index(i)]
                if u.ufl_shape == ():
                    vec.append(u)
                else:
                    vec.extend(u[idx] for idx in numpy.ndindex(u.ufl_shape))

            # So now we have a new representation for the solution
            # vector in the old problem. For the fields we're going
            # to solve for, it points to a new Function (which wraps
            # the original pieces). For the rest, it points to the
            # pieces from the original Function.
            # IOW, we've reinterpreted our original mixed solution
            # function as being made up of some spaces we're still
            # solving for, and some spaces that have just become
            # coefficients in the new form.
            u = as_vector(vec)
            F = replace(F, {problem.u_restrict: u})
            J = replace(J, {problem.u_restrict: u})
            if problem.Jp is not None:
                Jp = splitter.split(problem.Jp, argument_indices=(field, field))
                Jp = replace(Jp, {problem.u_restrict: u})
            else:
                Jp = None
            bcs = []
            for bc in problem.bcs:
                if isinstance(bc, DirichletBC):
                    bc_temp = bc.reconstruct(field=field, V=V, g=bc.function_arg, sub_domain=bc.sub_domain)
                elif isinstance(bc, EquationBC):
                    bc_temp = bc.reconstruct(V, subu, u, field, False)
                if bc_temp is not None:
                    bcs.append(bc_temp)
            new_problem = NLVP(F, subu, bcs=bcs, J=J, Jp=Jp,
                               form_compiler_parameters=problem.form_compiler_parameters)
            new_problem._constant_jacobian = problem._constant_jacobian
            splits.append(type(self)(new_problem, mat_type=self.mat_type, pmat_type=self.pmat_type,
                                     appctx=self.appctx,
                                     transfer_manager=self.transfer_manager))
        return self._splits.setdefault(tuple(fields), splits)

    @staticmethod
    def form_function(snes, X, F):
        r"""Form the residual for this problem

        :arg snes: a PETSc SNES object
        :arg X: the current guess (a Vec)
        :arg F: the residual at X (a Vec)
        """
        dm = snes.getDM()
        ctx = dmhooks.get_appctx(dm)
        # X may not be the same vector as the vec behind self._x, so
        # copy guess in from X.
        with ctx._x.dat.vec_wo as v:
            X.copy(v)

        if ctx._pre_function_callback is not None:
            ctx._pre_function_callback(X)

        ctx._assemble_residual(tensor=ctx._F)

        if ctx._post_function_callback is not None:
            with ctx._F.dat.vec as F_:
                ctx._post_function_callback(X, F_)

        # F may not be the same vector as self._F, so copy
        # residual out to F.
        with ctx._F.dat.vec_ro as v:
            v.copy(F)

    @staticmethod
    def form_jacobian(snes, X, J, P):
        r"""Form the Jacobian for this problem

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
        ctx._assemble_jac(ctx._jac)

        if ctx._post_jacobian_callback is not None:
            ctx._post_jacobian_callback(X, J)

        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac(ctx._pjac)

        ises = problem.J.arguments()[0].function_space()._ises
        ctx.set_nullspace(ctx._nullspace, ises, transpose=False, near=False)
        ctx.set_nullspace(ctx._nullspace_T, ises, transpose=True, near=False)
        ctx.set_nullspace(ctx._near_nullspace, ises, transpose=False, near=True)

    @staticmethod
    def compute_operators(ksp, J, P):
        r"""Form the Jacobian for this problem

        :arg ksp: a PETSc KSP object
        :arg J: the Jacobian (a Mat)
        :arg P: the preconditioner matrix (a Mat)
        """
        from firedrake.bcs import DirichletBC
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
            manager = dmhooks.get_transfer_manager(fine._x.function_space().dm)
            manager.inject(fine._x, ctx._x)

            for bc in chain(*ctx._problem.bcs):
                if isinstance(bc, DirichletBC):
                    bc.apply(ctx._x)

        ctx._assemble_jac(ctx._jac)
        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac(ctx._pjac)

    @cached_property
    def _assembler_jac(self):
        from firedrake.assemble import get_assembler
        return get_assembler(self.J, bcs=self.bcs_J, form_compiler_parameters=self.fcp, mat_type=self.mat_type, options_prefix=self.options_prefix, appctx=self.appctx)

    @cached_property
    def _jac(self):
        return self._assembler_jac.allocate()

    @cached_property
    def _assemble_jac(self):
        return self._assembler_jac.assemble

    @cached_property
    def is_mixed(self):
        return self._jac.block_shape != (1, 1)

    @cached_property
    def _assembler_pjac(self):
        from firedrake.assemble import get_assembler
        if self.mat_type != self.pmat_type or self._problem.Jp is not None:
            return get_assembler(self.Jp, bcs=self.bcs_Jp, form_compiler_parameters=self.fcp, mat_type=self.pmat_type, options_prefix=self.options_prefix, appctx=self.appctx)
        else:
            return self._assembler_jac

    @cached_property
    def _pjac(self):
        if self.mat_type != self.pmat_type or self._problem.Jp is not None:
            return self._assembler_pjac.allocate()
        else:
            return self._jac

    @cached_property
    def _assemble_pjac(self):
        return self._assembler_pjac.assemble

    @cached_property
    def _F(self):
        return cofunction.Cofunction(self.F.arguments()[0].function_space().dual())
