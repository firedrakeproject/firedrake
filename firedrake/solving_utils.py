import numpy

from pyop2 import op2
from firedrake import function, dmhooks
from firedrake.exceptions import ConvergenceError
from firedrake.petsc import PETSc
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.utils import cached_property
from ufl import VectorElement


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
    :arg pre_function_callback: User-defined function called immediately
        before residual assembly
    :arg options_prefix: The options prefix of the SNES.

    The idea here is that the SNES holds a shell DM which contains
    this object as "user context".  When the SNES calls back to the
    user form_function code, we pull the DM out of the SNES and then
    get the context (which is one of these objects) to find the
    Firedrake level information.
    """
    def __init__(self, problem, mat_type, pmat_type, appctx=None,
                 pre_jacobian_callback=None, pre_function_callback=None,
                 options_prefix=None):
        from firedrake.assemble import create_assembly_callable
        from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
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

        self.fcp = problem.form_compiler_parameters
        # Function to hold current guess
        self._x = problem.u

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
        def is_Jp_eq_J(bcs):
            v = True
            for bc in bcs:
                if isinstance(bc, EquationBC):
                    v = v and bc.Jp_eq_J and is_Jp_eq_J(bc.bcs)
            return v

        Jp_eq_J = (problem.Jp is None) and is_Jp_eq_J(problem.bcs)

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

        if problem.bcs is None:
            self.bcs_F = None
            self.bcs_J = None
            self.bcs_Jp = None
        else:
            # For each of (F, J, Jp), we need to make deep
            # copy if bc objects themselves have .bcs;
            # see bcs.py.
            def create_bc_tree(ebc, form_type):
                ebcsplit = EquationBCSplit(ebc, getattr(ebc, form_type))
                for bbc in ebc.bcs:
                    if isinstance(bbc, DirichletBC):
                        ebcsplit.add(bbc)
                    elif isinstance(bbc, EquationBC):
                        ebcsplit.add(create_bc_tree(bbc, form_type))
                return ebcsplit

            self.bcs_F = []
            self.bcs_J = []
            self.bcs_Jp = []
            for bc in self._problem.bcs:
                if isinstance(bc, DirichletBC):
                    self.bcs_F.append(bc)
                    self.bcs_J.append(bc)
                    self.bcs_Jp.append(bc)
                elif isinstance(bc, EquationBC):
                    self.bcs_F.append(create_bc_tree(bc, 'F'))
                    self.bcs_J.append(create_bc_tree(bc, 'J'))
                    self.bcs_Jp.append(create_bc_tree(bc, 'Jp'))

        self._assemble_residual = create_assembly_callable(self.F,
                                                           tensor=self._F,
                                                           bcs=self.bcs_F,
                                                           form_compiler_parameters=self.fcp)

        self._jacobian_assembled = False
        self._splits = {}
        self._coarse = None
        self._fine = None

        self._nullspace = None
        self._nullspace_T = None
        self._near_nullspace = None

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
            us = problem.u.split()
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
            # Permutation from field indexing to indexing of pieces
            field_renumbering = dict([f, i] for i, f in enumerate(field))
            vec = []
            for i, u in enumerate(us):
                if i in field:
                    # If this is a field we're keeping, get it from
                    # the new function. Otherwise just point to the
                    # old data.
                    u = subsplit[field_renumbering[i]]
                if u.ufl_shape == ():
                    vec.append(u)
                else:
                    for idx in numpy.ndindex(u.ufl_shape):
                        vec.append(u[idx])

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
            F = replace(F, {problem.u: u})
            J = replace(J, {problem.u: u})
            if problem.Jp is not None:
                Jp = splitter.split(problem.Jp, argument_indices=(field, field))
                Jp = replace(Jp, {problem.u: u})
            else:
                Jp = None

            # Recursively add DirichletBCs/EquationBCs
            def rcollect_bcs(bc):
                Vbc = bc.function_space()
                if Vbc.parent is not None and isinstance(Vbc.parent.ufl_element(), VectorElement):
                    index = Vbc.parent.index
                else:
                    index = Vbc.index
                cmpt = Vbc.component
                # TODO: need to test this logic
                if index in field:
                    if len(field) == 1:
                        W = V
                    else:
                        W = V.sub(field_renumbering[index])
                    if cmpt is not None:
                        W = W.sub(cmpt)
                    if isinstance(bc, DirichletBC):
                        return DirichletBC(W,
                                           bc.function_arg,
                                           bc.sub_domain,
                                           method=bc.method)
                    elif isinstance(bc, EquationBC):
                        bc_F = replace(splitter.split(bc.F, argument_indices=(field, )), {bc.u: u})
                        bc_J = replace(splitter.split(bc.J, argument_indices=(field, field)), {bc.u: u})
                        bc_Jp = None if bc.Jp is None else replace(splitter.split(bc.Jp, argument_indices=(field, field)), {bc.u: u})
                        ebc = EquationBC(bc_F == 0,
                                         subu,
                                         bc.sub_domain,
                                         method=bc.method,
                                         bcs=None,
                                         J=bc_J,
                                         Jp=None if bc.Jp_eq_J else bc_Jp,
                                         V=W,
                                         is_linear=bc.is_linear)
                        for bbc in bc.bcs:
                            bc_temp = rcollect_bcs(bbc)
                            # Due to the "if index", bc_temp can be None
                            if bc_temp is not None:
                                ebc.add(bc_temp)
                        return ebc
                    else:
                        raise TypeError("Unknown BC type")

            bcs = []
            for bc in problem.bcs:
                bc_temp = rcollect_bcs(bc)
                if bc_temp is not None:
                    bcs.append(bc_temp)
            new_problem = NLVP(F, subu, bcs=bcs, J=J, Jp=Jp,
                               form_compiler_parameters=problem.form_compiler_parameters)
            new_problem._constant_jacobian = problem._constant_jacobian
            splits.append(type(self)(new_problem, mat_type=self.mat_type, pmat_type=self.pmat_type,
                                     appctx=self.appctx))
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

        ctx._assemble_residual()

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

        ctx._assemble_jac()
        ctx._jac.force_evaluation()

        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac()
            ctx._pjac.force_evaluation()

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
        from firedrake.bcs import DirichletBC, EquationBC
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
            _, _, inject = dmhooks.get_transfer_operators(fine._x.function_space().dm)
            inject(fine._x, ctx._x)

            def rapply(bcs, u):
                for bc in bcs:
                    if isinstance(bc, DirichletBC):
                        bc.apply(u)
                    elif isinstance(bc, EquationBC):
                        rapply(bc.bcs, u)

            rapply(ctx._problem.bcs, ctx._x)

        ctx._assemble_jac()
        ctx._jac.force_evaluation()
        if ctx.Jp is not None:
            assert P.handle == ctx._pjac.petscmat.handle
            ctx._assemble_pjac()
            ctx._pjac.force_evaluation()

    @cached_property
    def _jac(self):
        from firedrake.assemble import allocate_matrix
        return allocate_matrix(self.J,
                               bcs=self.bcs_J,
                               form_compiler_parameters=self.fcp,
                               mat_type=self.mat_type,
                               appctx=self.appctx,
                               options_prefix=self.options_prefix)

    @cached_property
    def _assemble_jac(self):
        from firedrake.assemble import create_assembly_callable
        return create_assembly_callable(self.J,
                                        tensor=self._jac,
                                        bcs=self.bcs_J,
                                        form_compiler_parameters=self.fcp,
                                        mat_type=self.mat_type)

    @cached_property
    def is_mixed(self):
        return self._jac.block_shape != (1, 1)

    @cached_property
    def _pjac(self):
        if self.mat_type != self.pmat_type or self._problem.Jp is not None:
            from firedrake.assemble import allocate_matrix
            return allocate_matrix(self.Jp,
                                   bcs=self.bcs_Jp,
                                   form_compiler_parameters=self.fcp,
                                   mat_type=self.pmat_type,
                                   appctx=self.appctx,
                                   options_prefix=self.options_prefix)
        else:
            return self._jac

    @cached_property
    def _assemble_pjac(self):
        from firedrake.assemble import create_assembly_callable
        return create_assembly_callable(self.Jp,
                                        tensor=self._pjac,
                                        bcs=self.bcs_Jp,
                                        form_compiler_parameters=self.fcp,
                                        mat_type=self.pmat_type)

    @cached_property
    def _F(self):
        return function.Function(self.F.arguments()[0].function_space())
