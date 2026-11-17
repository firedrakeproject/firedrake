import abc
import functools

import firedrake.dmhooks as dmhooks
from firedrake.preconditioners import PCBase
from firedrake.petsc import PETSc


class SCBase(PCBase):
    """A general-purpose base class for static condensation
    interfaces.
    """

    @abc.abstractmethod
    def forward_elimination(self, pc, x):
        """Perform the forward elimination of fields and
        provide the reduced right-hand side for the condensed
        system.

        :arg pc: a Preconditioner instance.
        :arg x: a PETSc vector containing the incoming right-hand side.
        """

        raise NotImplementedError("Forward elimination not implemented")

    @abc.abstractmethod
    def backward_substitution(self, pc, y):
        """Perform the backwards recovery of eliminated fields.

        :arg pc: a Preconditioner instance.
        :arg y: a PETSc vector for placing the resulting fields.
        """

        raise NotImplementedError("Backward substitution not implemented")

    def initialize_condensed_ksp(self, pc, prefix, Vc, S_expr, bcs, nullspace_key, S_pc_expr=None, monitor=None):
        """Assemble the condensed operator and create the KSP that solves the condensed system.

        Parameters
        ----------
        pc : PETSc.PC
            The preconditioner of the original mixed system.
        prefix : str
            The options prefix of the condensed system.
        Vc : firedrake.functionspaceimpl.WithGeometry
            The function space of the condensed system.
        S_expr : firedrake.slate.slate.TensorBase
            The condensed operator.
        bcs : list of firedrake.bcs.DirichletBC
            The boundary conditions on ``Vc``.
        nullspace_key : str
            The key of the application context that can give a callback, which
            returns the null space of the condensed operator on ``Vc``.
        S_pc_expr : firedrake.slate.slate.TensorBase, optional
            The condensed operator from which the preconditioner is built.
            If it is not given, ``S_expr`` is used.
        monitor : optional
            A KSP monitor that can reconstruct the eliminated subspaces with
            :meth:`backward_substitution`.
        """
        from firedrake.assemble import get_assembler

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")
        fcp = self.ctx.fc_params
        appctx = self.get_appctx(pc)

        form_assembler = get_assembler(S_expr, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type, options_prefix=prefix, appctx=appctx)
        self.S = form_assembler.allocate()
        self._assemble_S = form_assembler.assemble
        with PETSc.Log.Event("SCOperatorAssembly"):
            self._assemble_S(tensor=self.S)
        Smat = self.S.petscmat

        if S_pc_expr is None:
            self.S_pc = None
            Smat_pc = Smat
        else:
            form_assembler = get_assembler(S_pc_expr, bcs=bcs, form_compiler_parameters=fcp, mat_type=mat_type, options_prefix=prefix, appctx=appctx)
            self.S_pc = form_assembler.allocate()
            self._assemble_S_pc = form_assembler.assemble
            self._assemble_S_pc(tensor=self.S_pc)
            Smat_pc = self.S_pc.petscmat

        nullspace = self.ctx.appctx.get(nullspace_key, None)
        if nullspace is not None:
            nsp = nullspace(Vc)
            Smat.setNullSpace(nsp.nullspace())

        self._ctx_ref = self.new_snes_ctx(pc, S_expr, bcs, mat_type, fcp, options_prefix=prefix)

        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.incrementTabLevel(1, parent=pc)
        ksp.setDM(Vc.dm)
        ksp.setDMActive(PETSc.KSP.DMActive.ALL, False)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(A=Smat, P=Smat_pc)
        if monitor:
            monitor.add_reconstructor(self.backward_substitution)
            ksp.setMonitor(monitor)
        self.condensed_ksp = ksp

        with dmhooks.add_hooks(Vc.dm, self, appctx=self._ctx_ref, save=False):
            ksp.setFromOptions()

    def local_solver_callbacks(self, rhs, x, elim_fields, schur_builder):
        """Return the callbacks that recover the eliminated subspaces.

        Parameters
        ----------
        rhs : firedrake.cofunction.Cofunction
            The right-hand side.
        x : firedrake.function.Function
            The solution, whose retained subspaces are known.
        elim_fields : collections.abc.Sequence of int
            The indices of the eliminated subspaces in the space of ``x``.
        schur_builder : firedrake.slate.static_condensation.la_utils.SchurComplementBuilder
            The builder of the local inverses.

        Returns
        -------
        list of functools.partial
            The callbacks, in the order in which they must be called.
        """
        from firedrake.assemble import get_assembler
        from firedrake.slate.static_condensation.la_utils import backward_solve

        local_solvers = []
        for local_system in backward_solve(rhs, x, schur_builder, elim_fields):
            local_solve = local_system.lhs * local_system.rhs
            assembler = get_assembler(local_solve, form_compiler_parameters=self.ctx.fc_params)
            local_solvers.append(functools.partial(assembler.assemble, tensor=x.sub(local_system.field_idx)))
        return local_solvers

    @PETSc.Log.EventDecorator("SCUpdate")
    def update(self, pc):
        """Assemble the condensed operator again."""
        self._assemble_S(tensor=self.S)
        if self.S_pc is not None:
            self._assemble_S_pc(tensor=self.S_pc)

    def sc_solve(self, pc):
        """Solve the condensed system for the retained subspaces.

        Parameters
        ----------
        pc : PETSc.PC
            The preconditioner of the original mixed system.
        """
        dm = self.condensed_ksp.getDM()
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            with self.condensed_rhs.dat.vec_ro as rhs:
                if self.condensed_ksp.getInitialGuessNonzero():
                    acc = self.solution.sub(self.c).dat.vec
                else:
                    acc = self.solution.sub(self.c).dat.vec_wo
                with acc as sol:
                    self.condensed_ksp.solve(rhs, sol)

    def apply(self, pc, x, y):
        """Applies the static condensation preconditioner.

        :arg pc: a Preconditioner instance.
        :arg x: A PETSc vector containing the incoming right-hand side.
        :arg y: A PETSc vector for the result.
        """

        with PETSc.Log.Event("SCForwardElim"):
            self.forward_elimination(pc, x)

        with PETSc.Log.Event("SCSolve"):
            self.sc_solve(pc)

        with PETSc.Log.Event("SCBackSub"):
            self.backward_substitution(pc, y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""

        raise NotImplementedError("Transpose application is not implemented.")
