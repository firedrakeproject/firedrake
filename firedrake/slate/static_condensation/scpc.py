import firedrake.dmhooks as dmhooks

from firedrake.slate.static_condensation.sc_base import SCBase
from firedrake.slate.static_condensation.la_utils import generate_static_condensation_expressions
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor
from pyop2.profiling import timed_function


__all__ = ['SCPC']


class SCPC(SCBase):

    needs_python_pmat = True

    """A Slate-based python preconditioner implementation of
    static condensation for problems with up to three fields.
    """

    @timed_function("SCPCInit")
    def initialize(self, pc):
        """Set up the problem context. This takes the incoming
        three-field system and constructs the static
        condensation operators using Slate expressions.

        A KSP is created for the reduced system. The eliminated
        variables are recovered via back-substitution.
        """

        from firedrake.assemble import (allocate_matrix,
                                        create_assembly_callable)
        from firedrake.bcs import DirichletBC
        from firedrake.function import Function
        from firedrake.functionspace import FunctionSpace
        from firedrake.interpolation import interpolate

        prefix = pc.getOptionsPrefix() + "condensed_field_"
        A, P = pc.getOperators()
        self.cxt = A.getPythonContext()
        if not isinstance(self.cxt, ImplicitMatrixContext):
            raise ValueError("Context must be an ImplicitMatrixContext")

        self.bilinear_form = self.cxt.a

        # Retrieve the mixed function space
        W = self.bilinear_form.arguments()[0].function_space()
        if len(W) > 3:
            raise NotImplementedError("Only supports up to three function spaces.")

        elim_fields = PETSc.Options().getString(pc.getOptionsPrefix()
                                                + "pc_sc_eliminate_fields",
                                                None)
        if elim_fields:
            elim_fields = [int(i) for i in elim_fields.split(',')]
        else:
            # By default, we condense down to the last field in the
            # mixed space.
            elim_fields = [i for i in range(0, len(W) - 1)]

        condensed_fields = list(set(range(len(W))) - set(elim_fields))
        if len(condensed_fields) != 1:
            raise NotImplementedError("Cannot condense to more than one field")

        c_field, = condensed_fields

        # Need to duplicate a space which is NOT
        # associated with a subspace of a mixed space.
        Vc = FunctionSpace(W.mesh(), W[c_field].ufl_element())
        bcs = []
        cxt_bcs = self.cxt.row_bcs
        for bc in cxt_bcs:
            if bc.function_space().index != c_field:
                raise NotImplementedError("Strong BC set on unsupported space")
            if isinstance(bc.function_arg, Function):
                bc_arg = interpolate(bc.function_arg, Vc)
            else:
                # Constants don't need to be interpolated
                bc_arg = bc.function_arg
            bcs.append(DirichletBC(Vc, bc_arg, bc.sub_domain))

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")

        self.c_field = c_field
        self.condensed_rhs = Function(Vc)
        self.residual = Function(W)
        self.solution = Function(W)

        # Get expressions for the condensed linear system and local recovery
        A_tensor = Tensor(self.bilinear_form)
        slate_contexts = generate_static_condensation_expressions(A_tensor,
                                                                  self.residual,
                                                                  self.solution,
                                                                  elim_fields)
        reduced_sys, backsolve = slate_contexts

        S_expr = reduced_sys.lhs
        r_expr = reduced_sys.rhs

        # Construct the condensed right-hand side
        self._assemble_Srhs = create_assembly_callable(
            r_expr,
            tensor=self.condensed_rhs,
            form_compiler_parameters=self.cxt.fc_params)

        # Allocate and set the condensed operator
        self.S = allocate_matrix(S_expr,
                                 bcs=bcs,
                                 form_compiler_parameters=self.cxt.fc_params,
                                 mat_type=mat_type,
                                 options_prefix=prefix,
                                 appctx=self.get_appctx(pc))

        self._assemble_S = create_assembly_callable(
            S_expr,
            tensor=self.S,
            bcs=bcs,
            form_compiler_parameters=self.cxt.fc_params,
            mat_type=mat_type)

        self._assemble_S()
        Smat = self.S.petscmat

        # If a different matrix is used for preconditioning,
        # assemble this as well
        if A != P:
            self.cxt_pc = P.getPythonContext()
            P_tensor = Tensor(self.cxt_pc.a)
            P_reduced_sys = self.condensed_system(P_tensor,
                                                  self.residual,
                                                  elim_fields)
            S_pc_expr = P_reduced_sys.lhs
            self.S_pc_expr = S_pc_expr

            # Allocate and set the condensed operator
            self.S_pc = allocate_matrix(S_expr,
                                        bcs=bcs,
                                        form_compiler_parameters=self.cxt.fc_params,
                                        mat_type=mat_type,
                                        options_prefix=prefix,
                                        appctx=self.get_appctx(pc))

            self._assemble_S_pc = create_assembly_callable(
                S_pc_expr,
                tensor=self.S_pc,
                bcs=bcs,
                form_compiler_parameters=self.cxt.fc_params,
                mat_type=mat_type)

            self._assemble_S_pc()
            Smat_pc = self.S_pc.petscmat

        else:
            self.S_pc_expr = S_expr
            Smat_pc = Smat

        # Get nullspace for the condensed operator (if any).
        # This is provided as a user-specified callback which
        # returns the basis for the nullspace.
        nullspace = self.cxt.appctx.get("condensed_field_nullspace", None)
        if nullspace is not None:
            nsp = nullspace(Vc)
            Smat.setNullSpace(nsp.nullspace(comm=pc.comm))

        # Create a SNESContext for the DM associated with the trace problem
        self._ctx_ref = self.new_snes_ctx(pc,
                                          S_expr,
                                          bcs,
                                          mat_type,
                                          self.cxt.fc_params,
                                          options_prefix=prefix)

        # Push new context onto the dm associated with the condensed problem
        c_dm = Vc.dm

        # Set up ksp for the condensed problem
        c_ksp = PETSc.KSP().create(comm=pc.comm)
        c_ksp.incrementTabLevel(1, parent=pc)

        # Set the dm for the condensed solver
        c_ksp.setDM(c_dm)
        c_ksp.setDMActive(False)
        c_ksp.setOptionsPrefix(prefix)
        c_ksp.setOperators(A=Smat, P=Smat_pc)
        self.condensed_ksp = c_ksp

        with dmhooks.add_hooks(c_dm, self,
                               appctx=self._ctx_ref,
                               save=False):
            c_ksp.setFromOptions()

        # Set up local solver for backwards substitution
        Ae = backsolve.lhs
        be = backsolve.rhs
        local_solve = Ae.solve(be, decomposition="PartialPivLU")
        # FIXME: Should use a temporary function here?
        self._local_solve = create_assembly_callable(
            local_solve,
            tensor=self.solution,
            form_compiler_parameters=self.cxt.fc_params)

    @timed_function("SCPCUpdate")
    def update(self, pc):
        """Update by assembling into the KSP operator. No
        need to reconstruct symbolic objects.
        """

        self._assemble_S()

        # Only reassemble if a preconditioning operator
        # is provided for the condensed system
        if hasattr(self, "S_pc"):
            self._assemble_S_pc()

    def forward_elimination(self, pc, x):
        """Perform the forward elimination of fields and
        provide the reduced right-hand side for the condensed
        system.

        :arg pc: a Preconditioner instance.
        :arg x: a PETSc vector containing the incoming right-hand side.
        """

        with self.residual.dat.vec_wo as v:
            x.copy(v)

        # Now assemble residual for the reduced problem
        self._assemble_Srhs()

    def sc_solve(self, pc):
        """Solve the condensed linear system for the
        condensed field.

        :arg pc: a Preconditioner instance.
        """

        dm = self.condensed_ksp.getDM()

        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):

            with self.condensed_rhs.dat.vec_ro as rhs:
                if self.condensed_ksp.getInitialGuessNonzero():
                    acc = self.solution.split()[self.c_field].dat.vec
                else:
                    acc = self.solution.split()[self.c_field].dat.vec_wo
                with acc as sol:
                    self.condensed_ksp.solve(rhs, sol)

    def backward_substitution(self, pc, y):
        """Perform the backwards recovery of eliminated fields.

        :arg pc: a Preconditioner instance.
        :arg y: a PETSc vector for placing the resulting fields.
        """

        # Recover eliminated unknowns
        self._local_solve()

        with self.solution.dat.vec_ro as w:
            w.copy(y)

    def view(self, pc, viewer=None):
        """Viewer calls for the various configurable objects in this PC."""

        super(SCPC, self).view(pc, viewer)
        if hasattr(self, "condensed_ksp"):
            viewer.printfASCII("Solving linear system using static condensation.\n")
            self.condensed_ksp.view(viewer=viewer)
            viewer.printfASCII("Locally reconstructing unknowns.\n")
