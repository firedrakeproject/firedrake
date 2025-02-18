import functools
import firedrake.dmhooks as dmhooks
import numpy as np
from firedrake.slate.static_condensation.sc_base import SCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor


__all__ = ['SCPC']


class SCPC(SCBase):

    needs_python_pmat = True

    """A Slate-based python preconditioner implementation of
    static condensation for problems with up to three fields.
    """

    @PETSc.Log.EventDecorator("SCPCInit")
    def initialize(self, pc):
        """Set up the problem context. This takes the incoming
        three-field system and constructs the static
        condensation operators using Slate expressions.

        A KSP is created for the reduced system. The eliminated
        variables are recovered via back-substitution.
        """

        from firedrake.assemble import get_assembler
        from firedrake.bcs import DirichletBC
        from firedrake.function import Function
        from firedrake.cofunction import Cofunction
        from firedrake.functionspace import FunctionSpace
        from firedrake.parloops import par_loop, INC
        from ufl import dx

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
            bcs.append(DirichletBC(Vc, 0, bc.sub_domain))

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")

        self.c_field = c_field
        self.condensed_rhs = Cofunction(Vc.dual())
        self.residual = Function(W)
        self.solution = Cofunction(W.dual())

        shapes = (Vc.finat_element.space_dimension(),
                  np.prod(Vc.shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        self.weight = Function(Vc)
        par_loop((domain, instructions), dx, {"w": (self.weight, INC)})
        with self.weight.dat.vec as wc:
            wc.reciprocal()

        # Get expressions for the condensed linear system
        A_tensor = Tensor(self.bilinear_form)
        reduced_sys, schur_builder = self.condensed_system(A_tensor, self.residual, elim_fields, prefix, pc)
        S_expr = reduced_sys.lhs
        r_expr = reduced_sys.rhs

        # Construct the condensed right-hand side
        self._assemble_Srhs = get_assembler(r_expr, bcs=bcs, form_compiler_parameters=self.cxt.fc_params).assemble

        # Allocate and set the condensed operator
        form_assembler = get_assembler(S_expr, bcs=bcs, form_compiler_parameters=self.cxt.fc_params, mat_type=mat_type, options_prefix=prefix, appctx=self.get_appctx(pc))
        self.S = form_assembler.allocate()
        self._assemble_S = form_assembler.assemble

        self._assemble_S(tensor=self.S)
        Smat = self.S.petscmat

        # If a different matrix is used for preconditioning,
        # assemble this as well
        if A != P:
            self.cxt_pc = P.getPythonContext()
            P_tensor = Tensor(self.cxt_pc.a)
            P_reduced_sys, _ = self.condensed_system(P_tensor,
                                                     self.residual,
                                                     elim_fields)
            S_pc_expr = P_reduced_sys.lhs
            self.S_pc_expr = S_pc_expr

            # Allocate and set the condensed operator
            form_assembler = get_assembler(S_pc_expr, bcs=bcs, form_compiler_parameters=self.cxt.fc_params, mat_type=mat_type, options_prefix=prefix, appctx=self.get_appctx(pc))
            self.S_pc = form_assembler.allocate()
            self._assemble_S_pc = form_assembler.assemble

            self._assemble_S_pc(tensor=self.S_pc)
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
            Smat.setNullSpace(nsp.nullspace())

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

        # Set up local solvers for backwards substitution
        self.local_solvers = self.local_solver_calls(A_tensor,
                                                     self.residual,
                                                     self.solution,
                                                     elim_fields,
                                                     schur_builder)

    def condensed_system(self, A, rhs, elim_fields, prefix, pc):
        """Forms the condensed linear system by eliminating
        specified unknowns.

        :arg A: A Slate Tensor containing the mixed bilinear form.
        :arg rhs: A firedrake function for the right-hand side.
        :arg elim_fields: An iterable of field indices to eliminate.
        :arg prefix: an option prefix for the condensed field.
        :arg pc: a Preconditioner instance.
        """

        from firedrake.slate.static_condensation.la_utils import condense_and_forward_eliminate

        return condense_and_forward_eliminate(A, rhs, elim_fields, prefix, pc)

    def local_solver_calls(self, A, rhs, x, elim_fields, schur_builder):
        """Provides solver callbacks for inverting local operators
        and reconstructing eliminated fields.

        :arg A: A Slate Tensor containing the mixed bilinear form.
        :arg rhs: A firedrake function for the right-hand side.
        :arg x: A firedrake function for the solution.
        :arg elim_fields: An iterable of eliminated field indices
                          to recover.
        :arg schur_builder: a `SchurComplementBuilder`.
        """
        from firedrake.assemble import get_assembler
        from firedrake.slate.static_condensation.la_utils import backward_solve

        fields = x.subfunctions
        systems = backward_solve(A, rhs, x, schur_builder, reconstruct_fields=elim_fields)

        local_solvers = []
        for local_system in systems:
            Aeinv = local_system.lhs
            be = local_system.rhs
            i, = local_system.field_idx
            local_solve = Aeinv * be
            solve_call = functools.partial(get_assembler(local_solve, form_compiler_parameters=self.cxt.fc_params).assemble, tensor=fields[i])
            local_solvers.append(solve_call)

        return local_solvers

    @PETSc.Log.EventDecorator("SCPCUpdate")
    def update(self, pc):
        """Update by assembling into the KSP operator. No
        need to reconstruct symbolic objects.
        """

        self._assemble_S(tensor=self.S)

        # Only reassemble if a preconditioning operator
        # is provided for the condensed system
        if hasattr(self, "S_pc"):
            self._assemble_S_pc(tensor=self.S_pc)

    def forward_elimination(self, pc, x):
        """Perform the forward elimination of fields and
        provide the reduced right-hand side for the condensed
        system.

        :arg pc: a Preconditioner instance.
        :arg x: a PETSc vector containing the incoming right-hand side.
        """

        with self.residual.dat.vec_wo as v:
            x.copy(v)

        # Disassemble the incoming right-hand side
        with self.residual.subfunctions[self.c_field].dat.vec as vc, self.weight.dat.vec_ro as wc:
            vc.pointwiseMult(vc, wc)

        # Now assemble residual for the reduced problem
        self._assemble_Srhs(tensor=self.condensed_rhs)

    def sc_solve(self, pc):
        """Solve the condensed linear system for the
        condensed field.

        :arg pc: a Preconditioner instance.
        """

        dm = self.condensed_ksp.getDM()

        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):

            with self.condensed_rhs.dat.vec_ro as rhs:
                if self.condensed_ksp.getInitialGuessNonzero():
                    acc = self.solution.subfunctions[self.c_field].dat.vec
                else:
                    acc = self.solution.subfunctions[self.c_field].dat.vec_wo
                with acc as sol:
                    self.condensed_ksp.solve(rhs, sol)

    def backward_substitution(self, pc, y):
        """Perform the backwards recovery of eliminated fields.

        :arg pc: a Preconditioner instance.
        :arg y: a PETSc vector for placing the resulting fields.
        """

        # Recover eliminated unknowns
        for local_solver_call in self.local_solvers:
            local_solver_call()

        with self.solution.dat.vec_ro as w:
            w.copy(y)

    def view(self, pc, viewer=None):
        """Viewer calls for the various configurable objects in this PC."""

        super(SCPC, self).view(pc, viewer)
        if hasattr(self, "condensed_ksp"):
            viewer.printfASCII("Solving linear system using static condensation.\n")
            self.condensed_ksp.view(viewer=viewer)
            viewer.printfASCII("Locally reconstructing unknowns.\n")
