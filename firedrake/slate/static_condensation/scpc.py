import numpy as np
from firedrake.slate.static_condensation.sc_base import SCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor


__all__ = ['SCPC']


class SCPC(SCBase):

    needs_python_pmat = True

    """A Slate-based python preconditioner implementation of
    static condensation for problems with any number of subspaces.
    """

    @PETSc.Log.EventDecorator("SCPCInit")
    def initialize(self, pc: PETSc.PC) -> None:
        """Construct the condensed system and the local recovery of the eliminated subspaces.

        Parameters
        ----------
        pc : PETSc.PC
            The preconditioner that supplies the original mixed operator.
        """

        from firedrake.assemble import get_assembler
        from firedrake.function import Function
        from firedrake.cofunction import Cofunction
        from firedrake.parloops import par_loop, INC
        from ufl import dx

        prefix = (pc.getOptionsPrefix() or "") + "condensed_field_"
        A, P = pc.getOperators()
        self.ctx = A.getPythonContext()
        if not isinstance(self.ctx, ImplicitMatrixContext):
            raise ValueError("Context must be an ImplicitMatrixContext")

        self.bilinear_form = self.ctx.a

        # Retrieve the mixed function space
        W = self.bilinear_form.arguments()[0].function_space()

        elim_option = (pc.getOptionsPrefix() or "") + "pc_sc_eliminate_fields"
        # By default, we condense down to the last field in the mixed space.
        elim_fields = PETSc.Options().getIntArray(elim_option, range(len(W)-1))
        elim_fields = list(map(int, elim_fields))

        c = tuple(i for i in range(len(W)) if i not in elim_fields)

        # The condensed operator is defined on the retained subspaces.
        Vc = W.sub(c)
        bcs = []
        for bc in self.ctx.row_bcs:
            if bc.function_space_index(W) not in c:
                raise NotImplementedError("Strong BC set on unsupported space")
            bcs.append(bc.reconstruct(field=c, V=Vc, g=0))

        self.c = c
        self.condensed_rhs = Cofunction(Vc.dual())
        self.residual = Cofunction(W.dual())
        self.solution = Function(W)

        # Count the cells that share each degree of freedom of the retained subspaces.
        self.weight = Function(Vc)
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        for Vi, wi in zip(Vc, self.weight.subfunctions):
            shapes = (Vi.finat_element.space_dimension(), np.prod(Vi.shape))
            domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
            par_loop((domain, instructions), dx, {"w": (wi, INC)})
        with self.weight.dat.vec_rw as wc:
            wc.reciprocal()

        # Get expressions for the condensed linear system
        A_tensor = Tensor(self.bilinear_form)
        reduced_sys, schur_builder = self.condensed_system(A_tensor, self.residual, elim_fields, prefix, pc)
        S_expr = reduced_sys.lhs
        r_expr = reduced_sys.rhs

        # Construct the condensed right-hand side
        self._assemble_Srhs = get_assembler(r_expr, bcs=bcs, form_compiler_parameters=self.ctx.fc_params).assemble

        # If a different matrix is used for preconditioning, condense it as well.
        S_pc_expr = None
        if A != P:
            P_tensor = Tensor(P.getPythonContext().a)
            S_pc_expr = self.condensed_system(P_tensor, self.residual, elim_fields, prefix, pc)[0].lhs

        self.initialize_condensed_ksp(pc, prefix, Vc, S_expr, bcs, "condensed_field_nullspace", S_pc_expr=S_pc_expr)

        # Set up local solvers for backwards substitution
        self.local_solvers = self.local_solver_callbacks(self.residual, self.solution, elim_fields, schur_builder)

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
        with self.residual.sub(self.c).dat.vec as vc, self.weight.dat.vec_ro as wc:
            vc.pointwiseMult(vc, wc)

        # Now assemble residual for the reduced problem
        self._assemble_Srhs(tensor=self.condensed_rhs)

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
