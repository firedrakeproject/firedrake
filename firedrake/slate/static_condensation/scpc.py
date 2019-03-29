from firedrake.slate.static_condensation.sc_base import SCBase
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
        _, P = pc.getOperators()
        self.cxt = P.getPythonContext()
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

        # Get expressions for the condensed linear system
        A = Tensor(self.bilinear_form)
        reduced_sys = self.condensed_system(A, self.residual, elim_fields)
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
                                 mat_type=mat_type)
        self._assemble_S = create_assembly_callable(
            S_expr,
            tensor=self.S,
            bcs=bcs,
            form_compiler_parameters=self.cxt.fc_params,
            mat_type=mat_type)

        self._assemble_S()
        self.S.force_evaluation()
        Smat = self.S.petscmat

        # Get nullspace for the condensed operator (if any).
        # This is provided as a user-specified callback which
        # returns the basis for the nullspace.
        nullspace = self.cxt.appctx.get("condensed_field_nullspace", None)
        if nullspace is not None:
            nsp = nullspace(Vc)
            Smat.setNullSpace(nsp.nullspace(comm=pc.comm))

        # Set up ksp for the condensed problem
        c_ksp = PETSc.KSP().create(comm=pc.comm)
        c_ksp.incrementTabLevel(1, parent=pc)
        c_ksp.setOptionsPrefix(prefix)
        c_ksp.setOperators(Smat)
        c_ksp.setUp()
        c_ksp.setFromOptions()
        self.condensed_ksp = c_ksp

        # Set up local solvers for backwards substitution
        self.local_solvers = self.local_solver_calls(A, self.residual,
                                                     self.solution,
                                                     elim_fields)

    def condensed_system(self, A, rhs, elim_fields):
        """Forms the condensed linear system by eliminating
        specified unknowns.

        :arg A: A Slate Tensor containing the mixed bilinear form.
        :arg rhs: A firedrake function for the right-hand side.
        :arg elim_fields: An iterable of field indices to eliminate.
        """

        from firedrake.slate.static_condensation.la_utils import condense_and_forward_eliminate

        return condense_and_forward_eliminate(A, rhs, elim_fields)

    def local_solver_calls(self, A, rhs, x, elim_fields):
        """Provides solver callbacks for inverting local operators
        and reconstructing eliminated fields.

        :arg A: A Slate Tensor containing the mixed bilinear form.
        :arg rhs: A firedrake function for the right-hand side.
        :arg x: A firedrake function for the solution.
        :arg elim_fields: An iterable of eliminated field indices
                          to recover.
        """

        from firedrake.slate.static_condensation.la_utils import backward_solve
        from firedrake.assemble import create_assembly_callable

        fields = x.split()
        systems = backward_solve(A, rhs, x, reconstruct_fields=elim_fields)

        local_solvers = []
        for local_system in systems:
            Ae = local_system.lhs
            be = local_system.rhs
            i, = local_system.field_idx
            local_solve = Ae.solve(be, decomposition="PartialPivLU")
            solve_call = create_assembly_callable(
                local_solve,
                tensor=fields[i],
                form_compiler_parameters=self.cxt.fc_params)
            local_solvers.append(solve_call)

        return local_solvers

    @timed_function("SCPCUpdate")
    def update(self, pc):
        """Update by assembling into the KSP operator. No
        need to reconstruct symbolic objects.
        """

        self._assemble_S()
        self.S.force_evaluation()

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
        for local_solver_call in self.local_solvers:
            local_solver_call()

        with self.solution.dat.vec_ro as w:
            w.copy(y)

    def view(self, pc, viewer=None):
        """Viewer calls for the various configurable objects in this PC."""

        viewer.printfASCII("Static condensation preconditioner\n")
        viewer.printfASCII("KSP to solve the reduced system:\n")
        self.condensed_ksp.view(viewer=viewer)
