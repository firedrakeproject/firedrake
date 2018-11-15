from firedrake.slate.static_condensation.sc_base import SCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor
from pyop2.profiling import timed_function


__all__ = ['SCPC']


class SCPC(SCBase):
    """A Slate-based python preconditioner implementation of
    static condensation for three-field problems. This also
    includes the local recovery of the eliminated unknowns.
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
        if len(W) != 3:
            raise NotImplementedError("Only supports three function spaces.")

        # Extract space of the condensed field
        # TODO: Make more general.
        T = W[2]

        # Need to duplicate a space which is NOT
        # associated with a subspace of a mixed space.
        Tr = FunctionSpace(T.mesh(), T.ufl_element())
        bcs = []
        cxt_bcs = self.cxt.row_bcs
        for bc in cxt_bcs:
            assert bc.function_space() == T
            if isinstance(bc.function_arg, Function):
                bc_arg = interpolate(bc.function_arg, Tr)
            else:
                # Constants don't need to be interpolated
                bc_arg = bc.function_arg
            bcs.append(DirichletBC(Tr, bc_arg, bc.sub_domain))

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")

        self.r_lambda = Function(T)
        self.residual = Function(W)
        self.solution = Function(W)

        A = Tensor(self.bilinear_form)
        reduced_sys = self.condensed_system(A, elim_fields=(0, 1))
        S_expr = reduced_sys.lhs
        r_lambda_expr = reduced_sys.rhs

        self.local_solvers = self.local_solver_calls(A, reconstruct_fields=(0, 1))

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

        # Set up ksp for the trace problem
        trace_ksp = PETSc.KSP().create(comm=pc.comm)
        trace_ksp.incrementTabLevel(1, parent=pc)
        trace_ksp.setOptionsPrefix(prefix)
        trace_ksp.setOperators(Smat)
        trace_ksp.setUp()
        trace_ksp.setFromOptions()
        self.trace_ksp = trace_ksp

        self._assemble_Srhs = create_assembly_callable(
            r_lambda_expr,
            tensor=self.r_lambda,
            form_compiler_parameters=self.cxt.fc_params)

    def condensed_system(self, A, elim_fields):
        """
        """

        from firedrake.slate.static_condensation.la_utils import condense_and_forward_eliminate

        return condense_and_forward_eliminate(A, self.residual, elim_fields)

    def local_solver_calls(self, A, reconstruct_fields):
        """
        """

        from firedrake.slate.static_condensation.la_utils import backward_solve
        from firedrake.assemble import create_assembly_callable

        fields = self.solution.split()
        systems = backward_solve(A, self.residual, self.solution,
                                 reconstruct_fields=reconstruct_fields)

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
        """
        """

        with self.residual.dat.vec_wo as v:
            x.copy(v)

        # Now assemble residual for the reduced problem
        self._assemble_Srhs()

    def sc_solve(self, pc):
        """
        """

        with self.r_lambda.dat.vec_ro as b:
            if self.trace_ksp.getInitialGuessNonzero():
                acc = self.solution.split()[2].dat.vec
            else:
                acc = self.solution.split()[2].dat.vec_wo
            with acc as x_trace:
                self.trace_ksp.solve(b, x_trace)

    def backward_substitution(self, pc, y):
        """
        """

        # Recover eliminated unknowns
        for local_solver_call in self.local_solvers:
            local_solver_call()

        with self.solution.dat.vec_ro as w:
            w.copy(y)

    def view(self, pc, viewer=None):
        viewer.printfASCII("Static condensation preconditioner\n")
        viewer.printfASCII("KSP to solve the reduced system:\n")
        self.trace_ksp.view(viewer=viewer)
