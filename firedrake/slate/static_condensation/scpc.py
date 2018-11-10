from firedrake.slate.static_condensation.sc_base import SCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor, AssembledVector
from firedrake.utils import cached_property
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

        # Retrieve the mixed function space
        W = self.cxt.a.arguments()[0].function_space()
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

        # Perform symbolics only once
        S_expr, r_lambda_expr, u_h_expr, q_h_expr = self._slate_expressions

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

        q_h, u_h, lambda_h = self.solution.split()

        # Assemble u_h using lambda_h
        self._assemble_u = create_assembly_callable(
            u_h_expr,
            tensor=u_h,
            form_compiler_parameters=self.cxt.fc_params)

        # Recover q_h using both u_h and lambda_h
        self._assemble_q = create_assembly_callable(
            q_h_expr,
            tensor=q_h,
            form_compiler_parameters=self.cxt.fc_params)

    @cached_property
    def _slate_expressions(self):
        """Returns all the relevant Slate expressions
        for the static condensation and local recovery
        procedures.
        """
        # This operator has the form:
        # | A  B  C |
        # | D  E  F |
        # | G  H  J |
        # NOTE: It is often the case that D = B.T,
        # G = C.T, H = F.T, and J = 0, but we're not making
        # that assumption here.
        _O = Tensor(self.cxt.a)
        O = _O.blocks

        # Extract sub-block:
        # | A B |
        # | D E |
        # which has block row indices (0, 1) and block
        # column indices (0, 1) as well.
        M = O[:2, :2]

        # Extract sub-block:
        # | C |
        # | F |
        # which has block row indices (0, 1) and block
        # column indices (2,)
        K = O[:2, 2]

        # Extract sub-block:
        # | G H |
        # which has block row indices (2,) and block column
        # indices (0, 1)
        L = O[2, :2]

        # And the final block J has block row-column
        # indices (2, 2)
        J = O[2, 2]

        # Schur complement for traces
        S = J - L * M.inv * K

        # Create mixed function for residual computation.
        # This projects the non-trace residual bits into
        # the trace space:
        # -L * M.inv * | v1 v2 |^T
        _R = AssembledVector(self.residual)
        R = _R.blocks
        v1v2 = R[:2]
        v3 = R[2]
        r_lambda = v3 - L * M.inv * v1v2

        # Reconstruction expressions
        q_h, u_h, lambda_h = self.solution.split()

        # Local tensors needed for reconstruction
        A = O[0, 0]
        B = O[0, 1]
        C = O[0, 2]
        D = O[1, 0]
        E = O[1, 1]
        F = O[1, 2]
        Se = E - D * A.inv * B
        Sf = F - D * A.inv * C

        v1, v2, v3 = self.residual.split()

        # Solve locally using LU (with partial pivoting)
        u_h_expr = Se.solve(AssembledVector(v2) -
                            D * A.inv * AssembledVector(v1) -
                            Sf * AssembledVector(lambda_h),
                            decomposition="PartialPivLU")

        q_h_expr = A.solve(AssembledVector(v1) -
                           B * AssembledVector(u_h) -
                           C * AssembledVector(lambda_h),
                           decomposition="PartialPivLU")

        return (S, r_lambda, u_h_expr, q_h_expr)

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
        self._assemble_u()
        self._assemble_q()

        with self.solution.dat.vec_ro as w:
            w.copy(y)

    def view(self, pc, viewer=None):
        viewer.printfASCII("Static condensation preconditioner\n")
        viewer.printfASCII("KSP to solve the reduced system:\n")
        self.trace_ksp.view(viewer=viewer)
