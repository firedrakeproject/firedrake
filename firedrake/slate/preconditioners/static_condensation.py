import ufl

from firedrake.logging import log, WARNING
from firedrake.matrix_free.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor, AssembledVector
from firedrake.slate.preconditioners.pc_utils import (create_sc_nullspace,
                                                      get_transfer_kernels)
from firedrake.parloops import par_loop, READ, WRITE
from pyop2.profiling import timed_region, timed_function


__all__ = ['StaticCondensationPC']


class StaticCondensationPC(PCBase):
    """A Slate-based python preconditioner that solves an
    H1-conforming problem using static condensation.
    """

    @timed_function("SCInit")
    def initialize(self, pc):
        """Set up the problem context. Take the original
        H1-problem and partition the spaces/functions
        into 'interior' and 'facet' parts.

        A KSP is created for the reduced system after
        static condensation is applied.
        """
        from firedrake import (FunctionSpace, Function,
                               TrialFunction, TestFunction,
                               DirichletBC, interpolate)
        from firedrake.assemble import (allocate_matrix,
                                        create_assembly_callable)
        from ufl.algorithms.replace import replace

        # Extract python context
        prefix = pc.getOptionsPrefix() + "static_condensation_"
        _, P = pc.getOperators()
        self.cxt = P.getPythonContext()

        if not isinstance(self.cxt, ImplicitMatrixContext):
            raise ValueError("Context must be an ImplicitMatrixContext")

        test, trial = self.cxt.a.arguments()
        V = test.function_space()
        mesh = V.mesh()

        if len(V) > 1:
            raise ValueError("Cannot use this PC for mixed problems.")

        if V.ufl_element().sobolev_space().name != "H1":
            raise ValueError("Expecting an H1-conforming element.")

        if not V.ufl_element().cell().is_simplex():
            raise NotImplementedError("Only simplex meshes are implemented.")

        top_dim = V.finat_element._element.ref_el.get_dimension()
        if not V.finat_element.entity_dofs()[top_dim][0]:
            raise RuntimeError("There are no interior dofs to eliminate.")

        # We decompose the space into an interior part and facet part
        interior_element = V.ufl_element()["interior"]
        facet_element = V.ufl_element()["facet"]
        V_int = FunctionSpace(mesh, interior_element)
        V_facet = FunctionSpace(mesh, facet_element)

        # Get transfer kernel for moving data
        self._transfer_kernel = get_transfer_kernels({'h1-space': V,
                                                      'interior-space': V_int,
                                                      'facet-space': V_facet})

        # Set up functions for the H1 functions and the interior/trace parts
        self.trace_solution = Function(V_facet)
        self.interior_solution = Function(V_int)
        self.h1_solution = Function(V)
        self.h1_residual = Function(V)
        self.interior_residual = Function(V_int)
        self.trace_residual = Function(V_facet)

        # Collect BCs for the facet problem
        bcs = []
        for bc in self.cxt.row_bcs:
            if isinstance(bc.function_arg, Function):
                g = interpolate(bc.function_arg, V_facet)
            else:
                g = bc.function_arg
            bcs.append(DirichletBC(V_facet, g, bc.sub_domain))

        if bcs:
            msg = ("Currently strong bcs are not handled correctly. "
                   "The solver may still converge with this PC if an "
                   "appropriate iterative method is used. ")
            log(WARNING, msg)

        self.bcs = bcs

        A00 = Tensor(replace(self.cxt.a, {test: TestFunction(V_int),
                                          trial: TrialFunction(V_int)}))
        A01 = Tensor(replace(self.cxt.a, {test: TestFunction(V_int),
                                          trial: TrialFunction(V_facet)}))
        A10 = Tensor(replace(self.cxt.a, {test: TestFunction(V_facet),
                                          trial: TrialFunction(V_int)}))
        A11 = Tensor(replace(self.cxt.a, {test: TestFunction(V_facet),
                                          trial: TrialFunction(V_facet)}))

        # Schur complement operator
        S = A11 - A10 * A00.inv * A01
        self.S = allocate_matrix(S, bcs=self.bcs,
                                 form_compiler_parameters=self.cxt.fc_params)
        self._assemble_S = create_assembly_callable(
            S,
            tensor=self.S,
            bcs=self.bcs,
            form_compiler_parameters=self.cxt.fc_params)

        self._assemble_S()
        self.S.force_evaluation()
        Smat = self.S.petscmat

        # Nullspace for the reduced system
        nullspace = create_sc_nullspace(P, V, V_facet, pc.comm)

        if nullspace:
            Smat.setNullSpace(nullspace)

        # Set up KSP for the reduced problem
        sc_ksp = PETSc.KSP().create(comm=pc.comm)
        sc_ksp.setOptionsPrefix(prefix)
        sc_ksp.setOperators(Smat)
        sc_ksp.setUp()
        sc_ksp.setFromOptions()
        self.sc_ksp = sc_ksp

        # Set up rhs for the reduced problem
        F0 = AssembledVector(self.interior_residual)
        self.sc_rhs = Function(V_facet)
        self.sc_rhs_thunk = Function(V_facet)
        self._assemble_sc_rhs_thunk = create_assembly_callable(
            -A10 * A00.inv * F0,
            tensor=self.sc_rhs_thunk,
            form_compiler_parameters=self.cxt.fc_params)

        # Reconstruction calls
        u_facet = AssembledVector(self.trace_solution)
        self._assemble_interior_u = create_assembly_callable(
            A00.inv * (F0 - A01 * u_facet),
            tensor=self.interior_solution,
            form_compiler_parameters=self.cxt.fc_params)

    @timed_function("SCRecon")
    def _reconstruct(self):
        """Locally solve for the interior degrees of
        freedom using the computed unknowns for the facets.
        A transfer kernel is used to join the interior and
        facet solutions together.
        """
        with timed_region("SCAssembleInterior"):
            self._assemble_interior_u()

        u_int = self.interior_solution
        u_facet = self.trace_solution

        with timed_region("SCReconSolution"):
            par_loop(self._transfer_kernel.join,
                     ufl.dx,
                     {"x": (self.h1_solution, WRITE),
                      "x_int": (u_int, READ),
                      "x_facet": (u_facet, READ)})

    @timed_function("SCTransferResidual")
    def _partition_residual(self):
        """Partition the incoming right-hand side residual
        into 'interior' and 'facet' sections.
        """
        r_int = self.interior_residual
        r_facet = self.trace_residual

        par_loop(self._transfer_kernel.partition,
                 ufl.dx,
                 {"x_int": (r_int, WRITE),
                  "x_facet": (r_facet, WRITE),
                  "x": (self.h1_residual, READ)})

    @timed_function("SCUpdate")
    def update(self, pc):
        """Update by assembling into the KSP operator. No
        need to reconstruct symbolic objects.
        """
        self._assemble_S()
        self.S.force_evaluation()

    def apply(self, pc, x, y):
        """Solve the reduced system for the facet degrees of
        freedom after static condensation is applied. The
        computed solution is used to solve element-wise problems
        for the interior degrees of freedom.
        """
        with timed_region("SCTransfer"):
            with self.h1_residual.dat.vec_wo as v:
                x.copy(v)

            # Partition residual data into interior and facet sections
            self._partition_residual()

        # Now that the residual data is transfered, we assemble
        # the RHS for the reduced system
        with timed_region("SCRHS"):
            self._assemble_sc_rhs_thunk()

            # Assemble the RHS of the reduced system:
            # If r = [F, G] is the incoming residual separated
            # into "facet" and "interior" restrictions, then
            # the Schur complement RHS is:
            # G - A10 * A00.inv * F.
            # This is assembled point-wise, with -A10 * A00.inv * F
            # precomputed element-wise using Slate.
            self.sc_rhs.assign(self.trace_residual
                               + self.sc_rhs_thunk)

        with timed_region("SCSolve"):
            # Solve the reduced problem
            with self.sc_rhs.dat.vec_ro as b:
                if self.sc_ksp.getInitialGuessNonzero():
                    acc = self.trace_solution.dat.vec
                else:
                    acc = self.trace_solution.dat.vec_wo
                with acc as x_trace:
                    self.sc_ksp.solve(b, x_trace)

        with timed_region("SCRecover"):
            self._reconstruct()

            with self.h1_solution.dat.vec_ro as v:
                v.copy(y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""
        raise NotImplementedError("Transpose not implemented.")
