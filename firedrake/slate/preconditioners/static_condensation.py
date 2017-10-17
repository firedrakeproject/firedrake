import ufl

from firedrake.matrix_free.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor, AssembledVector
from firedrake.parloops import par_loop, READ, INC
from pyop2.profiling import timed_region, timed_function

import numpy as np


__all__ = ['StaticCondensationPC']


class StaticCondensationPC(PCBase):
    """
    """

    @timed_function("SCInit")
    def initialize(self, pc):
        """
        """
        from firedrake import (FunctionSpace, Function,
                               TrialFunction, TestFunction)
        from firedrake.assemble import (allocate_matrix,
                                        create_assembly_callable)
        from ufl.algorithms.replace import replace

        # Extract python context
        prefix = pc.getOptionsPrefix() + "static_condensation_"
        _, P = pc.getOperators()
        self.cxt = P.getPythonContext()

        if not isinstance(self.cxt, ImplicitMatrixContext):
            raise ValueError("The python context must be an ImplicitMatrixContext")

        test, trial = self.cxt.a.arguments()
        V = test.function_space()
        mesh = V.mesh()

        if len(V) > 1:
            raise ValueError("Cannot use this PC for mixed problems.")

        if V.ufl_element().sobolev_space().name != "H1":
            raise ValueError("Expecting an H1-conforming element.")

        if not V.ufl_element().cell().is_simplex():
            raise NotImplementedError(
                "Only simplex meshes are implemented."
            )

        # We decompose the space into an interior part and facet part
        interior_element = V.ufl_element()["interior"]
        facet_element = V.ufl_element()["facet"]
        V_int = FunctionSpace(mesh, interior_element)
        V_facet = FunctionSpace(mesh, facet_element)

        # Set up functions for the H1 functions and the interior/trace parts
        self.trace_solution = Function(V_facet)
        self.interior_solution = Function(V_int)
        self.h1_solution = Function(V)
        self.h1_residual = Function(V)
        self.interior_residual = Function(V_int)
        self.trace_residual = Function(V_facet)

        # TODO: I think strong BCs just need to be applied to the facet space
        if self.cxt.row_bcs:
            raise NotImplementedError("Strong BCs not yet implemented.")

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
        self.S = allocate_matrix(S, bcs=None,
                                 form_compiler_parameters=self.cxt.fc_params)
        self._assemble_S = create_assembly_callable(
            S,
            tensor=self.S,
            bcs=None,
            form_compiler_parameters=self.cxt.fc_params)

        self._assemble_S()
        self.S.force_evaluation()
        Smat = self.S.petscmat

        # TODO: Nullspace

        # Set up KSP for the reduced problem
        sc_ksp = PETSc.KSP().create(comm=pc.comm)
        sc_ksp.setOptionsPrefix(prefix)
        sc_ksp.setOperators(Smat)
        sc_ksp.setUp()
        sc_ksp.setFromOptions()
        self.sc_ksp = sc_ksp

        # Set up rhs for the reduced problem
        F0 = AssembledVector(self.interior_residual)
        F1 = AssembledVector(self.trace_residual)
        self.sc_rhs = Function(V_facet)
        self._assemble_sc_rhs = create_assembly_callable(
            F1 - A10 * A00.inv * F0,
            tensor=self.sc_rhs,
            form_compiler_parameters=self.cxt.fc_params)

        # Reconstruction calls
        u_facet = AssembledVector(self.trace_solution)
        self._assemble_interior_u = create_assembly_callable(
            A00.inv * (F0 - A01 * u_facet),
            tensor=self.interior_solution,
            form_compiler_parameters=self.cxt.fc_params)

    @timed_function("SCRecon")
    def _reconstruct(self):
        """
        """
        with timed_region("SCAssembleInterior"):
            self._assemble_interior_u()

        u_int = self.interior_solution
        u_facet = self.trace_solution
        Vo = u_int.function_space()
        Vd = u_facet.function_space()
        V = self.h1_solution.function_space()

        # Offset for interior dof mapping is determined by inspecting the
        # entity dofs of V (original FE space) and the dofs of V_o. For
        # example, degree 5 CG element has entity dofs:
        #
        # {0: {0: [0], 1: [1], 2: [2]}, 1: {0: [3, 4, 5, 6], 1: [7, 8, 9, 10],
        #  2: [11, 12, 13, 14]}, 2: {0: [15, 16, 17, 18, 19, 20]}}.
        #
        # Looking at the cell dofs, we have a starting dof index of 15. The
        # interior element has dofs:
        #
        # {0: {0: [], 1: [], 2: []}, 1: {0: [], 1:[], 2:[]},
        #  2: {0: [0, 1, 2, 3, 4, 5]}}
        #
        # with a starting dof index of 0. So the par_loop will need to be
        # adjusted by the difference: i + 15. The skeleton dofs do no need
        # any offsets.

        # TODO: There must be a cleaner way of getting the offset
        offset = V.finat_element.entity_dofs()[2][0][0]
        args = (Vo.finat_element.space_dimension(), np.prod(Vo.shape),
                offset,
                Vd.finat_element.space_dimension(), np.prod(Vd.shape))

        kernel = """
        for (int i=0; i<%d; ++i){
            for (int j=0; j<%d; ++j){
                uh[i + %d][j] = u_int[i][j];
            }
        }

        for (int i=0; i<%d; ++i){
            for (int j=0; j<%d; ++j){
                uh[i][j] = u_facet[i][j];
            }
        }""" % args

        with timed_region("SCReconSolution"):
            par_loop(kernel, ufl.dx, {"uh": (self.h1_solution, INC),
                                      "u_int": (u_int, READ),
                                      "u_facet": (u_facet, READ)})

    @timed_function("SCTransferResidual")
    def _transfer_residual(self):
        """
        """
        r_int = self.interior_residual
        r_facet = self.trace_residual
        Vo = r_int.function_space()
        Vd = r_facet.function_space()
        V = self.h1_residual.function_space()

        # TODO: There must be a cleaner way of getting the offset
        offset = V.finat_element.entity_dofs()[2][0][0]
        args = (Vo.finat_element.space_dimension(), np.prod(Vo.shape),
                offset,
                Vd.finat_element.space_dimension(), np.prod(Vd.shape))

        kernel = """
        for (int i=0; i<%d; ++i){
            for (int j=0; j<%d; ++j){
                r_int[i][j] = r_h[i + %d][j];
            }
        }

        for (int i=0; i<%d; ++i){
            for (int j=0; j<%d; ++j){
                r_facet[i][j] = r_h[i][j];
            }
        }""" % args

        par_loop(kernel, ufl.dx, {"r_int": (r_int, INC),
                                  "r_facet": (r_facet, INC),
                                  "r_h": (self.h1_residual, READ)})

    @timed_function("SCUpdate")
    def update(self, pc):
        """
        """
        self._assemble_S()
        self.S.force_evaluation()

    def apply(self, pc, x, y):
        """
        """
        with timed_region("SCTransfer"):
            with self.h1_residual.dat.vec_wo as v:
                x.copy(v)

            # Transfer residual data into interior and facet
            # functions
            self._transfer_residual()

        # Now that the residual data is transfered, we assemble
        # the RHS for the reduced system
        with timed_region("SCRHS"):
            self._assemble_sc_rhs()

        with timed_region("SCSolve"):
            # Solve the reduced problem
            with self.sc_rhs.dat.vec_ro as b:
                if self.trace_solution.getInitialGuessNonzero():
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
