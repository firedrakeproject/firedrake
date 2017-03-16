"""This module provides custom python preconditioners utilizing
the Slate language.
"""

from __future__ import absolute_import, print_function, division

import ufl

from firedrake.matrix_free.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor


__all__ = ['HybridizationPC']


class HybridizationPC(PCBase):
    """A Slate-based python preconditioner that solves a
    mixed H(div)-conforming problem using hybridization.
    Currently, this preconditioner supports the hybridization
    of the RT and BDM mixed methods of arbitrary degree.

    The forward eliminations and backwards reconstructions
    are performed element-local using the Slate language.
    """
    def initialize(self, pc):
        """Set up the problem context. Take the original
        mixed problem and reformulate the problem as a
        hybridized mixed system.

        A KSP is created for the Lagrange multiplier system.
        """
        from firedrake import (FunctionSpace, Function, Constant,
                               TrialFunction, TrialFunctions, TestFunction,
                               DirichletBC, Projector)
        from firedrake.assemble import (allocate_matrix,
                                        create_assembly_callable)
        from firedrake.formmanipulation import split_form
        from ufl.algorithms.replace import replace

        # Extract the problem context
        prefix = pc.getOptionsPrefix() + "hybridization_"
        _, P = pc.getOperators()
        self.cxt = P.getPythonContext()

        assert isinstance(self.cxt, ImplicitMatrixContext), (
            "The python context must be an ImplicitMatrixContext!"
        )

        test, trial = self.cxt.a.arguments()

        V = test.function_space()
        mesh = V.mesh()
        if mesh.cell_set._extruded:
            # TODO: Merge FIAT branch to support TPC trace elements
            raise NotImplementedError("Not implemented on extruded meshes.")

        assert len(V) == 2, (
            "Can only hybridize a mixed system with two spaces."
        )

        # TODO: Future update to include more general spaces
        if all(Vi.ufl_element().value_shape() for Vi in V):
            raise ValueError(
                "Expecting an H(div) x L2 pair of spaces. "
                "Both spaces cannot be vector-valued."
            )

        # Automagically determine which spaces are vector and scalar
        for i, Vi in enumerate(V):
            if Vi.ufl_element().value_shape():
                self.vidx = i
            else:
                self.pidx = i

        # Create the space of approximate traces.
        # TODO: Once extruded and tensor product trace elements
        # are ready, this logic will be updated.
        W = V[self.vidx]
        hdiv_family = W.ufl_element().family()
        if hdiv_family == "Raviart-Thomas":
            tdegree = W.ufl_element().degree() - 1

        elif hdiv_family == "Brezzi-Douglas-Marini":
            tdegree = W.ufl_element().degree()

        else:
            raise ValueError(
                "%s not supported at the moment." % W.ufl_element().family()
            )

        TraceSpace = FunctionSpace(mesh, "HDiv Trace", tdegree)

        # We zero out the contribution of the trace variables on the exterior
        # boundary.
        # NOTE: For extruded, we will need to add "on_top" and "on_bottom"
        zero_trace_condition = [DirichletBC(TraceSpace, Constant(0.0),
                                            "on_boundary")]

        # Break the function spaces and define fully discontinuous spaces
        broken_elements = ufl.MixedElement([ufl.BrokenElement(Vi.ufl_element())
                                            for Vi in V])
        V_d = FunctionSpace(mesh, broken_elements)

        # Set up the functions for the original, hybridized
        # and schur complement systems
        self.broken_solution = Function(V_d)
        self.broken_rhs = Function(V_d)
        self.trace_solution = Function(TraceSpace)
        self.unbroken_solution = Function(V)
        self.unbroken_rhs = Function(V)

        # Create the symbolic Schur-reduction:
        # Original mixed operator replaced with "broken"
        # arguments
        arg_map = {test: TestFunction(V_d),
                   trial: TrialFunction(V_d)}
        Atilde = Tensor(replace(self.cxt.a, arg_map))
        gammar = TestFunction(TraceSpace)
        n = ufl.FacetNormal(mesh)
        sigma = TrialFunctions(V_d)[self.vidx]

        # NOTE: Once extruded is ready, this will change slightly
        # to include both horizontal and vertical interior facets
        K = Tensor(gammar('+') * ufl.dot(sigma, n) * ufl.dS)

        # Assemble the Schur complement operator and right-hand side
        self.schur_rhs = Function(TraceSpace)
        self._assemble_Srhs = create_assembly_callable(
            K * Atilde.inv * self.broken_rhs,
            tensor=self.schur_rhs,
            form_compiler_parameters=self.cxt.fc_params)

        schur_comp = K * Atilde.inv * K.T

        self.S = allocate_matrix(schur_comp,
                                 bcs=zero_trace_condition,
                                 form_compiler_parameters=self.cxt.fc_params)
        self._assemble_S = create_assembly_callable(
            schur_comp,
            tensor=self.S,
            bcs=zero_trace_condition,
            form_compiler_parameters=self.cxt.fc_params)

        self._assemble_S()
        self.S.force_evaluation()
        Smat = self.S.petscmat

        # Nullspace for the multiplier problem
        nullspace = create_schur_nullspace(P, K * Atilde.inv,
                                           V, V_d, TraceSpace,
                                           pc.comm)
        if nullspace:
            Smat.setNullSpace(nullspace)

        # Set up the KSP for the system of Lagrange multipliers
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(Smat)
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp

        split_mixed_op = dict(split_form(Atilde.form))
        split_trace_op = dict(split_form(K.form))
        opts = PETSc.Options()
        fact_type = opts.getString(prefix + "fieldsplit_schur_fact_type",
                                   "default")

        # Generate reconstruction calls
        self._reconstruction_calls(split_mixed_op, split_trace_op, fact_type)

        # Set up the projectors
        data_params = {"ksp_type": "preonly",
                       "pc_type": "bjacobi",
                       "sub_pc_type": "ilu"}
        self.data_projector = Projector(self.unbroken_rhs.split()[self.vidx],
                                        self.broken_rhs.split()[self.vidx],
                                        solver_parameters=data_params)

        # NOTE: Tolerance is very important here and so we provide
        # the user a way to specify projector tolerance
        tol = opts.getReal(prefix + "projector_tolerance", 1e-8)
        self.projector = Projector(self.broken_solution.split()[self.vidx],
                                   self.unbroken_solution.split()[self.vidx],
                                   solver_parameters={"ksp_type": "cg",
                                                      "ksp_rtol": tol})

    def _reconstruction_calls(self, split_mixed_op, split_trace_op, fact_type):
        """This generates the reconstruction calls for the unknowns using the
        Lagrange multipliers.

        :arg split_mixed_op: a ``dict`` of split forms that make up the broken
                             mixed operator from the original problem.
        :arg split_trace_op: a ``dict`` of split forms that make up the trace
                             contribution in the hybridized mixed system.
        :arg fact_type: a string denoting the order in which we eliminate
                        unknowns to generate the reconstruction expressions.

                        For example, "lower" will perform a lower Schur
                        complement on the broken mixed operator, which
                        eliminates the first unknown to generate an
                        expression for the second one.

                        If "upper", we perform an upper Schur complement which
                        eliminates the second unknown to arrive at solvable
                        expression for the first.

                        If the user does not specify an elimination order,
                        the defaulted behavior ("default") will eliminate
                        velocity first.
        """
        from firedrake.assemble import create_assembly_callable

        # NOTE: By construction of the matrix system, changing
        # from lower to upper eliminations requires only a simple
        # change in indices
        if fact_type == "default":
            id0, id1 = (self.vidx, self.pidx)
        elif fact_type == "lower":
            id0, id1 = (0, 1)
        elif fact_type == "upper":
            id0, id1 = (1, 0)
        else:
            raise ValueError(
                "%s not a recognized schur-fact type for this PC"
                % fact_type
            )

        # TODO: When PyOP2 is able to write into mixed dats,
        # the reconstruction expressions can simplify into
        # one clean expression.
        A = Tensor(split_mixed_op[(id0, id0)])
        B = Tensor(split_mixed_op[(id0, id1)])
        C = Tensor(split_mixed_op[(id1, id0)])
        D = Tensor(split_mixed_op[(id1, id1)])
        K_0 = Tensor(split_trace_op[(0, id0)])
        K_1 = Tensor(split_trace_op[(0, id1)])

        # Split functions and reconstruct each bit separately
        split_rhs = self.broken_rhs.split()
        split_sol = self.broken_solution.split()
        g = split_rhs[id0]
        f = split_rhs[id1]
        sigma = split_sol[id0]
        u = split_sol[id1]
        lambdar = self.trace_solution

        M = D - C * A.inv * B
        R = K_1.T - C * A.inv * K_0.T
        u_rec = M.inv * f - M.inv * (R * lambdar + C * A.inv * g)
        self._assemble_sub_unknown = create_assembly_callable(
            u_rec,
            tensor=u,
            form_compiler_parameters=self.cxt.fc_params)

        sigma_rec = A.inv * g - A.inv * (B * u + K_0.T * lambdar)
        self._assemble_elim_unknown = create_assembly_callable(
            sigma_rec,
            tensor=sigma,
            form_compiler_parameters=self.cxt.fc_params)

    def _reconstruct(self):
        """Reconstructs the system unknowns using the multipliers.
        Note that the reconstruction calls are assumed to be
        initialized at this point.
        """
        # We assemble the unknown which is an expression
        # of the first eliminated variable.
        self._assemble_sub_unknown()
        # Recover the eliminated unknown
        self._assemble_elim_unknown()

    def update(self, pc):
        """Update by assembling into the operator. No need to
        reconstruct symbolic objects.
        """
        self._assemble_S()
        self.S.force_evaluation()
        self._assemble_Srhs()

    def apply(self, pc, x, y):
        """We solve the forward eliminated problem for the
        approximate traces of the scalar solution (the multipliers)
        and reconstruct the "broken flux and scalar variable."

        Lastly, we project the broken solutions into the mimetic
        non-broken finite element space.
        """
        with self.unbroken_rhs.dat.vec as v:
            x.copy(v)

        # Transfer unbroken_rhs into broken_rhs
        unbroken_scalar_data = self.unbroken_rhs.split()[self.pidx]
        broken_scalar_data = self.broken_rhs.split()[self.pidx]
        self.data_projector.project()
        unbroken_scalar_data.dat.copy(broken_scalar_data.dat)

        # Compute the rhs for the multiplier system
        self._assemble_Srhs()

        # Solve the system for the Lagrange multipliers
        with self.schur_rhs.dat.vec_ro as b:
            with self.trace_solution.dat.vec as x:
                self.ksp.solve(b, x)

        # Reconstruct the unknowns
        self._reconstruct()

        # Project the broken solution into non-broken spaces
        broken_pressure = self.broken_solution.split()[self.pidx]
        unbroken_pressure = self.unbroken_solution.split()[self.pidx]
        broken_pressure.dat.copy(unbroken_pressure.dat)
        self.projector.project()

        with self.unbroken_solution.dat.vec_ro as v:
            v.copy(y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""
        raise NotImplementedError(
            "The transpose application of this PC"
            "is not implemented."
        )

    def view(self, pc, viewer=None):
        super(HybridizationPC, self).view(pc, viewer)
        viewer.printfASCII("Solves K * P^-1 * K.T using local eliminations.\n")
        viewer.pushASCIITab()
        viewer.printfASCII("KSP solver for the multipliers:\n")
        viewer.pushASCIITab()
        self.ksp.view(viewer)
        viewer.popASCIITab()


def create_schur_nullspace(P, forward, V, V_d, TraceSpace, comm):
    """Gets the nullspace vectors corresponding to the Schur complement
    system for the multipliers.

    :arg P: The mixed operator from the ImplicitMatrixContext.
    :arg forward: A Slate expression denoting the forward elimination
                  operator.
    :arg V: The original "unbroken" space.
    :arg V_d: The broken space.
    :arg TraceSpace: The space of approximate traces.

    Returns: A nullspace (if there is one) for the Schur-complement system.
    """
    from firedrake import assemble, Function, project

    nullspace = P.getNullSpace()
    if nullspace.handle == 0:
        # No nullspace
        return None

    vecs = nullspace.getVecs()
    tmp = Function(V)
    tmp_b = Function(V_d)
    tnsp_tmp = Function(TraceSpace)
    forward_action = forward * tmp_b
    new_vecs = []
    for v in vecs:
        with tmp.dat.vec as t:
            v.copy(t)

        project(tmp, tmp_b)
        assemble(forward_action, tensor=tnsp_tmp)
        with tnsp_tmp.dat.vec_ro as v:
            new_vecs.append(v.copy())

    schur_nullspace = PETSc.NullSpace().create(vectors=new_vecs,
                                               comm=comm)
    return schur_nullspace
