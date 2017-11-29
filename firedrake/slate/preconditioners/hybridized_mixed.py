import ufl

from firedrake.matrix_free.preconditioners import PCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor, AssembledVector
from firedrake.slate.preconditioners.pc_utils import create_trace_nullspace
from pyop2.profiling import timed_region, timed_function
from pyop2.utils import as_tuple

__all__ = ['HybridizationPC']


class HybridizationPC(PCBase):
    """A Slate-based python preconditioner that solves a
    mixed H(div)-conforming problem using hybridization.
    Currently, this preconditioner supports the hybridization
    of the RT and BDM mixed methods of arbitrary degree.

    The forward eliminations and backwards reconstructions
    are performed element-local using the Slate language.
    """

    @timed_function("HybridInit")
    def initialize(self, pc):
        """Set up the problem context. Take the original
        mixed problem and reformulate the problem as a
        hybridized mixed system.

        A KSP is created for the Lagrange multiplier system.
        """
        from firedrake import (FunctionSpace, Function, Constant,
                               TrialFunction, TrialFunctions, TestFunction,
                               DirichletBC, assemble)
        from firedrake.assemble import (allocate_matrix,
                                        create_assembly_callable)
        from firedrake.formmanipulation import split_form
        from ufl.algorithms.replace import replace

        # Extract the problem context
        prefix = pc.getOptionsPrefix() + "hybridization_"
        _, P = pc.getOperators()
        self.cxt = P.getPythonContext()

        if not isinstance(self.cxt, ImplicitMatrixContext):
            raise ValueError("Context must be an ImplicitMatrixContext")

        test, trial = self.cxt.a.arguments()

        V = test.function_space()
        mesh = V.mesh()

        if len(V) != 2:
            raise ValueError("Expecting two function spaces.")

        if all(Vi.ufl_element().value_shape() for Vi in V):
            raise ValueError("Expecting an H(div) x L2 pair of spaces.")

        # Automagically determine which spaces are vector and scalar
        for i, Vi in enumerate(V):
            if Vi.ufl_element().sobolev_space().name == "HDiv":
                self.vidx = i
            else:
                assert Vi.ufl_element().sobolev_space().name == "L2"
                self.pidx = i

        # Create the space of approximate traces.
        W = V[self.vidx]
        if W.ufl_element().family() == "Brezzi-Douglas-Marini":
            tdegree = W.ufl_element().degree()

        else:
            try:
                # If we have a tensor product element
                h_deg, v_deg = W.ufl_element().degree()
                tdegree = (h_deg - 1, v_deg - 1)

            except TypeError:
                tdegree = W.ufl_element().degree() - 1

        TraceSpace = FunctionSpace(mesh, "HDiv Trace", tdegree)

        # Break the function spaces and define fully discontinuous spaces
        broken_elements = [ufl.BrokenElement(Vi.ufl_element()) for Vi in V]
        V_d = FunctionSpace(mesh, ufl.MixedElement(broken_elements))

        # Set up the functions for the original, hybridized
        # and schur complement systems
        self.broken_solution = Function(V_d)
        self.broken_residual = Function(V_d)
        self.trace_solution = Function(TraceSpace)
        self.unbroken_solution = Function(V)
        self.unbroken_residual = Function(V)

        # Set up the KSP for the hdiv residual projection
        hdiv_mass_ksp = PETSc.KSP().create(comm=pc.comm)
        hdiv_mass_ksp.setOptionsPrefix(prefix + "hdiv_residual_")

        # HDiv mass operator
        p = TrialFunction(V[self.vidx])
        q = TestFunction(V[self.vidx])
        mass = ufl.dot(p, q)*ufl.dx
        M = assemble(mass, form_compiler_parameters=self.cxt.fc_params)
        M.force_evaluation()
        Mmat = M.petscmat

        hdiv_mass_ksp.setOperators(Mmat)
        hdiv_mass_ksp.setUp()
        hdiv_mass_ksp.setFromOptions()
        self.hdiv_mass_ksp = hdiv_mass_ksp

        # Storing the result of A.inv * r, where A is the HDiv
        # mass matrix and r is the HDiv residual
        self._primal_r = Function(V[self.vidx])

        tau = TestFunction(V_d[self.vidx])
        self._assemble_broken_r = create_assembly_callable(
            ufl.dot(self._primal_r, tau)*ufl.dx,
            tensor=self.broken_residual.split()[self.vidx],
            form_compiler_parameters=self.cxt.fc_params)

        # Create the symbolic Schur-reduction:
        # Original mixed operator replaced with "broken"
        # arguments
        arg_map = {test: TestFunction(V_d),
                   trial: TrialFunction(V_d)}
        Atilde = Tensor(replace(self.cxt.a, arg_map))
        gammar = TestFunction(TraceSpace)
        n = ufl.FacetNormal(mesh)
        sigma = TrialFunctions(V_d)[self.vidx]

        if mesh.cell_set._extruded:
            Kform = (gammar('+') * ufl.dot(sigma, n) * ufl.dS_h +
                     gammar('+') * ufl.dot(sigma, n) * ufl.dS_v)
        else:
            Kform = (gammar('+') * ufl.dot(sigma, n) * ufl.dS)

        # Here we deal with boundaries. If there are Neumann
        # conditions (which should be enforced strongly for
        # H(div)xL^2) then we need to add jump terms on the exterior
        # facets. If there are Dirichlet conditions (which should be
        # enforced weakly) then we need to zero out the trace
        # variables there as they are not active (otherwise the hybrid
        # problem is not well-posed).

        # If boundary conditions are contained in the ImplicitMatrixContext:
        if self.cxt.row_bcs:
            # Find all the subdomains with neumann BCS
            # These are Dirichlet BCs on the vidx space
            neumann_subdomains = set()
            for bc in self.cxt.row_bcs:
                if bc.function_space().index == self.pidx:
                    raise NotImplementedError("Dirichlet conditions for scalar variable not supported. Use a weak bc")
                if bc.function_space().index != self.vidx:
                    raise NotImplementedError("Dirichlet bc set on unsupported space.")
                # append the set of sub domains
                subdom = bc.sub_domain
                if isinstance(subdom, str):
                    neumann_subdomains |= set([subdom])
                else:
                    neumann_subdomains |= set(as_tuple(subdom, int))

            # separate out the top and bottom bcs
            extruded_neumann_subdomains = neumann_subdomains & {"top", "bottom"}
            neumann_subdomains = neumann_subdomains.difference(extruded_neumann_subdomains)

            integrand = gammar * ufl.dot(sigma, n)
            measures = []
            trace_subdomains = []
            if mesh.cell_set._extruded:
                ds = ufl.ds_v
                for subdomain in extruded_neumann_subdomains:
                    measures.append({"top": ufl.ds_t, "bottom": ufl.ds_b}[subdomain])
                trace_subdomains.extend(sorted({"top", "bottom"} - extruded_neumann_subdomains))
            else:
                ds = ufl.ds
            if "on_boundary" in neumann_subdomains:
                measures.append(ds)
            else:
                measures.append(ds(tuple(neumann_subdomains)))
                dirichlet_subdomains = set(mesh.exterior_facets.unique_markers) - neumann_subdomains
                trace_subdomains.append(sorted(dirichlet_subdomains))

            for measure in measures:
                Kform += integrand*measure

            trace_bcs = [DirichletBC(TraceSpace, Constant(0.0), subdomain) for subdomain in trace_subdomains]

        else:
            # No bcs were provided, we assume weak Dirichlet conditions.
            # We zero out the contribution of the trace variables on
            # the exterior boundary. Extruded cells will have both
            # horizontal and vertical facets
            trace_subdomains = ["on_boundary"]
            if mesh.cell_set._extruded:
                trace_subdomains.extend(["bottom", "top"])
            trace_bcs = [DirichletBC(TraceSpace, Constant(0.0), subdomain) for subdomain in trace_subdomains]

        # Make a SLATE tensor from Kform
        K = Tensor(Kform)

        # Assemble the Schur complement operator and right-hand side
        self.schur_rhs = Function(TraceSpace)
        self._assemble_Srhs = create_assembly_callable(
            K * Atilde.inv * AssembledVector(self.broken_residual),
            tensor=self.schur_rhs,
            form_compiler_parameters=self.cxt.fc_params)

        schur_comp = K * Atilde.inv * K.T
        self.S = allocate_matrix(schur_comp, bcs=trace_bcs,
                                 form_compiler_parameters=self.cxt.fc_params)
        self._assemble_S = create_assembly_callable(schur_comp,
                                                    tensor=self.S,
                                                    bcs=trace_bcs,
                                                    form_compiler_parameters=self.cxt.fc_params)

        self._assemble_S()
        self.S.force_evaluation()
        Smat = self.S.petscmat

        # Nullspace for the multiplier problem
        forward = -K * Atilde
        comm = pc.comm
        nullspace = create_trace_nullspace(P=P,
                                           forward=forward,
                                           V=V, V_d=V_d,
                                           TraceSpace=TraceSpace,
                                           comm=comm)
        if nullspace:
            Smat.setNullSpace(nullspace)

        # Set up the KSP for the system of Lagrange multipliers
        trace_ksp = PETSc.KSP().create(comm=pc.comm)
        trace_ksp.setOptionsPrefix(prefix)
        trace_ksp.setOperators(Smat)
        trace_ksp.setUp()
        trace_ksp.setFromOptions()
        self.trace_ksp = trace_ksp

        split_mixed_op = dict(split_form(Atilde.form))
        split_trace_op = dict(split_form(K.form))

        # Generate reconstruction calls
        self._reconstruction_calls(split_mixed_op, split_trace_op)

        # Set up the projection KSP
        hdiv_projection_ksp = PETSc.KSP().create(comm=pc.comm)
        hdiv_projection_ksp.setOptionsPrefix(prefix + 'hdiv_projection_')

        # Reuse the mass operator from the hdiv_mass_ksp
        hdiv_projection_ksp.setOperators(Mmat)

        # Construct the RHS for the projection stage
        self._projection_rhs = Function(V[self.vidx])
        self._assemble_projection_rhs = create_assembly_callable(
            ufl.dot(self.broken_solution.split()[self.vidx], q)*ufl.dx,
            tensor=self._projection_rhs,
            form_compiler_parameters=self.cxt.fc_params)

        # Finalize ksp setup
        hdiv_projection_ksp.setUp()
        hdiv_projection_ksp.setFromOptions()
        self.hdiv_projection_ksp = hdiv_projection_ksp

    def _reconstruction_calls(self, split_mixed_op, split_trace_op):
        """This generates the reconstruction calls for the unknowns using the
        Lagrange multipliers.

        :arg split_mixed_op: a ``dict`` of split forms that make up the broken
                             mixed operator from the original problem.
        :arg split_trace_op: a ``dict`` of split forms that make up the trace
                             contribution in the hybridized mixed system.
        """
        from firedrake.assemble import create_assembly_callable

        # We always eliminate the velocity block first
        id0, id1 = (self.vidx, self.pidx)

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
        split_residual = self.broken_residual.split()
        split_sol = self.broken_solution.split()
        g = AssembledVector(split_residual[id0])
        f = AssembledVector(split_residual[id1])
        sigma = split_sol[id0]
        u = split_sol[id1]
        lambdar = AssembledVector(self.trace_solution)

        M = D - C * A.inv * B
        R = K_1.T - C * A.inv * K_0.T
        u_rec = M.inv * (f - C * A.inv * g - R * lambdar)
        self._sub_unknown = create_assembly_callable(u_rec,
                                                     tensor=u,
                                                     form_compiler_parameters=self.cxt.fc_params)

        sigma_rec = A.inv * (g - B * AssembledVector(u) - K_0.T * lambdar)
        self._elim_unknown = create_assembly_callable(sigma_rec,
                                                      tensor=sigma,
                                                      form_compiler_parameters=self.cxt.fc_params)

    @timed_function("HybridRecon")
    def _reconstruct(self):
        """Reconstructs the system unknowns using the multipliers.
        Note that the reconstruction calls are assumed to be
        initialized at this point.
        """
        # We assemble the unknown which is an expression
        # of the first eliminated variable.
        self._sub_unknown()
        # Recover the eliminated unknown
        self._elim_unknown()

    @timed_function("HybridUpdate")
    def update(self, pc):
        """Update by assembling into the operator. No need to
        reconstruct symbolic objects.
        """
        self._assemble_S()
        self.S.force_evaluation()

    def apply(self, pc, x, y):
        """Solve the forward eliminated problem for the
        approximate traces of the scalar solution (the multipliers)
        and reconstruct the "broken flux and scalar variable."

        Once the hybridized solutions are computed, the data
        is projected back into the original non-broken finite element
        spaces.
        """

        with timed_region("HybridBreak"):
            with self.unbroken_residual.dat.vec_wo as v:
                x.copy(v)

            # Transfer unbroken_rhs into broken_rhs
            # NOTE: Scalar space is already "broken" so no need for
            # any projections
            unbroken_scalar_data = self.unbroken_residual.split()[self.pidx]
            broken_scalar_data = self.broken_residual.split()[self.pidx]
            unbroken_scalar_data.dat.copy(broken_scalar_data.dat)

        with timed_region("HybridProject"):
            # Handle the incoming HDiv residual:
            # Solve (approximately) for `g = A.inv * r`, where `A` is the HDiv
            # mass matrix and `r` is the HDiv residual.
            # Once `g` is obtained, we take the inner product against broken
            # HDiv test functions to obtain a broken residual.
            with self.unbroken_residual.split()[self.vidx].dat.vec_ro as r:
                with self._primal_r.dat.vec_wo as g:
                    self.hdiv_mass_ksp.solve(r, g)

        with timed_region("HybridRHS"):
            # Now assemble the new "broken" hdiv residual
            self._assemble_broken_r()

            # Compute the rhs for the multiplier system
            self._assemble_Srhs()

        with timed_region("HybridSolve"):
            # Solve the system for the Lagrange multipliers
            with self.schur_rhs.dat.vec_ro as b:
                if self.trace_ksp.getInitialGuessNonzero():
                    acc = self.trace_solution.dat.vec
                else:
                    acc = self.trace_solution.dat.vec_wo
                with acc as x_trace:
                    self.trace_ksp.solve(b, x_trace)

        # Reconstruct the unknowns
        self._reconstruct()

        with timed_region("HybridRecover"):
            # Project the broken solution into non-broken spaces
            broken_pressure = self.broken_solution.split()[self.pidx]
            unbroken_pressure = self.unbroken_solution.split()[self.pidx]
            broken_pressure.dat.copy(unbroken_pressure.dat)

            # Compute the hdiv projection of the broken hdiv solution
            self._assemble_projection_rhs()
            with self._projection_rhs.dat.vec_ro as b_proj:
                with self.unbroken_solution.split()[self.vidx].dat.vec_wo as sol:
                    self.hdiv_projection_ksp.solve(b_proj, sol)

            with self.unbroken_solution.dat.vec_ro as v:
                v.copy(y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""
        raise NotImplementedError("The transpose application of the PC is not implemented.")

    def view(self, pc, viewer=None):
        """Viewer calls for the various configurable objects in this PC."""
        super(HybridizationPC, self).view(pc, viewer)
        viewer.pushASCIITab()
        viewer.printfASCII("Construct the broken HDiv residual.\n")
        viewer.printfASCII("KSP solver for computing the primal map g = A.inv * r:\n")
        self.hdiv_mass_ksp.view(viewer)
        viewer.popASCIITab()
        viewer.printfASCII("Solves K * P^-1 * K.T using local eliminations.\n")
        viewer.printfASCII("KSP solver for the multipliers:\n")
        viewer.pushASCIITab()
        self.trace_ksp.view(viewer)
        viewer.popASCIITab()
        viewer.printfASCII("Locally reconstructing the broken solutions from the multipliers.\n")
        viewer.pushASCIITab()
        viewer.printfASCII("Project the broken hdiv solution into the HDiv space.\n")
        viewer.printfASCII("KSP for the HDiv projection stage:\n")
        viewer.pushASCIITab()
        self.hdiv_projection_ksp.view(viewer)
        viewer.popASCIITab()
