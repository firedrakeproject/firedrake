import functools
import numbers

import numpy as np
import ufl

import firedrake.dmhooks as dmhooks
from firedrake.slate.static_condensation.sc_base import SCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.parloops import par_loop, READ, INC
from firedrake.slate.slate import Tensor, AssembledVector
from pyop2.utils import as_tuple


__all__ = ['HybridizationPC']


class HybridizationPC(SCBase):

    needs_python_pmat = True

    """A Slate-based python preconditioner that solves a
    mixed H(div)-conforming problem using hybridization.
    Currently, this preconditioner supports the hybridization
    of the RT and BDM mixed methods of arbitrary degree.

    The forward eliminations and backwards reconstructions
    are performed element-local using the Slate language.
    """

    @PETSc.Log.EventDecorator("HybridInit")
    def initialize(self, pc):
        """Set up the problem context. Take the original
        mixed problem and reformulate the problem as a
        hybridized mixed system.

        A KSP is created for the Lagrange multiplier system.
        """
        from firedrake import (FunctionSpace, Function, Constant,
                               TrialFunction, TrialFunctions, TestFunction,
                               DirichletBC, assemble)
        from firedrake.assemble import allocate_matrix
        from firedrake.formmanipulation import split_form
        from ufl.algorithms.replace import replace

        # Extract the problem context
        prefix = pc.getOptionsPrefix() + "hybridization_"
        _, P = pc.getOperators()
        self.ctx = P.getPythonContext()

        if not isinstance(self.ctx, ImplicitMatrixContext):
            raise ValueError("The python context must be an ImplicitMatrixContext")

        test, trial = self.ctx.a.arguments()

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
        broken_elements = ufl.MixedElement([ufl.BrokenElement(Vi.ufl_element()) for Vi in V])
        V_d = FunctionSpace(mesh, broken_elements)

        # Set up the functions for the original, hybridized
        # and schur complement systems
        self.broken_solution = Function(V_d)
        self.broken_residual = Function(V_d)
        self.trace_solution = Function(TraceSpace)
        self.unbroken_solution = Function(V)
        self.unbroken_residual = Function(V)

        shapes = (V[self.vidx].finat_element.space_dimension(),
                  np.prod(V[self.vidx].shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        self.weight = Function(V[self.vidx])
        par_loop((domain, instructions), ufl.dx, {"w": (self.weight, INC)},
                 is_loopy_kernel=True)

        instructions = """
        for i, j
            vec_out[i,j] = vec_out[i,j] + vec_in[i,j]/w[i,j]
        end
        """
        self.average_kernel = (domain, instructions)

        # Create the symbolic Schur-reduction:
        # Original mixed operator replaced with "broken"
        # arguments
        arg_map = {test: TestFunction(V_d),
                   trial: TrialFunction(V_d)}
        Atilde = Tensor(replace(self.ctx.a, arg_map))
        gammar = TestFunction(TraceSpace)
        n = ufl.FacetNormal(mesh)
        sigma = TrialFunctions(V_d)[self.vidx]

        if mesh.cell_set._extruded:
            Kform = (gammar('+') * ufl.jump(sigma, n=n) * ufl.dS_h
                     + gammar('+') * ufl.jump(sigma, n=n) * ufl.dS_v)
        else:
            Kform = (gammar('+') * ufl.jump(sigma, n=n) * ufl.dS)

        # Here we deal with boundaries. If there are Neumann
        # conditions (which should be enforced strongly for
        # H(div)xL^2) then we need to add jump terms on the exterior
        # facets. If there are Dirichlet conditions (which should be
        # enforced weakly) then we need to zero out the trace
        # variables there as they are not active (otherwise the hybrid
        # problem is not well-posed).

        # If boundary conditions are contained in the ImplicitMatrixContext:
        if self.ctx.row_bcs:
            # Find all the subdomains with neumann BCS
            # These are Dirichlet BCs on the vidx space
            neumann_subdomains = set()
            for bc in self.ctx.row_bcs:
                if bc.function_space().index == self.pidx:
                    raise NotImplementedError("Dirichlet conditions for scalar variable not supported. Use a weak bc")
                if bc.function_space().index != self.vidx:
                    raise NotImplementedError("Dirichlet bc set on unsupported space.")
                # append the set of sub domains
                subdom = bc.sub_domain
                if isinstance(subdom, str):
                    neumann_subdomains |= set([subdom])
                else:
                    neumann_subdomains |= set(as_tuple(subdom, numbers.Integral))

            # separate out the top and bottom bcs
            extruded_neumann_subdomains = neumann_subdomains & {"top", "bottom"}
            neumann_subdomains = neumann_subdomains - extruded_neumann_subdomains

            integrand = gammar * ufl.dot(sigma, n)
            measures = []
            trace_subdomains = []
            if mesh.cell_set._extruded:
                ds = ufl.ds_v
                for subdomain in sorted(extruded_neumann_subdomains):
                    measures.append({"top": ufl.ds_t, "bottom": ufl.ds_b}[subdomain])
                trace_subdomains.extend(sorted({"top", "bottom"} - extruded_neumann_subdomains))
            else:
                ds = ufl.ds
            if "on_boundary" in neumann_subdomains:
                measures.append(ds)
            else:
                measures.extend((ds(sd) for sd in sorted(neumann_subdomains)))
                markers = [int(x) for x in mesh.exterior_facets.unique_markers]
                dirichlet_subdomains = set(markers) - neumann_subdomains
                trace_subdomains.extend(sorted(dirichlet_subdomains))

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

        # Split forms
        split_mixed_op = dict(split_form(Atilde.form))
        id0, id1 = (self.vidx, self.pidx)
        A00 = Tensor(split_mixed_op[(id0, id0)])
        A01 = Tensor(split_mixed_op[(id0, id1)])
        A10 = Tensor(split_mixed_op[(id1, id0)])
        A11 = Tensor(split_mixed_op[(id1, id1)])
        list_split_mixed_ops = [A00, A01, A10, A11]

        split_trace_op = dict(split_form(K.form))
        K0 = Tensor(split_trace_op[(0, id0)])
        K1 = Tensor(split_trace_op[(0, id1)])
        list_split_trace_ops = [K0, K1]

        # Build schur complement operator and right hand side
        nested = PETSc.Options(prefix).getBool("nested_schur", False)
        schur_rhs, schur_comp = self.build_schur(Atilde, K, list_split_mixed_ops,
                                                 list_split_trace_ops, nested=nested)

        # Assemble the Schur complement operator and right-hand side
        self.schur_rhs = Function(TraceSpace)
        self._assemble_Srhs = functools.partial(assemble,
                                                schur_rhs,
                                                tensor=self.schur_rhs,
                                                form_compiler_parameters=self.ctx.fc_params,
                                                assembly_type="residual")

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")
        self.S = allocate_matrix(schur_comp, bcs=trace_bcs,
                                 form_compiler_parameters=self.ctx.fc_params,
                                 mat_type=mat_type,
                                 options_prefix=prefix,
                                 appctx=self.get_appctx(pc))
        self._assemble_S = functools.partial(assemble,
                                             schur_comp,
                                             tensor=self.S,
                                             bcs=trace_bcs,
                                             form_compiler_parameters=self.ctx.fc_params,
                                             mat_type=mat_type,
                                             assembly_type="residual")

        with PETSc.Log.Event("HybridOperatorAssembly"):
            self._assemble_S()

        Smat = self.S.petscmat

        nullspace = self.ctx.appctx.get("trace_nullspace", None)
        if nullspace is not None:
            nsp = nullspace(TraceSpace)
            Smat.setNullSpace(nsp.nullspace(comm=pc.comm))

        # Create a SNESContext for the DM associated with the trace problem
        self._ctx_ref = self.new_snes_ctx(pc,
                                          schur_comp,
                                          trace_bcs,
                                          mat_type,
                                          self.ctx.fc_params,
                                          options_prefix=prefix)

        # dm associated with the trace problem
        trace_dm = TraceSpace.dm

        # KSP for the system of Lagrange multipliers
        trace_ksp = PETSc.KSP().create(comm=pc.comm)
        trace_ksp.incrementTabLevel(1, parent=pc)

        # Set the dm for the trace solver
        trace_ksp.setDM(trace_dm)
        trace_ksp.setDMActive(False)
        trace_ksp.setOptionsPrefix(prefix)
        trace_ksp.setOperators(Smat, Smat)

        # Option to add custom monitor
        monitor = self.ctx.appctx.get('custom_monitor', None)
        if monitor:
            monitor.add_reconstructor(self.backward_substitution)
            trace_ksp.setMonitor(monitor)

        self.trace_ksp = trace_ksp

        with dmhooks.add_hooks(trace_dm, self,
                               appctx=self._ctx_ref,
                               save=False):
            trace_ksp.setFromOptions()

        # Generate reconstruction calls
        self._reconstruction_calls(list_split_mixed_ops, list_split_trace_ops)

    def _reconstruction_calls(self, list_split_mixed_ops, list_split_trace_ops):
        """This generates the reconstruction calls for the unknowns using the
        Lagrange multipliers.

        :arg split_mixed_op: a ``dict`` of split forms that make up the broken
                             mixed operator from the original problem.
        :arg split_trace_op: a ``dict`` of split forms that make up the trace
                             contribution in the hybridized mixed system.
        """
        from firedrake import assemble

        # We always eliminate the velocity block first
        id0, id1 = (self.vidx, self.pidx)

        # TODO: When PyOP2 is able to write into mixed dats,
        # the reconstruction expressions can simplify into
        # one clean expression.
        A, B, C, D = list_split_mixed_ops
        K_0, K_1 = list_split_trace_ops

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
        u_rec = M.solve(f - C * A.inv * g - R * lambdar,
                        decomposition="PartialPivLU")
        self._sub_unknown = functools.partial(assemble,
                                              u_rec,
                                              tensor=u,
                                              form_compiler_parameters=self.ctx.fc_params,
                                              assembly_type="residual")

        sigma_rec = A.solve(g - B * AssembledVector(u) - K_0.T * lambdar,
                            decomposition="PartialPivLU")
        self._elim_unknown = functools.partial(assemble,
                                               sigma_rec,
                                               tensor=sigma,
                                               form_compiler_parameters=self.ctx.fc_params,
                                               assembly_type="residual")

    def build_schur(self, Atilde, K, list_split_mixed_ops, list_split_trace_ops, nested=False):
        """The Schur complement in the operators of the trace solve contains
        the inverse on a mixed system.  Users may want this inverse to be treated
        with another schur complement.

        Let the mixed matrix Atilde be called A here,
        then the "nested" options rewrites with a Schur decomposition
        as the following.

        .. code-block:: text

                A.inv = [[I, -A00.inv * A01]    *   [[A00.inv, 0    ]   *   [[I,             0]
                        [0,  I             ]]       [0,        S.inv]]      [-A10* A00.inv,  I]]
                        --------------------        -----------------       -------------------
                        block1                      block2                  block3

        with the (inner) schur complement S = A11 - A10 * A00.inv * A01
        """

        if nested:
            A00, A01, A10, A11 = list_split_mixed_ops
            K0, K1 = list_split_trace_ops
            broken_residual = self.broken_residual.split()
            split_broken_res = [AssembledVector(broken_residual[self.vidx]),
                                AssembledVector(broken_residual[self.pidx])]

            # inner schur complement
            S = (A11 - A10 * A00.inv * A01)
            # K * block1
            K_Ainv_block1 = [K0, -K0 * A00.inv * A01 + K1]
            # K * block1 * block2
            K_Ainv_block2 = [K_Ainv_block1[0] * A00.inv, K_Ainv_block1[1] * S.inv]
            # K * block1 * block2 * block3
            K_Ainv_block3 = [K_Ainv_block2[0] - K_Ainv_block2[1] * A10 * A00.inv, K_Ainv_block2[1]]
            # K * block1 * block2 * block3 * broken residual
            schur_rhs = (K_Ainv_block3[0] * split_broken_res[0] + K_Ainv_block3[1] * split_broken_res[1])
            # K * block1 * block2 * block3 * K.T
            schur_comp = K_Ainv_block3[0] * K0.T + K_Ainv_block3[1] * K1.T
        else:
            schur_rhs = K * Atilde.inv * AssembledVector(self.broken_residual)
            schur_comp = K * Atilde.inv * K.T
        return schur_rhs, schur_comp

    @PETSc.Log.EventDecorator("HybridUpdate")
    def update(self, pc):
        """Update by assembling into the operator. No need to
        reconstruct symbolic objects.
        """
        self._assemble_S()

    def forward_elimination(self, pc, x):
        """Perform the forward elimination of fields and
        provide the reduced right-hand side for the condensed
        system.

        :arg pc: a Preconditioner instance.
        :arg x: a PETSc vector containing the incoming right-hand side.
        """

        with PETSc.Log.Event("HybridBreak"):
            with self.unbroken_residual.dat.vec_wo as v:
                x.copy(v)

            # Transfer unbroken_rhs into broken_rhs
            # NOTE: Scalar space is already "broken" so no need for
            # any projections
            unbroken_scalar_data = self.unbroken_residual.split()[self.pidx]
            broken_scalar_data = self.broken_residual.split()[self.pidx]
            unbroken_scalar_data.dat.copy(broken_scalar_data.dat)

            # Assemble the new "broken" hdiv residual
            # We need a residual R' in the broken space that
            # gives R'[w] = R[w] when w is in the unbroken space.
            # We do this by splitting the residual equally between
            # basis functions that add together to give unbroken
            # basis functions.
            unbroken_res_hdiv = self.unbroken_residual.split()[self.vidx]
            broken_res_hdiv = self.broken_residual.split()[self.vidx]
            broken_res_hdiv.assign(0)
            par_loop(self.average_kernel, ufl.dx,
                     {"w": (self.weight, READ),
                      "vec_in": (unbroken_res_hdiv, READ),
                      "vec_out": (broken_res_hdiv, INC)},
                     is_loopy_kernel=True)

        with PETSc.Log.Event("HybridRHS"):
            # Compute the rhs for the multiplier system
            self._assemble_Srhs()

    def sc_solve(self, pc):
        """Solve the condensed linear system for the
        condensed field.

        :arg pc: a Preconditioner instance.
        """

        dm = self.trace_ksp.getDM()

        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):

            # Solve the system for the Lagrange multipliers
            with self.schur_rhs.dat.vec_ro as b:
                if self.trace_ksp.getInitialGuessNonzero():
                    acc = self.trace_solution.dat.vec
                else:
                    acc = self.trace_solution.dat.vec_wo
                with acc as x_trace:
                    self.trace_ksp.solve(b, x_trace)

    def backward_substitution(self, pc, y):
        """Perform the backwards recovery of eliminated fields.

        :arg pc: a Preconditioner instance.
        :arg y: a PETSc vector for placing the resulting fields.
        """

        # We assemble the unknown which is an expression
        # of the first eliminated variable.
        self._sub_unknown()
        # Recover the eliminated unknown
        self._elim_unknown()

        with PETSc.Log.Event("HybridProject"):
            # Project the broken solution into non-broken spaces
            broken_pressure = self.broken_solution.split()[self.pidx]
            unbroken_pressure = self.unbroken_solution.split()[self.pidx]
            broken_pressure.dat.copy(unbroken_pressure.dat)

            # Compute the hdiv projection of the broken hdiv solution
            broken_hdiv = self.broken_solution.split()[self.vidx]
            unbroken_hdiv = self.unbroken_solution.split()[self.vidx]
            unbroken_hdiv.assign(0)

            par_loop(self.average_kernel, ufl.dx,
                     {"w": (self.weight, READ),
                      "vec_in": (broken_hdiv, READ),
                      "vec_out": (unbroken_hdiv, INC)},
                     is_loopy_kernel=True)

        with self.unbroken_solution.dat.vec_ro as v:
            v.copy(y)

    def view(self, pc, viewer=None):
        """Viewer calls for the various configurable objects in this PC."""
        super(HybridizationPC, self).view(pc, viewer)
        if hasattr(self, "trace_ksp"):
            viewer.printfASCII("Applying hybridization to mixed problem.\n")
            viewer.printfASCII("Statically condensing to trace system.\n")
            viewer.printfASCII("KSP solver for the multipliers:\n")
            self.trace_ksp.view(viewer)
            viewer.printfASCII("Locally reconstructing solutions.\n")
            viewer.printfASCII("Projecting broken flux into HDiv space.\n")
