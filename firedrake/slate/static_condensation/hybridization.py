import functools

import ufl
import finat.ufl

import firedrake.dmhooks as dmhooks
from firedrake.slate.static_condensation.sc_base import SCBase
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.petsc import PETSc
from firedrake.parloops import par_loop, READ, INC
from firedrake.slate.slate import Tensor, AssembledVector
from firedrake.slate.static_condensation.la_utils import SchurComplementBuilder
from firedrake.ufl_expr import adjoint

__all__ = ['HybridizationPC', 'SchurComplementBuilder']


class HybridizationPC(SCBase):
    """Solve a mixed H(div)-conforming problem using hybridization.

    Group the H(div) fields and the L2 fields into two blocks. Eliminate
    both blocks locally with Slate and enforce continuity through one
    trace field for each H(div) field.

    Notes
    -----
    Configure the trace solver with the ``hybridization_`` options prefix.
    By default, the local solver puts the H(div) subspaces in ``fieldsplit_0``
    and the L2 subspaces in ``fieldsplit_1``. Set
    ``hybridization_localsolve_pc_fieldsplit_0_fields`` and
    ``hybridization_localsolve_pc_fieldsplit_1_fields`` to change the splits.
    """

    needs_python_pmat = True

    @PETSc.Log.EventDecorator("HybridInit")
    def initialize(self, pc: PETSc.PC) -> None:
        """Construct the hybridized system and its local reconstruction.

        Parameters
        ----------
        pc : PETSc.PC
            The preconditioner that supplies the original mixed operator.
        """
        from firedrake import (FunctionSpace, Cofunction, Function,
                               TrialFunction, TrialFunctions, TestFunction, TestFunctions,
                               DirichletBC)
        from firedrake.assemble import get_assembler
        from ufl.algorithms.replace import replace

        # Extract the problem context
        prefix = (pc.getOptionsPrefix() or "") + "hybridization_"
        _, P = pc.getOperators()
        self.ctx = P.getPythonContext()

        if not isinstance(self.ctx, ImplicitMatrixContext):
            raise ValueError("The python context must be an ImplicitMatrixContext")

        test, trial = self.ctx.a.arguments()

        V = test.function_space()
        mesh = V.mesh()
        if len(set(mesh)) == 1:
            mesh_unique = mesh.unique()
        else:
            raise NotImplementedError("Not implemented for general mixed meshes")

        self.vidx = tuple(i for i, Vi in enumerate(V) if Vi.ufl_element().sobolev_space == ufl.HDiv)
        self.pidx = tuple(i for i, Vi in enumerate(V) if Vi.ufl_element().sobolev_space == ufl.L2)
        if not self.vidx or not self.pidx or len(self.vidx) + len(self.pidx) != len(V):
            raise ValueError("Expecting H(div) and L2 spaces.")

        # Create the space of approximate traces.
        W = V.sub(self.vidx)
        TraceSpace = FunctionSpace(W.mesh(), div_to_trace(W.ufl_element()))

        # Break the function spaces and define fully discontinuous spaces
        V_d = V.broken_space()

        # Set up the functions for the original, hybridized
        # and schur complement systems
        self.broken_solution = Function(V_d)
        self.broken_residual = Cofunction(V_d.dual())
        self.trace_solution = Function(TraceSpace)
        self.unbroken_solution = Function(V)
        self.unbroken_residual = Cofunction(V.dual())

        self.weight = Function(W)
        self.average_kernels = []
        for Wi, weight in zip(W, self.weight.subfunctions):
            shapes = (Wi.finat_element.space_dimension(), Wi.block_size)
            domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
            instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
            par_loop((domain, instructions), ufl.dx, {"w": (weight, INC)})

            instructions = """
        for i, j
            vec_out[i,j] = vec_out[i,j] + vec_in[i,j]/w[i,j]
        end
        """
            self.average_kernels.append((domain, instructions))

        # Create the symbolic Schur-reduction:
        # Original mixed operator replaced with "broken"
        # arguments
        arg_map = {test: TestFunction(V_d),
                   trial: TrialFunction(V_d)}
        Atilde = Tensor(replace(self.ctx.a, arg_map))
        n = ufl.FacetNormal(mesh_unique)
        trial_fields = TrialFunctions(V_d)

        for bc in self.ctx.row_bcs:
            if bc.function_space_index(V) in self.pidx:
                raise NotImplementedError("Dirichlet conditions for scalar variable not supported. Use a weak bc")
            if bc.function_space_index(V) not in self.vidx:
                raise NotImplementedError("Dirichlet bc set on unsupported space.")

        Kform = 0
        trace_bcs = []
        for i, gammar, trace_space in zip(self.vidx, TestFunctions(TraceSpace), TraceSpace):
            sigma = trial_fields[i]
            if mesh_unique.extruded:
                Kform += (ufl.inner(ufl.jump(sigma, n=n), gammar('+')) * ufl.dS_h
                          + ufl.inner(ufl.jump(sigma, n=n), gammar('+')) * ufl.dS_v)
            else:
                Kform += ufl.inner(ufl.jump(sigma, n=n), gammar('+')) * ufl.dS

            # Here we deal with boundaries. If there are Neumann
            # conditions (which should be enforced strongly for
            # H(div)xL^2) then we need to add jump terms on the exterior
            # facets. If there are Dirichlet conditions (which should be
            # enforced weakly) then we need to zero out the trace
            # variables there as they are not active (otherwise the hybrid
            # problem is not well-posed).

            # If boundary conditions on this subspace are contained in the ImplicitMatrixContext:
            field_bcs = [bc for bc in self.ctx.row_bcs if bc.function_space_index(V) == i]
            if field_bcs:
                # Find all the subdomains with neumann BCS
                # These are Dirichlet BCs on the vidx space
                neumann_subdomains = set()
                for bc in field_bcs:
                    # append the set of sub domains
                    neumann_subdomains |= set(bc.sub_domain)

                # separate out the top and bottom bcs
                extruded_neumann_subdomains = neumann_subdomains & {"top", "bottom"}
                neumann_subdomains = neumann_subdomains - extruded_neumann_subdomains

                integrand = ufl.inner(ufl.dot(sigma, n), gammar)
                measures = []
                trace_subdomains = []
                if mesh_unique.extruded:
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
                    markers = [int(x) for x in mesh_unique.facet_markers]
                    dirichlet_subdomains = set(markers) - neumann_subdomains
                    trace_subdomains.extend(sorted(dirichlet_subdomains))

                for measure in measures:
                    Kform += integrand*measure

                trace_bcs.extend(DirichletBC(trace_space, 0, subdomain) for subdomain in trace_subdomains)

            else:
                # No bcs were provided, we assume weak Dirichlet conditions.
                # We zero out the contribution of the trace variables on
                # the exterior boundary. We don't need to do this for boundary-less
                # domains (like a sphere).
                trace_subdomains = []
                if mesh_unique.extruded:
                    if mesh_unique.exterior_facets_vert.global_size > 0:
                        trace_subdomains.append("on_boundary")
                    if not mesh_unique.extruded_periodic:
                        trace_subdomains.extend(["bottom", "top"])
                elif mesh_unique.exterior_facets.global_size > 0:
                    trace_subdomains.append("on_boundary")
                trace_bcs.extend(DirichletBC(trace_space, 0, subdomain) for subdomain in trace_subdomains)

        # Make a SLATE tensor from Kform
        K = Tensor(Kform)
        KT = Tensor(adjoint(Kform))

        # Build schur complement operator and right hand side
        self.schur_builder = SchurComplementBuilder(prefix, Atilde, K, KT, pc, self.vidx, self.pidx)
        schur_rhs, schur_comp = self.schur_builder.build_schur(AssembledVector(self.broken_residual))

        # Assemble the Schur complement operator and right-hand side
        self.schur_rhs = Cofunction(TraceSpace.dual())
        self._assemble_Srhs = get_assembler(schur_rhs, form_compiler_parameters=self.ctx.fc_params).assemble

        mat_type = PETSc.Options().getString(prefix + "mat_type", "aij")
        form_assembler = get_assembler(schur_comp, bcs=trace_bcs, form_compiler_parameters=self.ctx.fc_params, mat_type=mat_type, options_prefix=prefix, appctx=self.get_appctx(pc))
        self.S = form_assembler.allocate()
        self._assemble_S = form_assembler.assemble

        with PETSc.Log.Event("HybridOperatorAssembly"):
            self._assemble_S(tensor=self.S)

        Smat = self.S.petscmat

        nullspace = self.ctx.appctx.get("trace_nullspace", None)
        if nullspace is not None:
            nsp = nullspace(TraceSpace)
            Smat.setNullSpace(nsp.nullspace())

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
        trace_ksp.setDMActive(PETSc.KSP.DMActive.ALL, False)
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
        self._reconstruction_calls()

    def _reconstruction_calls(self):
        """This generates the reconstruction calls for the unknowns using the
        Lagrange multipliers.
        """
        from firedrake.assemble import get_assembler

        # Eliminate fieldsplit_0 first.
        id0, id1 = (self.schur_builder.vidx, self.schur_builder.pidx)

        # TODO: When PyOP2 is able to write into mixed dats,
        # the reconstruction expressions can simplify into
        # one clean expression.

        # reuse work from trace operator build
        A, B, C, _ = self.schur_builder.list_split_mixed_ops
        K_0, K_1 = self.schur_builder.list_split_trace_ops
        Ahat = self.schur_builder.A00_inv_hat
        S = self.schur_builder.inner_S

        # Select the H(div) and L2 groups for reconstruction.
        g = AssembledVector(self.broken_residual.sub(id0))
        f = AssembledVector(self.broken_residual.sub(id1))
        sigma = self.broken_solution.sub(id0)
        u = self.broken_solution.sub(id1)
        lambdar = AssembledVector(self.trace_solution)

        R = K_1.T - C * Ahat * K_0.T
        rhs = f - C * Ahat * g - R * lambdar
        if self.schur_builder.schur_approx or self.schur_builder.jacobi_S:
            Shat = self.schur_builder.inner_S_approx_inv_hat
            if self.schur_builder.preonly_S:
                S = Shat
            else:
                S = Shat * S
                rhs = Shat * rhs

        u_rec = S.solve(rhs, decomposition="PartialPivLU")
        self._sub_unknown = functools.partial(get_assembler(u_rec, form_compiler_parameters=self.ctx.fc_params).assemble, tensor=u)

        sigma_rec = A.solve(g - B * AssembledVector(u) - K_0.T * lambdar,
                            decomposition="PartialPivLU")
        self._elim_unknown = functools.partial(get_assembler(sigma_rec, form_compiler_parameters=self.ctx.fc_params).assemble, tensor=sigma)

    @PETSc.Log.EventDecorator("HybridUpdate")
    def update(self, pc):
        """Update by assembling into the operator. No need to
        reconstruct symbolic objects.
        """
        self._assemble_S(tensor=self.S)

    def forward_elimination(self, pc: PETSc.PC, x: PETSc.Vec) -> None:
        """Eliminate the mixed fields to form the right-hand side on the trace space.

        Parameters
        ----------
        pc : PETSc.PC
            The preconditioner for the original mixed system.
        x : PETSc.Vec
            The right-hand side of the original mixed system.
        """

        with PETSc.Log.Event("HybridBreak"):
            with self.unbroken_residual.dat.vec_wo as v:
                x.copy(v)

            # Copy the residual on the L2 fields, which are already discontinuous.
            unbroken_l2_residual = self.unbroken_residual.sub(self.pidx)
            broken_l2_residual = self.broken_residual.sub(self.pidx)
            broken_l2_residual.dat.assign(unbroken_l2_residual.dat, eager=True, eager_strategy="array")

            # Assemble the new "broken" hdiv residual
            # We need a residual R' in the broken space that
            # gives R'[w] = R[w] when w is in the unbroken space.
            # We do this by splitting the residual equally between
            # basis functions that add together to give unbroken
            # basis functions.
            for i, weight, kernel in zip(self.vidx, self.weight.subfunctions, self.average_kernels):
                unbroken_res_hdiv = self.unbroken_residual.sub(i)
                broken_res_hdiv = self.broken_residual.sub(i)
                broken_res_hdiv.assign(0)
                par_loop(kernel, ufl.dx,
                         {"w": (weight, READ),
                          "vec_in": (unbroken_res_hdiv, READ),
                          "vec_out": (broken_res_hdiv, INC)})

        with PETSc.Log.Event("HybridRHS"):
            # Compute the rhs for the multiplier system
            self._assemble_Srhs(tensor=self.schur_rhs)

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
                    acc = self.trace_solution.dat.vec_rw
                else:
                    acc = self.trace_solution.dat.vec_wo
                with acc as x_trace:
                    self.trace_ksp.solve(b, x_trace)

    def backward_substitution(self, pc: PETSc.PC, y: PETSc.Vec) -> None:
        """Recover the eliminated fields from the trace solution.

        Parameters
        ----------
        pc : PETSc.PC
            The preconditioner for the original mixed system.
        y : PETSc.Vec
            The vector in which to store the recovered solution.
        """
        # We assemble the unknown which is an expression
        # of the first eliminated variable.
        with PETSc.Log.Event("RecoverFirstElim"):
            self._sub_unknown()
        # Recover the eliminated unknown
        self._elim_unknown()

        with PETSc.Log.Event("HybridProject"):
            # Project the broken solution into non-broken spaces
            broken_l2_solution = self.broken_solution.sub(self.pidx)
            unbroken_l2_solution = self.unbroken_solution.sub(self.pidx)
            unbroken_l2_solution.dat.assign(broken_l2_solution.dat, eager=True, eager_strategy="array")

            # Compute the hdiv projection of the broken hdiv solution
            for i, weight, kernel in zip(self.vidx, self.weight.subfunctions, self.average_kernels):
                broken_hdiv = self.broken_solution.sub(i)
                unbroken_hdiv = self.unbroken_solution.sub(i)
                unbroken_hdiv.assign(0)
                par_loop(kernel, ufl.dx,
                         {"w": (weight, READ),
                          "vec_in": (broken_hdiv, READ),
                          "vec_out": (unbroken_hdiv, INC)})

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

    def getSchurComplementBuilder(self):
        return self.schur_builder


def div_to_trace(element: finat.ufl.FiniteElementBase) -> finat.ufl.FiniteElementBase:
    """Construct the trace element for the normal component of an H(div) element.

    Parameters
    ----------
    element : finat.ufl.FiniteElementBase
        An H(div) element or a mixed element of H(div) elements.

    Returns
    -------
    finat.ufl.FiniteElementBase
        The trace element with the same mixed, vector, or tensor structure.
    """
    if isinstance(element, (finat.ufl.VectorElement, finat.ufl.TensorElement)):
        return element.reconstruct(sub_element=div_to_trace(element._sub_element))
    elif isinstance(element, finat.ufl.MixedElement):
        return type(element)(*map(div_to_trace, element.sub_elements))
    elif isinstance(element, finat.ufl.RestrictedElement):
        return element.reconstruct(element=div_to_trace(element._element))
    else:
        degree = element.degree()
        if element.family() != "Brezzi-Douglas-Marini":
            if isinstance(degree, tuple):
                degree = tuple(d - 1 for d in degree)
            else:
                degree -= 1
        return finat.ufl.FiniteElement("HDiv Trace", element.cell, degree)
