import ufl
import finat.ufl

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

        # The hybridized solution and residual are on the broken subspaces,
        # followed by the trace subspaces.
        Z = V.broken_space() * TraceSpace
        d = tuple(range(len(V)))
        self.c = tuple(range(len(V), len(Z)))
        Vd = Z.sub(d)
        Vc = Z.sub(self.c)
        self.solution = Function(Z)
        self.residual = Cofunction(Z.dual())
        self.condensed_rhs = Cofunction(Vc.dual())
        self.unbroken_solution = Function(V)
        self.unbroken_residual = Cofunction(V.dual())

        self.weight = Function(W)
        self.average_kernels = []
        for i, Wi in enumerate(W):
            shapes = (Wi.finat_element.space_dimension(), Wi.block_size)
            domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
            instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
            par_loop((domain, instructions), ufl.dx, {"w": (self.weight.subfunctions[i], INC)})

            instructions = """
        for i, j
            vec_out[i,j] = vec_out[i,j] + vec_in[i,j]/w[i,j]
        end
        """
            self.average_kernels.append((domain, instructions))

        # Create the symbolic Schur-reduction:
        # Original mixed operator replaced with "broken"
        # arguments
        arg_map = {test: TestFunction(Vd),
                   trial: TrialFunction(Vd)}
        Atilde = Tensor(replace(self.ctx.a, arg_map))
        gammar = TestFunctions(Vc)
        n = ufl.FacetNormal(mesh_unique)
        sigma = TrialFunctions(Vd)

        for bc in self.ctx.row_bcs:
            if bc.function_space_index(V) in self.pidx:
                raise NotImplementedError("Dirichlet conditions for scalar variable not supported. Use a weak bc")
            if bc.function_space_index(V) not in self.vidx:
                raise NotImplementedError("Dirichlet bc set on unsupported space.")

        Kform = 0
        trace_bcs = []
        for i, j in enumerate(self.vidx):
            if mesh_unique.extruded:
                Kform += (ufl.inner(ufl.jump(sigma[j], n=n), gammar[i]('+')) * ufl.dS_h
                          + ufl.inner(ufl.jump(sigma[j], n=n), gammar[i]('+')) * ufl.dS_v)
            else:
                Kform += ufl.inner(ufl.jump(sigma[j], n=n), gammar[i]('+')) * ufl.dS

            # Here we deal with boundaries. If there are Neumann
            # conditions (which should be enforced strongly for
            # H(div)xL^2) then we need to add jump terms on the exterior
            # facets. If there are Dirichlet conditions (which should be
            # enforced weakly) then we need to zero out the trace
            # variables there as they are not active (otherwise the hybrid
            # problem is not well-posed).

            # If boundary conditions on this subspace are contained in the ImplicitMatrixContext:
            field_bcs = [bc for bc in self.ctx.row_bcs if bc.function_space_index(V) == j]
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

                integrand = ufl.inner(ufl.dot(sigma[j], n), gammar[i])
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

                trace_bcs.extend(DirichletBC(Vc[i], 0, subdomain) for subdomain in trace_subdomains)

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
                trace_bcs.extend(DirichletBC(Vc[i], 0, subdomain) for subdomain in trace_subdomains)

        # Make a SLATE tensor from Kform
        K = Tensor(Kform)
        KT = Tensor(adjoint(Kform))

        # Build schur complement operator and right hand side
        self.schur_builder = SchurComplementBuilder(prefix, Atilde, K, KT, pc, self.vidx, self.pidx)
        schur_rhs, schur_comp = self.schur_builder.build_schur(AssembledVector(self.residual.sub(d)))
        self._assemble_Srhs = get_assembler(schur_rhs, form_compiler_parameters=self.ctx.fc_params).assemble

        monitor = self.ctx.appctx.get("custom_monitor", None)
        self.initialize_condensed_ksp(pc, prefix, Vc, schur_comp, trace_bcs, "trace_nullspace", monitor=monitor)

        # Set up local solvers for backwards substitution
        self.local_solvers = self.local_solver_callbacks(self.residual, self.solution, d, self.schur_builder)

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
            broken_l2_residual = self.residual.sub(self.pidx)
            broken_l2_residual.dat.assign(unbroken_l2_residual.dat, eager=True, eager_strategy="array")

            # Assemble the new "broken" hdiv residual
            # We need a residual R' in the broken space that
            # gives R'[w] = R[w] when w is in the unbroken space.
            # We do this by splitting the residual equally between
            # basis functions that add together to give unbroken
            # basis functions.
            for i, j in enumerate(self.vidx):
                unbroken_res_hdiv = self.unbroken_residual.sub(j)
                broken_res_hdiv = self.residual.sub(j)
                broken_res_hdiv.assign(0)
                par_loop(self.average_kernels[i], ufl.dx,
                         {"w": (self.weight.subfunctions[i], READ),
                          "vec_in": (unbroken_res_hdiv, READ),
                          "vec_out": (broken_res_hdiv, INC)})

        with PETSc.Log.Event("HybridRHS"):
            # Compute the rhs for the multiplier system
            self._assemble_Srhs(tensor=self.condensed_rhs)

    def backward_substitution(self, pc: PETSc.PC, y: PETSc.Vec) -> None:
        """Recover the eliminated fields from the trace solution.

        Parameters
        ----------
        pc : PETSc.PC
            The preconditioner for the original mixed system.
        y : PETSc.Vec
            The vector in which to store the recovered solution.
        """
        for local_solver_call in self.local_solvers:
            local_solver_call()

        with PETSc.Log.Event("HybridProject"):
            # Project the broken solution into non-broken spaces
            broken_l2_solution = self.solution.sub(self.pidx)
            unbroken_l2_solution = self.unbroken_solution.sub(self.pidx)
            unbroken_l2_solution.dat.assign(broken_l2_solution.dat, eager=True, eager_strategy="array")

            # Compute the hdiv projection of the broken hdiv solution
            for i, j in enumerate(self.vidx):
                broken_hdiv = self.solution.sub(j)
                unbroken_hdiv = self.unbroken_solution.sub(j)
                unbroken_hdiv.assign(0)
                par_loop(self.average_kernels[i], ufl.dx,
                         {"w": (self.weight.subfunctions[i], READ),
                          "vec_in": (broken_hdiv, READ),
                          "vec_out": (unbroken_hdiv, INC)})

        with self.unbroken_solution.dat.vec_ro as v:
            v.copy(y)

    def view(self, pc, viewer=None):
        """Viewer calls for the various configurable objects in this PC."""
        super(HybridizationPC, self).view(pc, viewer)
        if hasattr(self, "condensed_ksp"):
            viewer.printfASCII("Applying hybridization to mixed problem.\n")
            viewer.printfASCII("Statically condensing to trace system.\n")
            viewer.printfASCII("KSP solver for the multipliers:\n")
            self.condensed_ksp.view(viewer)
            viewer.printfASCII("Locally reconstructing solutions.\n")
            viewer.printfASCII("Projecting broken flux into HDiv space.\n")

    @property
    def trace_ksp(self) -> PETSc.KSP:
        """Return the KSP that solves the system for the traces."""
        return self.condensed_ksp

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
