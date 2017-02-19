"""This module provides custom python preconditioners utilizing
the Slate language.
"""

from __future__ import absolute_import, print_function, division

import ufl

from firedrake.matrix_free.preconditioners import PCBase
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor


__all__ = ['HybridizationPC']


class HybridizationPC(PCBase):
    """A Slate-based python preconditioner that solves a
    mixed saddle-point problem using a hybridizable DG scheme.

    The forward eliminations and backwards reconstructions
    are performed element-local using the Slate language.
    """
    def initialize(self, pc):
        """Set up the problem context. Take the original
        mixed problem and reformulate the problem as a
        hybridized DG one.

        A KSP is created for the Lagrange multiplier system.
        """
        from ufl.algorithms.map_integrands import map_integrand_dags
        from firedrake import (FunctionSpace, TrialFunction,
                               TrialFunctions, TestFunction, Function,
                               BrokenElement, MixedElement,
                               FacetNormal, Constant, DirichletBC,
                               Projector)
        from firedrake.assemble import (allocate_matrix,
                                        create_assembly_callable)
        from firedrake.formmanipulation import ArgumentReplacer, split_form

        # Extract the problem context
        prefix = pc.getOptionsPrefix()
        _, P = pc.getOperators()
        context = P.getPythonContext()
        test, trial = context.a.arguments()

        # Break the function spaces and define fully discontinuous spaces
        V = test.function_space()
        if V.mesh().cell_set._extruded:
            raise NotImplementedError("Not implemented on extruded meshes.")

        broken_elements = [BrokenElement(Vi.ufl_element()) for Vi in V]
        elem = MixedElement(broken_elements)
        V_d = FunctionSpace(V.mesh(), elem)
        arg_map = {test: TestFunction(V_d),
                   trial: TrialFunction(V_d)}

        # Replace the problems arguments with arguments defined
        # on the new discontinuous spaces
        replacer = ArgumentReplacer(arg_map)
        new_form = map_integrand_dags(replacer, context.a)

        # Create the space of approximate traces:
        # the space of Lagrange multipliers
        # NOTE: Once extruded is ready, we will need to be aware of this
        # and construct the appropriate trace space for the HDiv element
        if V[0].ufl_element().family() == "Brezzi-Douglas-Marini":
            tdegree = V_d.ufl_element().degree()
        else:
            tdegree = V_d.ufl_element().degree() - 1

        TraceSpace = FunctionSpace(V.mesh(), "HDiv Trace", tdegree)

        # For extruded, we will need to add the flags "on_top" and "on_bottom"
        trace_conditions = [DirichletBC(TraceSpace, Constant(0.0),
                                        "on_boundary")]

        # Broken flux and scalar terms (solution via reconstruction)
        self.broken_solution = Function(V_d)

        # Broken RHS of the fully discontinuous problem
        self.broken_rhs = Function(V_d)

        # Solution of the system for the Lagrange multipliers
        self.trace_solution = Function(TraceSpace)

        # unbroken solutions and rhs
        self.unbroken_solution = Function(V)
        self.unbroken_rhs = Function(V)

        # Create the symbolic Schur-reduction of the discontinuous
        # problem in Slate. Weakly enforce continuity via the Lagrange
        # multipliers
        Atilde = Tensor(new_form)
        gammar = TestFunction(TraceSpace)
        n = FacetNormal(V.mesh())
        sigma, _ = TrialFunctions(V_d)

        # NOTE: Once extruded is ready, this will change slightly
        # to include both horizontal and vertical interior facets
        K = Tensor(gammar('+') * ufl.dot(sigma, n) * ufl.dS)

        # Assemble the Schur complement operator and right-hand side
        # in a cell-local way
        self.schur_rhs = Function(TraceSpace)
        self._assemble_Srhs = create_assembly_callable(
            K * Atilde.inv * self.broken_rhs,
            tensor=self.schur_rhs,
            form_compiler_parameters=context.fc_params)

        schur_comp = K * Atilde.inv * K.T
        self.S = allocate_matrix(schur_comp,
                                 bcs=trace_conditions,
                                 form_compiler_parameters=context.fc_params)
        self._assemble_S = create_assembly_callable(
            schur_comp,
            tensor=self.S,
            bcs=trace_conditions,
            form_compiler_parameters=context.fc_params)

        self._assemble_S()
        self.S.force_evaluation()
        Smat = self.S.petscmat

        # Transfer nullspace (if any)
        # TODO: Have a think about this...Is this right?
        Smat.setNullSpace(P.getNullSpace())

        # Set up the KSP for the system of Lagrange multipliers
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setOptionsPrefix(prefix + "trace_")
        ksp.setTolerances(rtol=1e-13)
        ksp.setOperators(Smat)
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp

        # Now we construct the local tensors for the reconstruction stage
        # TODO: Get Slate to support mixed tensors and these variables
        # become unnecessary
        split_forms = split_form(new_form)
        A = Tensor(next(sf.form for sf in split_forms
                        if sf.indices == (0, 0)))
        B = Tensor(next(sf.form for sf in split_forms
                        if sf.indices == (1, 0)))
        C = Tensor(next(sf.form for sf in split_forms
                        if sf.indices == (1, 1)))
        trial = TrialFunction(V_d[0])
        K_local = Tensor(gammar('+') * ufl.dot(trial, n) * ufl.dS)

        # Split the solution function and reconstruct
        # each bit separately
        sigma_h, u_h = self.broken_solution.split()

        # TODO: Write out full expression for general RHS of the form
        # RHS = [g; f]
        _, f = self.broken_rhs.split()

        # Pressure reconstruction
        M = B * A.inv * B.T + C
        u_sol = M.inv * f + M.inv * (B * A.inv *
                                     K_local.T * self.trace_solution)
        self._assemble_pressure = create_assembly_callable(
            u_sol,
            tensor=u_h,
            form_compiler_parameters=context.fc_params)

        # Velocity reconstructions
        sigma_sol = A.inv * (B.T * u_h -
                             K_local.T * self.trace_solution)
        self._assemble_velocity = create_assembly_callable(
            sigma_sol,
            tensor=sigma_h,
            form_compiler_parameters=context.fc_params)

        # Set up the projector for projecting the broken solution
        # into the unbroken finite element spaces
        # NOTE: Tolerance here matters!
        sigma_b, _ = self.broken_solution.split()
        sigma_u, _ = self.unbroken_solution.split()
        self.projector = Projector(sigma_b,
                                   sigma_u,
                                   solver_parameters={"ksp_type": "cg",
                                                      "ksp_rtol": 1e-13})

    def update(self, pc):
        """Update by assembling into the operator. No need to
        reconstruct symbolic objects.
        """
        # TODO: Is this correct? Does this PC even have an "update"?
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
        from firedrake import project

        # Transfer non-broken x into a firedrake function
        with self.unbroken_rhs.dat.vec as v:
            x.copy(v)

        # Transfer unbroken_rhs into broken_rhs
        field0, field1 = self.unbroken_rhs.split()
        bfield0, bfield1 = self.broken_rhs.split()

        # This updates broken_rhs
        project(field0, bfield0)
        field1.dat.copy(bfield1.dat)

        # Compute the rhs for the multiplier system
        self._assemble_Srhs()

        # Solve the system for the Lagrange multipliers
        with self.schur_rhs.dat.vec_ro as b:
            with self.trace_solution.dat.vec as x:
                self.ksp.solve(b, x)

        # Assemble the pressure and velocity (in that order)
        # using the Lagrange multipliers
        self._assemble_pressure()
        self._assemble_velocity()

        # Project the broken solution into non-broken spaces
        sigma_h, u_h = self.broken_solution.split()
        sigma_u, u_u = self.unbroken_solution.split()
        u_h.dat.copy(u_u.dat)
        self.projector.project()
        with self.unbroken_solution.dat.vec_ro as v:
            v.copy(y)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner."""
        raise NotImplementedError("Not implemented yet, sorry!")

    def view(self, pc, viewer=None):
        """View the KSP solver to monitor the Lagrange multiplier
        system.
        """
        super(HybridizationPC, self).view(pc, viewer)
        viewer.printfASCII("KSP solver for the Lagrange multipliers\n")
        viewer.pushASCIITab()
        self.ksp.view(viewer)
        viewer.popASCIITab()
