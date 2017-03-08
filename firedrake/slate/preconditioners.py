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
    mixed saddle-point problem using hybridization.

    The forward eliminations and backwards reconstructions
    are performed element-local using the Slate language.
    """
    def initialize(self, pc):
        """Set up the problem context. Take the original
        mixed problem and reformulate the problem as a
        hybridized mixed system.

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

        V = test.function_space()
        if V.mesh().cell_set._extruded:
            # TODO: Merge FIAT branch to support TPC trace elements
            raise NotImplementedError("Not implemented on extruded meshes.")

        # Break the function spaces and define fully discontinuous spaces
        broken_elements = [BrokenElement(Vi.ufl_element()) for Vi in V]
        elem = MixedElement(broken_elements)
        V_d = FunctionSpace(V.mesh(), elem)
        arg_map = {test: TestFunction(V_d),
                   trial: TrialFunction(V_d)}

        # Replace the problems arguments with arguments defined
        # on the new discontinuous spaces
        replacer = ArgumentReplacer(arg_map)
        new_form = map_integrand_dags(replacer, context.a)

        # Create the space of approximate traces.
        # The vector function space will have a non-empty value_shape
        W = next(v for v in V if bool(v.ufl_element().value_shape()))
        if W.ufl_element().family() in ["Raviart-Thomas", "RTCF"]:
            tdegree = W.ufl_element().degree() - 1

        else:
            tdegree = W.ufl_element().degree()

        # NOTE: Once extruded is ready, we will need to be aware of this
        # and construct the appropriate trace space for the HDiv element
        TraceSpace = FunctionSpace(V.mesh(), "HDiv Trace", tdegree)

        # NOTE: For extruded, we will need to add "on_top" and "on_bottom"
        trace_conditions = [DirichletBC(TraceSpace, Constant(0.0),
                                        "on_boundary")]

        # Set up the functions for the original, hybridized
        # and schur complement systems
        self.broken_solution = Function(V_d)
        self.broken_rhs = Function(V_d)
        self.trace_solution = Function(TraceSpace)
        self.unbroken_solution = Function(V)
        self.unbroken_rhs = Function(V)

        # Create the symbolic Schur-reduction
        Atilde = Tensor(new_form)
        gammar = TestFunction(TraceSpace)
        n = FacetNormal(V.mesh())

        # Vector trial function will have a non-empty ufl_shape
        sigma = next(f for f in TrialFunctions(V_d) if bool(f.ufl_shape))

        # NOTE: Once extruded is ready, this will change slightly
        # to include both horizontal and vertical interior facets
        K = Tensor(gammar('+') * ufl.dot(sigma, n) * ufl.dS)

        # Assemble the Schur complement operator and right-hand side
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

        # Nullspace for the multiplier problem
        nullsp = P.getNullSpace()
        if nullsp.handle != 0:
            new_vecs = get_trace_nullspace_vecs(K * Atilde.inv, nullsp,
                                                V, V_d, TraceSpace)
            tr_nullsp = PETSc.NullSpace().create(vectors=new_vecs,
                                                 comm=pc.comm)
            Smat.setNullSpace(tr_nullsp)

        # Set up the KSP for the system of Lagrange multipliers
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setOptionsPrefix(prefix + "trace_")
        ksp.setTolerances(rtol=1e-13)
        ksp.setOperators(Smat)
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp

        # Now we construct the local tensors for the reconstruction stage
        # TODO: Add support for mixed tensors and these variables
        # become unnecessary
        split_forms = split_form(new_form)
        A = Tensor(next(sf.form for sf in split_forms
                        if sf.indices == (0, 0)))
        B = Tensor(next(sf.form for sf in split_forms
                        if sf.indices == (1, 0)))
        C = Tensor(next(sf.form for sf in split_forms
                        if sf.indices == (1, 1)))
        trial = TrialFunction(FunctionSpace(V.mesh(),
                                            BrokenElement(W.ufl_element())))
        K_local = Tensor(gammar('+') * ufl.dot(trial, n) * ufl.dS)

        # Split functions and reconstruct each bit separately
        sigma_h, u_h = self.broken_solution.split()
        g, f = self.broken_rhs.split()

        # Pressure reconstruction
        M = B * A.inv * B.T + C
        u_sol = M.inv * f + M.inv * (B * A.inv *
                                     K_local.T * self.trace_solution
                                     - B * A.inv * g)
        self._assemble_pressure = create_assembly_callable(
            u_sol,
            tensor=u_h,
            form_compiler_parameters=context.fc_params)

        # Velocity reconstruction
        sigma_sol = A.inv * g + A.inv * (B.T * u_h -
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
        raise NotImplementedError(
            "The transpose application of this PC"
            "is not implemented."
        )

    def view(self, pc, viewer=None):
        super(HybridizationPC, self).view(pc, viewer)
        viewer.printfASCII("Hybridizing mixed system:\n")
        viewer.pushASCIITab()
        viewer.printfASCII("KSP solver for the multipliers:\n")
        viewer.pushASCIITab()
        self.ksp.view(viewer)
        viewer.popASCIITab()


def get_trace_nullspace_vecs(forward, nullspace, V, V_d, TraceSpace):
    """Gets the nullspace vectors corresponding to the Schur complement
    system for the multipliers.

    :arg forward: A Slate expression denoting the forward elimination
                  operator.
    :arg nullspace: A nullspace for the original mixed problem
    :arg V: The original "unbroken" space.
    :arg V_d: The broken space.
    :arg TraceSpace: The space of approximate traces.

    Returns: A list of vectors describing the nullspace of the multiplier
             system
    """
    from firedrake import project, assemble, Function

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

    return new_vecs
