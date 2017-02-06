"""This module provides custom python preconditioners utilizing
the Slate language.
"""

from __future__ import absolute_import, print_function, division

import ufl

from firedrake import assemble
from firedrake.matrix_free.preconditioners import PCBase
from firedrake.petsc import PETSc
from firedrake.slate.slate import Tensor


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
        from firedrake import (FunctionSpace, TrialFunction, TestFunction,
                               Function, TrialFunctions, TestFunctions,
                               BrokenElement, MixedElement, dS, FacetNormal)
        from firedrake.formmanipulation import ArgumentReplacer
        from ufl.algorithms.map_integrands import map_integrand_dags

        # Extract the problem context
        prefix = pc.getOptionsPrefix()
        _, P = pc.getOperators()
        context = P.getPythonContext()
        test, trial = context.a.arguments()

        # Break the function spaces and define fully discontinuous spaces
        V = test.function_space()
        broken_elements = [BrokenElement(Vi.ufl_element()) for Vi in V]
        if len(broken_elements) == 1:
            V_d = FunctionSpace(V.mesh(), broken_elements[0])
            arg_map = {test: TestFunction(V_d),
                       trial: TrialFunction(V_d)}
        else:
            V_d = FunctionSpace(V.mesh(), MixedElement(broken_elements))
            arg_map = {test: TestFunctions(V_d),
                       trial: TrialFunctions(V_d)}

        # Replace the problems arguments with arguments defined
        # on the new discontinuous spaces
        replacer = ArgumentReplacer(arg_map)
        new_form = map_integrand_dags(replacer, context.a)

        # Create the space of approximate traces:
        # the space of Lagrange multipliers
        TraceSpace = FunctionSpace(V.mesh(), "HDiv Trace",
                                   V_d.ufl_element().degree() - 1)

        # Broken flux and scalar terms (solution via reconstruction)
        self.broken_solution = Function(V_d)
        # Broken RHS of the fully discontinuous problem
        self.broken_rhs = Function(V_d)
        # Solution of the system for the Lagrange multipliers
        self.trace_solution = Function(TraceSpace)

        # Create the symbolic Schur-reduction of the discontinuous
        # problem in Slate. Weakly enforce continuity via the Lagrange
        # multipliers
        A = Tensor(new_form)
        gammar = TestFunction(TraceSpace)
        n = FacetNormal(V.mesh())
        K = Tensor(gammar('+') * ufl.dot(trial[0], n) * dS)
        self.schur_comp = K * A.inv * K.T
        self.schur_rhs = K * A.inv * self.broken_rhs

        # Set up the KSP for the system of Lagrange multipliers
        ksp = PETSc.KSP().create(comm=pc.comm)
        ksp.setOperators(self.schur_comp, self.schur_rhs)
        ksp.setOptionsPrefix(prefix + "trace_")
        ksp.setUp()
        ksp.setFromOptions()
        self.ksp = ksp

    def update(self, pc):
        """Update by assembling into the operator. No need to
        reconstruct symbolic objects.
        """
        assemble(self.schur_comp, tensor=self.schur_comp)
