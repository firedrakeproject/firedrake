import ufl
import ufl.algorithms
from pyadjoint.optimization.constraints import (
    Constraint, EqualityConstraint, InequalityConstraint,
)

import firedrake


class UFLConstraint(Constraint):
    """
    Easily implement scalar constraints using UFL.

    The form must be a 0-form that depends on a Function control.
    """

    def __init__(self, form, control):

        if not isinstance(control.control, firedrake.Function):
            raise NotImplementedError("Only implemented for Function controls")

        args = ufl.algorithms.extract_arguments(form)
        if len(args) != 0:
            raise ValueError("Must be a rank-zero form, i.e. a functional")

        u = control.control
        self.V = u.function_space()
        # We want to make a copy of the control purely for use
        # in the constraint, so that our writing it isn't
        # bothering anyone else
        self.u = firedrake.Function(self.V)
        self.form = ufl.replace(form, {u: self.u})

        self.test = firedrake.TestFunction(self.V)
        self.dform = firedrake.derivative(self.form, self.u, self.test)
        if len(ufl.algorithms.extract_arguments(
            ufl.algorithms.expand_derivatives(self.dform)
        )) == 0:
            raise ValueError("Form must depend on control")

        self.trial = firedrake.TrialFunction(self.V)
        self.hess = ufl.algorithms.expand_derivatives(
            firedrake.derivative(self.dform, self.u, self.trial)
        )
        self.zero_hess = len(ufl.algorithms.extract_arguments(self.hess)) == 0

    def update_control(self, m):
        if isinstance(m, list):
            assert len(m) == 1
            m = m[0]

        if isinstance(m, firedrake.Function):
            self.u.assign(m)
        else:
            self.u._ad_assign_numpy(self.u, m, 0)

    def function(self, m):
        self.update_control(m)
        b = firedrake.assemble(self.form)
        return firedrake.Constant(b)

    def jacobian(self, m):
        if isinstance(m, list):
            assert len(m) == 1
            m = m[0]

        self.update_control(m)
        out = [firedrake.assemble(self.dform)]
        return out

    def jacobian_action(self, m, dm, result):
        """Computes the Jacobian action of c(m) in direction dm.

        Stores the result in result.
        """

        if isinstance(m, list):
            assert len(m) == 1
            m = m[0]
        self.update_control(m)

        form = firedrake.action(self.dform, dm)
        result.assign(firedrake.assemble(form))

    def jacobian_adjoint_action(self, m, dp, result):
        """Computes the Jacobian adjoint action of c(m) in direction dp.

        Stores the result in result.
        """

        if isinstance(m, list):
            assert len(m) == 1
            m = m[0]
        self.update_control(m)

        asm = firedrake.assemble(
            dp * ufl.replace(self.dform, {self.trial: self.test})
        )
        if isinstance(result, firedrake.Cofunction):
            result.assign(asm)
        else:
            raise NotImplementedError("Do I need to untangle all controls?")

    def hessian_action(self, m, dm, dp, result):
        """Computes the Hessian action of c(m) in direction dm and dp.

        Stores the result in result.
        """

        if isinstance(m, list):
            assert len(m) == 1
            m = m[0]
        self.update_control(m)

        H = dm * ufl.replace(self.hess, {self.trial: dp})
        if isinstance(result, firedrake.Function):
            if self.zero_hess:
                result.assign(0)
            else:
                result.assign(firedrake.assemble(H))

        else:
            raise NotImplementedError("Do I need to untangle all controls?")

    def output_workspace(self):
        """Return an object like the output of c(m) for calculations."""

        return firedrake.Constant(firedrake.assemble(self.form))

    def _get_constraint_dim(self):
        """Returns the number of constraint components."""
        return 1


class UFLEqualityConstraint(UFLConstraint, EqualityConstraint):
    pass


class UFLInequalityConstraint(UFLConstraint, InequalityConstraint):
    pass
