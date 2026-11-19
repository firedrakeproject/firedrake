from types import SimpleNamespace

from firedrake import inner, assemble
from firedrake.external_operators import AbstractExternalOperator, assemble_method
from firedrake.norms import errornorm
from firedrake.matrix import AssembledMatrix
from pyadjoint.optimization.tao_solver import (
    ReducedFunctionalMat,
    RFOperation,
)


def ReducedFunctionalOperator(Jhat, update_eps=None):
    operands = tuple(c.control for c in Jhat.controls) + tuple(Jhat.parameters)
    return ReducedFunctionalOperatorImpl(
        operands[0],
        function_space=Jhat.functional.function_space(),
        operator_data=SimpleNamespace(Jhat=Jhat,
                                      update_eps=update_eps,
                                      parameters=tuple(Jhat.parameters)),
    )


class ReducedFunctionalOperatorImpl(AbstractExternalOperator):
    @property
    def Jhat(self):
        return self.operator_data.Jhat

    @property
    def update_eps(self):
        return self.operator_data.update_eps

    @property
    def num_controls(self):
        return len(self.Jhat.controls)
    
    def update(self):
        controls = self.ufl_operands[:self.num_controls]
        parameters = self.ufl_operands[self.num_controls:]
        if self.update_eps is None:
            to_update = True
        else:
            xold = (tuple(c.tape_value() for c in self.Jhat.controls) + tuple(self.Jhat.parameters))
            xnew = self.ufl_operands
            to_update = assemble(inner(xnew - xold, xnew - xold) * dx(domain=self.function_space().mesh()))**0.5 > self.update_eps
            # xold = self.Jhat.control[0].data()
            # xnew = controls[0]
            # to_update = errornorm(xold, xnew) > self.update_eps

        if to_update:
            if parameters:
                self.Jhat.update_parameters(parameters)
            self.Jhat(controls)
            # self.Jhat(*self.ufl_operands)
            # self.Jhat.update_parameters(parameters)

    @assemble_method(0, (0,))
    def assemble_J(self, *args, **kwargs):
        return self.Jhat(*self.ufl_operands)

    @assemble_method(1, (0, None))
    def assemble_derivative_action(self, *args, **kwargs):
        self.update()
        return self.Jhat.tlm(self.argument_slots()[1])

    @assemble_method(1, (None, 0))
    def assemble_adjoint_derivative_action(self, *args, **kwargs):
        self.update()
        return self.Jhat.derivative(adj_input=self.argument_slots()[0])

    @assemble_method(2, (0, None, None))
    def assemble_hessian_action(self, *args, **kwargs):
        # TODO: Is this correct? Should we check that the functional is
        #       a Real so the hessian is a 2-form not a 3rd order tensor?
        self.update()
        return self.Jhat.hessian(
            m_dot=self.argument_slots()[2], hessian_input=self.argument_slots()[1]
        )

    @assemble_method(1, (0, 1))
    def assemble_derivative(self, *args, **kwargs):
        # TODO: Is this correct?
        # TODO: ReducedFunctionalMat should probably have option:
        #       -mat_rf_update_schedule (always|never|eps)
        return AssembledMatrix(
            args=self.arguments(),
            petscmat=ReducedFunctionalMat(
                self.Jhat,
                RFOperation.TLM,
                comm=self.ufl_operands[0].function_space().mesh().comm,
                always_update_tape=True,
            ),
        )

    @assemble_method(1, (1, 0))
    def assemble_adjoint_derivative(self, *args, **kwargs):
        # TODO: Is this correct?
        # TODO: ReducedFunctionalMat should probably have option:
        #       -mat_rf_update_schedule (always|never|eps)
        return AssembledMatrix(
            args=self.arguments(),
            petscmat=ReducedFunctionalMat(
                self.Jhat,
                RFOperation.ADJOINT,
                comm=self.ufl_operands[0].function_space().mesh().comm,
                always_update_tape=True,
            ),
        )
