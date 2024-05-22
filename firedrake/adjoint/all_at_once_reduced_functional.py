from pyadjoint import ReducedFunctional, stop_annotating, Control, \
    overloaded_type
from firedrake import norm

class AllAtOnceReducedFunctional(ReducedFunctional):

    def __init__(self, control: Control, weak_constraint=True):
        self.weak_constraint = weak_constraint
        self.controls = [control.block_variable]
        self.states = [control.block_variable]
        self.observations = [None]
        self.forward_model_stages = []

    def set_observation(self, state: overloaded_type, observation_fn):
        self.states.append(state.block_variable)
        if self.weak_constraint:
            with stop_annotating(modifies=state):
                self.forward_model_stages.append(
                    ReducedFunctional(state,
                                      controls=Control(self.controls[-1]))
                )
        else:
            raise NotImplementedError
        self.controls.append(state.block_variable)
        self.observations.append(
            ReducedFunctional(observation_fn(state), control(state))
        )

    def __call__(self, control_value):
        if self.weak_constraint:
            J = 0.0
            end_state = None
            # Assumes control_value is iterable over observation stages.
            for stage, control, observation in zip(self.forward_model_stages,
                                                   control_value,
                                                   self.observations):
                if end_state is None:
                    # first stage
                    J += norm(self.controls[0] - control)
                else:
                    J += norm(end_state - control)
                end_state = stage(control)
                J += norm(observation(end_state))
            return J
        else:
            raise NotImplementedError

    def derivative(self, adj_input=1):
        # All the magic goes here.
        raise NotImplementedError

    def hessian(self, m_dot):
        raise NotImplementedError
    
    def hessian_matrix(self):
        # Other reduced functionals don't have this.
        raise NotImplementedError