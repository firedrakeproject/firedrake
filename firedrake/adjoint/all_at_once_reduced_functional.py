from pyadjoint import ReducedFunctional, stop_annotating, Control, \
    overloaded_type
from firedrake import norm

class AllAtOnceReducedFunctional(ReducedFunctional):

    def __init__(self, control: Control, observation_fn=None, weak_constraint=True):
        """
        :arg control: initial guess at initial condition of timeseries (background)
        :arg weak_constraint: whether to use the weak or strong constraint 4DVar formulation
        """
        self.weak_constraint = weak_constraint
        if self.weak_constraint:
            self.controls = [control.block_variable]  # The solution at the beginning of each time-chunk
            self.states = [control.block_variable]    # The model propogation at the end of each time-chunk
            self.forward_model_stages = []            # ReducedFunctional for each model propogation
            self.observations = []                    # ReducedFunctional for each observation set
            self.initial_observations = False
            if observation_fn is not None:
                self.initial_observations = True
                self.observations.append(ReducedFunctional(observation_fn(control), control))
        else:
            self.control = control  # Background state
            self.functional = 0     # Strong constraint functional to be converted to ReducedFunctional later

    def set_observation(self, state: overloaded_type, observation_fn):
        if self.weak_constraint:
            self.states.append(state.block_variable)
            with stop_annotating(modifies=state):
                self.forward_model_stages.append(
                    ReducedFunctional(state,
                                      controls=Control(self.controls[-1]))
                )
            self.controls.append(state.block_variable)
            self.observations.append(
                ReducedFunctional(observation_fn(state), control(state))
            )
        else:
            if hasattr(self, "strong_reduced_functional"):
                msg = "Cannot add observations once strong constraint ReducedFunctional instantiated"
                raise ValueError(msg)
            self.functional += observation_fn(state)

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

            # Assumes control_value is iterable over observation stages.
            for i in range(len(self.forward_model_stages):
                if i==0:
                    # background 'error'
                    J += norm(self.controls[0] - control_value[0])
                    # observation error
                    if self.initial_observations:
                        J += norm(self.observations[0](control_value[0]))
                else:
                    # model error
                    stage = self.forward_model_stages[i-1]
                    end_state = stage(control_value[i-1])
                    J += norm(end_state - control)

                    # observation error
                    if self.initial_observations:
                        J += norm(self.observations[i](control_value[i]))
                    else:
                        J += norm(self.observations[i-1](control_value[i-1]))


            return J
        else:

            if not hasattr(self, "strong_reduced_functional"):
                self.strong_reduced_functional = ReducedFunctional(
                    self.functional, self.control)
            return self.strong_reduced_functional(control_value)

    def derivative(self, adj_input=1):
        if self.weak_constraint:
            # All the magic goes here.
            raise NotImplementedError
        else:
            return self.strong_reduced_functional.derivative(adj_input=adj_input)

    def hessian(self, m_dot):
        if self.weak_constraint:
            raise NotImplementedError
        else:
            return self.strong_reduced_functional.hessian(adj_input=adj_input)
    
    def hessian_matrix(self):
        # Other reduced functionals don't have this.
        if self.weak_constraint:
            raise NotImplementedError
        else:
            return self.strong_reduced_functional.hessian_matrix()
