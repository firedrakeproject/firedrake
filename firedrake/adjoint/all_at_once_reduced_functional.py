from pyadjoint import ReducedFunctional, stop_annotating, Control, \
    overloaded_type
from pyadjoint.enlisting import Enlist
from firedrake import norm


class AllAtOnceReducedFunctional(ReducedFunctional):

    def __init__(self, control: Control, background_fn=None, observation_fn=None, weak_constraint=True):
        """
        :arg control: initial guess at initial condition of timeseries (background)
        :arg background_fn: function that takes in the initial conditions and returns the background functional. Only required if using strong constraint.
        :arg observation_fn: function that takes in a state and returns the observation functional at time 0. Optional.
        :arg weak_constraint: whether to use the weak or strong constraint 4DVar formulation
        """
        self.weak_constraint = weak_constraint
        self.initial_observations = observation_fn is not None
        if self.weak_constraint:
            self.controls = [control.block_variable]  # The solution at the beginning of each time-chunk
            self.states = [control.block_variable]    # The model propogation at the end of each time-chunk
            self.forward_model_stages = []            # ReducedFunctional for each model propogation
            self.observations = []                    # ReducedFunctional for each observation set
            if self.initial_observations:
                self.observations.append(ReducedFunctional(observation_fn(control), control))
        else:
            self.controls = Enlist(control)  # initial conditions guess to be updated

            # Strong constraint functional to be converted to ReducedFunctional later
            self.functional = background_fn(control.control)  # penalty for straying from prior
            if self.initial_observations:
                self.functional += observation_fn(control.control)  # penalty for not hitting observations

    def set_observation(self, state: overloaded_type, observation_fn):
        """
        :arg state: state at current observation time.
        :arg observation_fn: function that takes in a state and returns the observation functional at the current time.
        """
        if self.weak_constraint:
            self.states.append(state.block_variable)
            with stop_annotating(modifies=state):
                self.forward_model_stages.append(
                    ReducedFunctional(state,
                                      controls=Control(self.controls[-1]))
                )
            self.controls.append(state.block_variable)
            # should the control for the observations be the state at the end of the previous time-chunk
            # or the state at the beginning of the next time-chunk? These won't be equal once we start
            # the optimisation iterations.
            self.observations.append(
                ReducedFunctional(observation_fn(state), Control(state))
            )

        else:
            if hasattr(self, "strong_reduced_functional"):
                msg = "Cannot add observations once strong constraint ReducedFunctional instantiated"
                raise ValueError(msg)
            self.functional += observation_fn(state)

    def setup_strong_constraint(self):
        if not hasattr(self, "strong_reduced_functional"):
            self.strong_reduced_functional = ReducedFunctional(
                self.functional, self.controls)

    def __call__(self, control_value):
        if self.weak_constraint:
            J = 0.0
            # Assumes control_value is iterable over observation stages.
            for i in range(len(self.forward_model_stages)+1):  # +1 for the initial time
                if i == 0:
                    # background 'error' - how far from the prior ic?
                    J += norm(control_value[0] - self.controls[0])
                    # observation error - do we match the 'real world'?
                    if self.initial_observations:
                        J += norm(self.observations[0](control_value[0]))
                else:
                    # model error - does propogation from last control match this control?
                    stage = self.forward_model_stages[i-1]
                    end_state = stage(control_value[i-1])
                    J += norm(control_value[i] - end_state)

                    # observation error - do we match the 'real world'?
                    if self.initial_observations:
                        J += norm(self.observations[i](control_value[i]))
                    else:
                        J += norm(self.observations[i-1](control_value[i]))
            return J

        else:
            self.setup_strong_constraint()
            return self.strong_reduced_functional(control_value)

    def derivative(self, *args, **kwargs):
        if self.weak_constraint:
            # All the magic goes here.
            raise NotImplementedError
        else:
            self.setup_strong_constraint()
            return self.strong_reduced_functional.derivative(*args, **kwargs)

    def hessian(self, *args, **kwargs):
        if self.weak_constraint:
            raise NotImplementedError
        else:
            self.setup_strong_constraint()
            return self.strong_reduced_functional.hessian(*args, **kwargs)

    def hessian_matrix(self):
        # Other reduced functionals don't have this.
        if self.weak_constraint:
            raise NotImplementedError
        else:
            raise AttributeError("Strong constraint 4DVar does not form a Hessian matrix")
