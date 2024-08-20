from pyadjoint import ReducedFunctional, stop_annotating, Control, \
    overloaded_type
from pyadjoint.enlisting import Enlist
from pyop2.utils import cached_property
from firedrake import assemble, inner, dx
from functools import wraps

__all__ = ['AllAtOnceReducedFunctional']


def sc_passthrough(func):
    """
    Wraps standard ReducedFunctional methods to differentiate strong or
    weak constraint implementations.

    If using strong constraint, makes sure strong_reduced_functional
    is instantiated then passes args/kwargs through to the
    corresponding strong_reduced_functional method.

    If using weak constraint, returns the AllAtOnceReducedFunctional
    method definition.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.weak_constraint:
            return func(self, *args, **kwargs)
        else:
            sc_func = getattr(self.strong_reduced_functional, func.__name__)
            return sc_func(*args, **kwargs)
    return wrapper


def l2prod(x):
    return assemble(inner(x, x)*dx)


class AllAtOnceReducedFunctional(ReducedFunctional):

    def __init__(self, control: Control, background_iprod=None,
                 observation_err=None, observation_iprod=None,
                 weak_constraint=True):
        """
        :arg control: initial guess at initial condition of timeseries (background)
        :arg background_iprod: Inner product callable for background error (possibly including error covariance).
        :arg observation_err: callable mapping state to the initial observation error. Optional.
        :arg observation_iprod: Inner product callable for initial observation error (possibly including error covariance). Optional.
        :arg weak_constraint: whether to use the weak or strong constraint 4DVar formulation.
        """
        self.weak_constraint = weak_constraint
        self.initial_observations = observation_err is not None

        # default to l2 inner products for all functionals
        background_iprod = background_iprod or l2prod
        observation_iprod = observation_iprod or l2prod

        # We need a copy for the prior, but this shouldn't be part of the tape
        with stop_annotating():
            self.background = control.copy_data()

        if self.weak_constraint:
            self.background_iprod = background_iprod  # Inner product for background error (possibly including error covariance)
            self.controls = [control]                 # The solution at the beginning of each time-chunk
            self.states = []                          # The model propogation at the end of each time-chunk
            self.forward_model_stages = []            # ReducedFunctional for each model propogation (returns state)
            self.forward_model_iprods = []            # Inner product for model errors (possibly including error covariance)
            self.observations = []                    # ReducedFunctional for each observation set (returns observation error)
            self.observation_iprods = []              # Inner product for observation errors (possibly including error covariance)

            if self.initial_observations:
                self.observations.append(ReducedFunctional(observation_err(control), control))
                self.observation_iprods.append(observation_iprod)

        else:
            # initial conditions guess to be updated
            self.controls = Enlist(control)

            # Strong constraint functional to be converted to ReducedFunctional later

            # penalty for straying from prior
            self.functional = background_iprod(control.control - self.background)

            # penalty for not hitting observations at initial time
            if self.initial_observations:
                self.functional += observation_iprod(observation_err(control.control))

    def set_observation(self, state: overloaded_type, observation_err,
                        observation_iprod=None, forward_model_iprod=None):
        """
        :arg state: state at current observation time.
        :arg observation_err: function that takes in a state and returns the observation error at the current time.
        """
        observation_iprod = observation_iprod or l2prod
        if self.weak_constraint:

            forward_model_iprod = forward_model_iprod or l2prod

            self.states.append(state.block_variable)
            self.forward_model_iprods.append(forward_model_iprod)

            # Cut the tape into seperate time-chunks.
            # State is output from previous control i.e. forward model
            # propogation over previous time-chunk.
            with stop_annotating(modifies=state):
                self.forward_model_stages.append(
                    ReducedFunctional(state,
                                      controls=self.controls[-1])
                )
            # Beginning of next time-chunk is the control for this observation
            # and the state at the end of the next time-chunk.
            next_control = Control(state)
            self.controls.append(next_control)

            # Observations after tape cut because this is now a control, not a state
            self.observations.append(
                ReducedFunctional(observation_err(state), next_control)
            )
            self.observation_iprods.append(observation_iprod)

        else:

            if hasattr(self, "_strong_reduced_functional"):
                msg = "Cannot add observations once strong constraint ReducedFunctional instantiated"
                raise ValueError(msg)
            self.functional += observation_iprod(observation_err(state))

    @cached_property
    def strong_reduced_functional(self):
        if self.weak_constraint:
            msg = "Strong constraint ReducedFunctional not instantiated for weak constraint 4DVar"
            raise AttributeError(msg)
        self._strong_reduced_functional = ReducedFunctional(
            self.functional, self.controls)
        return self._strong_reduced_functional

    @sc_passthrough
    def __call__(self, control_value):
        # update controls so derivative etc is evaluated at correct point
        for old, new in zip(self.controls, control_value):
            old.update(new)

        controls = self.controls

        # Shift lists so indexing matches standard nomenclature:
        # index 0 is initial condition. Model i propogates from i-1 to i.

        forward_models = [None, *self.forward_model_stages]
        model_iprods = [None, *self.forward_model_iprods]

        observations = (self.observations if self.initial_observations
                        else [None, *self.observations])
        observation_iprods = (self.observation_iprods if self.initial_observations
                              else [None, *self.observation_iprods])

        # Initial condition functionals
        J = self.background_iprod(controls[0].control - self.background)

        if self.initial_observations:
            J += observation_iprods[0](observations[0](controls[0]))

        for i in range(1, len(forward_models)):
            # Propogate forward over previous time-chunk
            end_state = forward_models[i](controls[i-1])

            # Cache end state here so we can reuse it in other functions
            self.states[i-1] = end_state.block_variable

            # Model error - does propogation from last control match this control?
            model_err = end_state - controls[i].control
            J += model_iprods[i](model_err)

            # observation error - do we match the 'real world'?
            obs_err = observations[i](controls[i])
            J += observation_iprods[i](obs_err)

        return J

    @sc_passthrough
    def derivative(self, *args, **kwargs):
        # All the magic goes here.
        raise NotImplementedError

    @sc_passthrough
    def hessian(self, *args, **kwargs):
        raise NotImplementedError

    def hessian_matrix(self):
        # Other reduced functionals don't have this.
        if not self.weak_constraint:
            raise AttributeError("Strong constraint 4DVar does not form a Hessian matrix")
        raise NotImplementedError
