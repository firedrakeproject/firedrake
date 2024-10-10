from pyadjoint import ReducedFunctional, OverloadedType, Control, \
    stop_annotating, no_annotations, get_working_tape, Tape
from pyadjoint.enlisting import Enlist
from firedrake import assemble, inner, dx
from functools import wraps, cached_property

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
    """ReducedFunctional for 4DVar data assimilation.

    Creates either the strong constraint or weak constraint system incrementally
    by logging observations through the initial forward model run.

    Warning: Weak constraint 4DVar not implemented yet.

    Parameters
    ----------
    control : pyadjoint.Control
        The initial condition :math:`x_{0}`. Starting value is used as the
        background (prior) data :math:`x_{b}`.
    background_iprod : Optional[Callable[[pyadjoint.OverloadedType], pyadjoint.AdjFloat]]
        The inner product to calculate the background error functional from the
        background error :math:`x_{0} - x_{b}`. Can include the error covariance matrix.
        Defaults to L2.
    observation_err : Optional[Callable[[pyadjoint.OverloadedType], pyadjoint.OverloadedType]]
        Given a state :math:`x`, returns the observations error
        :math:`y_{0} - \\mathcal{H}_{0}(x)` where :math:`y_{0}` are the observations at
        the initial time and :math:`\\mathcal{H}_{0}` is the observation operator for
        the initial time.
    observation_iprod : Optional[Callable[[pyadjoint.OverloadedType], pyadjoint.AdjFloat]]
        The inner product to calculate the observation error functional from the
        observation error :math:`y_{0} - \\mathcal{H}_{0}(x)`. Can include the error
        covariance matrix. Ignored if observation_err not provided.
        Defaults to L2.
    weak_constraint : bool
        Whether to use the weak or strong constraint 4DVar formulation.
    tape : pyadjoint.Tape
        The tape to record on.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    """

    def __init__(self, control: Control, background_iprod=None,
                 observation_err=None, observation_iprod=None,
                 weak_constraint=True, tape=None,
                 _annotate_accumulation=False):

        self.tape = get_working_tape() if tape is None else tape

        self.weak_constraint = weak_constraint
        self.initial_observations = observation_err is not None

        # default to l2 inner products for all functionals
        background_iprod = background_iprod or l2prod
        observation_iprod = observation_iprod or l2prod

        # We need a copy for the prior, but this shouldn't be part of the tape
        with stop_annotating():
            self.background = control.copy_data()

        if self.weak_constraint:
            self._annotate_accumulation = _annotate_accumulation

            background_err = background_iprod(control.control - self.background)
            self.background_rf = ReducedFunctional(
                background_err, control, tape=self.tape)
            self._accumulate_functional(background_err)

            self.controls = [control]       # The solution at the beginning of each time-chunk
            self.states = []                # The model propogation at the end of each time-chunk
            self.forward_model_stages = []  # ReducedFunctional for each model propogation (returns state)
            self.forward_model_rfs = []     # Inner product for model errors (possibly including error covariance)
            self.observations = []          # ReducedFunctional for each observation set (returns observation error)
            self.observation_rfs = []       # Inner product for observation errors (possibly including error covariance)

            if self.initial_observations:
                obs_err = observation_err(control.control)
                self.observations.append(
                    ReducedFunctional(obs_err, control, tape=self.tape)
                )
                obs_err = observation_iprod(obs_err)
                self.observation_rfs.append(
                    ReducedFunctional(obs_err, control, tape=self.tape)
                )
                self._accumulate_functional(obs_err)

        else:
            self._annotate_accumulation = True

            # initial conditions guess to be updated
            self.controls = Enlist(control)

            # Strong constraint functional to be converted to ReducedFunctional later

            # penalty for straying from prior
            self._accumulate_functional(
                background_iprod(control.control - self.background))

            # penalty for not hitting observations at initial time
            if self.initial_observations:
                self._accumulate_functional(
                    observation_iprod(observation_err(control.control)))

    def set_observation(self, state: OverloadedType, observation_err,
                        observation_iprod=None, forward_model_iprod=None):
        """
        Record an observation at the time of `state`.

        Parameters
        ----------

        state: pyadjoint.OverloadedType
            The state at the current observation time.
        observation_err : Callable[[pyadjoint.OverloadedType], pyadjoint.OverloadedType]
            Given a state :math:`x`, returns the observations error
            :math:`y_{i} - \\mathcal{H}_{i}(x)` where :math:`y_{i}` are the observations
            at the current observation time and :math:`\\mathcal{H}_{i}` is the
            observation operator for the current observation time.
        observation_iprod : Optional[Callable[[pyadjoint.OverloadedType], pyadjoint.AdjFloat]]
            The inner product to calculate the observation error functional from the
            observation error :math:`y_{i} - \\mathcal{H}_{i}(x)`. Can include the error
            covariance matrix.
            Defaults to L2.
        forward_model_iprod : Optional[Callable[[pyadjoint.OverloadedType], pyadjoint.AdjFloat]]
            The inner product to calculate the model error functional from the
            model error :math:`x_{i} - \\mathcal{M}_{i}(x_{i-1})`. Can include the error
            covariance matrix. Ignored if using the strong constraint formulation.
            Defaults to L2.
        """
        observation_iprod = observation_iprod or l2prod
        if self.weak_constraint:

            forward_model_iprod = forward_model_iprod or l2prod

            # save propogated value for model error calculation after tape cut
            self.states.append(state.block_variable.output)

            # Cut the tape into seperate time-chunks.
            # State is output from previous control i.e. forward model
            # propogation over previous time-chunk.
            prev_control = self.controls[-1]
            self.forward_model_stages.append(
                ReducedFunctional(state, controls=prev_control, tape=self.tape)
            )

            # Beginning of next time-chunk is the control for this observation
            # and the state at the end of the next time-chunk.
            with stop_annotating():
                # smuggle initial guess at this time into the control without the tape seeing
                next_control = Control(state._ad_copy())
                next_control.control.topological.rename(f"Control {len(self.controls)}")
                self.controls.append(next_control)

            # model error links time-chunks by depending on both the
            # previous and current controls
            model_err = forward_model_iprod(state - next_control.control)
            self.forward_model_rfs.append(
                ReducedFunctional(model_err, self.controls[-2:], tape=self.tape)
            )
            self._accumulate_functional(model_err)

            # Observations after tape cut because this is now a control, not a state
            obs_err = observation_err(next_control.control)
            self.observations.append(
                ReducedFunctional(obs_err, next_control, tape=self.tape)
            )
            obs_err = observation_iprod(obs_err)
            self.observation_rfs.append(
                ReducedFunctional(obs_err, next_control, tape=self.tape)
            )
            self._accumulate_functional(obs_err)

            # Look we're starting this time-chunk from an "unrelated" value... really!
            state.assign(next_control.control)

        else:

            if hasattr(self, "_strong_reduced_functional"):
                msg = "Cannot add observations once strong constraint ReducedFunctional instantiated"
                raise ValueError(msg)
            self._accumulate_functional(
                observation_iprod(observation_err(state)))

    @cached_property
    def strong_reduced_functional(self):
        """A :class:`pyadjoint.ReducedFunctional` for the strong constraint 4DVar system.

        Only instantiated if using the strong constraint formulation, and cannot be used
        before all observations are recorded.
        """
        if self.weak_constraint:
            msg = "Strong constraint ReducedFunctional not instantiated for weak constraint 4DVar"
            raise AttributeError(msg)
        self._strong_reduced_functional = ReducedFunctional(
            self._total_functional, self.controls, tape=self.tape)
        return self._strong_reduced_functional

    def __getattr__(self, attr):
        """
        If using strong constraint then grab attributes from self.strong_reduced_functional.
        """
        if self.weak_constraint:
            raise AttributeError
        else:
            return getattr(self.strong_reduced_functional, attr)

    @sc_passthrough
    @no_annotations
    def __call__(self, values):
        """Computes the reduced functional with supplied control value.

        Parameters
        ----------
        values : pyadjoint.OverloadedType
            If you have multiple controls this should be a list of
            new values for each control in the order you listed the controls to the constructor.
            If you have a single control it can either be a list or a single object.
            Each new value should have the same type as the corresponding control.

        Returns
        -------
        pyadjoint.OverloadedType
            The computed value. Typically of instance of :class:`pyadjoint.AdjFloat`.

        """
        # controls are updated by the sub ReducedFunctionals
        # so we don't need to do it ourselves

        # Shift lists so indexing matches standard nomenclature:
        # index 0 is initial condition.
        # Model i propogates from i-1 to i.
        # Observation i is at i.

        model_rfs = [None, *self.forward_model_rfs]

        observation_rfs = (self.observation_rfs if self.initial_observations
                           else [None, *self.observation_rfs])

        # Initial condition functionals
        J = self.background_rf(values[0])

        if self.initial_observations:
            J += observation_rfs[0](values[0])

        for i in range(1, len(model_rfs)):
            # Model error - does propogation from last control match this control?
            prev_control = values[i-1]
            this_control = values[i]

            J += model_rfs[i]([prev_control, this_control])

            # observation error - do we match the 'real world'?
            J += observation_rfs[i](this_control)

        return J

    @sc_passthrough
    @no_annotations
    def derivative(self, adj_input=1.0, options={}):
        """Returns the derivative of the functional w.r.t. the control.
        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the
        control, is computed and returned.

        Parameters
        ----------
        adj_input : float
            The adjoint input.
        options : dict
            Additional options for the derivative computation.

        Returns
        -------
        pyadjoint.OverloadedType
            The derivative with respect to the control.
            Should be an instance of the same type as the control.
        """
        # create a list of overloaded types to put derivative into
        derivatives = []

        kwargs = {'adj_input': adj_input, 'options': options}

        # Shift lists so indexing matches standard nomenclature:
        # index 0 is initial condition. Model i propogates from i-1 to i.
        model_rfs = [None, *self.forward_model_rfs]

        observation_rfs = (self.observation_rfs if self.initial_observations
                           else [None, *self.observation_rfs])

        # initial condition derivatives
        derivatives.append(
            self.background_rf.derivative(**kwargs))

        if self.initial_observations:
            derivatives[0] += observation_rfs[0].derivative(**kwargs)

        for i in range(1, len(model_rfs)):
            derivatives.append(observation_rfs[i].derivative(**kwargs))

            mderivs = model_rfs[i].derivative(**kwargs)

            derivatives[i-1] += mderivs[0]
            derivatives[i] += mderivs[1]

        return derivatives

    @sc_passthrough
    @no_annotations
    def hessian(self, m_dot, options={}):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Parameters
        ----------
        m_dot : pyadjoint.OverloadedType
            The direction in which to compute the action of the Hessian.
        options : dict
            A dictionary of options. To find a list of available options
            have a look at the specific control type.

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.
        """
        # create a list of overloaded types to put hessian into
        hessians = []

        kwargs = {'options': options}

        # Shift lists so indexing matches standard nomenclature:
        # index 0 is initial condition. Model i propogates from i-1 to i.
        model_rfs = [None, *self.forward_model_rfs]

        observation_rfs = (self.observation_rfs if self.initial_observations
                           else [None, *self.observation_rfs])

        # initial condition hessians
        hessians.append(
            self.background_rf.hessian(m_dot[0], **kwargs))

        if self.initial_observations:
            hessians[0] += observation_rfs[0].hessian(m_dot[0], **kwargs)

        for i in range(1, len(model_rfs)):
            hessians.append(observation_rfs[i].hessian(m_dot[i], **kwargs))

            mhess = model_rfs[i].hessian(m_dot[i-1:i+1], **kwargs)

            hessians[i-1] += mhess[0]
            hessians[i] += mhess[1]

        return hessians

    @no_annotations
    def hessian_matrix(self):
        # Other reduced functionals don't have this.
        if not self.weak_constraint:
            raise AttributeError("Strong constraint 4DVar does not form a Hessian matrix")
        raise NotImplementedError

    @sc_passthrough
    @no_annotations
    def optimize_tape(self):
        all_rfs = [
            self.background_rf,
            *self.forward_model_rfs,
            *self.observation_rfs
        ]
        for rf in all_rfs:
            rf.tape = Tape(rf.tape._blocks, rf.tape._package_data)
            rf.optimize_tape()

    def _accumulate_functional(self, val):
        if not self._annotate_accumulation:
            return
        if hasattr(self, '_total_functional'):
            self._total_functional += val
        else:
            self._total_functional = val
