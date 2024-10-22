from pyadjoint import ReducedFunctional, OverloadedType, Control, Tape, AdjFloat, \
    stop_annotating, no_annotations, get_working_tape, set_working_tape
from pyadjoint.enlisting import Enlist
from firedrake import assemble, inner, dx, Function
from functools import wraps, cached_property
from typing import Callable, Optional

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
    control : The initial condition :math:`x_{0}`. Starting value is used as the
            background (prior) data :math:`x_{b}`.
    background_iprod : The inner product to calculate the background error functional
                     from the background error :math:`x_{0} - x_{b}`. Can include the
                     error covariance matrix.
    observation_err : Given a state :math:`x`, returns the observations error
                    :math:`y_{0} - \\mathcal{H}_{0}(x)` where :math:`y_{0}` are the
                    observations at the initial time and :math:`\\mathcal{H}_{0}` is
                    the observation operator for the initial time. Optional.
    observation_iprod : The inner product to calculate the observation error functional
                      from the observation error :math:`y_{0} - \\mathcal{H}_{0}(x)`.
                      Can include the error covariance matrix. Must be provided if
                      observation_err is provided.
    weak_constraint : Whether to use the weak or strong constraint 4DVar formulation.
    tape : The tape to record on.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    """

    def __init__(self, control: Control,
                 background_iprod: Callable[[OverloadedType], AdjFloat],
                 observation_err: Optional[Callable[[OverloadedType], OverloadedType]] = None,
                 observation_iprod: Optional[Callable[[OverloadedType], AdjFloat]] = None,
                 weak_constraint: bool = True,
                 tape: Optional[Tape] = None,
                 bkg_split: Optional[str] = 'full',
                 _annotate_accumulation: bool = False):

        self.tape = get_working_tape() if tape is None else tape

        self.weak_constraint = weak_constraint
        self.initial_observations = observation_err is not None

        # default to l2 inner products for all functionals
        background_iprod = background_iprod or l2prod
        observation_iprod = observation_iprod or l2prod

        assert bkg_split in ('full', 'split')

        # We need a copy for the prior, but this shouldn't be part of the tape
        with stop_annotating():
            self.background = control.copy_data()

        if self.weak_constraint:
            self._annotate_accumulation = _annotate_accumulation

            self.bkg_split = bkg_split

            # Full background error reduction
            if self.bkg_split == 'full':
                with set_working_tape() as tape:
                    # start from independent control
                    with stop_annotating():
                        control_copy = control.copy_data()
                        control_copy.rename("Control_0_bkg_copy")
                    bkg_err_vec = Function(control_copy.function_space(),
                                           name="bkg_err_vec")
                    bkg_err_vec.assign(control_copy - self.background,
                                       ad_block_tag="bkg_err_sub")
                    bkg_err = background_iprod(bkg_err_vec)
                    self.background_rf_full = ReducedFunctional(
                        bkg_err, Control(control_copy), tape=tape,
                        name="Background_RF", claim_block_variables=True)

            elif self.bkg_split == 'split':
                # new tape for background error vector
                with set_working_tape() as tape:
                    # start from a control independent of any other tapes
                    with stop_annotating():
                        control_copy = control.copy_data()
                        control_copy.rename("Control_0_bkg_copy")

                    # vector of x_0 - x_b
                    bkg_err_vec = Function(control_copy.function_space(),
                                           name="bkg_err_vec")
                    bkg_err_vec.assign(control_copy - self.background,
                                       ad_block_tag="bkg_err_sub")

                    # bkg_err_vec = assemble(control_copy - self.background)
                    # bkg_err_vec.rename("bkg_err_vec")

                    # RF to recover x_0 - x_b
                    self.background_error = ReducedFunctional(
                        bkg_err_vec, Control(control_copy), tape=tape,
                        name="Background_vector_RF", claim_block_variables=True)

                # new tape for background error reduction
                with set_working_tape() as tape:
                    # start from a control independent of any other tapes
                    with stop_annotating():
                        bkg_err_vec_copy = bkg_err_vec.copy(deepcopy=True)
                        bkg_err_vec_copy.rename("bkg_err_vec_copy")

                    # inner product |x_0 - x_b|_B
                    bkg_err = background_iprod(bkg_err_vec_copy)

                    # RF to recover |x_0 - x_b|_B
                    self.background_rf = ReducedFunctional(
                        bkg_err, Control(bkg_err_vec_copy), tape=tape,
                        name="Background_reduction_RF", claim_block_variables=True)

            self.controls = [control]       # The solution at the beginning of each time-chunk
            self.states = []                # The model propogation at the end of each time-chunk
            self.forward_model_stages = []  # ReducedFunctional for each model propogation (returns state)
            self.forward_model_errors = []     # Inner product for model errors (possibly including error covariance)
            self.forward_model_rfs = []     # Inner product for model errors (possibly including error covariance)
            self.forward_model_rfs_full = []     # Inner product for model errors (possibly including error covariance)
            self.observation_errors = []    # ReducedFunctional for each observation set (returns observation error)
            self.observation_rfs = []       # Inner product for observation errors (possibly including error covariance)
            self.observation_rfs_full = []       # Inner product for observation errors (possibly including error covariance)

            if self.initial_observations:

                # Full observation error reduction
                # with set_working_tape() as tape:
                #     obs_err = observation_iprod(observation_err(control.control))
                #     self.observation_rfs_full.append(
                #         ReducedFunctional(obs_err, control, tape=tape,
                #         name="Observation_0_RF", claim_block_variables=True)
                #     )
                self.observation_rfs_full.append(None)

                # new tape for observation error vector
                with set_working_tape() as tape:
                    # start from a control independent of any other tapes
                    with stop_annotating():
                        control_copy = control.copy_data()
                        control_copy.rename("Control_0_obs_copy")

                    # vector of H(x_0) - y_0
                    obs_err_vec = observation_err(control_copy)
                    obs_err_vec.rename("obs_err_vec_0")

                    # RF to recover H(x_0) - y_0
                    self.observation_errors.append(ReducedFunctional(
                        obs_err_vec, Control(control_copy), tape=tape,
                        name="Observation_vector_0_RF", claim_block_variables=True)
                    )

                # new tape for observation error reduction
                with set_working_tape() as tape:
                    # start from a control independent of any othe tapes
                    with stop_annotating():
                        obs_err_vec_copy = obs_err_vec.copy(deepcopy=True)
                        obs_err_vec_copy.rename("obs_err_vec_0_copy")

                    # inner product |H(x_0) - y_0|_R
                    obs_err = observation_iprod(obs_err_vec_copy)

                    # RF to recover |H(x_0) - y_0|_R
                    self.observation_rfs.append(ReducedFunctional(
                        obs_err, Control(obs_err_vec_copy), tape=tape,
                        name="Observation_reduction_0_RF", claim_block_variables=True)
                    )

                # new tape for the next stage
                set_working_tape()
                self._stage_tape = get_working_tape()

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

    def set_observation(self, state: OverloadedType,
                        observation_err: Callable[[OverloadedType], OverloadedType],
                        observation_iprod: Callable[[OverloadedType], AdjFloat],
                        forward_model_iprod: Optional[Callable[[OverloadedType], AdjFloat]]):
        """
        Record an observation at the time of `state`.

        Parameters
        ----------

        state: The state at the current observation time.
        observation_err : Given a state :math:`x`, returns the observations error
                        :math:`y_{i} - \\mathcal{H}_{i}(x)` where :math:`y_{i}` are
                        the observations at the current observation time and
                        :math:`\\mathcal{H}_{i}` is the observation operator for the
                        current observation time.
        observation_iprod : The inner product to calculate the observation error
                          functional from the observation error
                          :math:`y_{i} - \\mathcal{H}_{i}(x)`. Can include the error
                          covariance matrix.
        forward_model_iprod : The inner product to calculate the model error functional
                            from the model error
                            :math:`x_{i} - \\mathcal{M}_{i}(x_{i-1})`. Can include the
                            error covariance matrix. Ignored if using the strong
                            constraint formulation.
        """
        observation_iprod = observation_iprod or l2prod
        if self.weak_constraint:

            forward_model_iprod = forward_model_iprod or l2prod

            stage_index = len(self.controls)

            # Cut the tape into seperate time-chunks.
            # State is output from previous control i.e. forward model
            # propogation over previous time-chunk.

            # get the tape used for this stage and make sure its the right one
            prev_stage_tape = get_working_tape()
            if prev_stage_tape is not self._stage_tape:
                msg = "working tape canot be changed during observation stage"
                raise ValueError(msg)

            # # record forward propogation
            with set_working_tape(prev_stage_tape.copy()) as tape:
                # this_state = state.copy(deepcopy=True)
                prev_control = self.controls[-1]
                self.forward_model_stages.append(ReducedFunctional(
                    state.copy(deepcopy=True), controls=prev_control, tape=tape,
                    name=f"Model_forward_{stage_index}_RF", claim_block_variables=True)
                )

            # with stop_annotating():
            #     print(f"set_observation {stage_index} entry")
            #     for i, fms in enumerate(self.forward_model_stages):
            #         print(f"Testing forward model stage {i} at set_observation entry")
            #         v = Function(state.function_space()).zero()
            #         fms(v)
            #         fms.derivative()
            #     print()

            # Beginning of next time-chunk is the control for this observation
            # and the state at the end of the next time-chunk.
            with stop_annotating():
                # smuggle initial guess at this time into the control without the tape seeing
                next_control_state = state.copy(deepcopy=True)
                next_control_state.rename(f"Control_{len(self.controls)}")
            next_control = Control(next_control_state)
            self.controls.append(next_control)

            # model error links time-chunks by depending on both the
            # previous and current controls

            # with set_working_tape(prev_stage_tape.copy()) as tape:
            #     model_err = forward_model_iprod(state - next_control.control)
            #     self.forward_model_rfs_full.append(ReducedFunctional(
            #         model_err, self.controls[-2:], tape=tape,
            #         name=f"Model_{stage_index}_RF", claim_block_variables=False)
            #     )

            # new tape for model error vector
            self.forward_model_rfs_full.append(None)
            with set_working_tape() as tape:
                # start from a control independent of any other tapes
                with stop_annotating():
                    state_copy = state.copy(deepcopy=True)
                    next_control_copy = next_control.copy_data()

                # vector of M_i - x_i
                model_err_vec = Function(state_copy.function_space())
                model_err_vec.assign(state_copy - next_control_copy,
                                     ad_block_tag=f"model_err_sub_{stage_index}")
                model_err_vec.rename(f"model_err_vec_{stage_index}")

                # RF to recover M_i - x_i
                fmcontrols = [Control(state_copy), Control(next_control_copy)]
                self.forward_model_errors.append(ReducedFunctional(
                    model_err_vec, fmcontrols, tape=tape,
                    name="Model_vector_{stage_index}_RF", claim_block_variables=True)
                )

            # new tape for model error reduction
            with set_working_tape() as tape:
                # start from a control independent of any othe tapes
                with stop_annotating():
                    model_err_vec_copy = model_err_vec.copy(deepcopy=True)

                # inner product |M_i - x_i|_Q
                model_err = forward_model_iprod(model_err_vec_copy)

                # RF to recover |M_i - x_i|_Q
                self.forward_model_rfs.append(ReducedFunctional(
                    model_err, Control(model_err_vec_copy), tape=tape,
                    name=f"Model_reduction_{stage_index}_RF", claim_block_variables=True)
                )

            # Observations after tape cut because this is now a control, not a state

            # with set_working_tape() as tape:
            #     obs_err = observation_iprod(observation_err(next_control.control))
            #     self.observation_rfs_full.append(
            #         ReducedFunctional(obs_err, next_control, tape=tape,
            #         name=f"Observation_{stage_index}_RF", claim_block_variables=False)
            #     )

            # new tape for observation error vector
            self.observation_rfs_full.append(None)
            with set_working_tape() as tape:
                # start from a control independent of any other tapes
                with stop_annotating():
                    next_control_copy = next_control.copy_data()
                    next_control_copy.rename(f"Control_{stage_index}_copy")

                # vector of H(x_i) - y_i
                obs_err_vec = observation_err(next_control_copy)
                obs_err_vec.rename(f"obs_err_vec_{stage_index}")

                # RF to recover H(x_i) - y_i
                self.observation_errors.append(ReducedFunctional(
                    obs_err_vec, Control(next_control_copy), tape=tape,
                    name=f"Observation_vector_{stage_index}_RF", claim_block_variables=True)
                )

            # new tape for observation error reduction
            with set_working_tape() as tape:
                # start from a control independent of any othe tapes
                with stop_annotating():
                    obs_err_vec_copy = obs_err_vec.copy(deepcopy=True)
                    obs_err_vec_copy.rename(f"obs_err_vec_{stage_index}_copy")

                # inner product |H(x_i) - y_i|_R
                obs_err = observation_iprod(obs_err_vec_copy)

                # RF to recover |H(x_i) - y_i|_R
                self.observation_rfs.append(ReducedFunctional(
                    obs_err, Control(obs_err_vec_copy), tape=tape,
                    name=f"Observation_reduction_{stage_index}_RF", claim_block_variables=True)
                )

            # new tape for the next stage

            set_working_tape()
            self._stage_tape = get_working_tape()

            # Look we're starting this time-chunk from an "unrelated" value... really!
            # with stop_annotating(modifies=state):
            #     pass
            state.assign(next_control.control)

            # with stop_annotating():
            #     print(f"set_observation {stage_index} exit")
            #     for i, fms in enumerate(self.forward_model_stages):
            #         print(f"Testing forward model stage {i} at set_observation exit")
            #             v = Function(state.function_space()).zero()
            #         fms(v)
            #         fms.derivative()
            #     print()

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
            raise AttributeError(f"'{type(self)}' object has no attribute '{attr}'")
        else:
            return getattr(self.strong_reduced_functional, attr)

    @sc_passthrough
    @no_annotations
    def __call__(self, values: OverloadedType):
        """Computes the reduced functional with supplied control value.

        Parameters
        ----------
        values : If you have multiple controls this should be a list of new values
               for each control in the order you listed the controls to the constructor.
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

        for c, v in zip(self.controls, values):
            c.control.assign(v)

        model_stages = [None, *self.forward_model_stages]
        model_errors = [None, *self.forward_model_errors]
        model_rfs = [None, *self.forward_model_rfs]
        model_rfs_full = [None, *self.forward_model_rfs_full]

        observation_rfs = (self.observation_rfs if self.initial_observations
                           else [None, *self.observation_rfs])
        observation_rfs_full = (self.observation_rfs_full if self.initial_observations
                                else [None, *self.observation_rfs_full])

        # Initial condition functionals
        if self.bkg_split == 'full':
            J = self.background_rf_full(values[0])
        elif self.bkg_split == 'split':
            bkg_err_vec = self.background_error(values[0])
            J = self.background_rf(bkg_err_vec)

        if self.initial_observations:
            # J += observation_rfs_full[0](values[0])

            obs_err_vec = self.observation_errors[0](values[0])
            J += self.observation_rfs[0](obs_err_vec)

        for i in range(1, len(model_rfs_full)):
            # Model error - does propogation from last control match this control?
            prev_control = values[i-1]
            this_control = values[i]

            # J += model_rfs_full[i]([prev_control, this_control])

            Mi = model_stages[i](prev_control)
            model_err_vec = model_errors[i]([Mi, this_control])
            J += model_rfs[i](model_err_vec)

            # observation error - do we match the 'real world'?
            # J += observation_rfs_full[i](this_control)

            obs_err_vec = self.observation_errors[i](values[i])
            J += self.observation_rfs[i](obs_err_vec)

        return J

    @sc_passthrough
    @no_annotations
    def derivative(self, adj_input: float = 1.0, options: dict = {}):
        """Returns the derivative of the functional w.r.t. the control.
        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the
        control, is computed and returned.

        Parameters
        ----------
        adj_input : The adjoint input.
        options : Additional options for the derivative computation.

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
        model_stages = [None, *self.forward_model_stages]
        model_errors = [None, *self.forward_model_errors]
        model_rfs = [None, *self.forward_model_rfs]
        model_rfs_full = [None, *self.forward_model_rfs_full]

        observation_rfs = (self.observation_rfs if self.initial_observations
                           else [None, *self.observation_rfs])
        observation_rfs_full = (self.observation_rfs_full if self.initial_observations
                                else [None, *self.observation_rfs_full])

        # initial condition derivatives
        # print("Evaluating background derivative")
        if self.bkg_split == 'full':
            derivatives.append(
                self.background_rf_full.derivative(**kwargs))
        elif self.bkg_split == 'split':
            bkg_deriv = self.background_rf.derivative(**kwargs)
            bkg_deriv = bkg_deriv.riesz_representation()
            derivatives.append(
                self.background_error.derivative(adj_input=bkg_deriv))

        if self.initial_observations:
            # print("Evaluating observation 0 derivative")
            # derivatives[0] += observation_rfs_full[0].derivative(**kwargs)

            obs_deriv = self.observation_rfs[0].derivative(**kwargs)
            obs_deriv = obs_deriv.riesz_representation()
            derivatives[0] += self.observation_errors[0].derivative(adj_input=obs_deriv)

        for i in range(1, len(model_rfs_full)):
            # print(f"Evaluating observation {i} derivative")
            # derivatives.append(observation_rfs_full[i].derivative(**kwargs))

            obs_deriv = self.observation_rfs[i].derivative(**kwargs)
            obs_deriv = obs_deriv.riesz_representation()
            derivatives.append(self.observation_errors[i].derivative(adj_input=obs_deriv))

            # # print(f"Evaluating model {i} derivative")
            # mderivs = model_rfs_full[i].derivative(**kwargs)

            # derivatives[i-1] += mderivs[0]
            # derivatives[i] += mderivs[1]

            # print(f"Evaluating model {i} derivative")
            model_deriv = model_rfs[i].derivative(**kwargs)
            # print(f"Evaluating model error {i} derivative")
            model_err_derivs = model_errors[i].derivative(adj_input=model_deriv.riesz_representation())
            # print(f"Evaluating model stage {i} derivative")
            model_stage_deriv = model_stages[i].derivative(adj_input=model_err_derivs[0].riesz_representation())

            derivatives[i-1] += model_stage_deriv
            derivatives[i] += model_err_derivs[1]

        return derivatives

    @sc_passthrough
    @no_annotations
    def hessian(self, m_dot: OverloadedType, options: dict = {}):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Parameters
        ----------
        m_dot : The direction in which to compute the action of the Hessian.
        options : A dictionary of options. To find a list of available options
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
        for rf in (self.background_error,
                   self.background_rf,
                   *self.observation_errors,
                   *self.observation_rfs,
                   *self.forward_model_stages,
                   *self.forward_model_rfs):
            rf.optimize_tape()

    def _accumulate_functional(self, val):
        if not self._annotate_accumulation:
            return
        if hasattr(self, '_total_functional'):
            self._total_functional += val
        else:
            self._total_functional = val
