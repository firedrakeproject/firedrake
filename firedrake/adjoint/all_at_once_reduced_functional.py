from pyadjoint import ReducedFunctional, OverloadedType, Control, Tape, AdjFloat, \
    stop_annotating, get_working_tape, set_working_tape
from pyadjoint.enlisting import Enlist
from functools import wraps, cached_property
from typing import Callable, Optional
from contextlib import contextmanager
from mpi4py import MPI

__all__ = ['AllAtOnceReducedFunctional']


# @set_working_tape()  # ends up using old_tape = None because evaluates when imported - need separate decorator
def isolated_rf(operation, control,
                functional_name=None,
                control_name=None):
    """
    Return a ReducedFunctional where the functional is `operation` applied
    to a copy of `control`, and the tape contains only `operation`.
    """
    with set_working_tape():
        controls = Enlist(control)
        control_names = Enlist(control_name)

        with stop_annotating():
            control_copies = [control._ad_copy() for control in controls]

            if control_names:
                for control, name in zip(control_copies, control_names):
                    _rename(control, name)

        if len(control_copies) == 1:
            functional = operation(control_copies[0])
            control = Control(control_copies[0])
        else:
            functional = operation(control_copies)
            control = [Control(control) for control in control_copies]

        if functional_name:
            _rename(functional, functional_name)

        return ReducedFunctional(
            functional, control)


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


def _rename(obj, name):
    if hasattr(obj, "rename"):
        obj.rename(name)


def _ad_sub(left, right):
    result = right._ad_copy()
    result._ad_imul(-1)
    result._ad_iadd(left)
    return result


def _intermediate_options(final_options):
    """
    Options set for the intermediate stages of a chain of ReducedFunctionals

    Takes all elements of the final_options except riesz_representation,
    which is set to prevent returning derivatives to the primal space.
    """
    return {
        'riesz_representation': None,
        **{k: v for k, v in final_options.items()
           if (k != 'riesz_representation')}
    }


class AllAtOnceReducedFunctional(ReducedFunctional):
    """ReducedFunctional for 4DVar data assimilation.

    Creates either the strong constraint or weak constraint system incrementally
    by logging observations through the initial forward model run.

    Warning: Weak constraint 4DVar not implemented yet.

    Parameters
    ----------

    control
        The :class:`EnsembleFunction` for the control x_{i} at the initial
        condition and at the end of each observation stage.

    background_iprod
        The inner product to calculate the background error functional
        from the background error :math:`x_{0} - x_{b}`. Can include the
        error covariance matrix. Only used on ensemble rank 0.

    background
        The background (prior) data for the initial condition :math:`x_{b}`.
        If not provided, the value of the first Function on the first ensemble
        of the EnsembleFunction will be used.

    observation_err
        Given a state :math:`x`, returns the observations error
        :math:`y_{0} - \\mathcal{H}_{0}(x)` where :math:`y_{0}` are the
        observations at the initial time and :math:`\\mathcal{H}_{0}` is
        the observation operator for the initial time. Only used on
        ensemble rank 0. Optional.

    observation_iprod
        The inner product to calculate the observation error functional
        from the observation error :math:`y_{0} - \\mathcal{H}_{0}(x)`.
        Can include the error covariance matrix. Must be provided if
        observation_err is provided. Only used on ensemble rank 0

    weak_constraint
        Whether to use the weak or strong constraint 4DVar formulation.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    """

    def __init__(self, control: Control,
                 background_iprod: Optional[Callable[[OverloadedType], AdjFloat]],
                 background: Optional[OverloadedType] = None,
                 observation_err: Optional[Callable[[OverloadedType], OverloadedType]] = None,
                 observation_iprod: Optional[Callable[[OverloadedType], AdjFloat]] = None,
                 weak_constraint: bool = True,
                 tape: Optional[Tape] = None,
                 _annotate_accumulation: bool = False):

        self.tape = get_working_tape() if tape is None else tape

        self.weak_constraint = weak_constraint
        self.initial_observations = observation_err is not None

        with stop_annotating():
            if background:
                self.background = background._ad_copy()
            else:
                self.background = control.control.subfunctions[0]._ad_copy()
            _rename(self.background, "Background")

        if self.weak_constraint:
            self._annotate_accumulation = _annotate_accumulation
            self._accumulation_started = False

            ensemble = control.ensemble
            self.ensemble = ensemble
            self.trank = ensemble.ensemble_comm.rank if ensemble else 0
            self.nchunks = ensemble.ensemble_comm.size if ensemble else 1

            self._cbuf = control.copy()
            _x = self._cbuf.subfunctions
            self._x = _x
            self._controls = tuple(Control(xi) for xi in _x)

            self.control = control
            self.controls = [control]

            # first control on rank 0 is initial conditions, not end of observation stage
            self.nlocal_stages = len(_x) - (1 if self.trank == 0 else 0)

            self.stages = []    # The record of each observation stage

            # first rank sets up functionals for background initial observations
            if self.trank == 0:

                # RF to recalculate error vector (x_0 - x_b)
                self.background_error = isolated_rf(
                    operation=lambda x0: _ad_sub(x0, self.background),
                    control=_x[0],
                    functional_name="bkg_err_vec",
                    control_name="Control_0_bkg_copy")

                # RF to recalculate inner product |x_0 - x_b|_B
                self.background_norm = isolated_rf(
                    operation=background_iprod,
                    control=self.background_error.functional,
                    control_name="bkg_err_vec_copy")

                if self.initial_observations:

                    # RF to recalculate error vector (H(x_0) - y_0)
                    self.initial_observation_error = isolated_rf(
                        operation=observation_err,
                        control=_x[0],
                        functional_name="obs_err_vec_0",
                        control_name="Control_0_obs_copy")

                    # RF to recalculate inner product |H(x_0) - y_0|_R
                    self.initial_observation_norm = isolated_rf(
                        operation=observation_iprod,
                        control=self.initial_observation_error.functional,
                        functional_name="obs_err_vec_0_copy")

            # create halo for previous state
            if self.ensemble and self.trank != 0:
                with stop_annotating():
                    self.xprev = _x[0]._ad_copy()
                self._control_prev = Control(self.xprev)

            # halo for the derivative from the next chunk
            if self.ensemble and self.trank != self.nchunks - 1:
                with stop_annotating():
                    self.xnext = _x[0]._ad_copy()

        else:
            self._annotate_accumulation = True
            self._accumulation_started = False

            # initial conditions guess to be updated
            self.controls = Enlist(control)

            self.tape = get_working_tape() if tape is None else tape

            # Strong constraint functional to be converted to ReducedFunctional later

            # penalty for straying from prior
            self._accumulate_functional(
                background_iprod(control.control - self.background))

            # penalty for not hitting observations at initial time
            if self.initial_observations:
                self._accumulate_functional(
                    observation_iprod(observation_err(control.control)))

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
        # hasattr calls getattr, so check self.__dir__ directly here to avoid recursion
        if self.weak_constraint or "_strong_reduced_functional" not in dir(self):
            raise AttributeError(f"'{type(self)}' object has no attribute '{attr}'")
        return getattr(self.strong_reduced_functional, attr)

    @sc_passthrough
    @stop_annotating()
    def __call__(self, values: OverloadedType):
        """Computes the reduced functional with supplied control value.

        Parameters
        ----------

        values
            If you have multiple controls this should be a list of new values
            for each control in the order you listed the controls to the constructor.
            If you have a single control it can either be a list or a single object.
            Each new value should have the same type as the corresponding control.

        Returns
        -------
        pyadjoint.OverloadedType
            The computed value. Typically of instance of :class:`pyadjoint.AdjFloat`.

        """
        value = values[0] if isinstance(values, list) else values

        if not isinstance(value, type(self.control.control)):
            raise ValueError(f"Value must be of type {type(self.control.control)} not type {type(value)}")

        self.control.update(value)
        self._cbuf.assign(value)

        trank = self.trank

        # first "control" for later ranks is the halo
        if self.ensemble and trank != 0:
            x = [self.xprev, *self._x]
        else:
            x = [*self._x]

        # post messages for control of forward model propogation on next chunk
        if self.ensemble:
            src = trank - 1
            dst = trank + 1

            if trank != self.nchunks - 1:
                self.ensemble.isend(
                    x[-1], dest=dst, tag=dst)

            if trank != 0:
                recv_reqs = self.ensemble.irecv(
                    self.xprev, source=src, tag=trank)

        # Initial condition functionals
        if trank == 0:
            Jlocal = (
                self.background_norm(
                    self.background_error(x[0])))

            # observations at time 0
            if self.initial_observations:
                Jlocal += (
                    self.initial_observation_norm(
                        self.initial_observation_error(x[0])))
        else:
            Jlocal = 0.

        # evaluate all stages on chunk except first
        for i in range(1, len(self.stages)):
            Jlocal += self.stages[i](x[i:i+2])

        # wait for halo swap to finish
        if trank != 0:
            MPI.Request.Waitall(recv_reqs)

        # evaluate first stage model on chunk now we have data
        Jlocal += self.stages[0](x[0:2])

        # sum all stages
        if self.ensemble:
            J = self.ensemble.ensemble_comm.allreduce(Jlocal)
        else:
            J = Jlocal

        return J

    @sc_passthrough
    @stop_annotating()
    def derivative(self, adj_input: float = 1.0, options: dict = {}):
        """Returns the derivative of the functional w.r.t. the control.
        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the
        control, is computed and returned.

        Parameters
        ----------
        adj_input
            The adjoint input.

        options
            Additional options for the derivative computation.

        Returns
        -------
        pyadjoint.OverloadedType
            The derivative with respect to the control.
            Should be an instance of the same type as the control.
        """
        trank = self.trank

        # chaining ReducedFunctionals means we need to pass Cofunctions not Functions
        options = options or {}
        intermediate_options = _intermediate_options(options)

        # evaluate first forward model, which contributes to previous chunk
        sderiv0 = self.stages[0].derivative(
            adj_input=adj_input, options=options)

        # create the derivative in the right primal or dual space
        from ufl.duals import is_primal, is_dual
        if is_primal(sderiv0[0]):
            from firedrake.ensemblefunction import EnsembleFunction
            derivatives = EnsembleFunction(
                self.ensemble, self.control.local_function_spaces)
        else:
            if not is_dual(sderiv0[0]):
                raise ValueError(
                    "Do not know how to handle stage derivative which is not primal or dual")
            from firedrake.ensemblefunction import EnsembleCofunction
            derivatives = EnsembleCofunction(
                self.ensemble, [V.dual() for V in self.control.local_function_spaces])

        derivatives.zero()

        if self.ensemble:
            with stop_annotating():
                xprev = derivatives.subfunctions[0]._ad_copy()
                xnext = derivatives.subfunctions[0]._ad_copy()
                xprev.zero()
                xnext.zero()
            if trank != 0:
                derivs = [xprev, *derivatives.subfunctions]
            else:
                derivs = [*derivatives.subfunctions]

        # start accumulating the complete derivative
        derivs[0] += sderiv0[0]
        derivs[1] += sderiv0[1]

        # post the derivative halo exchange
        if self.ensemble:
            # halos sent backward in time
            src = trank + 1
            dst = trank - 1

            if trank != 0:
                self.ensemble.isend(
                    derivs[0], dest=dst, tag=dst)

            if trank != self.nchunks - 1:
                recv_reqs = self.ensemble.irecv(
                    xnext, source=src, tag=trank)

        # initial condition derivatives
        if trank == 0:
            bkg_deriv = self.background_norm.derivative(
                adj_input=adj_input, options=intermediate_options)

            derivs[0] += self.background_error.derivative(
                adj_input=bkg_deriv, options=options)

            # observations at time 0
            if self.initial_observations:
                obs_deriv = self.initial_observation_norm.derivative(
                    adj_input=adj_input, options=intermediate_options)

                derivs[0] += self.initial_observation_error.derivative(
                    adj_input=obs_deriv, options=options)

        # # evaluate all forward models on chunk except first while halo in flight
        for i in range(1, len(self.stages)):
            sderiv = self.stages[i].derivative(
                adj_input=adj_input, options=options)

            derivs[i] += sderiv[0]
            derivs[i+1] += sderiv[1]

        # finish the derivative halo exchange
        if self.ensemble:
            if trank != self.nchunks - 1:
                MPI.Request.Waitall(recv_reqs)
                derivs[-1] += xnext

        return derivatives

    @sc_passthrough
    @stop_annotating()
    def hessian(self, m_dot: OverloadedType, options: dict = {}):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        options
            A dictionary of options. To find a list of available options
            have a look at the specific control type.

        rtype:
            Whether to evaluate:
                - the model error ('model'),
                - the observation error ('obs'),
                - both model and observation errors (None).

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.
        """
        raise ValueError("Not implemented yet")

    @stop_annotating()
    def hessian_matrix(self):
        # Other reduced functionals don't have this.
        if not self.weak_constraint:
            raise AttributeError("Strong constraint 4DVar does not form a Hessian matrix")
        raise NotImplementedError

    def _accumulate_functional(self, val):
        if not self._annotate_accumulation:
            return
        if self._accumulation_started:
            self._total_functional += val
        else:
            self._total_functional = val
            self._accumulation_started = True

    @contextmanager
    def recording_stages(self, sequential=True, **stage_kwargs):
        if not sequential:
            raise ValueError("Recording stages concurrently not yet implemented")

        # record over ensemble
        if self.weak_constraint:

            trank = self.trank

            # index of "previous" stage and observation in global series
            global_index = -1
            observation_index = 0 if self.initial_observations else -1
            with stop_annotating():
                xhalo = self._x[0]._ad_copy()

            # add our data onto the user's context data
            ekwargs = {k: v for k, v in stage_kwargs.items()}
            ekwargs['global_index'] = global_index
            ekwargs['observation_index'] = observation_index

            ekwargs['xhalo'] = xhalo

            # proceed one ensemble rank at a time
            with self.ensemble.sequential(**ekwargs) as ectx:

                # later ranks start from halo
                if trank == 0:
                    controls = self._controls
                else:
                    controls = [self._control_prev, *self._controls]
                    with stop_annotating():
                        controls[0].assign(ectx.xhalo)

                # grab the user's data from the ensemble context
                local_stage_kwargs = {
                    k: getattr(ectx, k) for k in stage_kwargs.keys()
                }

                # initialise iterator for local stages
                stage_sequence = ObservationStageSequence(
                    controls, self, ectx.global_index,
                    ectx.observation_index,
                    local_stage_kwargs, sequential)

                # let the user record the local stages
                yield stage_sequence

                # send the state forward
                with stop_annotating():
                    state = self.stages[-1].controls[1].control
                    ectx.xhalo.assign(state)
                    # grab the user's information to send forward
                    for k in local_stage_kwargs.keys():
                        setattr(ectx, k, getattr(stage_sequence.ctx, k))
                    # increment the global indices for the last local stage
                    ectx.global_index = self.stages[-1].global_index
                    ectx.observation_index = self.stages[-1].observation_index

        else:  # strong constraint

            yield ObservationStageSequence(
                self.controls, self, stage_kwargs, sequential=True)


class ObservationStageSequence:
    def __init__(self, controls: Control,
                 aaorf: AllAtOnceReducedFunctional,
                 global_index: int,
                 observation_index: int,
                 stage_kwargs: dict = None,
                 sequential: bool = True):
        self.controls = controls
        self.nstages = len(controls) - 1
        self.aaorf = aaorf
        self.ctx = StageContext(**(stage_kwargs or {}))
        self.weak_constraint = aaorf.weak_constraint
        self.global_index = global_index
        self.observation_index = observation_index
        self.local_index = -1

    def __iter__(self):
        return self

    def __next__(self):

        if self.weak_constraint:
            stages = self.aaorf.stages

            # increment global indices
            self.local_index += 1
            self.global_index += 1
            self.observation_index += 1

            # start of the next stage
            next_control = self.controls[self.local_index]

            # smuggle state forward and increment stage indices
            if self.local_index > 0:
                state = stages[-1].controls[1].control
                with stop_annotating():
                    next_control.control.assign(state)

            # stop after we've recorded all stages
            if self.local_index >= self.nstages:
                raise StopIteration

            stage = WeakObservationStage(next_control,
                                         local_index=self.local_index,
                                         global_index=self.global_index,
                                         observation_index=self.observation_index)
            stages.append(stage)

        else:  # strong constraint

            # increment stage indices
            self.local_index += 1
            self.global_index += 1
            self.observation_index += 1

            # stop after we've recorded all stages
            if self.index >= self.nstages:
                raise StopIteration
            self.index += 1

            # dummy control to "start" stage from
            control = (self.aaorf.controls[0].control if self.index == 0
                       else self._prev_stage.state)

            stage = StrongObservationStage(control, self.aaorf)
            self._prev_stage = stage

        return stage, self.ctx


class StageContext:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class StrongObservationStage:
    """
    Record an observation for strong constraint 4DVar at the time of `state`.

    Parameters
    ----------

    aaorf
        The strong constraint AllAtOnceReducedFunctional.

    """

    def __init__(self, control: OverloadedType,
                 aaorf: AllAtOnceReducedFunctional):
        self.aaorf = aaorf
        self.control = control

    def set_observation(self, state: OverloadedType,
                        observation_err: Callable[[OverloadedType], OverloadedType],
                        observation_iprod: Callable[[OverloadedType], AdjFloat]):
        """
        Record an observation at the time of `state`.

        Parameters
        ----------

        state
            The state at the current observation time.

        observation_err
            Given a state :math:`x`, returns the observations error
            :math:`y_{i} - \\mathcal{H}_{i}(x)` where :math:`y_{i}` are
            the observations at the current observation time and
            :math:`\\mathcal{H}_{i}` is the observation operator for the
            current observation time.

        observation_iprod
            The inner product to calculate the observation error functional
            from the observation error :math:`y_{i} - \\mathcal{H}_{i}(x)`.
            Can include the error covariance matrix.
        """
        if hasattr(self.aaorf, "_strong_reduced_functional"):
            raise ValueError("Cannot add observations once strong"
                             " constraint ReducedFunctional instantiated")
        self.aaorf._accumulate_functional(
            observation_iprod(observation_err(state)))
        self.state = state


class WeakObservationStage:
    """
    A single stage for weak constraint 4DVar at the time of `state`.

    Records the forward model propogation from the control at the beginning
    of the stage, and the model and observation errors at the end of the stage.

    Parameters
    ----------

    control
        The control x_{i-1} at the beginning of the stage

    local_index
        The index of this stage in the timeseries on the
        local ensemble member.

    global_index
        The index of this stage in the global timeseries.

    observation_index
        The index of the observation at the end of this stage in
        the global timeseries. May be different from global_index if
        an observation is taken at the initial time.

    """
    def __init__(self, control: Control,
                 local_index: Optional[int] = None,
                 global_index: Optional[int] = None,
                 observation_index: Optional[int] = None):
        # "control" to use as initial condition.
        # Not actual `Control` for consistency with strong constraint
        self.control = control.control

        self.controls = Enlist(control)
        self.local_index = local_index
        self.global_index = global_index
        self.observation_index = observation_index
        set_working_tape()
        self._stage_tape = get_working_tape()

    def set_observation(self, state: OverloadedType,
                        observation_err: Callable[[OverloadedType], OverloadedType],
                        observation_iprod: Callable[[OverloadedType], AdjFloat],
                        forward_model_iprod: Callable[[OverloadedType], AdjFloat]):
        """
        Record an observation at the time of `state`.

        Parameters
        ----------

        state
            The state at the current observation time.

        observation_err
            Given a state :math:`x`, returns the observations error
            :math:`y_{i} - \\mathcal{H}_{i}(x)` where :math:`y_{i}` are
            the observations at the current observation time and
            :math:`\\mathcal{H}_{i}` is the observation operator for the
            current observation time.

        observation_iprod
            The inner product to calculate the observation error functional
            from the observation error :math:`y_{i} - \\mathcal{H}_{i}(x)`.
            Can include the error covariance matrix.

        forward_model_iprod
            The inner product to calculate the model error functional from
            the model error :math:`x_{i} - \\mathcal{M}_{i}(x_{i-1})`. Can
            include the error covariance matrix.
        """
        # get the tape used for this stage and make sure its the right one
        stage_tape = get_working_tape()
        if stage_tape is not self._stage_tape:
            raise ValueError(
                "Working tape at the end of the observation stage"
                " differs from the tape at the stage beginning."
            )

        # record forward propogation
        with set_working_tape(stage_tape.copy()) as tape:
            self.forward_model = ReducedFunctional(
                state._ad_copy(), controls=self.controls[0], tape=tape)

        # Beginning of next time-chunk is the control for this observation
        # and the state at the end of the next time-chunk.
        with stop_annotating():
            # smuggle initial guess at this time into the control without the tape seeing
            self.controls.append(Control(state._ad_copy()))
            if self.global_index:
                _rename(self.controls[-1].control, f"Control_{self.global_index}")

        # model error links time-chunks by depending on both the
        # previous and current controls

        # RF to recalculate error vector (M_i - x_i)
        names = {
            'functional_name': f"model_err_vec_{self.global_index}",
            'control_name': [f"state_{self.global_index}_copy",
                             f"Control_{self.global_index}_model_copy"]
        } if self.global_index else {}

        self.model_error = isolated_rf(
            operation=lambda controls: _ad_sub(*controls),
            control=[state, self.controls[-1].control],
            **names)

        # RF to recalculate inner product |M_i - x_i|_Q
        names = {
            'control_name': f"model_err_vec_{self.global_index}_copy"
        } if self.global_index else {}

        self.model_norm = isolated_rf(
            operation=forward_model_iprod,
            control=self.model_error.functional,
            **names)

        # Observations after tape cut because this is now a control, not a state

        # RF to recalculate error vector (H(x_i) - y_i)
        names = {
            'functional_name': f"obs_err_vec_{self.global_index}",
            'control_name': f"Control_{self.global_index}_obs_copy"
        } if self.global_index else {}

        self.observation_error = isolated_rf(
            operation=observation_err,
            control=self.controls[-1],
            **names)

        # RF to recalculate inner product |H(x_i) - y_i|_R
        names = {
            'functional_name': "obs_err_vec_{self.global_index}_copy"
        } if self.global_index else {}
        self.observation_norm = isolated_rf(
            operation=observation_iprod,
            control=self.observation_error.functional,
            **names)

        # remove the stage initial condition "control" now we've finished recording
        delattr(self, "control")

        # stop the stage tape recording anything else
        set_working_tape()

    @stop_annotating()
    def __call__(self, values: OverloadedType,
                 rftype: Optional[str] = None):
        """Computes the reduced functional with supplied control value.

        Parameters
        ----------

        values
            If you have multiple controls this should be a list of new values
            for each control in the order you listed the controls to the constructor.
            If you have a single control it can either be a list or a single object.
            Each new value should have the same type as the corresponding control.

        rtype:
            Whether to evaluate:
                - the model error ('model'),
                - the observation error ('obs'),
                - both model and observation errors (None).

        Returns
        -------
        pyadjoint.OverloadedType
            The computed value. Typically of instance of :class:`pyadjoint.AdjFloat`.

        """
        J = 0.0

        # evaluate model error
        if (rftype is None) or (rftype == 'model'):
            Mi = self.forward_model(values[0])
            J += self.model_norm(self.model_error([Mi, values[1]]))

        # evaluate observation errors
        if (rftype is None) or (rftype == 'obs'):
            J += self.observation_norm(self.observation_error(values[1]))

        return J

    @stop_annotating()
    def derivative(self, adj_input: float = 1.0, options: dict = {},
                   rftype: Optional[str] = None):
        """Returns the derivative of the functional w.r.t. the control.
        Using the adjoint method, the derivative of the functional with
        respect to the control, around the last supplied value of the
        control, is computed and returned.

        Parameters
        ----------
        adj_input
            The adjoint input.

        options
            Additional options for the derivative computation.

        rtype:
            Whether to evaluate:
                - the model error ('model'),
                - the observation error ('obs'),
                - both model and observation errors (None).

        Returns
        -------
        pyadjoint.OverloadedType
            The derivative with respect to the control.
            Should be an instance of the same type as the control.
        """
        # create a list of overloaded types to put derivative into
        derivatives = []

        # chaining ReducedFunctionals means we need to pass Cofunctions not Functions
        options = options or {}
        intermediate_options = _intermediate_options(options)

        if (rftype is None) or (rftype == 'model'):
            # derivative of reduction
            dm_norm = self.model_norm.derivative(adj_input=adj_input,
                                                 options=intermediate_options)

            # derivative of difference splits into (Mi, xi)
            dm_errors = self.model_error.derivative(adj_input=dm_norm,
                                                    options=intermediate_options)

            # derivative through the forward model wrt to xprev
            dm_forward = self.forward_model.derivative(adj_input=dm_errors[0],
                                                       options=options)

            derivatives.append(dm_forward)

            # dm_errors is still in the dual space, so we need to convert it to the
            # type that the user has requested - this will be the type of dm_forward.
            derivatives.append(dm_forward._ad_convert_type(dm_errors[1], options))

        if (rftype is None) or (rftype == 'obs'):
            # derivative of reduction
            do_norm = self.observation_norm.derivative(adj_input=adj_input,
                                                       options=intermediate_options)
            # derivative of error
            do_error = self.observation_error.derivative(adj_input=do_norm,
                                                         options=options)

            if len(derivatives) == 0:
                derivatives.append(None)
                derivatives.append(do_error)
            else:
                derivatives[1] += do_error

        return derivatives

    @stop_annotating()
    def hessian(self, m_dot: OverloadedType, options: dict = {},
                rftype: Optional[str] = None):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        options
            A dictionary of options. To find a list of available options
            have a look at the specific control type.

        rtype:
            Whether to evaluate:
                - the model error ('model'),
                - the observation error ('obs'),
                - both model and observation errors (None).

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.
        """
        pass
