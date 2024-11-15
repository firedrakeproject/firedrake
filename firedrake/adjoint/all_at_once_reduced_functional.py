from pyadjoint import ReducedFunctional, OverloadedType, Control, Tape, AdjFloat, \
    stop_annotating, no_annotations, get_working_tape, set_working_tape
from pyadjoint.enlisting import Enlist
from firedrake import Ensemble
from functools import wraps, cached_property
from typing import Callable, Optional
from contextlib import contextmanager
from mpi4py import MPI

__all__ = ['AllAtOnceReducedFunctional']


@set_working_tape(decorator=True)
def isolated_rf(operation, control,
                functional_name=None,
                control_name=None):
    """
    Return a ReducedFunctional where the functional is `operation` applied
    to a copy of `control`, and the tape contains only `operation`.
    """
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


def _scalarSend(comm, x, **kwargs):
    from numpy import ones
    comm.Send(x*ones(1, dtype=type(x)), **kwargs)


def _scalarRecv(comm, dtype=float, **kwargs):
    from numpy import zeros
    xtmp = zeros(1, dtype=dtype)
    comm.Recv(xtmp, **kwargs)
    return xtmp[0]


class AllAtOnceReducedFunctional(ReducedFunctional):
    """ReducedFunctional for 4DVar data assimilation.

    Creates either the strong constraint or weak constraint system incrementally
    by logging observations through the initial forward model run.

    Warning: Weak constraint 4DVar not implemented yet.

    Parameters
    ----------

    control
        The initial condition :math:`x_{0}`. Starting value is used as the
        background (prior) data :math:`x_{b}`.

    nlocal_stages
        The number of observation stages on the local ensemble member.

    background_iprod
        The inner product to calculate the background error functional
        from the background error :math:`x_{0} - x_{b}`. Can include the
        error covariance matrix.

    observation_err
        Given a state :math:`x`, returns the observations error
        :math:`y_{0} - \\mathcal{H}_{0}(x)` where :math:`y_{0}` are the
        observations at the initial time and :math:`\\mathcal{H}_{0}` is
        the observation operator for the initial time. Optional.

    observation_iprod
        The inner product to calculate the observation error functional
        from the observation error :math:`y_{0} - \\mathcal{H}_{0}(x)`.
        Can include the error covariance matrix. Must be provided if
        observation_err is provided.

    weak_constraint
        Whether to use the weak or strong constraint 4DVar formulation.

    ensemble
        The ensemble communicator to parallelise over. None for no time parallelism.
        If `ensemble` is provided, then `background_iprod`, `observation_err` and
        `observation_iprod` must only be provided on ensemble rank 0.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    """

    def __init__(self, control: Control,
                 nlocal_stages: int,
                 background_iprod: Optional[Callable[[OverloadedType], AdjFloat]],
                 observation_err: Optional[Callable[[OverloadedType], OverloadedType]] = None,
                 observation_iprod: Optional[Callable[[OverloadedType], AdjFloat]] = None,
                 weak_constraint: bool = True,
                 tape: Optional[Tape] = None,
                 _annotate_accumulation: bool = False,
                 ensemble: Optional[Ensemble] = None):

        self.tape = get_working_tape() if tape is None else tape

        self.weak_constraint = weak_constraint
        self.initial_observations = observation_err is not None

        # We need a copy for the prior, but this shouldn't be part of the tape
        with stop_annotating():
            self.background = control.copy_data()

        if self.weak_constraint:
            self._annotate_accumulation = _annotate_accumulation
            self._accumulation_started = False

            self.nlocal_stages = nlocal_stages

            self.ensemble = ensemble
            self.trank = ensemble.ensemble_comm.rank if ensemble else 0
            self.nchunks = ensemble.ensemble_comm.size if ensemble else 1

            self.stages = []    # The record of each observation stage
            self.controls = []  # The solution at the beginning of each time-chunk

            # first rank sets up functionals for background initial observations
            if self.trank == 0:
                self.controls.append(control)

                # RF to recalculate error vector (x_0 - x_b)
                self.background_error = isolated_rf(
                    operation=lambda x0: _ad_sub(x0, self.background),
                    control=control,
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
                        control=control,
                        functional_name="obs_err_vec_0",
                        control_name="Control_0_obs_copy")

                    # RF to recalculate inner product |H(x_0) - y_0|_R
                    self.initial_observation_norm = isolated_rf(
                        operation=observation_iprod,
                        control=self.initial_observation_error.functional,
                        functional_name="obs_err_vec_0_copy")

            else:
                # create halo for previous state
                with stop_annotating():
                    self.xprev = control.copy_data()
                self.control_prev = Control(self.xprev)

                if background_iprod is not None:
                    raise ValueError("Only the first ensemble rank needs `background_iprod`")
                if observation_iprod is not None:
                    raise ValueError("Only the first ensemble rank needs `observation_iprod`")
                if observation_err is not None:
                    raise ValueError("Only the first ensemble rank needs `observation_err`")

            # create all controls on local ensemble member
            with stop_annotating():
                for _ in range(nlocal_stages):
                    self.controls.append(Control(control.copy_data()))

            # halo for the derivative from the next chunk
            if self.ensemble and self.trank != self.nchunks - 1:
                self.xnext = control.copy_data()

            # new tape for the initial stage
            if self.trank == 0:
                self.stages.append(
                    WeakObservationStage(self.controls[0], index=0))
            else:
                self._stage_tape = None

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
    @no_annotations
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
        for c, v in zip(self.controls, values):
            c.control.assign(v)

        # post messages for control of forward model propogation on next chunk
        trank = self.trank
        if self.ensemble:
            src = trank - 1
            dst = trank + 1

            if trank != self.nchunks - 1:
                self.ensemble.isend(
                    self.controls[-1].control, dest=dst, tag=dst)

            if trank != 0:
                recv_reqs = self.ensemble.irecv(
                    self.xprev, source=src, tag=trank)

        # first "control" is the halo
        if self.ensemble and trank != 0:
            values = [self.xprev, *values]

        # Initial condition functionals
        if trank == 0:
            Jlocal = (
                self.background_norm(
                    self.background_error(values[0]))
            )

            # observations at time 0
            if self.initial_observations:
                Jlocal += (
                    self.initial_observation_norm(
                        self.initial_observation_error(values[0]))
                )
        else:
            Jlocal = 0.

        # evaluate all stages on chunk except first
        for i in range(1, len(self.stages)):
            Jlocal += self.stages[i](values[i:i+2])

        # wait for halo swap to finish
        if trank != 0:
            MPI.Request.Waitall(recv_reqs)

        # evaluate first stage model on chunk now we have data
        Jlocal += self.stages[0](values[0:2])

        # sum all stages
        if self.ensemble:
            J = self.ensemble.ensemble_comm.allreduce(Jlocal)
        else:
            J = Jlocal

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
        # create a list of overloaded types to put derivative into
        derivatives = []

        # chaining ReducedFunctionals means we need to pass Cofunctions not Functions
        options = options or {}
        intermediate_options = {
            'riesz_representation': None,
            **{k: v for k, v in options.items()
               if (k != 'riesz_representation')}
        }

        # initial condition derivatives
        if self.trank == 0:
            bkg_deriv = self.background_norm.derivative(adj_input=adj_input,
                                                        options=intermediate_options)
            derivatives.append(self.background_error.derivative(adj_input=bkg_deriv,
                                                                options=options))

            # observations at time 0
            if self.initial_observations:
                obs_deriv = self.initial_observation_norm.derivative(adj_input=adj_input,
                                                                     options=intermediate_options)
                derivatives[0] += self.initial_observation_error.derivative(adj_input=obs_deriv,
                                                                            options=options)

        # evaluate first forward model, which contributes to previous chunk
        derivs = self.stages[0].derivative(adj_input=adj_input, options=options)

        if self.trank == 0:
            derivatives[0] += derivs[0]
        else:
            derivatives.append(derivs[0])
        derivatives.append(derivs[1])

        # post the derivative halo exchange
        if self.ensemble:
            src = self.trank + 1
            dst = self.trank - 1

            if self.trank != 0:
                self.ensemble.isend(
                    derivatives[0], dest=dst, tag=dst)

            if self.trank != self.nchunks - 1:
                recv_reqs = self.ensemble.irecv(
                    self.xnext, source=src, tag=self.trank)

        # # evaluate all forward models on chunk except first while halo in flight
        for i in range(1, len(self.stages)):
            derivs = self.stages[i].derivative(adj_input=adj_input, options=options)
            derivatives[i] += derivs[0]
            derivatives.append(derivs[1])

        # finish the derivative halo exchange
        if self.ensemble:
            if self.trank != self.nchunks - 1:
                MPI.Request.Waitall(recv_reqs)
                derivatives[-1] += self.xnext

            # we don't own the control for the halo, so remove it from the
            # list of local derivatives once the communication has finished
            if self.trank != 0:
                derivatives.pop(0)

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

    @no_annotations
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
    def recording_stages(self, sequential=True, **kwargs):
        if not sequential:
            raise ValueError("Recording stages concurrently not yet implemented")

        # indices of stage in global and local list
        stage_kwargs = {k: v for k, v in kwargs.items()}
        stage_kwargs['local_index'] = 0
        stage_kwargs['global_index'] = 0

        # record over ensemble
        if self.weak_constraint:

            trank = self.trank

            # later ranks recv forward state and kwargs
            if trank > 0:
                tcomm = self.ensemble.ensemble_comm
                src = trank-1
                with stop_annotating():
                    self.ensemble.recv(self.xprev, source=src, tag=trank+000)

                    for i, (k, v) in enumerate(stage_kwargs.items()):
                        stage_kwargs[k] = _scalarRecv(
                            tcomm, dtype=type(v), source=src, tag=trank+i*100)
                    # restart local stage counter
                    stage_kwargs['local_index'] = 0

            # subsequent ranks start from halo
            controls = self.controls if trank == 0 else [self.control_prev, *self.controls]

            stage_sequence = ObservationStageSequence(
                controls, self, stage_kwargs, sequential, weak_constraint=True)

            yield stage_sequence

            # grab the stages now they have been taped
            self.stages = stage_sequence.stages

            # send forward state and kwargs
            if self.ensemble and trank != self.nchunks - 1:
                with stop_annotating():
                    tcomm = self.ensemble.ensemble_comm
                    dst = trank+1

                    state = self.stages[-1].controls[1].control
                    self.ensemble.send(state, dest=dst, tag=dst+000)

                    for i, k in enumerate(stage_kwargs.keys()):
                        v = getattr(stage_sequence.ctx, k)
                        _scalarSend(
                            tcomm, v, dest=dst, tag=dst+i*100)

        else:  # strong constraint

            yield ObservationStageSequence(
                self.controls, self, stage_kwargs,
                sequential=True, weak_constraint=False)


class ObservationStageSequence:
    def __init__(self, controls: Control,
                 aaorf: AllAtOnceReducedFunctional,
                 stage_kwargs: dict = None,
                 sequential: bool = True,
                 weak_constraint: bool = True):
        self.controls = controls
        self.nstages = len(controls) - 1
        self.aaorf = aaorf
        self.ctx = StageContext(**(stage_kwargs or {}))
        self.index = 0
        self.weak_constraint = weak_constraint
        if weak_constraint:
            self.stages = []

    def __iter__(self):
        return self

    def __next__(self):

        if self.weak_constraint:
            # start of the next stage
            next_control = self.controls[self.index]

            # smuggle state forward and increment stage indices
            if self.index > 0:
                self.ctx.local_index += 1
                self.ctx.global_index += 1

                state = self.stages[-1].controls[1].control
                with stop_annotating():
                    next_control.control.assign(state)

            # stop after we've recorded all stages
            if self.index >= self.nstages:
                raise StopIteration
            self.index += 1

            stage = WeakObservationStage(next_control, index=self.ctx.global_index)
            self.stages.append(stage)

        else:  # strong constraint

            # increment stage indices
            if self.index > 0:
                self.ctx.local_index += 1
                self.ctx.global_index += 1

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

    index
        Optional integer to name controls and functionals with

    """
    def __init__(self, control: Control,
                 index: Optional[int] = None):
        # "control" to use as initial condition.
        # Not actual `Control` for consistency with strong constraint
        self.control = control.control

        self.controls = Enlist(control)
        self.index = index
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
            if self.index:
                _rename(self.controls[-1].control, f"Control_{self.index}")

        # model error links time-chunks by depending on both the
        # previous and current controls

        # RF to recalculate error vector (M_i - x_i)
        names = {
            'functional_name': f"model_err_vec_{self.index}",
            'control_name': [f"state_{self.index}_copy",
                             f"Control_{self.index}_model_copy"]
        } if self.index else {}

        self.model_error = isolated_rf(
            operation=lambda controls: _ad_sub(*controls),
            control=[state, self.controls[-1].control],
            **names)

        # RF to recalculate inner product |M_i - x_i|_Q
        names = {
            'control_name': f"model_err_vec_{self.index}_copy"
        } if self.index else {}

        self.model_norm = isolated_rf(
            operation=forward_model_iprod,
            control=self.model_error.functional,
            **names)

        # Observations after tape cut because this is now a control, not a state

        # RF to recalculate error vector (H(x_i) - y_i)
        names = {
            'functional_name': f"obs_err_vec_{self.index}",
            'control_name': f"Control_{self.index}_obs_copy"
        } if self.index else {}

        self.observation_error = isolated_rf(
            operation=observation_err,
            control=self.controls[-1],
            **names)

        # RF to recalculate inner product |H(x_i) - y_i|_R
        names = {
            'functional_name': "obs_err_vec_{self.index}_copy"
        } if self.index else {}
        self.observation_norm = isolated_rf(
            operation=observation_iprod,
            control=self.observation_error.functional,
            **names)

        # remove the stage initial condition "control" now we've finished recording
        delattr(self, "control")

    @no_annotations
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

    @no_annotations
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
        intermediate_options = {
            'riesz_representation': None,
            **{k: v for k, v in options.items()
               if (k != 'riesz_representation')}
        }

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
            derivatives.append(dm_errors[1].riesz_representation())

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

    @no_annotations
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
