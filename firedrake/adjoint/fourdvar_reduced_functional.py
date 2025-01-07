from pyadjoint import ReducedFunctional, OverloadedType, Control, Tape, AdjFloat, \
    stop_annotating, get_working_tape, set_working_tape
from pyadjoint.enlisting import Enlist
from firedrake.function import Function
from firedrake.ensemblefunction import EnsembleFunction, EnsembleCofunction
from firedrake.adjoint.composite_reduced_functional import (
    CompositeReducedFunctional, tlm, hessian, intermediate_options)

from functools import wraps, cached_property
from typing import Callable, Optional
from types import SimpleNamespace
from contextlib import contextmanager
from mpi4py import MPI

__all__ = ['FourDVarReducedFunctional']


# @set_working_tape()  # ends up using old_tape = None because evaluates when imported - need separate decorator
def isolated_rf(operation, control,
                functional_name=None,
                control_name=None):
    """
    Return a ReducedFunctional where the functional is `operation` applied
    to a copy of `control`, and the tape contains only `operation`.
    """
    with stop_annotating():
        controls = Enlist(control)
        control_copies = [control._ad_copy() for control in controls]

        if control_name:
            for control, name in zip(control_copies, Enlist(control_name)):
                _rename(control, name)

    with set_working_tape():
        functional = operation(controls.delist(control_copies))

        if functional_name:
            _rename(functional, functional_name)

        control = controls.delist([Control(control_copy)
                                   for control_copy in control_copies])

        return ReducedFunctional(
            functional, control)


def sc_passthrough(func):
    """
    Wraps standard ReducedFunctional methods to differentiate strong or
    weak constraint implementations.

    If using strong constraint, makes sure strong_reduced_functional
    is instantiated then passes args/kwargs through to the
    corresponding strong_reduced_functional method.

    If using weak constraint, returns the FourDVarReducedFunctional
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


class FourDVarReducedFunctional(ReducedFunctional):
    """ReducedFunctional for 4DVar data assimilation.

    Creates either the strong constraint or weak constraint system
    by logging observations through the initial time propagator run.

    Parameters
    ----------

    control
        The :class:`.EnsembleFunction` for the control x_{i} at the initial condition
        and at the end of each observation stage.

    background_iprod
        The inner product to calculate the background error functional
        from the background error :math:`x_{0} - x_{b}`. Can include the
        error covariance matrix. Only used on ensemble rank 0.

    background
        The background (prior) data for the initial condition :math:`x_{b}`.
        If not provided, the value of the first subfunction on the first ensemble
        member of the control :class:`.EnsembleFunction` will be used.

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

        if self.weak_constraint:
            self._annotate_accumulation = _annotate_accumulation
            self._accumulation_started = False

            if not isinstance(control.control, EnsembleFunction):
                raise TypeError(
                    "Control for weak constraint 4DVar must be an EnsembleFunction"
                )

            with stop_annotating():
                if background:
                    self.background = background._ad_copy()
                else:
                    self.background = control.control.subfunctions[0]._ad_copy()
                _rename(self.background, "Background")

            ensemble = control.ensemble
            self.ensemble = ensemble
            self.trank = ensemble.ensemble_comm.rank if ensemble else 0
            self.nchunks = ensemble.ensemble_comm.size if ensemble else 1

            # because we need to manually evaluate the different bits
            # of the functional, we need an internal set of controls
            # to use for the stage ReducedFunctionals
            self._cbuf = control.copy_data()
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

                # compose background reduced functionals to evaluate both together
                self.background_rf = CompositeReducedFunctional(
                    self.background_error, self.background_norm)

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

                    # compose initial observation reduced functionals to evaluate both together
                    self.initial_observation_rf = CompositeReducedFunctional(
                        self.initial_observation_error, self.initial_observation_norm)

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

            if not isinstance(control.control, Function):
                raise TypeError(
                    "Control for strong constraint 4DVar must be a Function"
                )

            with stop_annotating():
                if background:
                    self.background = background._ad_copy()
                else:
                    self.background = control.control._ad_copy()
                _rename(self.background, "Background")

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

    @cached_property
    def strong_reduced_functional(self):
        """A :class:`pyadjoint.ReducedFunctional` for the strong constraint 4DVar system.

        Only instantiated if using the strong constraint formulation, and cannot be used
        before all observations are recorded.
        """
        if self.weak_constraint:
            msg = "Strong constraint ReducedFunctional cannot be instantiated for weak constraint 4DVar"
            raise AttributeError(msg)
        self._strong_reduced_functional = ReducedFunctional(
            self._total_functional, self.controls.delist(), tape=self.tape)
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
        # put the new value into our internal set of controls to pass to each stage
        self._cbuf.assign(value)

        trank = self.trank

        # first "control" for later ranks is the halo
        if self.ensemble and trank != 0:
            x = [self.xprev, *self._x]
        else:
            x = [*self._x]

        # post messages for control of time propagator on next chunk
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
            Jlocal = self.background_rf(x[0])

            # observations at time 0
            if self.initial_observations:
                Jlocal += self.initial_observation_rf(x[0])
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

        # evaluate first time propagator, which contributes to previous chunk
        sderiv0 = self.stages[0].derivative(
            adj_input=adj_input, options=options)

        # create the derivative in the right primal or dual space
        from ufl.duals import is_primal, is_dual
        if is_primal(sderiv0[0]):
            derivatives = EnsembleFunction(
                self.ensemble, self.control.local_function_spaces)
        else:
            if not is_dual(sderiv0[0]):
                raise ValueError(
                    "Do not know how to handle stage derivative which is not primal or dual")
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
            derivs[0] += self.background_rf.derivative(
                adj_input=adj_input, options=options)

            # observations at time 0
            if self.initial_observations:
                derivs[0] += self.initial_observation_rf.derivative(
                    adj_input=adj_input, options=options)

        # # evaluate all time stages on chunk except first while halo in flight
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
        trank = self.trank

        hess = self.control.copy_data()
        hess.zero()

        # set up arrays including halos
        if trank == 0:
            hs = [*hess.subfunctions]
            mdot = [*m_dot[0].subfunctions]
        else:
            hprev = hess.subfunctions[0].copy(deepcopy=True)
            mprev = m_dot[0].subfunctions[0].copy(deepcopy=True)
            hs = [hprev, *hess.subfunctions]
            mdot = [mprev, *m_dot[0].subfunctions]

        if trank != self.nchunks - 1:
            hnext = hess.subfunctions[0].copy(deepcopy=True)

        # send m_dot halo forward
        if self.ensemble:
            src = trank - 1
            dst = trank + 1

            if trank != self.nchunks - 1:
                self.ensemble.isend(
                    mdot[-1], dest=dst, tag=dst)

            if trank != 0:
                recv_reqs = self.ensemble.irecv(
                    mdot[0], source=src, tag=trank)

        # hessian actions at the initial condition
        if trank == 0:
            hs[0] += self.background_rf.hessian(
                mdot[0], options=options)

            if self.initial_observations:
                hs[0] += self.initial_observation_rf.hessian(
                    mdot[0], options=options)

        # evaluate all stages on chunk except first
        for i in range(1, len(self.stages)):
            hms = self.stages[i].hessian(
                mdot[i:i+2], options=options)

            hs[i] += hms[0]
            hs[i+1] += hms[1]

        # wait for halo swap to finish
        if trank != 0:
            MPI.Request.Waitall(recv_reqs)

        # evaluate first stage on chunk now we have the halo
        hms = self.stages[0].hessian(
            mdot[:2], options=options)

        hs[0] += hms[0]
        hs[1] += hms[1]

        # send result halo backward
        if self.ensemble:
            src = trank + 1
            dst = trank - 1

            if trank != 0:
                self.ensemble.isend(
                    hs[0], dest=dst, tag=dst)

            if trank != self.nchunks - 1:
                recv_reqs = self.ensemble.irecv(
                    hnext, source=src, tag=trank)

        # finish the result halo
        if trank != self.nchunks - 1:
            MPI.Request.Waitall(recv_reqs)
            hs[-1] += hnext

        return hess

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
    def recording_stages(self, sequential=True, nstages=None, **stage_kwargs):
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

                    # make sure that self.control now holds the
                    # values of the initial timeseris
                    self.control.assign(self._cbuf)

        else:  # strong constraint

            yield ObservationStageSequence(
                self.controls, self, global_index=-1,
                observation_index=0 if self.initial_observations else -1,
                stage_kwargs=stage_kwargs, nstages=nstages)


class ObservationStageSequence:
    def __init__(self, controls: Control,
                 aaorf: FourDVarReducedFunctional,
                 global_index: int,
                 observation_index: int,
                 stage_kwargs: dict = None,
                 nstages: Optional[int] = None):
        self.controls = controls
        self.aaorf = aaorf
        self.ctx = SimpleNamespace(**(stage_kwargs or {}))
        self.weak_constraint = aaorf.weak_constraint
        self.global_index = global_index
        self.observation_index = observation_index
        self.local_index = -1
        self.nstages = (len(controls) - 1 if self.weak_constraint
                        else nstages)

    def __iter__(self):
        return self

    def __next__(self):

        # increment global indices.
        self.local_index += 1
        self.global_index += 1
        self.observation_index += 1

        if self.weak_constraint:
            stages = self.aaorf.stages

            # control for the start of the next stage.
            next_control = self.controls[self.local_index]

            # smuggle state forward into aaorf's next control.
            if self.local_index > 0:
                state = stages[-1].controls[1].control
                with stop_annotating():
                    next_control.control.assign(state)

            # now we know that the aaorf's controls have
            # been updated from the previous stage's controls,
            # we can check if we need to exit.
            if self.local_index >= self.nstages:
                raise StopIteration

            stage = WeakObservationStage(next_control,
                                         local_index=self.local_index,
                                         global_index=self.global_index,
                                         observation_index=self.observation_index)
            stages.append(stage)

        else:  # strong constraint

            # stop after we've recorded all stages
            if self.local_index >= self.nstages:
                raise StopIteration

            # dummy control to "start" stage from
            control = (self.aaorf.controls[0].control if self.local_index == 0
                       else self._prev_stage.state)

            stage = StrongObservationStage(
                control, self.aaorf,
                index=self.local_index,
                observation_index=self.observation_index)

            self._prev_stage = stage

        return stage, self.ctx


class StrongObservationStage:
    """
    Record an observation for strong constraint 4DVar at the time of `state`.

    Parameters
    ----------

    aaorf
        The strong constraint FourDVarReducedFunctional.

    """

    def __init__(self, control: OverloadedType,
                 aaorf: FourDVarReducedFunctional,
                 index: Optional[int] = None,
                 observation_index: Optional[int] = None):
        self.aaorf = aaorf
        self.control = control
        self.index = index
        self.observation_index = observation_index

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
        # save the user's state to hand back for beginning of next stage
        self.state = state


class WeakObservationStage:
    """
    A single stage for weak constraint 4DVar at the time of `state`.

    Records the time propagator from the control at the beginning
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

        # compose model error reduced functionals to evaluate both together
        self.model_error_rf = CompositeReducedFunctional(
            self.model_error, self.model_norm)

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

        # compose observation reduced functionals to evaluate both together
        self.observation_rf = CompositeReducedFunctional(
            self.observation_error, self.observation_norm)

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
        if rftype in (None, 'model'):
            J += self.model_error_rf(
                [self.forward_model(values[0]), values[1]])

        # evaluate observation errors
        if rftype in (None, 'obs'):
            J += self.observation_rf(values[1])

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
        ioptions = intermediate_options(options)

        if rftype in (None, 'model'):
            # derivative of reduction and difference
            model_err_derivs = self.model_error_rf.derivative(
                adj_input=adj_input, options=ioptions)

            # derivative through the time propagator wrt to xprev
            model_forward_deriv = self.forward_model.derivative(
                adj_input=model_err_derivs[0], options=options)

            derivatives.append(model_forward_deriv)

            # model_err_derivs is still in the dual space, so we need to convert it to the
            # type that the user has requested - this will be the type of model_forward_deriv.
            derivatives.append(
                model_forward_deriv._ad_convert_type(
                    model_err_derivs[1], options))

        if rftype in (None, 'obs'):
            obs_deriv = self.observation_rf.derivative(
                adj_input=adj_input, options=options)

            if len(derivatives) == 0:
                derivatives.append(None)
                derivatives.append(obs_deriv)
            else:
                derivatives[1] += obs_deriv

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
        hessian_value = []

        if rftype in (None, 'model'):
            hessian_value.extend(self._model_hessian(
                m_dot, options=options))

        if rftype in (None, 'obs'):
            obs_hessian = self.observation_rf.hessian(
                m_dot[1], options=options)
            if len(hessian_value) == 0:
                hessian_value.append(None)
                hessian_value.append(obs_hessian)
            else:
                hessian_value[1] += obs_hessian

        return hessian_value

    def _model_hessian(self, m_dot, options):
        iopts = intermediate_options(options)

        # TLM for model from mdot[0]
        forward_tlm = tlm(self.forward_model, m_dot[0],
                          options=iopts)

        # combine model TLM and mdot[1]
        mdot_error = [forward_tlm, m_dot[1]]

        # Hessian (dual) for error
        error_hessian = self.model_error_rf.hessian(
            mdot_error, options=iopts, evaluate_tlm=True)

        # Hessian for model
        model_hessian = hessian(
            self.forward_model, options=options,
            hessian_value=error_hessian[0])

        # combine model Hessian and converted error Hessian
        return [
            model_hessian,
            model_hessian._ad_convert_type(error_hessian[1],
                                           options=options)
        ]
