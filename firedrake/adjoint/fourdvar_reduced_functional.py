from pyadjoint import ReducedFunctional, OverloadedType, Control, Tape, AdjFloat, \
    stop_annotating, get_working_tape, set_working_tape
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint.enlisting import Enlist
from firedrake.assemble import get_assembler
from firedrake.function import Function
from firedrake.ensemble import EnsembleFunction, EnsembleFunctionSpace
from firedrake import assemble, inner, dx, Constant, grad, TestFunction, TrialFunction, derivative, Cofunction, LinearVariationalProblem, LinearVariationalSolver, RieszMap
from .composite_reduced_functional import CompositeReducedFunctional
from .composite_reduced_functional import _rename, isolated_rf
from .ensemble_reduced_functional import (
    EnsembleReduceReducedFunctional, EnsembleTransformReducedFunctional)
from .ensemble_adjvec import EnsembleAdjVec
from .allatonce_reduced_functional import AllAtOnceReducedFunctional
from functools import wraps, cached_property, partial
from typing import Callable, Optional, Collection, Union
from types import SimpleNamespace
from contextlib import contextmanager
from firedrake.petsc import PETSc
from math import pi, sqrt
from abc import ABC, abstractmethod

__all__ = ['FourDVarReducedFunctional', 'CovarianceOperator']


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


class FourDVarReducedFunctional(AbstractReducedFunctional):
    """ReducedFunctional for 4DVar data assimilation.

    Creates either the strong constraint or weak constraint system
    by logging observations through the initial time propagator run.

    Parameters
    ----------

    control
        The :class:`.EnsembleFunction` for the control x_{i} at the initial condition
        and at the end of each observation stage.

    background_covariance
        The inner product to calculate the background error functional
        from the background error :math:`x_{0} - x_{b}`. Can include the
        error covariance matrix. Only used on ensemble rank 0.

    background
        The background (prior) data for the initial condition :math:`x_{b}`.
        If not provided, the value of the first subfunction on the first ensemble
        member of the control :class:`.EnsembleFunction` will be used.

    observation_error
        Given a state :math:`x`, returns the observations error
        :math:`y_{0} - \\mathcal{H}_{0}(x)` where :math:`y_{0}` are the
        observations at the initial time and :math:`\\mathcal{H}_{0}` is
        the observation operator for the initial time. Only used on
        ensemble rank 0. Optional.

    observation_covariance
        The inner product to calculate the observation error functional
        from the observation error :math:`y_{0} - \\mathcal{H}_{0}(x)`.
        Can include the error covariance matrix. Must be provided if
        observation_error is provided. Only used on ensemble rank 0

    weak_constraint
        Whether to use the weak or strong constraint 4DVar formulation.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, control: Control,
                 background_covariance: Union[Constant, tuple],
                 background: Optional[OverloadedType] = None,
                 observation_error: Optional[Callable[[OverloadedType], OverloadedType]] = None,
                 observation_covariance: Optional[Callable[[OverloadedType], AdjFloat]] = None,
                 weak_constraint: bool = True,
                 tape: Optional[Tape] = None,
                 _annotate_accumulation: bool = False):

        self.tape = get_working_tape() if tape is None else tape

        self.weak_constraint = weak_constraint
        self.initial_observations = observation_error is not None

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

            self.control_space = control.function_space()
            ensemble = self.control_space.ensemble
            self.ensemble = ensemble
            self.trank = ensemble.ensemble_rank if ensemble else 0
            self.nchunks = ensemble.ensemble_size if ensemble else 1

            # because we need to manually evaluate the different bits
            # of the functional, we need an internal set of controls
            # to use for the stage ReducedFunctionals
            self._cbuf = control.control._ad_copy()
            _x = self._cbuf.subfunctions
            self._x = _x
            self._controls_ = tuple(Control(xi) for xi in _x)

            self.control = control
            self._controls = Enlist(control)

            # first control on rank 0 is initial conditions, not end of observation stage
            self.nlocal_stages = len(_x) - (1 if self.trank == 0 else 0)

            if self.trank == 0 and self.initial_observations:
                self.nlocal_observations = self.nlocal_stages + 1
            else:
                self.nlocal_observations = self.nlocal_stages

            self.stages = []    # The record of each observation stage

            self.observation_rfs = []
            self.observation_norms = []

            self.model_rfs = []
            self.model_norms = []

            # first rank sets up functionals for background initial observations
            if self.trank == 0:

                # RF to recalculate inner product |x_0 - x_b|_B
                # self.background_norm = CovarianceNormReducedFunctional(
                #     self.background._ad_init_zero(),
                #     background_covariance, form="diffusion",
                #     control_name="bkg_err_vec_copy")

                self.background_norm = CovarianceReducedFunctional(
                    self.background._ad_init_zero(),
                    background_covariance)

                if self.initial_observations:

                    # RF to recalculate error vector (H(x_0) - y_0)
                    self.initial_observation_error = isolated_rf(
                        operation=observation_error,
                        control=_x[0],
                        functional_name="obs_err_vec_0",
                        control_name="Control_0_obs_copy")

                    # RF to recalculate inner product |H(x_0) - y_0|_R
                    # self.initial_observation_norm = CovarianceNormReducedFunctional(
                    #     self.initial_observation_error.functional,
                    #     observation_covariance, form="mass",
                    #     control_name="obs_err_vec_0_copy")

                    self.initial_observation_norm = CovarianceReducedFunctional(
                        self.initial_observation_error.functional,
                        observation_covariance)

                    self.observation_rfs.append(self.initial_observation_error)
                    self.observation_norms.append(self.initial_observation_norm)

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

            self.control_space = self.background.function_space()

            # initial conditions guess to be updated
            self._controls = Enlist(control)

            # Strong constraint functional to be converted to ReducedFunctional later

            # penalty for straying from prior
            self.background_norm = CovarianceReducedFunctional(
                control.control._ad_init_zero(),
                background_covariance)

            bkg_err = Function(self.control_space)
            bkg_err.assign(control.control - self.background)
            self._accumulate_functional(
                weighted_norm(bkg_err, background_covariance))

            # penalty for not hitting observations at initial time
            if self.initial_observations:
                self._accumulate_functional(
                    weighted_norm(
                        observation_error(control.control),
                        observation_covariance))

    @property
    def controls(self):
        return self._controls

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
    @PETSc.Log.EventDecorator()
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
        value = values[0] if isinstance(values, (list, tuple)) else values

        if not isinstance(value, type(self.control.control)):
            raise ValueError(f"Value must be of type {type(self.control.control)} not type {type(value)}")

        self.control.update(value)

        return self.Jmodel(value) + self.Jobservations(value)

    @sc_passthrough
    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def derivative(self, adj_input: float = 1.0, apply_riesz: bool = False):
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
        adj_input = adj_input[0] if isinstance(adj_input, (list, tuple)) else adj_input
        adj_args = {'adj_input': adj_input, 'apply_riesz': apply_riesz}
        return (
            self.Jobservations.derivative(**adj_args)
            + self.Jmodel.derivative(**adj_args)
        )

    @sc_passthrough
    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def tlm(self, m_dot: OverloadedType):
        m_dot = m_dot[0] if isinstance(m_dot, (list, tuple)) else m_dot
        return self.Jmodel.tlm(m_dot) + self.Jobservations.tlm(m_dot)

    @sc_passthrough
    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def hessian(self, m_dot: OverloadedType, hessian_input: Optional[OverloadedType] = None,
                evaluate_tlm: bool = True, apply_riesz: bool = False):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Parameters
        ----------

        m_dot
            The direction in which to compute the action of the Hessian.

        Returns
        -------
        pyadjoint.OverloadedType
            The action of the Hessian in the direction m_dot.
            Should be an instance of the same type as the control.
        """
        m_dot = m_dot[0] if isinstance(m_dot, (list, tuple)) else m_dot

        if evaluate_tlm:
            self.tlm(m_dot)

        hess_args = {'m_dot': None, 'hessian_input': hessian_input,
                     'evaluate_tlm': False, 'apply_riesz': False}

        return (
            self.Jobservations.hessian(**hess_args)
            + self.Jmodel.hessian(**hess_args)
        )

    def _accumulate_functional(self, val):
        if not self._annotate_accumulation:
            return
        if self._accumulation_started:
            self._total_functional += val
        else:
            self._total_functional = val
            self._accumulation_started = True

    @contextmanager
    @PETSc.Log.EventDecorator()
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
                    controls = self._controls_
                    with stop_annotating():
                        controls[0].assign(self.background)
                else:
                    controls = [self._control_prev, *self._controls_]
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

                for stage in self.stages:
                    self.observation_rfs.append(stage.observation_error)
                    self.observation_norms.append(stage.observation_norm)
                    self.model_rfs.append(stage.forward_model)
                    self.model_norms.append(stage.model_norm)

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

            with stop_annotating():
                # make sure that self.control now holds the
                # values of the initial timeseris
                self.control.assign(self._cbuf)

                # AdjVec for controls
                self._float_vec = EnsembleAdjVec(
                    [AdjFloat(0.) for _ in range(self.control_space.nlocal_spaces)],
                    ensemble=self.ensemble)

                # reduction rf
                self.Jsum = EnsembleReduceReducedFunctional(
                    AdjFloat(0.), Control(self._float_vec._ad_init_zero()))

                # # # # # #
                # Create observation ReducedFunctionals
                # # # # # #

                self.observation_space = EnsembleFunctionSpace(
                    [Jo.functional.function_space() for Jo in self.observation_rfs],
                    ensemble=self.ensemble)

                # (JH : V^n -> U^n) = (x -> y - H(x))
                observation_control = EnsembleFunction(self.control_space)
                observation_functional = EnsembleFunction(self.observation_space)

                self.JH = EnsembleTransformReducedFunctional(
                    self.observation_rfs,
                    observation_functional,
                    Control(observation_control))

                # (JR : U^n -> R^n) = (x -> x^T R^{-1} x)
                observation_control = EnsembleFunction(self.observation_space)
                observation_functional = self._float_vec._ad_init_zero()

                self.JR = EnsembleTransformReducedFunctional(
                    self.observation_norms,
                    observation_functional,
                    Control(observation_control))

                # (Jsum : R^n -> R) = (x -> \sum_{i} x_{i})
                JOsum = self.Jsum

                # (Jobs : V^n -> R) = (x -> \sum_{i} |y - H(x)|_{R})
                self.Jobservations = CompositeReducedFunctional(
                    CompositeReducedFunctional(self.JH, self.JR), JOsum)

                # # # # # #
                # Create model propagator ReducedFunctionals
                # # # # # #

                # (JL : V^n -> V^n) = (x -> x - <x_b, M(x)>)
                model_control = EnsembleFunction(self.control_space)
                model_functional = EnsembleFunction(self.control_space)

                self.JL = AllAtOnceReducedFunctional(
                    EnsembleFunction(self.control_space),
                    Control(EnsembleFunction(self.control_space)),
                    self.model_rfs, background=self.background)

                # (JDnorm : V^{n} -> R^{n}) = (x -> x^T <B,Q>^{-1} x)
                model_control = EnsembleFunction(self.control_space)
                model_functional = self._float_vec._ad_init_zero()

                model_norms = [norm for norm in self.model_norms]
                if self.trank == 0:
                    model_norms.insert(0, self.background_norm)

                self.JD = EnsembleTransformReducedFunctional(
                    model_norms,
                    model_functional,
                    Control(model_control))

                # (JMsum : R^n -> R) = (x -> \sum_{i} x_{i})
                JMsum = self.Jsum

                # (Jmod : V^n -> R) = (x -> \sum_{i} |x - <x_b, M(x)>|_{<B, Q>})
                self.Jmodel = CompositeReducedFunctional(
                    CompositeReducedFunctional(self.JL, self.JD), JMsum)

        else:  # strong constraint
            with stop_annotating():
                self.controls[0].control.assign(self.background)

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

    @PETSc.Log.EventDecorator()
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
        self.local_index = index
        self.global_index = index
        self.observation_index = observation_index

    @PETSc.Log.EventDecorator()
    def set_observation(self, state: OverloadedType,
                        observation_error: Callable[[OverloadedType], OverloadedType],
                        observation_covariance: Callable[[OverloadedType], AdjFloat]):
        """
        Record an observation at the time of `state`.

        Parameters
        ----------

        state
            The state at the current observation time.

        observation_error
            Given a state :math:`x`, returns the observations error
            :math:`y_{i} - \\mathcal{H}_{i}(x)` where :math:`y_{i}` are
            the observations at the current observation time and
            :math:`\\mathcal{H}_{i}` is the observation operator for the
            current observation time.

        observation_covariance
            The inner product to calculate the observation error functional
            from the observation error :math:`y_{i} - \\mathcal{H}_{i}(x)`.
            Can include the error covariance matrix.
        """
        if hasattr(self.aaorf, "_strong_reduced_functional"):
            raise ValueError("Cannot add observations once strong"
                             " constraint ReducedFunctional instantiated")
        self.aaorf._accumulate_functional(
            weighted_norm(
                observation_error(state),
                observation_covariance))

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

    @PETSc.Log.EventDecorator()
    def set_observation(self, state: OverloadedType,
                        observation_error: Callable[[OverloadedType], OverloadedType],
                        observation_covariance: Callable[[OverloadedType], AdjFloat],
                        forward_model_covariance: Callable[[OverloadedType], AdjFloat]):
        """
        Record an observation at the time of `state`.

        Parameters
        ----------

        state
            The state at the current observation time.

        observation_error
            Given a state :math:`x`, returns the observations error
            :math:`y_{i} - \\mathcal{H}_{i}(x)` where :math:`y_{i}` are
            the observations at the current observation time and
            :math:`\\mathcal{H}_{i}` is the observation operator for the
            current observation time.

        observation_covariance
            The inner product to calculate the observation error functional
            from the observation error :math:`y_{i} - \\mathcal{H}_{i}(x)`.
            Can include the error covariance matrix.

        forward_model_covariance
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

        # RF to recalculate inner product |M_i - x_i|_Q
        names = {
            'control_name': f"model_err_vec_{self.global_index}_copy"
        } if self.global_index else {}

        # self.model_norm = CovarianceNormReducedFunctional(
        #     state._ad_init_zero(),
        #     forward_model_covariance,
        #     **names, form="diffusion")

        self.model_norm = CovarianceReducedFunctional(
            state._ad_init_zero(),
            forward_model_covariance)

        # Observations after tape cut because this is now a control, not a state

        # RF to recalculate error vector (H(x_i) - y_i)
        names = {
            'functional_name': f"obs_err_vec_{self.global_index}",
            'control_name': f"Control_{self.global_index}_obs_copy"
        } if self.global_index else {}

        self.observation_error = isolated_rf(
            operation=observation_error,
            control=self.controls[-1],
            **names)

        # RF to recalculate inner product |H(x_i) - y_i|_R
        names = {
            'control_name': "obs_err_vec_{self.global_index}_copy"
        } if self.global_index else {}
        # self.observation_norm = CovarianceNormReducedFunctional(
        #     self.observation_error.functional,
        #     observation_covariance,
        #     **names, form="mass")

        self.observation_norm = CovarianceReducedFunctional(
            self.observation_error.functional,
            observation_covariance)

        # remove the stage initial condition "control" now we've finished recording
        delattr(self, "control")

        # stop the stage tape recording anything else
        set_working_tape()


def covariance_norm(x, covariance, form='diffusion'):
    pass
    # if isinstance(covariance, Collection):
    #     covariance, power = covariance
    # else:
    #     power = None
    power = None

    valid_forms = ("diffusion", "mass")
    if form not in valid_forms:
        raise ValueError(
            f"Unknown covariance form type {form}, should be "
            " or ".join(valid_forms))

    v = TestFunction(x.function_space())
    if form == 'diffusion':
        sigma, L, bcs = covariance
        # diffusion coefficient for lengthscale L
        nu = 0.5*L*L
        # normalisation for diffusion operator
        lambda_g = L*sqrt(2*pi)
        scale = lambda_g/sigma

        xB = scale*(inner(x, v) - nu*inner(grad(x), grad(v)))*dx
        val = assemble(xB(x))
    elif form == 'mass':
        if isinstance(covariance, (float, int)):
            w = Constant(1/covariance)
        else:
            w = 1/covariance
        xB = inner(x*w, v)*dx
        val = assemble(xB(x))
    else:
        raise ValueError("Forgot to handle a covariance form type")

    from pyadjoint import AdjFloat
    result = val if power is None else val**power
    assert type(result) is AdjFloat, f"{type(result).__name__ = } is not AdjFloat."
    return result


class CovarianceOperatorBase(ABC):
    @abstractmethod
    def norm(self, x):
        """Calculate (x^T)(B^-1)(x) : V x V -> V x V* -> R
        """
        pass

    @abstractmethod
    def action(self, x):
        """Calculate B*x : V* -> V
        """
        pass

    @property
    @abstractmethod
    def reduced_functional(self):
        pass


class FormCovarianceOperator(CovarianceOperatorBase):
    def __init__(self, V, sigma, bcs=None, implicit=True,
                 options_prefix=None, solver_parameters=None):
        self.V = V
        self.sigma = sigma
        self.bcs = bcs
        self.implicit = implicit

        self.x = Function(V)

        self._sol = Function(V)
        if implicit:
            self._b = Function(V)
            rhs = inner(self._b, v)*dx
        else:
            self._b = Cofunction(V.dual())
            rhs = self._b

        self.two_form = derivative(self.one_form, self.x)

        self.solver = LinearVariationalSolver(
            LinearVariationalProblem(
                self.two_form, rhs, self._sol,
                bcs=self.bcs, constant_jacobian=True),
            options_prefix=options_prefix,
            solver_parameters=solver_parameters)

        if implicit:
            self._assemble_xBx = get_assembler(
                inner(self._b, self._sol)*dx).assemble

            self._assemble_one_form = get_assembler(
                self.one_form, bcs=self.bcs).assemble

        else:
            self._assemble_norm = get_assembler(
                self.one_form(self.x)).assemble

    @property
    @abstractmethod
    def one_form(self):
        raise NotImplementedError

    def norm(self, x):
        if x.function_space() != self.V:
            raise ValueError("Covariance norm acts on a Function")

        if self.implicit:  # xT*L^{-1}*x
            self._b.assign(x)
            self.solver.solve()
            return self._assemble_xBx()

        else:  # xT*L*x
            self.x.assign(x)
            return self._assemble_norm()

    def action(self, x, tensor=None, apply_riesz=True):
        # B : V* -> V
        if x.function_space() != self.V.dual():
            raise ValueError("Covariance operator acts on a Cofunction")
        if tensor is None:
            tensor = Function(self.V if apply_riesz else self.V.dual())

        if self.implicit:  # L*x
            self.x.assign(x.riesz_representation())
            Bx = self._assemble_one_form()
            if apply_riesz:
                return tensor.assign(Bx.riesz_representation())
            else:
                return tensor.assign(Bx)

        else:  # L^{-1}*x
            self._b.assign(x)
            self.solver.solve()
            if apply_riesz:
                return tensor.assign(self._sol)
            else:
                return tensor.assign(self._sol.riesz_representation())

    @cached_property
    def reduced_functional(self):
        pass


class MassCovarianceOperator(CovarianceOperatorBase):
    @property
    def one_form(self):
        x = self.x
        v = TestFunction(V)
        w = self.sigma if self.implicit else 1/self.sigma
        return inner(x*w, v)*dx


class DiffusionCovarianceOperator(CovarianceOperatorBase):
    def __init__(self, V, sigma, L, bcs=None, implicit=True,
                 options_prefix=None, solver_parameters=None):
        self.L = L
        super().__init__(V, sigma, bcs=bcs, implicit=implicit,
                         options_prefix=options_prefix,
                         solver_parameters=solver_parameters)

    @property
    def one_form(self):
        L = self.L
        sigma = self.sigma
        nu = Constant(L*L/2)
        lambda_g = Constant(L*sqrt(2*pi))
        self.nu = nu
        self.lambda_g = lambda_g
        w = sigma/lambda_g if self.implicit else lambda_g/sigma
        x = self.x
        v = TestFunction(self.V)
        return w*(inner(x, v)*dx + inner(nu*grad(x), grad(v))*dx)


class CovarianceOperator:
    def __init__(self, V, form_type, **kwargs):
        valid_form_types = ("diffusion", "mass")
        if form_type not in valid_form_types:
            raise ValueError(
                f"Unknown form_type {form} for {type(self).__name__},"
                "should be "+" or ".join(valid_form_types))
        self.form_type = form_type
        self.V = V

        x = Function(V)
        u = TrialFunction(V)
        v = TestFunction(V)
        self.x = x
        self.u = u
        self.v = v

        sigma = kwargs["sigma"]
        self.sigma = sigma

        if form_type == "mass":
            w = 1/sigma
            one_form = inner(x*w, v)*dx

        elif form_type == "diffusion":
            L = kwargs["L"]
            nu = Constant(0.5*L*L)
            lambda_g = Constant(L*sqrt(2.*pi))
            self.L = L
            self.nu = nu
            self.lambda_g = lambda_g

            scale = sigma/lambda_g
            one_form = scale*(inner(x, v) + nu*inner(grad(x), grad(v)))*dx

        self.one_form = one_form
        self.two_form = derivative(self.one_form, x)

        self.bcs = kwargs.get("bcs", None) or []
        options_prefix = kwargs.get("options_prefix", None)
        solver_parameters = kwargs.get("solver_parameters", None)

        self._assemble_action = get_assembler(
            self.one_form, bcs=self.bcs).assemble

        self._b = Function(V)
        rhs = inner(self._b, v)*dx
        self._sol = Function(V)
        self.solver = LinearVariationalSolver(
            LinearVariationalProblem(
                self.two_form, rhs, self._sol,
                bcs=self.bcs, constant_jacobian=True),
            options_prefix=options_prefix,
            solver_parameters=solver_parameters)

    def norm(self, x):
        if x.function_space() != self.V:
            raise ValueError("Covariance norm taken over a Function")

        if self.form_type == "mass":
            self.x.assign(x)
            return assemble(self.one_form(self.x))

        elif self.form_type == "diffusion":
            self._b.assign(x)
            self.solver.solve()
            Binv_x = self._sol
            return assemble(inner(x, Binv_x)*dx)

    def action(self, x, tensor=None):
        if x.function_space() != self.V.dual():
            raise ValueError("Covariance operator acts on a Cofunction")

        if tensor is None:
            tensor = Function(self.V)

        if self.form_type == "mass":
            self._b.assign(x.riesz_representation())
            self._sol.zero()
            self.solver.solve()
            return tensor.assign(self._sol)

        elif self.form_type == "diffusion":
            self.x.assign(x.riesz_representation())
            Bx = assemble(self.one_form, bcs=self.bcs)
            return tensor.assign(Bx.riesz_representation())

    @cached_property
    def reduced_functional(self):
        return CovarianceReducedFunctional(self)  # TODO: ref cycle


def weighted_norm(x, w):
    return assemble(inner(w*x, x*w)*dx)


class CovarianceReducedFunctional(ReducedFunctional):
    def __init__(self, v, covariance):
        self.covariance = covariance
        rf = isolated_rf(
            lambda x: weighted_norm(x, covariance),
            v._ad_init_zero())
        super().__init__(
            rf.functional, rf.controls[0], tape=rf.tape)


class CovarianceNormReducedFunctional(ReducedFunctional):
    def __init__(self, x, covariance, form='diffusion',
                 functional_name=None,
                 control_name=None):
        # if isinstance(covariance, Collection):
        #     self.covariance, self.power = covariance
        # else:
        #     self.covariance = covariance
        #     self.power = None
        self.covariance = covariance
        cov_norm = partial(covariance_norm, covariance=covariance, form=form)
        rf = isolated_rf(cov_norm, x,
                         functional_name=functional_name,
                         control_name=control_name)
        super().__init__(rf.functional,
                         rf.controls.delist(rf.controls),
                         tape=rf.tape)
