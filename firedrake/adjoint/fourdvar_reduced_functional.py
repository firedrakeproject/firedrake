from pyadjoint import (
    ReducedFunctional, OverloadedType, Control, Tape, AdjFloat,
    stop_annotating, get_working_tape, set_working_tape, annotate_tape)
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint.enlisting import Enlist
from firedrake.function import Function
from firedrake.ensemble import EnsembleFunction, EnsembleFunctionSpace
from .reduced_functional_pipeline import ReducedFunctionalPipeline
from .ensemble_reduced_functional import (
    EnsembleReduceReducedFunctional, EnsembleTransformReducedFunctional)
from .ensemble_adjvec import EnsembleAdjVec
from .covariance_operator import CovarianceOperatorBase
from .allatonce_reduced_functional import AllAtOnceReducedFunctional
from functools import partial
from typing import Callable, Optional
from types import SimpleNamespace
from contextlib import contextmanager
from firedrake.petsc import PETSc


class WC4DVarReducedFunctional(AbstractReducedFunctional):
    """ReducedFunctional for weak constraint 4DVar data assimilation.

    Parameters
    ----------

    control
        The :class:`.EnsembleFunction` for the control x_{i} at the initial condition
        and at the end of each observation stage.

    background_covariance
        The inner product to calculate the background error functional
        from the background error :math:`x_{0} - x_{b}`. Can include the
        error covariance matrix. Only used on ensemble rank 0.

    observation_covariance
        The inner product to calculate the observation error functional
        from the observation error :math:`y_{0} - \\mathcal{H}_{0}(x)`.
        Can include the error covariance matrix. Must be provided if
        observation_error is provided. Only used on ensemble rank 0

    observation_error
        Given a state :math:`x`, returns the observations error
        :math:`y_{0} - \\mathcal{H}_{0}(x)` where :math:`y_{0}` are the
        observations at the initial time and :math:`\\mathcal{H}_{0}` is
        the observation operator for the initial time. Only used on
        ensemble rank 0. Optional.

    background
        The background (prior) data for the initial condition :math:`x_{b}`.
        If not provided, the value of the first subfunction on the first ensemble
        member of the control :class:`.EnsembleFunction` will be used.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, control: Control,
                 background_covariance: CovarianceOperatorBase,
                 observation_covariance: CovarianceOperatorBase,
                 observation_error: Callable[[OverloadedType], OverloadedType] = None,
                 background: Optional[OverloadedType] = None,
                 gauss_newton: bool = False):

        self.gauss_newton = gauss_newton

        if not isinstance(control.control, EnsembleFunction):
            raise TypeError(
                "Control for weak constraint 4DVar must be an EnsembleFunction"
            )

        with stop_annotating():
            if background:
                self.background = background._ad_copy()
            else:
                self.background = control.control.subfunctions[0]._ad_copy()

        self.control_space = control.function_space()
        self.ensemble = self.control_space.ensemble
        rank = self.ensemble.ensemble_rank

        self.control = control
        self._controls = Enlist(control)

        # first control on rank 0 is initial conditions, not end of observation stage
        self.nlocal_observations = self.control_space.nlocal_spaces
        self.nlocal_stages = self.control_space.nlocal_spaces - (rank == 0)

        self.observation_rfs = []
        self.observation_covariances = []

        self.model_rfs = []
        self.model_covariances = []

        # first rank sets up functionals for background initial observations
        if rank == 0:
            self.background_covariance = background_covariance
            self.observation_covariances.append(observation_covariance)

            # RF to recalculate error vector (H(x_0) - y_0)
            xi = self.background._ad_copy()
            with set_working_tape() as tape:
                self.observation_rfs.append(
                    ReducedFunctional(observation_error(xi), Control(xi), tape=tape))

    @property
    def controls(self):
        return self._controls

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

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def tlm(self, m_dot: OverloadedType):
        m_dot = m_dot[0] if isinstance(m_dot, (list, tuple)) else m_dot
        return self.Jmodel.tlm(m_dot) + self.Jobservations.tlm(m_dot)

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

        if self.gauss_newton:
            H, R = self.JH, self.JR
            L, D = self.JL, self.JD

            R = ReducedFunctionalPipeline(self.JR, self.Jsum).hessian
            D = ReducedFunctionalPipeline(self.JD, self.Jsum).hessian

            L = self.JL.tlm
            H = self.JH.tlm

            LT = partial(self.JL.derivative, apply_riesz=apply_riesz)
            HT = partial(self.JH.derivative, apply_riesz=apply_riesz)

            return LT(D(L(m_dot))) + HT(R(H(m_dot)))

        m_dot = m_dot[0] if isinstance(m_dot, (list, tuple)) else m_dot

        if evaluate_tlm:
            self.tlm(m_dot)

        hess_args = {'m_dot': None, 'hessian_input': hessian_input,
                     'evaluate_tlm': False, 'apply_riesz': apply_riesz}

        return (
            self.Jobservations.hessian(**hess_args)
            + self.Jmodel.hessian(**hess_args)
        )

    @contextmanager
    @PETSc.Log.EventDecorator()
    def recording_stages(self, sequential=True, nstages=None, **stage_kwargs):
        if not sequential:
            raise ValueError("Recording stages concurrently not yet implemented")

        rank = self.ensemble.ensemble_rank

        # index of "previous" stage and observation in global series
        global_index = -1
        observation_index = 0 if rank == 0 else -1
        with stop_annotating():
            xhalo = self.control.control.subfunctions[0]._ad_copy()
            _x = self.control.control._ad_copy()

        # add our data onto the user's context data
        ekwargs = {k: v for k, v in stage_kwargs.items()}
        ekwargs['global_index'] = global_index

        ekwargs['xhalo'] = xhalo

        # proceed one ensemble rank at a time
        with self.ensemble.sequential(**ekwargs) as ectx:

            # rank 0 starts from background, later ranks start from halo
            if rank == 0:
                controls = _x.subfunctions[1:]
                xstart = self.background
            else:
                controls = _x.subfunctions
                xstart = ectx.xhalo

            with stop_annotating():
                controls[0].assign(xstart)

            # grab the user's data from the ensemble context
            local_stage_kwargs = {
                k: getattr(ectx, k) for k in stage_kwargs.keys()
            }

            # initialise iterator for local stages
            stage_sequence = WC4DVarObservationStageSequence(
                controls, ectx.global_index,
                observation_index,
                local_stage_kwargs)

            # let the user record the local stages
            yield stage_sequence

            stages = stage_sequence.stages

            with stop_annotating():
                if rank == 0:
                    _x.subfunctions[0].assign(self.background)
                for c, stage in zip(controls, stages):
                    c.assign(stage.state)

            # grab what we need from the stages
            for stage in stages:
                self.observation_rfs.append(stage.observation_rf)
                self.observation_covariances.append(stage.observation_covariance)
                self.model_rfs.append(stage.forward_model)
                self.model_covariances.append(stage.model_covariance)

            # send the state forward
            with stop_annotating():
                ectx.xhalo.assign(stages[-1].state)
                # grab the user's information to send forward
                for k in local_stage_kwargs.keys():
                    setattr(ectx, k, getattr(stage_sequence.ctx, k))
                # increment the global indices for the last local stage
                ectx.global_index = stages[-1].global_index

        # Create a ReducedFunctional for each observation error covariance operator
        observation_norms = []
        for cov in self.observation_covariances:
            yi = Function(cov.function_space())
            with set_working_tape() as tape:
                observation_norms.append(
                    ReducedFunctional(cov.norm(yi), Control(yi), tape=tape))

        # Create a ReducedFunctional for each model error covariance operator
        model_covariances = [self.background_covariance] if rank == 0 else []
        model_covariances.extend(self.model_covariances)
        model_norms = []
        for cov in model_covariances:
            xi = Function(cov.function_space())
            with set_working_tape() as tape:
                model_norms.append(
                    ReducedFunctional(cov.norm(xi), Control(xi), tape=tape))

        # Now we have all the various per-stage components (M, H, B, D, R, etc)
        # we can build the space-time collective RFs that will evaluate the
        # objective across all stages.
        with stop_annotating():
            # make sure that self.control now holds the
            # values of the initial timeseries
            self.control.assign(_x)

            self.observation_space = EnsembleFunctionSpace(
                [Jo.functional.function_space() for Jo in self.observation_rfs],
                ensemble=self.ensemble)

            _x = EnsembleFunction(self.control_space)
            _y = EnsembleFunction(self.observation_space)
            _adjvec = EnsembleAdjVec(
                [AdjFloat(0.) for _ in range(self.control_space.nlocal_spaces)],
                ensemble=self.ensemble)

            # reduction rf
            Jsum = EnsembleReduceReducedFunctional(
                AdjFloat(0.), Control(_adjvec._ad_init_zero()))
            self.Jsum = Jsum

            # # # # # #
            # Create observation ReducedFunctionals
            # # # # # #

            # (JH : V^n -> U^n) = (x -> y - H(x))
            JH = EnsembleTransformReducedFunctional(
                _y._ad_init_zero(),
                Control(_x._ad_init_zero()),
                self.observation_rfs)
            self.JH = JH

            # (JR : U^n -> R^n) = (x -> x^T R^{-1} x)
            JR = EnsembleTransformReducedFunctional(
                _adjvec._ad_init_zero(),
                Control(_y._ad_init_zero()),
                observation_norms)
            self.JR = JR

            # (Jobs : V^n -> R) = (x -> \sum_{i} |y - H(x)|_{R})
            self.Jobservations = ReducedFunctionalPipeline(JH, JR, Jsum)

            # # # # # #
            # Create model propagator ReducedFunctionals
            # # # # # #

            # (JL : V^n -> V^n) = (x -> x - <x_b, M(x)>)
            JL = AllAtOnceReducedFunctional(
                _x._ad_init_zero(), Control(_x._ad_init_zero()),
                self.model_rfs, background=self.background)
            self.JL = JL

            # (JDnorm : V^{n} -> R^{n}) = (x -> x^T <B,Q>^{-1} x)
            JD = EnsembleTransformReducedFunctional(
                _adjvec._ad_init_zero(),
                Control(_x._ad_init_zero()),
                model_norms)
            self.JD = JD

            # (Jmod : V^n -> R) = (x -> \sum_{i} |x - <x_b, M(x)>|_{<B, Q>})
            self.Jmodel = ReducedFunctionalPipeline(JL, JD, Jsum)


class SC4DVarReducedFunctional(AbstractReducedFunctional):
    """ReducedFunctional for strong constraint 4DVar data assimilation.

    Parameters
    ----------

    control
        The :class:`.Function` for the control x_{0} at the initial condition.

    background_covariance
        The inner product to calculate the background error functional
        from the background error :math:`x_{0} - x_{b}`. Can include the
        error covariance matrix.

    observation_covariance
        The inner product to calculate the observation error functional
        from the observation error :math:`y_{0} - \\mathcal{H}_{0}(x)`.
        Can include the error covariance matrix. Must be provided if
        observation_error is provided.

    observation_error
        Given a state :math:`x`, returns the observations error
        :math:`y_{0} - \\mathcal{H}_{0}(x)` where :math:`y_{0}` are the
        observations at the initial time and :math:`\\mathcal{H}_{0}` is
        the observation operator for the initial time.

    background
        The background (prior) data for the initial condition :math:`x_{b}`.
        If not provided, the value of the control will be used.

    See Also
    --------
    :class:`pyadjoint.ReducedFunctional`.
    """
    def __init__(self, control: Control,
                 background: OverloadedType,
                 background_covariance: CovarianceOperatorBase,
                 observation_covariance: CovarianceOperatorBase,
                 observation_error: Callable[[OverloadedType], OverloadedType],
                 tape: Tape | None = None):

        if not isinstance(control.control, Function):
            raise TypeError(
                "Control for strong constraint 4DVar must be a Function.")
        if background is control.control:
            raise ValueError(
                "Background must be a different object to the control.")

        self.tape = get_working_tape() if tape is None else tape

        self._controls = Enlist(control)
        self._functional = AdjFloat(0.)
        self._background = background
        x0 = control.control

        self.background_covariance = background_covariance
        self.observation_covariances = [observation_covariance]
        self.observation_errors = [observation_error]

        # penalty for straying from prior
        background_error = x0.copy(deepcopy=True)
        background_error -= background
        self._functional += background_covariance.norm(background_error)

        # penalty for straying from initial observations
        self._functional += observation_covariance.norm(observation_error(x0))

    @property
    def controls(self):
        return self._controls

    @property
    def functional(self):
        return self._functional

    def background(self):
        return self._background

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def __call__(self, values: OverloadedType):
        return self.rf(values)

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def derivative(self, adj_input: float = 1.0, apply_riesz: bool = False):
        return self.rf.derivative(adj_input=adj_input, apply_riesz=apply_riesz)

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def tlm(self, m_dot: OverloadedType):
        return self.rf.tlm(m_dot)

    @stop_annotating()
    @PETSc.Log.EventDecorator()
    def hessian(self, m_dot: OverloadedType, hessian_input: Optional[OverloadedType] = None,
                evaluate_tlm: bool = True, apply_riesz: bool = False):
        return self.rf.hessian(m_dot, hessian_input=hessian_input,
                               evaluate_tlm=evaluate_tlm, apply_riesz=apply_riesz)

    @contextmanager
    @PETSc.Log.EventDecorator()
    def recording_stages(self, nstages=None, **stage_kwargs):
        yield SC4DVarObservationStageSequence(
            self.controls[0].control, self,
            stage_kwargs=stage_kwargs, nstages=nstages)

        self.rf = ReducedFunctional(
            self._functional, self.controls.delist(), tape=self.tape)


class WC4DVarObservationStageSequence:
    def __init__(self, controls: Control, global_index: int,
                 observation_index: int, stage_kwargs: dict = None):
        self.controls = controls
        self.ctx = SimpleNamespace(**(stage_kwargs or {}))
        self.global_index = global_index
        self.observation_index = observation_index
        self.local_index = -1
        self.nstages = len(controls)
        self.stages = []

    def __iter__(self):
        return self

    @PETSc.Log.EventDecorator()
    def __next__(self):

        # increment global indices.
        self.local_index += 1
        self.global_index += 1
        self.observation_index += 1

        # Now we can check if we need to exit.
        if self.local_index >= self.nstages:
            raise StopIteration

        # control for the start of the next stage.
        next_control = self.controls[self.local_index]

        # smuggle state forward into aaorf's next control.
        if self.local_index > 0:
            with stop_annotating():
                next_control.assign(self.stages[-1].state)

        stage = WC4DVarObservationStage(next_control,
                                        local_index=self.local_index,
                                        global_index=self.global_index,
                                        observation_index=self.observation_index)
        self.stages.append(stage)

        return stage, self.ctx


class WC4DVarObservationStage:
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
        self.control = control
        # where are we in the local/global timeseries?
        self.local_index = local_index
        self.global_index = global_index
        self.observation_index = observation_index
        # we set our own tape to record the propagator
        self._old_tape = get_working_tape()
        set_working_tape()
        self.tape = get_working_tape()

    @PETSc.Log.EventDecorator()
    def set_observation(self, state: OverloadedType,
                        observation_error: Callable[[OverloadedType], OverloadedType],
                        observation_covariance: CovarianceOperatorBase,
                        forward_model_covariance: CovarianceOperatorBase):
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
        if get_working_tape() is not self.tape:
            raise ValueError(
                "Working tape at the end of the observation stage"
                " differs from the tape at the stage beginning."
            )
        if not annotate_tape():
            raise ValueError(
                "Must have annotations switched on whilst"
                " recording the 4DVar observation stages.")

        # Take a copy so it doesn't matter if the user modifies state later
        self.state = state._ad_copy()
        # record forward propagation
        self.forward_model = ReducedFunctional(
            self.state, controls=Control(self.control), tape=self.tape)

        # Save the covariance operators
        self.observation_covariance = observation_covariance
        self.model_covariance = forward_model_covariance

        # Record our own tape of the observation operator so we can
        # rerun it without interfering with/interference from the user.
        with stop_annotating():
            x_copy = state._ad_copy()
        with set_working_tape() as tape:
            self.observation_rf = ReducedFunctional(
                observation_error(x_copy), Control(x_copy), tape=tape)

        # reset the tape to what it was before recording this stage.
        set_working_tape(self._old_tape)


class SC4DVarObservationStageSequence:
    def __init__(self, control: OverloadedType,
                 rf: SC4DVarReducedFunctional,
                 stage_kwargs: dict,
                 nstages: int):
        self.ctx = SimpleNamespace(**stage_kwargs)
        self.nstages = nstages
        self._index = 0
        self._current_stage = SC4DVarObservationStage(
            control, rf, index=self._index)

    def __iter__(self):
        return self

    @PETSc.Log.EventDecorator()
    def __next__(self):
        if self._index >= self.nstages:
            raise StopIteration

        if self._index > 0:
            self._current_stage = SC4DVarObservationStage(
                control=self._current_stage.state,
                rf=self._current_stage._rf,
                index=self._index)

        self._index += 1
        return self._current_stage, self.ctx


class SC4DVarObservationStage:
    """
    Record an observation for strong constraint 4DVar at the time of `state`.

    Parameters
    ----------

    control :
        The state at the beginning of this stage.
    rf :
        The SC4DVarReducedFunctional.
    index :
        The index of this stage, numbered from 0.
    """

    def __init__(self, control: OverloadedType,
                 rf: SC4DVarReducedFunctional, index: int):
        self._rf = rf
        self.control = control
        self.index = index

    @PETSc.Log.EventDecorator()
    def set_observation(self, state: OverloadedType,
                        observation_error: Callable[[OverloadedType], OverloadedType],
                        observation_covariance: CovarianceOperatorBase):
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
        # get the tape used for this stage and make sure its the right one
        if get_working_tape() is not self._rf.tape:
            raise ValueError(
                "Working tape at the end of the observation stage"
                " differs from the tape at the stage beginning."
            )
        if not annotate_tape():
            raise ValueError(
                "Must have annotations switched on whilst"
                " recording the 4DVar observation stages.")

        self._rf._functional += (
            observation_covariance.norm(observation_error(state)))

        self._rf.observation_covariances.append(observation_covariance)
        self._rf.observation_errors.append(observation_error)

        # save the user's state to hand back for beginning of next stage
        self.state = state
