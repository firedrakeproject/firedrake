from functools import partial, cached_property
from contextlib import contextmanager
from pyadjoint.reduced_functional import AbstractReducedFunctional
from pyadjoint import (
    ReducedFunctional, Control, AdjFloat, set_working_tape, annotate_tape,
    no_annotations, continue_annotation, pause_annotation)
from pyadjoint.enlisting import Enlist
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from .ensemble_adjvec import EnsembleAdjVec
from firedrake.ensemble.ensemble_function import (
    EnsembleFunctionSpace, EnsembleFunction, EnsembleFunctionBase)

__all__ = (
    "EnsembleReducedFunctional",
    "EnsembleBcastReducedFunctional",
    "EnsembleReduceReducedFunctional",
    "EnsembleTransformReducedFunctional",
)


def _local_subs(val):
    if isinstance(val, EnsembleFunctionBase):
        return val.subfunctions
    elif isinstance(val, EnsembleAdjVec):
        return val.subvec
    elif isinstance(val, (list, tuple)):
        return val
    else:
        raise TypeError(
            f"Cannot use {type(val).__name__} as an ensemble overloaded type.")


def _set_local_subs(dst, src):
    dst = _local_subs(dst)
    for i, s in enumerate(src):
        if hasattr(dst[i], 'assign'):
            dst[i].assign(s)
        else:
            dst[i] = s
    return dst


def _ad_sum(vals):
    vals = Enlist(vals)
    total = vals[0]._ad_init_zero()
    for v in vals:
        if isinstance(v, float):
            total = total._ad_add(v)
        else:
            total._ad_iadd(v)
    return total


class EnsembleReduceReducedFunctional(AbstractReducedFunctional):
    def __init__(self, functional, control, ensemble=None):
        if isinstance(functional, AbstractReducedFunctional):
            self.reduction_rf = functional
            self.functional = self.reduction_rf.functional
        else:
            self.reduction_rf = None
            self.functional = functional

        self._controls = Enlist(control)
        self._control = control

        if isinstance(functional, AdjFloat):
            if not isinstance(control.control, EnsembleAdjVec):
                raise TypeError(
                    f"Control for {type(self).__name__} must be an EnsembleAdjVec"
                    "  if using an AdjFloat functional.")
            if (ensemble is not None and ensemble is not control.control.ensemble):
                raise ValueError(
                    f"Ensemble provided to {type(self).__name__} must match"
                    " the ensemble of the control.")
            self.ensemble = ensemble
            self.nlocal_inputs = len(control.subvec)

        elif isinstance(functional, Function):
            if not isinstance(control.control, EnsembleFunction):
                raise TypeError(
                    f"Control for {type(self).__name__} must be an"
                    " EnsembleFunction if using a Function functional.")
            if not all([c.function_space() == functional.function_space()
                        for c in control.subfunctions]):
                raise ValueError(
                    f"All subfunctions of the {type(self).__name__} control"
                    " must have the same function space as the functional.")
            if (ensemble is not None and ensemble is not control.function_space().ensemble):
                raise ValueError(
                    f"Ensemble provided to {type(self).__name__} must match"
                    " the ensemble of the control.")
            self.ensemble = control.function_space().ensemble
            self.nlocal_inputs = len(control.subfunctions)

        else:
            raise ValueError(
                f"Do not know how to handle a {type(functional).__name__}"
                f" control for {type(self).__name__}.")

        _ = self._sum_rf

    @property
    def controls(self):
        return self._controls

    @no_annotations
    def __call__(self, values):
        if self.reduction_rf:
            return self.reduction_rf(self._allgather(_local_subs(values)))
        else:
            return self.ensemble.allreduce(
                self._sum_rf(_local_subs(values)))

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        if self.reduction_rf:
            dJ_global = self.reduction_rf.derivative(
                adj_input=adj_input, apply_riesz=False)
            dJ_local = [dJ_global[i] for i in range(self._local_indices)]
        else:
            dJ_local = (
                self.functional._ad_init_zero(dual=True)._ad_init_object(adj_input))
            dJ_local = [dJ_local for _ in range(self.nlocal_inputs)]

        dJ_global = self._control._ad_init_zero(dual=True)
        _set_local_subs(dJ_global, dJ_local)
        if apply_riesz:
            dJ_global = dJ_global._ad_convert_riesz(
                dJ_global, riesz_map=self._control.riesz_map)
        return dJ_global

    @no_annotations
    def tlm(self, m_dot):
        if isinstance(m_dot, (list, tuple)):
            if len(m_dot) != 1:
                raise ValueError(
                    f"m_dot for {self(type).__name__} TLM must"
                    f" only have one entry, not {len(m_dot)}")
            m_dot = m_dot[0]
        if self.reduction_rf:
            return self.reduction_rf.tlm(
                self._allgather(_local_subs(m_dot)))
        else:
            return self.ensemble.allreduce(
                self._sum_rf.tlm(_local_subs(m_dot)))

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        if evaluate_tlm:
            self.tlm(m_dot)
        hessian_input = 0.0 if hessian_input is None else hessian_input
        return self.derivative(adj_input=hessian_input, apply_riesz=apply_riesz)

    @cached_property
    def _sum_rf(self):
        controls = [c._ad_init_zero()
                    for c in _local_subs(self.controls[0].control)]
        J = self.functional._ad_init_zero()
        annotating = annotate_tape()
        if not annotating:
            continue_annotation()
        with set_working_tape() as tape:
            J = _ad_sum(controls)
            rf = ReducedFunctional(
                J, [Control(c) for c in controls], tape=tape)
        if not annotating:
            pause_annotation()
        return rf

    @cached_property
    def _local_indices(self):
        fs = self.functional.function_space()
        offset = self.ensemble.ensemble_comm.exscan(fs.nlocal_spaces())
        return tuple(offset + i for i in range(fs.nlocal_spaces))

    def _allgather(self, vals):
        pass


class EnsembleBcastReducedFunctional(AbstractReducedFunctional):
    def __init__(self, functional, control, root=None, ensemble=None):
        self.functional = functional
        self._controls = Enlist(control)
        self._control = control
        self.root = root

        if isinstance(control.control, AdjFloat):
            if not isinstance(functional, EnsembleAdjVec):
                raise TypeError(
                    f"Functional for {type(self).__name__} must be an EnsembleAdjVec"
                    " if using an AdjFloat control.")
            if ensemble is not None and ensemble is not functional.ensemble:
                raise ValueError(
                    f"Ensemble provided to {type(self).__name__} must match"
                    " the ensemble of the functional.")
            self.ensemble = functional.ensemble
            self.nlocal_outputs = len(functional.subvec)

        elif isinstance(control.control, Function):
            if not isinstance(functional, EnsembleFunction):
                raise TypeError(
                    f"Functional for {type(self).__name__} must be an"
                    " EnsembleFunction if using a Function control.")
            if not all([f.function_space() == control.function_space()
                        for f in functional.subfunctions]):
                raise ValueError(
                    f"All subfunctions of the {type(self).__name__} functional"
                    " must have the same function space as the control.")
            if (ensemble is not None and ensemble is not functional.function_space().ensemble):
                raise ValueError(
                    f"Ensemble provided to {type(self).__name__} must match"
                    " the ensemble of the functional.")
            self.ensemble = functional.function_space().ensemble
            self.nlocal_outputs = len(functional.subfunctions)

        else:
            raise ValueError(
                f"Do not know how to handle a {type(control).__name__}"
                f" control for {type(self).__name__}.")

    @property
    def controls(self):
        return self._controls

    @no_annotations
    def __call__(self, values):
        if self.root is None:
            val = values
        else:
            val = self.ensemble.bcast(values, root=self.root)

        J = self.functional._ad_init_zero()
        _set_local_subs(J, [val for _ in range(self.nlocal_outputs)])
        return J

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        local_adj = _local_subs(adj_input)
        dJ = self.ensemble.allreduce(_ad_sum(local_adj))

        if apply_riesz:
            return self._control._ad_convert_riesz(
                dJ, riesz_map=self._control.riesz_map)
        else:
            return dJ

    @no_annotations
    def tlm(self, m_dot):
        return self(m_dot)

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        if evaluate_tlm:
            self.tlm(m_dot)
        return self.derivative(hessian_input, apply_riesz=apply_riesz)


class EnsembleTransformReducedFunctional(AbstractReducedFunctional):
    def __init__(self, rfs, functional, control, ensemble=None):
        self.rfs = rfs
        self.functional = functional
        self._controls = Enlist(control)

        if isinstance(functional, EnsembleFunction):
            if not all(isinstance(c.control, EnsembleFunction)
                       for c in self.controls):
                raise TypeError(
                    f"Controls for {type(self).__name__} must be EnsembleFunctions")
            if ensemble is not None and ensemble is not functional.function_space().ensemble:
                raise ValueError(
                    f"Ensemble provided to {type(self).__name__} must"
                    " match the ensemble of the functional.")

            self.ensemble = functional.function_space().ensemble
            self.nlocal_outputs = len(functional.subfunctions)

        elif isinstance(functional, EnsembleAdjVec):
            if ensemble is not None and ensemble is not functional.ensemble:
                raise ValueError(
                    f"Ensemble provided to {type(self).__name__} must"
                    " match the ensemble of the functional.")

            self.ensemble = ensemble
            self.nlocal_outputs = len(functional.subvec)

        else:
            raise TypeError(
                f"Functional for {type(self).__name__} must be either"
                " an EnsembleFunction or EnsembleAdjVec")

    @property
    def controls(self):
        return self._controls

    @no_annotations
    def __call__(self, values):
        J = self.functional._ad_init_zero()

        with self._local_data(values, output=J) as (local_vals, output):

            output([rf(rf.controls.delist(vals))
                    for rf, vals in zip(self.rfs, local_vals)])

        return J

    @no_annotations
    def tlm(self, m_dot):
        tlm = self.functional._ad_init_zero()

        with self._local_data(m_dot, output=tlm) as (local_mdots, output):

            output([rf.tlm(rf.controls.delist(md))
                    for rf, md in zip(self.rfs, local_mdots)])

        return tlm

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        dJ = self.controls.delist(
            [c.control._ad_init_zero(dual=not apply_riesz)
             for c in self.controls])

        with self._local_data(adj_input, output=dJ) as (local_adjs, output):

            output([rf.derivative(adj_input=adj[0],
                                  apply_riesz=apply_riesz)
                    for rf, adj in zip(self.rfs, local_adjs)])

        return dJ

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        if evaluate_tlm:
            self.tlm(m_dot)

        hessian = self.controls.delist(
            [c.control._ad_init_zero(dual=not apply_riesz)
             for c in self.controls])

        with self._local_data(hessian_input, output=hessian) as (local_hessians, output):

            output([rf.hessian(m_dot=None, evaluate_tlm=False,
                               hessian_input=hess[0],
                               apply_riesz=apply_riesz)
                    for rf, hess in zip(self.rfs, local_hessians)])

        return hessian

    @contextmanager
    def _local_data(self, data, output):
        local_data = self._global_to_local_data(data)
        output_transform = partial(
            self._local_to_global_data, global_data=output)
        yield local_data, output_transform

    def _local_to_global_data(self, local_data, global_data):
        # N local lists of length n -> n global lists of length N
        # [(1,), (2,), (3,)]->  [(1, 2, 3)]
        # [(1, 11), (2, 12), (3, 13)] -> [(1, 2, 3), (11, 12, 13)]

        for j, global_group in enumerate(Enlist(global_data)):
            local_group = [Enlist(local_group)[j]
                           for local_group in local_data]
            _set_local_subs(global_group, local_group)

        return global_data

    def _global_to_local_data(self, global_data):
        # n global lists of length N -> N local lists of length n
        # [(1, 2, 3)] -> [(1,), (2,), (3,)]
        # [(1, 2, 3), (11, 12, 13)] -> [(1, 11), (2, 12), (3, 13)]

        local_groups = [
            ld for ld in zip(*map(_local_subs, Enlist(global_data)))]
        return local_groups


class EnsembleReducedFunctional(AbstractReducedFunctional):
    """Enable solving simultaneously reduced functionals in parallel.

    Consider a functional :math:`J` and its gradient :math:`\\dfrac{dJ}{dm}`,
    where :math:`m` is the control parameter. Let us assume that :math:`J` is the sum of
    :math:`N` functionals :math:`J_i(m)`, i.e.,

    .. math::

        J = \\sum_{i=1}^{N} J_i(m).

    The gradient over a summation is a linear operation. Therefore, we can write the gradient
    :math:`\\dfrac{dJ}{dm}` as

    .. math::

        \\frac{dJ}{dm} = \\sum_{i=1}^{N} \\frac{dJ_i}{dm},

    The :class:`EnsembleReducedFunctional` allows simultaneous evaluation of :math:`J_i` and
    :math:`\\dfrac{dJ_i}{dm}`. After that, the allreduce :class:`~.ensemble.Ensemble`
    operation is employed to sum the functionals and their gradients over an ensemble
    communicator.

    If gather_functional is present, then all the values of J are communicated to all ensemble
    ranks, and passed in a list to gather_functional, which is a reduced functional that expects
    a list of that size of the relevant types.

    Parameters
    ----------
    rfs : list[pyadjoint.ReducedFunctional]
        The reduced functionals for each term Ji.
    functional : pyadjoint.OverloadedType
        An instance of an OverloadedType, usually :class:`pyadjoint.AdjFloat`.
        This should be the functional that we want to calculate.
    control : pyadjoint.Control or list of pyadjoint.Control
        A single or a list of Control instances, which you want to map to the functional.
    ensemble : Ensemble
        An instance of the :class:`~.ensemble.Ensemble`. It is used to communicate the
        functionals and their derivatives between the ensemble members.
    bcast_control : bool
        Whether the control (or a list of controls) is broadcast over the ensemble
        communicator ``Ensemble.ensemble_comm`` before evaluating the reduced functionals.
        If True, then the control must be a :class:`Function` or :class:`pyadjoint.AdjFloat`
        defined on ``Ensemble.comm``, and is logically identical across all ensemble members.
        If False, then the control must be a :class:`EnsembleFunction` defined on the ensemble,
        and is logically collective over all ensemble members.
    reduce_functional : bool
        Whether the functional is reduced over the ensemble communicator
        ``Ensemble.ensemble_comm`` after evaluating the reduced functionals.
        If True, then the functional must be a :class:`Function` or :class:`pyadjoint.AdjFloat`
        defined on ``Ensemble.comm``, and is logically identical across all ensemble members.
        If False, then the functional must be a :class:`EnsembleFunction` defined on the
        ensemble, and is logically collective over all ensemble members.

    See Also
    --------
    :class:`~.ensemble.Ensemble`, :class:`pyadjoint.ReducedFunctional`.

    Notes
    -----
    The functionals :math:`J_i` and the control must be defined over a common
    `ensemble.comm` communicator. To understand more about how ensemble parallelism
    works, please refer to the `Firedrake manual
    <https://www.firedrakeproject.org/ensemble_parallelism.html>`_.
    """
    def __init__(self, rfs, functional, control, ensemble=None,
                 bcast_control=True, reduce_functional=True):
        self.rfs = rfs
        self._controls = Enlist(control)
        self.functional = functional
        self.bcast_control = bcast_control
        self.reduce_functional = reduce_functional
        self.ensemble = ensemble

        if bcast_control:
            # controls are Functions, so need EnsembleFunctions for Transform input
            ensemble_controls = []
            for cntrl in self.controls:
                if isinstance(cntrl.control, float):
                    ensemble_controls.append(
                        EnsembleAdjVec(
                            [AdjFloat(0.) for _ in range(len(rfs))],
                            ensemble))

                elif isinstance(cntrl.control, (Function, Cofunction)):
                    control_space = EnsembleFunctionSpace(
                        [cntrl.control.function_space() for _ in range(len(rfs))],
                        ensemble)
                    ensemble_controls.append(
                        EnsembleFunction(control_space))

                else:
                    TypeError(
                        f"Don't know how to broadcast controls of type {type(cntrl.control).__name__}")

            self.bcast_rfs = [
                EnsembleBcastReducedFunctional(
                    ec, Control(c.control._ad_copy()), ensemble=ensemble)
                for ec, c in zip(ensemble_controls, self.controls)]

            ensemble_controls = [Control(ec) for ec in ensemble_controls]

        else:
            ensemble_controls = self.controls

        if reduce_functional:
            # functional is Function or AdjFloat, so need EnsembleFunction for Transform output
            if isinstance(functional, float):
                ensemble_functional = EnsembleAdjVec(
                    [functional._ad_copy() for _ in range(len(rfs))],
                    ensemble=ensemble)
                reduce_control = Control(ensemble_functional)

            elif isinstance(functional, Function):
                ensemble_functional_space = EnsembleFunctionSpace(
                    [rf.functional.function_space() for rf in rfs], ensemble)

                ensemble_functional = EnsembleFunction(ensemble_functional_space)
                reduce_control = Control(ensemble_functional)

            else:
                raise TypeError(
                    f"Do not know how to handle a {type(functional).__name__}"
                    f" control for {type(self).__name__}")

            self.reduce_rf = EnsembleReduceReducedFunctional(
                functional._ad_copy(), reduce_control, ensemble=ensemble)

        else:
            ensemble_functional = functional

        self.transform_rf = EnsembleTransformReducedFunctional(
            rfs, ensemble_functional,
            self.controls.delist(ensemble_controls),
            ensemble=ensemble)

    @property
    def controls(self):
        return self._controls

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
        if self.bcast_control:
            values = self.controls.delist(
                [bcast(val) for bcast, val in zip(self.bcast_rfs, Enlist(values))])

        J = self.transform_rf(values)

        if self.reduce_functional:
            J = self.reduce_rf(J)

        return J

    @no_annotations
    def derivative(self, adj_input=1.0, apply_riesz=False):
        """Compute derivatives of a functional with respect to the control parameters.

        Parameters
        ----------
        adj_input : float
            The adjoint input.
        apply_riesz: bool
            If True, apply the Riesz map of each control in order to return
            a primal gradient rather than a derivative in the dual space.

        Returns
        -------
            dJdm_total : pyadjoint.OverloadedType
            The result of Allreduce operations of ``dJdm_local`` into ``dJdm_total`` over the`Ensemble.ensemble_comm`.

        See Also
        --------
        :meth:`~.ensemble.Ensemble.allreduce`, :meth:`pyadjoint.ReducedFunctional.derivative`.
        """
        if self.reduce_functional:
            adj_input = self.reduce_rf.derivative(
                adj_input=adj_input, apply_riesz=False)

        transform_riesz = False if self.bcast_control else apply_riesz

        dJ = self.transform_rf.derivative(
            adj_input=adj_input, apply_riesz=transform_riesz)

        if self.bcast_control:
            dJ = self.controls.delist(
                [bcast.derivative(adj_input=dj, apply_riesz=apply_riesz)
                 for bcast, dj in zip(self.bcast_rfs, Enlist(dJ))])

        return dJ

    @no_annotations
    def tlm(self, m_dot):
        """Returns the action of the tangent linear model of the functional w.r.t. the
        control on a vector m_dot, around the last supplied value of the control.

        Args:
            m_dot ([OverloadedType]): The direction in which to compute the
                action of the tangent linear model.
            options (dict): A dictionary of options. To find a list of
                available options have a look at the specific control type.

        Returns:
            OverloadedType: The action of the tangent linear model in the direction m_dot.
                Should be an instance of the same type as the functional.
        """
        if self.bcast_control:
            m_dot = self.controls.delist(
                [bcast.tlm(md)
                 for bcast, md in zip(self.bcast_rfs, Enlist(m_dot))])

        tlm = self.transform_rf.tlm(m_dot)

        if self.reduce_functional:
            tlm = self.reduce_rf.tlm(tlm)

        return tlm

    @no_annotations
    def hessian(self, m_dot, hessian_input=None, evaluate_tlm=True, apply_riesz=False):
        """Returns the action of the Hessian of the functional w.r.t. the control on a vector m_dot.

        Using the second-order adjoint method, the action of the Hessian of the
        functional with respect to the control, around the last supplied value
        of the control, is computed and returned.

        Args:
            m_dot ([OverloadedType]): The direction in which to compute the
                action of the Hessian.
            options (dict): A dictionary of options. To find a list of
                available options have a look at the specific control type.
            hessian_input (OverloadedType): The value to start the Hessian accumulation from after the TLM calculation.
            evaluate_tlm (bool): Whether or not to compute the forward (tlm) part of the Hessian calculation.
                If False, assumes that the tape has already been populated with required TLM values.

        Returns:
            OverloadedType: The action of the Hessian in the direction m_dot.
                Should be an instance of the same type as the control.
        """
        if evaluate_tlm:
            self.tlm(m_dot)

        if self.reduce_functional:
            hessian_input = self.reduce_rf.hessian(
                m_dot=None, evaluate_tlm=False,
                hessian_input=hessian_input,
                apply_riesz=False)

        transform_riesz = False if self.bcast_control else apply_riesz

        hessian = self.transform_rf.hessian(
            m_dot=None, evaluate_tlm=False,
            hessian_input=hessian_input,
            apply_riesz=transform_riesz)

        if self.bcast_control:
            hessian = self.controls.delist(
                [bcast.hessian(m_dot=None, evaluate_tlm=False,
                               hessian_input=hess,
                               apply_riesz=apply_riesz)
                 for bcast, hess in zip(self.bcast_rfs, Enlist(hessian))])

        return hessian
