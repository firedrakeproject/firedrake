from functools import partial, cached_property
from contextlib import contextmanager
from pyadjoint import ReducedFunctional, Control, set_working_tape, AdjFloat, no_annotations
from pyadjoint.enlisting import Enlist
from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.ensemble.ensemble_function import (
    EnsembleFunctionSpace, EnsembleFunction, EnsembleCofunction, EnsembleFunctionBase)

__all__ = (
    "EnsembleReducedFunctional",
    "EnsembleBcastReducedFunctional",
    "EnsembleReduceReducedFunctional",
    "EnsembleTransformReducedFunctional",
)


def _intermediate_options(options):
    iopts = {} if options is None else options.copy()
    iopts["riesz_representation"] = None
    return iopts


def _local_subs(val):
    return val.subfunctions if isinstance(val, EnsembleFunctionBase) else val


def _set_local_subs(dst, src):
    dst = _local_subs(dst)
    for i, s in enumerate(src):
        if hasattr(dst[i], 'assign'):
            dst[i].assign(s)
        else:
            dst[i] = s
    return dst


class EnsembleReducedFunctional(ReducedFunctional):
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
    <https://www.firedrakeproject.org/parallelism.html#ensemble-parallelism>`_.
    """
    def __init__(self, rfs, functional, control, ensemble=None,
                 bcast_control=True, reduce_functional=True):
        self.rfs = rfs
        self.controls = Enlist(control)
        self.functional = functional
        self.bcast_control = bcast_control
        self.reduce_functional = reduce_functional
        self.ensemble = ensemble

        if bcast_control:
            # controls are Functions, so need EnsembleFunctions for Transform input
            ensemble_control_spaces = [
                EnsembleFunctionSpace([c.control.function_space()
                                       for _ in range(len(rfs))], ensemble)
                for c in self.controls]

            ensemble_controls = [
                Control(EnsembleFunction(space)) for space in ensemble_control_spaces]

            self.bcast_rfs = [
                EnsembleBcastReducedFunctional(
                    ec.control, Control(c.control._ad_copy()), ensemble=ensemble)
                for ec, c in zip(ensemble_controls, self.controls)]

        else:
            ensemble_controls = self.controls

        if reduce_functional:
            # functional is Function or AdjFloat, so need EnsembleFunction for Transform output
            if isinstance(functional, float):
                ensemble_functional = [functional._ad_copy()
                                       for _ in range(len(rfs))]
                reduce_control = [Control(f) for f in ensemble_functional]

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

    def derivative(self, adj_input=1.0, options=None):
        """Compute derivatives of a functional with respect to the control parameters.

        Parameters
        ----------
        adj_input : float
            The adjoint input.
        options : dict
            Additional options for the derivative computation.

        Returns
        -------
            dJdm_total : pyadjoint.OverloadedType
            The result of Allreduce operations of ``dJdm_local`` into ``dJdm_total`` over the`Ensemble.ensemble_comm`.

        See Also
        --------
        :meth:`~.ensemble.Ensemble.allreduce`, :meth:`pyadjoint.ReducedFunctional.derivative`.
        """
        iopts = _intermediate_options(options)
        if self.reduce_functional:
            adj_input = self.reduce_rf.derivative(adj_input=adj_input,
                                                  options=iopts)

        transform_options = iopts if self.bcast_control else options

        dJ = self.transform_rf.derivative(adj_input=adj_input,
                                          options=transform_options)

        if self.bcast_control:
            dJ = Enlist(dJ)
            dJ = dJ.delist(
                [bcast.derivative(adj_input=dj, options=options)
                 for bcast, dj in zip(self.bcast_rfs, dJ)])

        return dJ

    @no_annotations
    def tlm(self, m_dot, options=None):
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
        iopts = _intermediate_options(options)

        if self.bcast_control:
            m_dot = Enlist(m_dot)
            m_dot = m_dot.delist(
                [bcast.tlm(md, options=iopts)
                 for bcast, md in zip(self.bcast_rfs, m_dot)])

        tlm_options = iopts if self.reduce_functional else options

        tlm = self.transform_rf.tlm(m_dot, options=tlm_options)

        if self.reduce_functional:
            tlm = self.reduce_rf.tlm(tlm, options=options)

        return tlm

    def hessian(self, m_dot, options=None, hessian_input=0.0, evaluate_tlm=True):
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
        iopts = _intermediate_options(options)

        if evaluate_tlm:
            self.tlm(m_dot, options=iopts)

        if self.reduce_functional:
            hessian_input = self.reduce_rf.hessian(
                m_dot=None, evaluate_tlm=False,
                hessian_input=hessian_input,
                options=iopts)

        hessian_options = iopts if self.bcast_control else options

        hessian = self.transform_rf.hessian(
            m_dot=None, evaluate_tlm=False,
            hessian_input=hessian_input,
            options=hessian_options)

        if self.bcast_control:
            hessian = Enlist(hessian)
            hessian = hessian.delist(
                [bcast.hessian(m_dot=None, evaluate_tlm=False,
                               hessian_input=hess, options=options)
                 for bcast, hess in zip(self.bcast_rfs, hessian)])

        return hessian


class FunctionOrFloatMPIMixin:
    # Should be replaced by Ensemble passing through non-Functions to ensemble_comm
    def _bcast(self, val, root=None):
        if root is None:
            return val
        if isinstance(val, float):
            val = self.ensemble.ensemble_comm.bcast(val, root=root)
        elif isinstance(val, Function):
            val = self.ensemble.bcast(val, root=self.root)
        else:
            raise NotImplementedError(f"Functionals of type {type(val).__name__} are not supported.")
        return val

    def _reduce(self, vals, root=None):
        vals = Enlist(vals)
        for v in vals:
            if not isinstance(v, (Function, Cofunction, float)):
                raise NotImplementedError(
                    f"Functionals of type {type(v).__name__} are not supported.")

        if isinstance(vals[0], float):
            comm = self.ensemble.ensemble_comm
            local_sum = sum(vals)
            if root is None:
                return comm.allreduce(local_sum)
            else:
                return comm.reduce(local_sum, root=root)
        else:
            comm = self.ensemble
            global_sum = vals[0]._ad_convert_type(0)
            local_sum = vals[0]._ad_convert_type(0)
            local_sum.assign(sum(vals))
            if root is None:
                return comm.allreduce(local_sum, global_sum)
            else:
                return comm.reduce(local_sum, global_sum, root=root)

    def _allgather(self, vals):
        pass


class EnsembleBcastReducedFunctional(ReducedFunctional, FunctionOrFloatMPIMixin):
    def __init__(self, functional, control, root=None, ensemble=None):
        self.functional = functional
        self.controls = Enlist(control)
        self.control = control
        self.root = root

        # for AdjFloats functional is a list
        self._functionals = Enlist(functional)

        if isinstance(control.control, AdjFloat):
            if not all([isinstance(f, AdjFloat) for f in functional]):
                raise TypeError(
                    f"Functional for {type(self).__name__} must be a list of"
                    " AdjFloats if using an AdjFloat control.")
            if ensemble is None:
                raise ValueError(
                    f"Must provide {type(self).__name__} an Ensemble"
                    " if using an AdjFloat control.")
            self.ensemble = ensemble
            self.nlocal_outputs = len(functional)

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
            self.nlocal_outputs = len(self.functional.subfunctions)

        else:
            raise ValueError(
                f"Do not know how to handle a {type(control).__name__}"
                f" control for {type(self).__name__}.")

    def __call__(self, values):
        val = self._bcast(values, root=self.root)

        # "bcast" val into all elements of J (0+val = val)
        J = self._functionals.delist([
            functional._ad_convert_type(0)
            for functional in self._functionals
        ])
        for Ji in _local_subs(J):
            Ji._ad_iadd(val)

        _set_local_subs(J, [val for _ in range(self.nlocal_outputs)])

        return J

    def derivative(self, adj_input=1.0, options=None):
        if isinstance(self.control.control, Function):
            if not isinstance(adj_input, EnsembleCofunction):
                raise TypeError(
                    f"{type(self).__name__} needs an EnsembleCofunction"
                    f" adj_input, not a {type(adj_input).__name__}")
        else:
            if not all([isinstance(adj, AdjFloat) for adj in adj_input]):
                raise TypeError(
                    f"{type(self).__name__} needs a list of AdjFloats"
                    f" adj_input, not a {type(adj_input).__name__}")

        dJ = self._reduce(_local_subs(adj_input),
                          root=self.root)

        return self.control._ad_convert_type(dJ, options=options)

    @no_annotations
    def tlm(self, m_dot, options=None):
        return self.functional._ad_convert_type(self(m_dot), options=options)

    @no_annotations
    def hessian(self, m_dot, options=None, hessian_input=0.0, evaluate_tlm=True):
        if evaluate_tlm:
            iopts = _intermediate_options(options)
            self.tlm(m_dot, options=iopts)
        return self.derivative(hessian_input, options=options)


class EnsembleReduceReducedFunctional(ReducedFunctional, FunctionOrFloatMPIMixin):
    def __init__(self, functional, control, ensemble=None):
        if isinstance(functional, ReducedFunctional):
            self.reduction_rf = functional
            self.functional = self.reduction_rf.functional
        else:
            self.reduction_rf = None
            self.functional = functional

        self.controls = Enlist(control)
        self.control = control

        if isinstance(functional, AdjFloat):
            if not all([isinstance(c.control, AdjFloat) for c in control]):
                raise TypeError(
                    f"Control for {type(self).__name__} must be a list of"
                    " AdjFloats if using an AdjFloat functional.")
            if ensemble is None:
                raise ValueError(
                    f"Must provide {type(self).__name__} an Ensemble"
                    " if using an AdjFloat functional.")
            self.ensemble = ensemble
            self.nlocal_inputs = len(control)

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
            self.nlocal_inputs = len(self.control.subfunctions)

        else:
            raise ValueError(
                f"Do not know how to handle a {type(functional).__name__}"
                f" control for {type(self).__name__}.")

        _ = self._sum_rf

    @no_annotations
    def __call__(self, values):
        if self.reduction_rf:
            return self.reduction_rf(self._allgather(_local_subs(values)))
        else:
            return self._reduce(
                self._sum_rf(_local_subs(values)))

    def derivative(self, adj_input=1.0, options=None):
        if self.reduction_rf:
            dJ_global = self.reduction_rf.derivative(
                adj_input=adj_input, options=options)
            dJ_local = [dJ_global[i] for i in range(self._local_indices)]
        else:
            dJ_local = self.functional._ad_convert_type(adj_input, options=options)
            dJ_local = [dJ_local for _ in range(self.nlocal_inputs)]

        dJ_global = self.controls.delist(
            [c.control._ad_convert_type(0, options=options)
             for c in self.controls])

        _set_local_subs(dJ_global, dJ_local)

        return dJ_global

    @no_annotations
    def tlm(self, m_dot, options=None):
        if self.reduction_rf:
            return self.reduction_rf.tlm(
                self._allgather(_local_subs(m_dot)), options=options)
        else:
            return self._reduce(
                self._sum_rf.tlm(_local_subs(m_dot), options=options))

    @no_annotations
    def hessian(self, m_dot, options=None, hessian_input=0.0, evaluate_tlm=True):
        if evaluate_tlm:
            iopts = _intermediate_options(options)
            self.tlm(m_dot, options=iopts)
        return self.derivative(hessian_input, options=options)

    @cached_property
    def _sum_rf(self):
        subs = self.controls.delist(
            [_local_subs(c.control) for c in self.controls])
        controls = [c._ad_convert_type(0.) for c in subs]
        J = self.functional._ad_convert_type(0.)
        with set_working_tape() as tape:
            if isinstance(J, float):
                J = sum(controls)
            else:
                for c in controls:
                    J = J._ad_add(c)
            rf = ReducedFunctional(
                J, [Control(c) for c in controls], tape=tape)
        return rf

    @cached_property
    def _local_indices(self):
        fs = self.functional.function_space()
        offset = self.ensemble.ensemble_comm.exscan(fs.nlocal_spaces())
        return tuple(offset + i for i in range(fs.nlocal_spaces))


class EnsembleTransformReducedFunctional(ReducedFunctional):
    def __init__(self, rfs, functional, control, ensemble=None):
        self.rfs = rfs
        self.functional = functional
        self.controls = Enlist(control)

        # AdjFloat functional is a list but need to treat as a single object
        self._flist = Enlist(functional)

        # list[EFunction] -> EFunction
        if isinstance(functional, EnsembleFunction):
            if not all(isinstance(c.control, EnsembleFunction)
                       for c in self.controls):
                raise TypeError(
                    f"Controls for {type(self).__name__} must be EnsembleFunctions")
            if ensemble is not None:
                ensembles = [c.control.function_space().ensemble
                             for c in self.controls]
                ensembles.append(functional.function_space().ensemble)
                if not all(e is ensemble for e in ensembles):
                    raise ValueError(
                        f"Ensemble provided to {type(self).__name__} must"
                        " match the ensemble of the controls and functional.")

            self.ensemble = functional.function_space().ensemble
            self.nlocal_outputs = len(functional.subfunctions)

        # list[EFunction] -> list[AdjFloat]
        elif isinstance(functional, list):
            if not all(isinstance(f, AdjFloat) for f in functional):
                raise TypeError(
                    f"Functional for {type(self).__name__} must be either"
                    " an EnsembleFunction or a list of AdjFloats")
            if ensemble is None:
                raise ValueError(
                    f"Must provide {type(self).__name__} an Ensemble"
                    " if using an AdjFloat functional.")

            self.ensemble = ensemble
            self.nlocal_outputs = len(functional)

        else:
            raise TypeError(
                f"Functional for {type(self).__name__} must be either"
                " an EnsembleFunction or a list of AdjFloats")

    @no_annotations
    def __call__(self, values):
        # For AdjFloat output, a single functional is a list of AdjFloats
        J = [f._ad_convert_type(0) for f in self._flist]
        if self._flist.listed:
            J = [J]

        with self._local_data(data=Enlist(values), output=J) as (local_vals, output):

            output([rf(rf.controls.delist(vals))
                    for rf, vals in zip(self.rfs, local_vals)])

        return J[0]

    @no_annotations
    def tlm(self, m_dot, options=None):
        tlm = [f._ad_convert_type(0) for f in self._flist]
        if self._flist.listed:
            tlm = [tlm]

        with self._local_data(data=Enlist(m_dot), output=tlm) as (local_mdots, output):

            output([rf.tlm(rf.controls.delist(md), options=options)
                    for rf, md in zip(self.rfs, local_mdots)])

        return tlm[0]

    def derivative(self, adj_input=1.0, options=None):
        dJ = [c.control._ad_convert_type(0, options=options)
              for c in self.controls]

        with self._local_data(data=[adj_input], output=dJ) as (local_adjs, output):
            output([rf.derivative(adj_input=adj[0], options=options)
                    for rf, adj in zip(self.rfs, local_adjs)])

        return self.controls.delist(dJ)

    @no_annotations
    def hessian(self, m_dot, options=None, hessian_input=0.0, evaluate_tlm=True):
        if evaluate_tlm:
            iopts = _intermediate_options(options)
            self.tlm(m_dot, options=iopts)

        hessian = [
            c.control._ad_convert_type(0, options=options)
            for c in self.controls]

        with self._local_data(data=[hessian_input], output=hessian) as (local_hessians, output):
            output([rf.hessian(m_dot=None, evaluate_tlm=False,
                               hessian_input=hess[0], options=options)
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

        for j, global_group in enumerate(global_data):
            local_group = [Enlist(local_group)[j]
                           for local_group in local_data]
            _set_local_subs(global_group, local_group)

        return global_data

    def _global_to_local_data(self, global_data):
        # n global lists of length N -> N local lists of length n
        # [(1, 2, 3)] -> [(1,), (2,), (3,)]
        # [(1, 2, 3), (11, 12, 13)] -> [(1, 11), (2, 12), (3, 13)]

        global_groups = [
            ld for ld in zip(*map(_local_subs, global_data))]

        local_groups = global_groups
        return local_groups
