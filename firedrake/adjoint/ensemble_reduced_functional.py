from functools import partial, cached_property
from contextlib import contextmanager
from pyadjoint import ReducedFunctional, Control, set_working_tape
from pyadjoint.enlisting import Enlist
from pyop2.mpi import MPI
from firedrake.function import Function
from firedrake.ensemble.ensemble_function import EnsembleFunction, EnsembleFunctionSpace


def _maybe_ensemble_function_input(val, n):
    if hasattr(val, 'subfunctions'):
        return val.subfunctions
    else:
        return [val for _ in range(n)]


class EnsembleReducedFunctionalOriginal(ReducedFunctional):
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
    J : pyadjoint.OverloadedType
        An instance of an OverloadedType, usually :class:`pyadjoint.AdjFloat`.
        This should be the functional that we want to reduce.
    control : pyadjoint.Control or list of pyadjoint.Control
        A single or a list of Control instances, which you want to map to the functional.
    ensemble : Ensemble
        An instance of the :class:`~.ensemble.Ensemble`. It is used to communicate the
        functionals and their derivatives between the ensemble members.
    scatter_control : bool
        Whether scattering a control (or a list of controls) over the ensemble communicator
        ``Ensemble.ensemble comm``.
    gather_functional : An instance of the :class:`pyadjoint.ReducedFunctional`.
        that takes in all of the Js.
    derivative_components : list of int
        The indices of the controls that the derivative should be computed with respect to.
        If present, it overwrites ``derivative_cb_pre`` and ``derivative_cb_post``.
    scale : float
        A scaling factor applied to the functional and its gradient(with respect to the control).
    tape : pyadjoint.Tape
        A tape object that the reduced functional will use to evaluate the functional and
        its gradients (or derivatives).
    eval_cb_pre : :func:
        Callback function before evaluating the functional. Input is a list of Controls.
    eval_cb_pos : :func:
        Callback function after evaluating the functional. Inputs are the functional value
        and a list of Controls.
    derivative_cb_pre : :func:
        Callback function before evaluating gradients (or derivatives). Input is a list of
        gradients (or derivatives). Should return a list of Controls (usually the same list as
        the input) to be passed to :func:`pyadjoint.compute_gradient`.
    derivative_cb_post : :func:
        Callback function after evaluating derivatives. Inputs are the functional, a list of
        gradients (or derivatives), and controls. All of them are the checkpointed versions.
        Should return a list of gradients (or derivatives) (usually the same list as the input)
        to be returned from ``self.derivative``.
    hessian_cb_pre : :func:
        Callback function before evaluating the Hessian. Input is a list of Controls.
    hessian_cb_post : :func:
        Callback function after evaluating the Hessian. Inputs are the functional, a list of
        Hessian, and controls.

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
    def __init__(self, J, control, ensemble, scatter_control=True,
                 gather_functional=None, derivative_components=None,
                 scale=1.0, tape=None, eval_cb_pre=lambda *args: None,
                 eval_cb_post=lambda *args: None,
                 derivative_cb_pre=lambda controls: controls,
                 derivative_cb_post=lambda checkpoint, derivative_components, controls: derivative_components,
                 hessian_cb_pre=lambda *args: None, hessian_cb_post=lambda *args: None):
        super(EnsembleReducedFunctional, self).__init__(
            J, control, derivative_components=derivative_components,
            scale=scale, tape=tape, eval_cb_pre=eval_cb_pre,
            eval_cb_post=eval_cb_post, derivative_cb_pre=derivative_cb_pre,
            derivative_cb_post=derivative_cb_post,
            hessian_cb_pre=hessian_cb_pre, hessian_cb_post=hessian_cb_post)

        self.ensemble = ensemble
        self.scatter_control = scatter_control
        self.gather_functional = gather_functional

    def _allgather_J(self, J):
        if isinstance(J, float):
            vals = self.ensemble.ensemble_comm.allgather(J)
        elif isinstance(J, Function):
            #  allgather not implemented in ensemble.py
            vals = []
            for i in range(self.ensemble.ensemble_comm.size):
                J0 = J.copy(deepcopy=True)
                vals.append(self.ensemble.bcast(J0, root=i))
        else:
            raise NotImplementedError(f"Functionals of type {type(J).__name__} are not supported.")
        return vals

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
        local_functional = super(EnsembleReducedFunctional, self).__call__(values)
        ensemble_comm = self.ensemble.ensemble_comm
        if self.gather_functional:
            controls_g = self._allgather_J(local_functional)
            total_functional = self.gather_functional(controls_g)
        # if gather_functional is None then we do a sum
        elif isinstance(local_functional, float):
            total_functional = ensemble_comm.allreduce(sendobj=local_functional, op=MPI.SUM)
        elif isinstance(local_functional, Function):
            total_functional = type(local_functional)(local_functional.function_space())
            total_functional = self.ensemble.allreduce(local_functional, total_functional)
        else:
            raise NotImplementedError("This type of functional is not supported.")
        return total_functional

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

        if self.gather_functional:
            dJg_dmg = self.gather_functional.derivative(adj_input=adj_input,
                                                        options=options)
            i = self.ensemble.ensemble_comm.rank
            adj_input = dJg_dmg[i]

        dJdm_local = super(EnsembleReducedFunctional, self).derivative(adj_input=adj_input, options=options)

        if self.scatter_control:
            dJdm_local = Enlist(dJdm_local)
            dJdm_total = []

            for dJdm in dJdm_local:
                if not isinstance(dJdm, (Function, float)):
                    raise NotImplementedError("This type of gradient is not supported.")

                dJdm_total.append(
                    self.ensemble.allreduce(dJdm, type(dJdm)(dJdm.function_space()))
                    if isinstance(dJdm, Function)
                    else self.ensemble.ensemble_comm.allreduce(sendobj=dJdm, op=MPI.SUM)
                )
            return dJdm_local.delist(dJdm_total)
        return dJdm_local

    def hessian(self, m_dot, options=None):
        """The Hessian is not yet implemented for ensemble reduced functional.

        Raises:
            NotImplementedError: This method is not yet implemented for ensemble reduced functional.
        """
        raise NotImplementedError("Hessian is not yet implemented for ensemble reduced functional.")


class EnsembleReducedFunctionalAdjFloat(ReducedFunctional):
    def __init__(self, rfs, functional, controls, ensemble, reduce_functional=None):
        self.rfs = rfs
        self.functional = functional
        self.controls = Enlist(controls)
        self.ensemble = ensemble
        self.reduce_functional = reduce_functional

    def __call__(self, values):
        Jlocals = [rf(values) for rf in self.rfs]

        if self.reduce_functional:
            J = self.reduce_functional(
                self.ensemble.ensemble_comm.allgather(Jlocals))
        else:
            J = self.ensemble.ensemble_comm.allreduce(sum(Jlocals))
        return J

    def tlm(self, m_dot, options=None):
        intermediate_options = None
        tlm_options = intermediate_options if self.reduce_functional else options

        tlm_local = [rf.tlm(m_dot, options=tlm_options) for rf in self.rfs]

        if self.reduce_functional:
            tlm = self.reduce_functional.tlm(
                self.ensemble.ensemble_comm.allgather(tlm_local),
                options=options)
        else:
            tlm = self.ensemble.ensemble_comm.allreduce(sum(tlm_local))

        return tlm

    def derivative(self, adj_input=1.0, options=None):

        if self.reduce_functional:
            intermediate_options = None
            adj_global = self.reduce_functional.derivative(
                adj_input=adj_input, options=intermediate_options)

            adj_inputs = [adj_global[i] for i in self._local_indices]
        else:
            adj_inputs = [adj_input for _ in range(len(self.rfs))]

        dJ_local = [Enlist(rf.derivative(adj, options=options))
                    for rf, adj in zip(self.rfs, adj_inputs)]

        dJ_global = []
        for djls in zip(*dJ_local):
            dJ_global.append(self.ensemble.ensemble_comm.allreduce(sum(djls)))

        return self.controls.delist(dJ_global)

    def hessian(self, m_dot, options=None, hessian_value=0.0, evaluate_tlm=True):
        intermediate_options = None

        if evaluate_tlm:
            self.tlm(m_dot, options=intermediate_options)

        if self.reduce_functional:
            hess_global = self.reduce_functional.hessian(
                m_dot=None, evaluate_tlm=False,
                hessian_value=hessian_value,
                options=intermediate_options)
            hessian_values = [hess_global[i] for i in self._local_indices]
        else:
            hessian_values = [hessian_value for _ in range(len(self.rfs))]

        hessian_local = [
            Enlist(rf.hessian(m_dot=None, evaluate_tlm=False,
                              hessian_value=hess, options=options))
            for rf, hess in zip(self.rfs, hessian_values)]

        hessian_global = []

        for hls in zip(*hessian_local):
            hessian_global.append(self.ensemble.ensemble_comm.allreduce(sum(hls)))

        return self.controls.delist(hessian_global)

    @cached_property
    def _local_indices(self):
        fs = self.functional.function_space()
        offset = self.ensemble.ensemble_comm.exscan(fs.nlocal_spaces())
        return tuple(offset + i for i in range(fs.nlocal_spaces))


class EnsembleReducedFunctional:
    def __init__(self, rfs, functional, control, ensemble=None,
                 bcast_control=False,
                 reduce_functional=False):
        self.rfs = rfs
        self.controls = Enlist(control)
        self.functional = functional
        self.bcast_control = bcast_control
        self.reduce_functional = reduce_functional
        self.ensemble = ensemble

        if bcast_control:
            control_spaces = (
                EnsembleFunctionSpace([c.function_space()
                                       for _ in range(len(rfs))], ensemble)
                for c in self.controls)

            ensemble_controls = (
                EnsembleFunction(space) for space in control_spaces)

            self.bcast_rfs = (
                EnsembleBcastReducedFunctional(ec, Control(c))
                for ec, c in zip(ensemble_controls, self.controls))
        else:
            ensemble_controls = self.controls

        if reduce_functional:
            functional_space = EnsembleFunctionSpace(
                [rf.functional.function_space() for rf in rfs], ensemble)
            ensemble_functional = EnsembleFunction(functional_space)
        else:
            ensemble_functional = functional

        self.full_rf = EnsembleReducedFunctionalImpl(
            rfs, ensemble_functional, self.controls.delist(ensemble_controls))

        if reduce_functional:
            self.reduce_rf = EnsembleReduceReducedFunctional(
                reduce_functional, Control(ensemble_functional._ad_copy()))

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
            values = Enlist(values)
            values = values.delist(
                [bcast(val)
                 for bcast, val in zip(self.bcast_rfs, values)])

        J = self.full_rf(values)

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
        intermediate_options = None
        if self.reduce_functional:
            adj_input = self.reduce_rf.derivative(adj_input=adj_input,
                                                  options=intermediate_options)

        derivative_options = intermediate_options if self.bcast_control else options

        dJ = self.full_rf.derivative(
            adj_input=adj_input,
            options=derivative_options)

        if self.bcast_control:
            dJ = Enlist(dJ)
            dJ = dJ.delist(
                [bcast.derivative(adj_input=dj, options=options)
                 for bcast, dj in zip(self.bcast_rfs, dJ)])

        return dJ

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
        intermediate_options = None

        if self.bcast_control:
            m_dot = Enlist(m_dot)
            m_dot = m_dot.delist(
                [bcast.tlm(md, options=intermediate_options)
                 for bcast, md in zip(self.bcast_rfs, m_dot)])

        tlm_options = intermediate_options if self.reduce_functional else options

        tlm = self.full_rf.tlm(m_dot, options=tlm_options)

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
        intermediate_options = None

        if evaluate_tlm:
            self.tlm(m_dot, options=intermediate_options)

        if self.reduce_functional:
            hessian_input = self.reduce_rf.hessian(
                m_dot=None, evaluate_tlm=False,
                hessian_input=hessian_input,
                options=intermediate_options)

        hessian_options = intermediate_options if self.reduce_functional else options

        hessian = self.full_rf.hessian(
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

    def _reduce(self, vals, result_type, root=None):
        vals = Enlist(vals)
        for v in vals:
            if not isinstance(v, (Function, float)):
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
            global_sum = Function(result_type.function_space())
            local_sum = Function(result_type.function_space())
            local_sum.assign(sum(vals))
            if root is None:
                return comm.allreduce(local_sum, global_sum)
            else:
                return comm.reduce(local_sum, global_sum, root=root)

    def _allgather(self, vals):
        pass


class EnsembleBcastReducedFunctional(FunctionOrFloatMPIMixin):
    def __init__(self, functional, control, root=None):
        self.functional = functional
        self.ensemble = functional.ensemble
        self.controls = Enlist(control)
        self.control = Enlist(control)
        self.root = root

    def __call__(self, values):
        val = self._bcast(values, root=self.root)
        for subfunc in self.functional.subfunctions:
            subfunc.assign(val)
        return self.functional._ad_copy()

    def derivative(self, adj_input=1.0, options=None):
        adj_inputs = _maybe_ensemble_function_input(
            adj_input, len(self.functional.subfunctions))
        return self._reduce(adj_inputs, result_type=self.control, root=self.root)

    def tlm(self, m_dot, options=None):
        return self.control._ad_convert_type(self(m_dot), options=options)

    def hessian(self, m_dot, options=None, hessian_input=0.0, evaluate_tlm=True):
        if evaluate_tlm:
            intermediate_options = None
            self.tlm(m_dot, options=intermediate_options)
        hessian_inputs = _maybe_ensemble_function_input(
            hessian_input, len(self.functional.subfunctions))
        return self.derivative(hessian_inputs, options=options)


class EnsembleReduceReducedFunctional(FunctionOrFloatMPIMixin):
    def __init__(self, functional, control):
        if isinstance(functional, ReducedFunctional):
            self.reduction_rf = functional
            self.functional = self.reduction_rf.functional
        else:
            self.reduction_rf = None
            self.functional = functional
        self.controls = Enlist(control)
        self.control = control
        self.ensemble = control.ensemble

    def __call__(self, values):
        if self.reduction_rf:
            return self.reduction_rf(self._allgather(values))
        else:
            return self._reduce(self._sum_rf(values.subfunctions),
                                result_type=self.functional)

    def derivative(self, adj_input=1.0, options=None):
        if self.reduction_rf:
            dJ_global = self.reduction_rf.derivative(
                adj_input=adj_input, options=options)
            dJ_local = [dJ_global[i] for i in range(self._local_indices)]
        else:
            dJ_local = self._sum_rf.derivative(adj_input, options=options)

        dJ_global = EnsembleReducedFunctionalImpl._new_efuncs(
            self, self.control.function_space(), options=options)

        for dJg, dJl in zip(dJ_global.subfunctions, dJ_local):
            dJg.assign(dJl)

        return dJ_global

    def tlm(self, m_dot, options=None):
        return self.control._ad_convert_type(self(m_dot), options=options)

    def hessian(self, m_dot, options=None, hessian_input=0.0, evaluate_tlm=True):
        if evaluate_tlm:
            intermediate_options = None
            self.tlm(m_dot, options=intermediate_options)
        return self.derivative(hessian_input, options=options)

    @cached_property
    def _sum_rf(self):
        controls = self.control._ad_copy().subfunctions
        with set_working_tape() as tape:
            J = Function(self.functional.function_space())
            J.assign(sum(controls))
            return ReducedFunctional(
                J, [Control(c) for c in controls], tape=tape)

    @cached_property
    def _local_indices(self):
        fs = self.functional.function_space()
        offset = self.ensemble.ensemble_comm.exscan(fs.nlocal_spaces())
        return tuple(offset + i for i in range(fs.nlocal_spaces))


class EnsembleReducedFunctionalImpl:
    def __init__(self, rfs, functional, controls):
        self.rfs = rfs
        self.controls = Enlist(controls)
        self.ensemble = self.controls[0].ensemble

        self.functional = functional

        self.functional_space = functional.function_space()
        self.control_spaces = [c.control.function_space()
                               for c in self.controls]

    def __call__(self, values):
        J = self.functional

        with self._local_data(data=values, output=J) as (local_vals, output):
            output([rf(rf.controls.delist(vals))
                    for rf, vals in zip(self.rfs, local_vals)])
        return J._ad_copy()

    def tlm(self, m_dot, options=None):
        tlm = self._new_efuncs(self.functional_space, options)

        with self._local_data(data=m_dot, output=tlm) as (local_mdot, output):
            output([rf.tlm(rf.controls.delist(md), options=options)
                    for rf, md in zip(self.rfs, local_mdot)])
        return tlm

    def derivative(self, adj_input=1.0, options=None):
        # adj_input may need to be bcast to a list
        adj_inputs = _maybe_ensemble_function_input(
            adj_input, len(self.functional.subfunctions))

        dJ = self._new_efuncs(self.control_spaces, options)

        with self._local_data(data=adj_inputs, output=dJ) as (local_adjs, output):
            output([rf.derivative(adj, options=options)
                    for rf, adj in zip(self.rfs, local_adjs)])
        return dJ

    def hessian(self, m_dot, options=None, hessian_input=0.0, evaluate_tlm=True):
        if evaluate_tlm:
            intermediate_options = None
            self.tlm(m_dot, options=intermediate_options)

        # hessian_input may need to be bcast to a list
        hessian_inputs = _maybe_ensemble_function_input(
            hessian_input, len(self.functional.subfunctions))

        hessian = self._new_efuncs(self.control_spaces, options)

        with self._local_data(data=hessian_inputs, output=hessian) as (local_hessian, output):
            output([rf.hessian(m_dot=None, evaluate_tlm=False,
                               hessian_input=hess, options=options)
                    for rf, hess in zip(self.rfs, local_hessian)])
        return hessian

    @contextmanager
    def _local_data(self, data, output):
        local_data = self._global_to_local_data(data)
        output_transform = partial(
            self._local_to_global_data, global_data=output)
        yield local_data, output_transform

    def _local_to_global_data(self, local_data, global_data):
        local_data = Enlist(local_data)
        global_data = Enlist(global_data)

        global_groups = [*zip(*(gdata.subfunctions
                                for gdata in global_data))]

        for new_group, dst_group in zip(local_data, global_groups):
            for new, dst in zip(new_group, dst_group):
                dst.assign(new)

        return global_data.delist()

    def _global_to_local_data(self, global_data, local_data=None):
        local_data = Enlist(local_data)
        global_data = Enlist(global_data)

        global_groups = [*zip(*(gdata.subfunctions
                                for gdata in Enlist(global_data)))]

        if local_data:
            for new_group, dst_group in zip(global_groups, local_data):
                for new, dst in zip(new_group, dst_group):
                    dst.assign(new)
            global_groups = local_data
        return global_data.delist(global_groups)

    def _new_efuncs(self, spaces, options=None):
        options = options or {}
        dual = (options.get("riesz_representation", "") in (None, "l2"))
        spaces = Enlist(spaces)
        return spaces.delist([EnsembleFunction(space.dual() if dual else space)
                              for space in spaces])
