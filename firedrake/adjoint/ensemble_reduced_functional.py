from pyadjoint import ReducedFunctional
from pyadjoint.enlisting import Enlist
from pyop2.mpi import MPI

import firedrake


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
    <https://www.firedrakeproject.org/parallelism.html#id8>`_.
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
        elif isinstance(J, firedrake.Function):
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
        elif isinstance(local_functional, firedrake.Function):
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
                if not isinstance(dJdm, (firedrake.Function, float)):
                    raise NotImplementedError("This type of gradient is not supported.")

                dJdm_total.append(
                    self.ensemble.allreduce(dJdm, type(dJdm)(dJdm.function_space()))
                    if isinstance(dJdm, firedrake.Function)
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
