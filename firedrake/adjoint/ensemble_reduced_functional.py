from pyadjoint import ReducedFunctional
from pyadjoint.enlisting import Enlist
from pyop2.mpi import MPI
import numpy as np

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

    If gather_functional is present, then all the values of J are communicated to all ensemble ranks, and passed in a list to gather_functional, which is a reduced functional that expects a list of that size of the relevant types.

    Parameters
    ----------
    J : pyadjoint.OverloadedType
        An instance of an OverloadedType, usually :class:`pyadjoint.AdjFloat`,
        or, a list of those (which should all have the same type).
        This should be the functional that we want to reduce.
    control : pyadjoint.Control or list of pyadjoint.Control
        A single or a list of Control instances, which you want to map to the functional(s).
    ensemble : Ensemble
        An instance of the :class:`~.ensemble.Ensemble`. It is used to communicate the
        functionals and their derivatives between the ensemble members.
    scatter_control : bool
        Whether scattering a control (or a list of controls) over the ensemble communicator
        ``Ensemble.ensemble comm``.
    gather_functional : An instance of the :class:`pyadjoint.ReducedFunctional`.
        that takes in all of the Js.


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
                 gather_functional=None, derivative_components=None):
        self.ensemble = ensemble
        self.controls = Enlist(control)
        J = Enlist(J)
        self.functional = J
        self.Jhats = []
        for i, J in enumerate(self.functional):
            self.Jhats.append(ReducedFunctional(J, self.controls),
                              derivative_components=derivative_components)
        self.scatter_control = scatter_control
        self.gather_functional = gather_functional

    def _allgather_J(self, J):
        vals = []
        J = Enlist(J)
        sizes = self.ensemble.ensemble_comm.allgather(len(J))
        rank = self.ensemble.ensemble_comm.rank
        # do a bit of type checking as things can hang and it is hard
        # to diagnose
        if not all(isinstance(J1, type(J[0])) for J1 in J):
            raise TypeError("All items in J should have the same type.")
        Jtype = type(J[0])
        Jtypes = self.ensemble.ensemble_comm.allgather(Jtype)
        if not all(issubclass(Jtype, Jtype1) for Jtype1 in Jtypes):
            raise TypeError("All items in J should have the same type globally.")
        # allgather a flattened list of all of the
        # functional values
        for i, size in enumerate(sizes):
            for j in range(size):
                if issubclass(Jtype, float):
                    if i == rank:
                        Jsend = J[j]
                    else:
                        Jsend = 0.
                    vals.append(self.ensemble.ensemble_comm.bcast(Jsend, root=i))
                elif issubclass(Jtype, firedrake.Function):
                    if i == rank:
                        Jsend = J[j].copy(deepcopy=True)
                    else:
                        Jsend = J[0].copy(deepcopy=True)
                    vals.append(self.ensemble.bcast(Jsend, root=i))
                else:
                    raise NotImplementedError("This type of functional is not supported: " + str(Jtype))
        return vals

    def __call__(self, values):
        local_functional = []
        for i, Jhat in enumerate(self.Jhats):
            local_functional.append(Jhat(values))
        ensemble_comm = self.ensemble.ensemble_comm
        if self.gather_functional:
            Controls_g = self._allgather_J(local_functional)
            total_functional = self.gather_functional(Controls_g)
        # if gather_functional is None then we do a sum
        else:
            if len(local_functional) > 1:
                raise TypeError("Only scalar functionals supported.")
            local_functional = local_functional[0]
            if isinstance(local_functional, float):
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
            adj_input = self.gather_functional.derivative(adj_input=adj_input,
                                                          options=options)
        i = self.ensemble.ensemble_comm.rank
        sizes = self.ensemble.ensemble_comm.allgather(len(self.functional))
        k = int(np.sum(sizes[:i]))
        for j, Jhat in enumerate(self.Jhats):
            if self.gather_functional:
                der = Jhat.derivative(adj_input=adj_input[k], options=options)
            else:
                der = Jhat.derivative(adj_input=adj_input, options=options)
            # we have the same controls for all local elements of the list
            # so the controls must be added
            if j == 0:
                if isinstance(der, list):
                    dJdm_local = []
                    for l, derl in enumerate(der):
                        dJdm_local.append(derl)
                else:
                    dJdm_local = der
            else:
                if isinstance(der, list):
                    for l, derl in enumerate(der):
                        dJdm_local[l] += derl
                else:
                    dJdm_local += der
            k += 1

        if self.scatter_control:
            dJdm_total = []
            dJdm_local = Enlist(dJdm_local)
            for dJdm in dJdm_local:
                if not isinstance(dJdm, (firedrake.Function, float)):
                    raise NotImplementedError("This type of gradient is not supported.")

                dJdm_total.append(
                    self.ensemble.allreduce(dJdm, type(dJdm)(dJdm.function_space()))
                    if isinstance(dJdm, firedrake.Function)
                    else self.ensemble.ensemble_comm.allreduce(sendobj=dJdm, op=MPI.SUM)
                )
            return self.controls.delist(dJdm_total)
        return self.controls.delist(dJdm_local)

    def hessian(self, m_dot, options=None):
        """The Hessian is not yet implemented for ensemble reduced functional.

        Raises:
            NotImplementedError: This method is not yet implemented for ensemble reduced functional.
        """
        raise NotImplementedError("Hessian is not yet implemented for ensemble reduced functional.")
