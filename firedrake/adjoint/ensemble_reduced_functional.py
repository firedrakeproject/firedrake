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
    allreduce : bool
        If True, the functionals and their derivatives are summed over the ensemble communicator
        `ensemble.ensemble_comm`. If False, the functionals and their derivatives computed in
        multiple ranks are not summed.

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
    def __init__(self, J, control, ensemble, allreduce=True):
        super(EnsembleReducedFunctional, self).__init__(J, control)
        self.ensemble = ensemble
        self.allreduce = allreduce

    def __call__(self, values):
        local_functional = super(EnsembleReducedFunctional, self).__call__(values)
        if not self.allreduce:
            return local_functional
        if isinstance(local_functional, float):
            total_functional = self.ensemble.ensemble_comm.allreduce(sendobj=local_functional, op=MPI.SUM)
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
        dJdm_local = super(EnsembleReducedFunctional, self).derivative(adj_input=adj_input, options=options)
        dJdm_local = Enlist(dJdm_local)
        dJdm_total = []
        if not self.allreduce:
            return dJdm_local
        for dJdm in dJdm_local:
            if not isinstance(dJdm, (firedrake.Function, float)):
                raise NotImplementedError("This type of gradient is not supported.")

            dJdm_total.append(
                self.ensemble.allreduce(dJdm, type(dJdm)(dJdm.function_space()))
                if isinstance(dJdm, firedrake.Function)
                else self.ensemble.ensemble_comm.allreduce(sendobj=dJdm, op=MPI.SUM)
            )
        return dJdm_local.delist(dJdm_total)

    def hessian(self, m_dot, options=None):
        """The Hessian is not yet implemented for ensemble reduced functional.

        Raises:
            NotImplementedError: This method is not yet implemented for ensemble reduced functional.
        """
        raise NotImplementedError("Hessian is not yet implemented for ensemble reduced functional.")
