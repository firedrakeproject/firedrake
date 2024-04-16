from pyadjoint import ReducedFunctional
from pyop2.mpi import MPI

import firedrake


class EnsembleReducedFunctional(ReducedFunctional):
    """Enable solving simultaneously reduced functionals in parallel.

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

    See Also
    --------
    :class:`~.ensemble.Ensemble`, :class:`pyadjoint.ReducedFunctional`.

    Notes
    -----
    To understand more about how ensemble parallelism works, please refer to the Firedrake
    `documentation <https://www.firedrakeproject.org/parallelism.html#id8>`_.
    """
    def __init__(self, J, control, ensemble):
        super(EnsembleReducedFunctional, self).__init__(J, control)
        self.ensemble = ensemble

    def __call__(self, values):
        local_functional = super(EnsembleReducedFunctional, self).__call__(values)
        if isinstance(local_functional, float):
            total_functional = self.ensemble.ensemble_comm.allreduce(sendobj=local_functional, op=MPI.SUM)
        elif isinstance(local_functional, firedrake.Function):
            total_functional = firedrake.Function(local_functional.function_space())
            total_functional = self.ensemble.allreduce(local_functional, total_functional)
        else:
            raise NotImplementedError("This type of functional is not supported.")
        return total_functional

    def derivative(self, adj_input=1.0, options=None, op=MPI.SUM):
        """Compute derivatives of a functional with respect to the control parameters.

        Parameters
        ----------
        adj_input : float
            The adjoint input. (Improve this description. Ask David for help.)
        options : dict
            Additional options for the derivative computation.
            (Improve this description. Ask David for help.)
        op : mpi4py.MPI.Op
            The employed MPI operation for the `Ensemble.allreduce` the derivatives.

        Returns
        -------
            dJdm_total : :class:`~.function.Function` or list of :class:`~.function.Function`
            The result of Allreduce operations of ``dJdm_local`` into ``dJdm_total`` over `Ensemble.ensemble_comm`.

        See Also
        --------
        :meth:`~.ensemble.Ensemble.allreduce`, :meth:`pyadjoint.ReducedFunctional.derivative`.
        """
        dJdm_local = super(EnsembleReducedFunctional, self).derivative(adj_input=adj_input, options=options)
        if isinstance(dJdm_local, list):
            dJdm_total = []
            for dJdm in dJdm_local:
                dJdm_total.append(
                    self.ensemble.allreduce(dJdm, firedrake.Function(dJdm.function_space()), op=op)
                )
        elif isinstance(dJdm_local, firedrake.Function):
            dJdm_total = firedrake.Function(dJdm_local.function_space())
            dJdm_total = self.ensemble.allreduce(dJdm_local, dJdm_total, op=op)
        else:
            raise NotImplementedError("This type of gradient is not supported.")

        return dJdm_total

    def hessian(self, m_dot, options=None):
        """The Hessian is not yet implemented for ensemble reduced functional.

        Raises:
            NotImplementedError: This method is not yet implemented for ensemble reduced functional.
        """
        raise NotImplementedError("Hessian is not yet implemented for ensemble reduced functional.")
