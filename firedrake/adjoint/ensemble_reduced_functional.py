from pyadjoint import ReducedFunctional
from pyop2.mpi import MPI

import firedrake


class EnsembleReducedFunctional(ReducedFunctional):
    """Ensemble of Reduced Functionals.

    This class is used to compute the ensemble of reduced functionals in parallel.

    Parameters
    ----------
    J : :obj:`pyadjoint.OverloadedType`
        An instance of an OverloadedType, usually :class:`pyadjoint.AdjFloat`. This should be the functional that we want to
        reduce.
    control : :obj:`pyadjoint.Control` or list of :obj:`pyadjoint.Control`
        A single or a list of Control instances, which you want to map to the functional.
    ensemble : Ensemble
        An instance of the `Ensemble`. This is used to communicate the reduced functional values across the ensemble.
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
        """Return the derivative of the ensemble of reduced functionals with respect to the control parameters.

        Parameters
        ----------
        adj_input : float
            The adjoint input for the derivative computation.
        options : dict
            Additional options for the derivative computation.
        op : mpi4py.MPI.Op
            The MPI operation used for the allreduce operation.

        Returns
        -------
            dJdm_total : firedrake.Function or list of firedrake.Function
            The total derivative of the ensemble reduced functional with respect to the model parameters.
        """
        dJdm_local = super(EnsembleReducedFunctional, self).derivative(adj_input=adj_input, options=options)
        if isinstance(dJdm_local, list):
            dJdm_total = []
            for dJdm in dJdm_local:
                dJdm_total.append(
                    self.ensemble.allreduce(dJdm, firedrake.Function(dJdm.function_space()), op=op)
                )
        elif isinstance(dJdm_local, (firedrake.Function, firedrake.Cofunction)):
            dJdm_total = firedrake.Function(dJdm_local.function_space())
            dJdm_total = self.ensemble.allreduce(dJdm_local, dJdm_total, op=op)
        else:
            raise NotImplementedError("This type of gradient is not supported.")

        return dJdm_total

    def hessian(self, m_dot, options=None):
        """Should return the Hessian of the ensemble of reduced functionals with respect to the control parameters.

        Raises:
            NotImplementedError: This method is not yet implemented for ensemble reduced functional.
        """
        raise NotImplementedError("Hessian is not yet implemented for ensemble reduced functional.")
