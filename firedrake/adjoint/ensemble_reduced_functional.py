from pyadjoint import ReducedFunctional
from pyop2.mpi import MPI

import firedrake


class EnsembleReducedFunctional(ReducedFunctional):
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

    def derivative(self, adj_input=1.0, options=None):
        dJdm_local = super(EnsembleReducedFunctional, self).derivative(adj_input=adj_input, options=options)
        if isinstance(dJdm_local, list):
            dJdm_total = []
            for dJdm in dJdm_local:
                dJdm_total.append(self.ensemble.allreduce(dJdm, firedrake.Function(dJdm.function_space())))
        elif isinstance(dJdm_local, (firedrake.Function, firedrake.Cofunction)):
            dJdm_total = firedrake.Function(dJdm_local.function_space())
            dJdm_total = self.ensemble.allreduce(dJdm_local, dJdm_total)
        else:
            raise NotImplementedError("This type of gradient is not supported.")

        return dJdm_total

    def hessian(self, m_dot, options=None):
        raise NotImplementedError("Hessian is not yet implemented for ensemble reduced functional.")
