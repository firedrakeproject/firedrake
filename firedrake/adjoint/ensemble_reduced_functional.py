from pyadjoint import ReducedFunctional
import numpy as np
from enum import Enum
from pyop2.mpi import MPI

import firedrake


class OutputGradientType(str, Enum):
    NUMPYARRAY = "numpy_array"
    FUNCTION = "function"


class EnsembleReducedFunctional(ReducedFunctional):
    def __init__(self, J, control, ensemble,
                 output_gradient_type=OutputGradientType.FUNCTION):
        super(EnsembleReducedFunctional, self).__init__(J, control)
        self.ensemble = ensemble
        self.output_gradient_type = output_gradient_type

    def __call__(self, values):
        local_functional = None
        if not isinstance(values, (np.ndarray, firedrake.Function, list)):
            raise ValueError("The values must be a numpy array, a firedrake function, or a list of firedrake functions.")
        
        if isinstance(values, np.ndarray):
            updated_control = [control.copy_data() for control in self.controls]
            offset = 0
            for i, control in enumerate(self.controls):
                updated_control[i], offset = control.assign_numpy(updated_control[i], values, offset)
            local_functional = super(EnsembleReducedFunctional, self).__call__(updated_control)
        else:
            local_functional = super(EnsembleReducedFunctional, self).__call__(values)

        if isinstance(local_functional, float):
            return self.ensemble.ensemble_comm.allreduce(np.array([local_functional]), op=MPI.SUM)[0]
        else:
            raise NotImplementedError("This type of functional is not supported.")

    def derivative(self, adj_input=1.0, options=None):
        dJdm_local = super(EnsembleReducedFunctional, self).derivative(adj_input=adj_input, options=options)
        if isinstance(dJdm_local, firedrake.Function):
            dJdm_total = dJdm_local.copy(deepcopy=True)
            dJdm_total = self.ensemble.allreduce(dJdm_local, dJdm_total)
        else:
            raise NotImplementedError("This type of gradient is not supported.")
        
        if self.output_gradient_type == OutputGradientType.FUNCTION:
            return dJdm_total
        elif self.output_gradient_type == OutputGradientType.NUMPYARRAY:
            return dJdm_total.dat.data_ro[:]
        else:
            raise NotImplementedError("This output gradient type is not yet supported.")

    def hessian(self, m_dot, options=None):

        raise NotImplementedError("Hessian is not yet implemented for ensemble reduced functional.")