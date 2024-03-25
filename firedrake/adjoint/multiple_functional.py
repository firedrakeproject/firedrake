from pyadjoint import ReducedFunctional

class EnsembleReducedFunctional(ReducedFunctional):
    def __init__(self, J, control, ensemble):
        super().__init__(J, control, ...)
        self.ensemble = ensemble
    
    def __call__(self, values):
        J_local = super().__call__(values)
        # Be careful with the __call__ output type
        J_total = self.ensemble.ensemble_comm.allreduce(J_local)
        return J_total
    
    def derivative(self):
        dJ_local = super().derivative(...)
        # define the dJ_total correctly
        self.ensemble.reduce(dJ_local, dJ_total)
        return dJ_total