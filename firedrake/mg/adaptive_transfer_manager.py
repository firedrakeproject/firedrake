"""
This module contains the AdaptiveTransferManager used to perform
transfer operations on AdaptiveMeshHierarchies
"""
from firedrake.mg.embedded import TransferManager
from firedrake.ufl_expr import action, TrialFunction
from firedrake.interpolation import interpolate


__all__ = ("AdaptiveTransferManager",)


class AdaptiveTransferManager(TransferManager):
    """
    TransferManager for adaptively refined mesh hierarchies
    """
    def __init__(self, *, native_transfers=None, use_averaging=True):
        if native_transfers is not None:
            raise NotImplementedError("Custom transfers not implemented.")
        super().__init__(native_transfers=native_transfers, use_averaging=use_averaging)
        self.cache = {}

    def get_interpolator(self, Vc, Vf):
        from firedrake.assemble import assemble
        key = (Vc, Vf)
        try:
            return self.cache[key]
        except KeyError:
            Iexpr = interpolate(TrialFunction(Vc), Vf)
            # TODO reusable matfree Interpolator
            I = assemble(Iexpr, mat_type="aij")
            return self.cache.setdefault(key, I)

    def forward(self, uc, uf):
        from firedrake.assemble import assemble
        Vc = uc.function_space()
        Vf = uf.function_space()
        I = self.get_interpolator(Vc, Vf)
        return assemble(action(I, uc), tensor=uf)

    def adjoint(self, uf, uc):
        from firedrake.assemble import assemble
        Vc = uc.function_space().dual()
        Vf = uf.function_space().dual()
        I = self.get_interpolator(Vc, Vf)
        return assemble(action(uf, I), tensor=uc)

    def prolong(self, uf, uc):
        return self.forward(uf, uc)

    def inject(self, uc, uf):
        return self.forward(uc, uf)

    def restrict(self, uc, uf):
        return self.adjoint(uc, uf)
