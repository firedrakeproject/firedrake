from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.adjfloat import AdjFloat
from firedrake.ensemble import Ensemble

__all__ = ("EnsembleAdjVec",)


class EnsembleAdjVec(OverloadedType):
    """
    Basic functionality for a list of AdjFloats distributed
    across multiple ensemble ranks to be an OverloadedType.
    """

    def __init__(self, subvec, ensemble):
        if not isinstance(ensemble, Ensemble):
            raise TypeError(
                f"EnsembleAdjVec needs an Ensemble, not a {type(ensemble).__name__}")
        if not all(isinstance(v, AdjFloat) for v in subvec):
            raise TypeError(
                f"EnsembleAdjVec must be instantiated with a list of AdjFloats, not {subvec}")
        self.subvec = subvec
        self.ensemble = ensemble
        OverloadedType.__init__(self)

    def _ad_init_zero(self, dual=False):
        return type(self)(
            [v._ad_init_zero(dual=dual) for v in self.subvec],
            self.ensemble)

    def _ad_dot(self, other, options=None):
        local_dot = sum(s._ad_dot(o)
                        for s, o in zip(self.subvec, other.subvec))
        global_dot = self.ensemble.ensemble_comm.allreduce(local_dot)
        return global_dot

    def _ad_add(self, other):
        return EnsembleAdjVec(
            [s._ad_add(o) for s, o in zip(self.subvec, other.subvec)],
            ensemble=self.ensemble)

    def _ad_mul(self, other):
        return EnsembleAdjVec(
            [s._ad_mul(o) for s, o in zip(self.subvec,
                                          self._maybe_scalar(other))],
            ensemble=self.ensemble)

    def _ad_iadd(self, other):
        for s, o in zip(self.subvec, other.subvec):
            s._ad_iadd(o)
        return self

    def _ad_imul(self, other):
        for s, o in zip(self.subvec, self._maybe_scalar(other)):
            s._ad_imul(o)
        return self

    def _maybe_scalar(self, val):
        if isinstance(val, EnsembleAdjVec):
            return val.subvec
        else:
            return [val for _ in self.subvec]

    def _ad_copy(self):
        return EnsembleAdjVec(
            [v._ad_copy() for v in self.subvec],
            ensemble=self.ensemble)

    def _ad_convert_riesz(self, value, riesz_map=None):
        return EnsembleAdjVec(
            [s._ad_convert_riesz(v, riesz_map=riesz_map)
             for s, v in zip(self.subvec, self._maybe_scalar(value))],
            ensemble=self.ensemble)
