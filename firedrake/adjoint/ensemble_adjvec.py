from functools import cached_property
from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.adjfloat import AdjFloat
from firedrake.ensemble import Ensemble
from firedrake.adjoint_utils.checkpointing import disk_checkpointing


class EnsembleAdjVec(OverloadedType):
    """
    A vector of :class:`pyadjoint.AdjFloat` distributed
    over an :class:`.Ensemble`.

    Analagous to the :class:`.EnsembleFunction` and
    :class:`.EnsembleCofunction` types but for :class:`~pyadjoint.AdjFloat`.

    Implements basic :class:`pyadjoint.OverloadedType` functionality
    to be used as a :class:`pyadjoint.Control` or functional for the
    :class:`~.ensemble_reduced_functional.EnsembleReducedFunctional` types.

    Parameters
    ----------
    subvec :
        The local part of the vector.
    ensemble :
        The :class:`.Ensemble` communicator.

    See Also
    --------
    :class:`~.Ensemble`
    :class:`~.EnsembleFunction`
    :class:`~.EnsembleCofunction`
    :class:`~.EnsembleReducedFunctional`
    """

    def __init__(self, subvec: list[AdjFloat], ensemble: Ensemble):
        if not isinstance(ensemble, Ensemble):
            raise TypeError(
                f"EnsembleAdjVec needs an Ensemble, not a {type(ensemble).__name__}")
        if not all(isinstance(v, (AdjFloat, float)) for v in subvec):
            raise TypeError(
                f"EnsembleAdjVec must be instantiated with a list of AdjFloats, not {subvec}")
        self._subvec = [AdjFloat(x) for x in subvec]
        self.ensemble = ensemble
        OverloadedType.__init__(self)

    @property
    def subvec(self) -> list[AdjFloat]:
        """The part of the vector on the local spatial comm."""
        return self._subvec

    @cached_property
    def local_size(self) -> int:
        """The length of the part of the vector on the local spatial comm."""
        return len(self._subvec)

    @cached_property
    def global_size(self) -> int:
        """The global length of vector."""
        return self.ensemble.allreduce(self.local_size)

    def _ad_init_zero(self, dual: bool = False) -> "EnsembleAdjVec":
        return type(self)(
            [v._ad_init_zero(dual=dual) for v in self.subvec],
            self.ensemble)

    def _ad_dot(self, other: OverloadedType) -> float:
        local_dot = sum(s._ad_dot(o)
                        for s, o in zip(self.subvec, other.subvec))
        global_dot = self.ensemble.ensemble_comm.allreduce(local_dot)
        return global_dot

    def _ad_add(self, other) -> "EnsembleAdjVec":
        return EnsembleAdjVec(
            [s._ad_add(o) for s, o in zip(self.subvec, other.subvec)],
            ensemble=self.ensemble)

    def _ad_mul(self, other) -> "EnsembleAdjVec":
        return EnsembleAdjVec(
            [s._ad_mul(o) for s, o in zip(self.subvec,
                                          self._maybe_scalar(other))],
            ensemble=self.ensemble)

    def _ad_iadd(self, other) -> "EnsembleAdjVec":
        for s, o in zip(self.subvec, other.subvec):
            s._ad_iadd(o)
        return self

    def _ad_imul(self, other) -> "EnsembleAdjVec":
        for s, o in zip(self.subvec, self._maybe_scalar(other)):
            s._ad_imul(o)
        return self

    def _maybe_scalar(self, val):
        if isinstance(val, EnsembleAdjVec):
            return val.subvec
        else:
            return [val for _ in self.subvec]

    def _ad_copy(self) -> "EnsembleAdjVec":
        return EnsembleAdjVec(
            [v._ad_copy() for v in self.subvec],
            ensemble=self.ensemble)

    def _ad_convert_riesz(self, value, riesz_map=None) -> "EnsembleAdjVec":
        return EnsembleAdjVec(
            [s._ad_convert_riesz(v, riesz_map=riesz_map)
             for s, v in zip(self.subvec, self._maybe_scalar(value))],
            ensemble=self.ensemble)

    def _ad_create_checkpoint(self):
        if disk_checkpointing():
            raise NotImplementedError(
                f"Disk checkpointing not implemented for {type(self).__name__}")
        else:
            return self._ad_copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        if type(checkpoint) is type(self):
            return checkpoint
        raise NotImplementedError(
            f"Disk checkpointing not implemented for {type(self).__name__}")
