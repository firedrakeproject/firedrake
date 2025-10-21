from pyadjoint.overloaded_type import OverloadedType
from firedrake.petsc import PETSc
from .checkpointing import disk_checkpointing

from functools import wraps


class EnsembleFunctionMixin(OverloadedType):
    """
    Basic functionality for EnsembleFunction to be OverloadedTypes.
    Note that currently no EnsembleFunction operations are taped.

    Enables EnsembleFunction to do the following:
    - Be a Control for a NumpyReducedFunctional (_ad_to_list and _ad_assign_numpy)
    - Be used with pyadjoint TAO solver (_ad_{to,from}_petsc)
    - Be used as a Control for Taylor tests (_ad_dot, _ad_add, _ad_mul)
    """

    @staticmethod
    def _ad_annotate_init(init):
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            OverloadedType.__init__(self)
            init(self, *args, **kwargs)
            self._ad_add = self.__add__
            self._ad_mul = self.__mul__
            self._ad_iadd = self.__iadd__
            self._ad_imul = self.__imul__
            self._ad_copy = self.copy
        return wrapper

    @staticmethod
    def _ad_to_list(m):
        with m.vec_ro() as gvec:
            lvec = PETSc.Vec().createSeq(gvec.size,
                                         comm=PETSc.COMM_SELF)
            PETSc.Scatter().toAll(gvec).scatter(
                gvec, lvec, addv=PETSc.InsertMode.INSERT_VALUES)
        return lvec.array_r.tolist()

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        with dst.vec_wo() as vec:
            begin, end = vec.owner_range
            vec.array[:] = src[offset + begin: offset + end]
            offset += vec.size
        return dst, offset

    def _ad_dot(self, other, options=None):
        local_dot = sum(uself._ad_dot(uother, options=options)
                        for uself, uother in zip(self.subfunctions,
                                                 other.subfunctions))
        return self.function_space().ensemble_comm.allreduce(local_dot)

    def _ad_convert_riesz(self, value, riesz_map=None):
        return value.riesz_representation(riesz_map=riesz_map or "L2")

    def _ad_init_zero(self, dual=False):
        from firedrake import EnsembleFunction, EnsembleCofunction
        if dual:
            return EnsembleCofunction(self.function_space().dual())
        else:
            return EnsembleFunction(self.function_space())

    def _ad_create_checkpoint(self):
        if disk_checkpointing():
            raise NotImplementedError(
                "Disk checkpointing not implemented for EnsembleFunctions")
        else:
            return self.copy()

    def _ad_restore_at_checkpoint(self, checkpoint):
        if type(checkpoint) is type(self):
            return checkpoint
        raise NotImplementedError(
            "Disk checkpointing not implemented for EnsembleFunctions")

    def _ad_from_petsc(self, vec):
        with self.vec_wo as self_v:
            vec.copy(self_v)

    def _ad_to_petsc(self, vec=None):
        with self.vec_ro as self_v:
            return self_v.copy(vec or self._vec.duplicate())
