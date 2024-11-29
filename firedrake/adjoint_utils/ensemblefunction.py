from pyadjoint.overloaded_type import OverloadedType
from firedrake.petsc import PETSc
from .checkpointing import disk_checkpointing

from functools import wraps


class EnsembleFunctionMixin(OverloadedType):

    @staticmethod
    def _ad_annotate_init(init):
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            OverloadedType.__init__(self)
            init(self, *args, **kwargs)
        return wrapper

    @staticmethod
    def _ad_to_list(m):
        with m.vec_ro() as gvec:
            lcomm = PETSc.COMM_SELF
            gsize = gvec.size
            lvec = PETSc.Vec().createSeq(gsize, comm=lcomm)
            is_ = PETSc.IS().createStride(gsize, 0, 1, comm=lcomm)

            mode = PETSc.InsertMode.INSERT_VALUES
            scatter = PETSc.Scatter().create(gvec, is_, lvec, None)
            scatter.scatterBegin(gvec, lvec, addv=mode)
            scatter.scatterEnd(gvec, lvec, addv=mode)

            return lvec.array_r.tolist()

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        with dst.vec_wo() as vec:
            begin, end = vec.owner_range
            src_array = src[offset + begin: offset + end]
            vec.array[:] = src_array
            offset += vec.size
            return dst, offset

    def _ad_dot(self, other, options=None):
        # local dot product
        ldot = sum(
            uself._ad_dot(uother, options=options)
            for uself, uother in zip(self.subfunctions,
                                     other.subfunctions))
        # global dot product
        gdot = self.ensemble.ensemble_comm.allreduce(ldot)
        return gdot

    def _ad_add(self, other):
        new = self.copy()
        new += other
        return new

    def _ad_mul(self, other):
        new = self.copy()
        # `self` can be a Cofunction in which case only left multiplication with a scalar is allowed.
        other = other._fbuf if type(other) is type(self) else other
        new._fbuf.assign(other*new._fbuf)
        return new

    def _ad_iadd(self, other):
        self += other
        return self

    def _ad_imul(self, other):
        self *= other
        return self

    def _ad_copy(self):
        return self.copy()

    def _ad_convert_riesz(self, value, options=None):
        raise ValueError("NotImplementedYet")

    def _ad_create_checkpoint(self):
        if disk_checkpointing():
            raise NotImplementedError(
                "Disk checkpointing not implemented for EnsembleFunctions")
        else:
            return self.copy()

    def _ad_from_petsc(self, vec):
        with self.vec_wo as self_v:
            vec.copy(result=self_v)

    def _ad_to_petsc(self, vec=None):
        with self.vec_ro as self_v:
            if vec:
                self_v.copy(result=vec)
            else:
                vec = self_v.copy()
        return vec
