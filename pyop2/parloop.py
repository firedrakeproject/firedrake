import abc
import itertools
import operator
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import loopy as lp
import numpy as np
from petsc4py import PETSc

from pyop2 import mpi, profiling
from pyop2.configuration import configuration
from pyop2.datatypes import as_numpy_dtype
from pyop2.exceptions import KernelTypeError, MapValueError, SetTypeError
from pyop2.global_kernel import (GlobalKernelArg, DatKernelArg, MixedDatKernelArg,
                                 MatKernelArg, MixedMatKernelArg, PassthroughKernelArg, GlobalKernel)
from pyop2.local_kernel import LocalKernel, CStringLocalKernel, LoopyLocalKernel
from pyop2.types import (Access, Global, AbstractDat, Dat, DatView, MixedDat, Mat, Set,
                         MixedSet, ExtrudedSet, Subset, Map, ComposedMap, MixedMap)
from pyop2.types.data_carrier import DataCarrier
from pyop2.utils import cached_property


class ParloopArg(abc.ABC):

    @staticmethod
    def check_map(m):
        if configuration["type_check"]:
            if isinstance(m, ComposedMap):
                for m_ in m.maps_:
                    ParloopArg.check_map(m_)
            elif m.iterset.total_size > 0 and len(m.values_with_halo) == 0:
                raise MapValueError(f"{m} is not initialized")


@dataclass
class GlobalParloopArg(ParloopArg):
    """Class representing a :class:`Global` argument to a :class:`Parloop`."""

    data: Global

    @property
    def _kernel_args_(self):
        return self.data._kernel_args_

    @property
    def map_kernel_args(self):
        return ()

    @property
    def maps(self):
        return ()


@dataclass
class DatParloopArg(ParloopArg):
    """Class representing a :class:`Dat` argument to a :class:`Parloop`."""

    data: Dat
    map_: Optional[Map] = None

    def __post_init__(self):
        if self.map_ is not None:
            self.check_map(self.map_)

    @property
    def _kernel_args_(self):
        return self.data._kernel_args_

    @property
    def map_kernel_args(self):
        return self.map_._kernel_args_ if self.map_ else ()

    @property
    def maps(self):
        if self.map_ is not None:
            return self.map_,
        else:
            return ()


@dataclass
class MixedDatParloopArg(ParloopArg):
    """Class representing a :class:`MixedDat` argument to a :class:`Parloop`."""

    data: MixedDat
    map_: MixedMap

    def __post_init__(self):
        self.check_map(self.map_)

    @property
    def _kernel_args_(self):
        return self.data._kernel_args_

    @property
    def map_kernel_args(self):
        return self.map_._kernel_args_ if self.map_ else ()

    @property
    def maps(self):
        return self.map_,


@dataclass
class MatParloopArg(ParloopArg):
    """Class representing a :class:`Mat` argument to a :class:`Parloop`."""

    data: Mat
    maps: Tuple[Map, Map]
    lgmaps: Optional[Any] = None

    def __post_init__(self):
        for m in self.maps:
            self.check_map(m)

    @property
    def _kernel_args_(self):
        return self.data._kernel_args_

    @property
    def map_kernel_args(self):
        rmap, cmap = self.maps
        return tuple(itertools.chain(rmap._kernel_args_, cmap._kernel_args_))


@dataclass
class MixedMatParloopArg(ParloopArg):
    """Class representing a mixed :class:`Mat` argument to a :class:`Parloop`."""

    data: Mat
    maps: Tuple[MixedMap, MixedMap]
    lgmaps: Any = None

    def __post_init__(self):
        for m in self.maps:
            self.check_map(m)

    @property
    def _kernel_args_(self):
        return self.data._kernel_args_

    @property
    def map_kernel_args(self):
        rmap, cmap = self.maps
        return tuple(itertools.chain(rmap._kernel_args_, cmap._kernel_args_))


@dataclass
class PassthroughParloopArg(ParloopArg):
    # a pointer
    data: int

    @property
    def _kernel_args_(self):
        return (self.data,)

    @property
    def map_kernel_args(self):
        return ()

    @property
    def maps(self):
        return ()


class Parloop:
    """A parallel loop invocation.

    :arg global_knl: The :class:`GlobalKernel` to be executed.
    :arg iterset: The iteration :class:`Set` over which the kernel should be executed.
    :arguments: Iterable of arguments to the parloop.
    """

    def __init__(self, global_knl, iterset, arguments):
        if len(global_knl.arguments) != len(arguments):
            raise ValueError("You are trying to pass in a different number of "
                             "arguments than the kernel is expecting")

        # Performing checks on dtypes is difficult for C-string kernels because PyOP2
        # will happily pass any type into a kernel with void* arguments.
        if (isinstance(global_knl.local_kernel, LoopyLocalKernel)
                and not all(as_numpy_dtype(a.dtype) == as_numpy_dtype(b.data.dtype)
                            for a, b in zip(global_knl.local_kernel.arguments, arguments))):
            raise ValueError("The argument dtypes do not match those for the local kernel")

        self.check_iterset(iterset, global_knl, arguments)
        self._check_frozen_access_modes(global_knl.local_kernel, arguments)

        self.global_kernel = global_knl
        self.iterset = iterset
        self.comm = mpi.internal_comm(iterset.comm, self)
        self.arguments, self.reduced_globals = self.prepare_reduced_globals(arguments, global_knl)

    @property
    def local_kernel(self):
        return self.global_kernel.local_kernel

    @property
    def accesses(self):
        return self.local_kernel.accesses

    @property
    def arglist(self):
        """Prepare the argument list for calling generated code."""
        arglist = self.iterset._kernel_args_
        for d in self.arguments:
            arglist += d._kernel_args_

        # Collect an ordered set of maps (ignore duplicates)
        maps = {m: None for d in self.arguments for m in d.map_kernel_args}
        return arglist + tuple(maps.keys())

    @property
    def zipped_arguments(self):
        return self.zip_arguments(self.global_kernel, self.arguments)

    def replace_data(self, index, new_argument):
        self.arguments[index].data = new_argument

    def _compute_event(self):
        return profiling.timed_region(f"Parloop_{self.iterset.name}_{self.global_kernel.name}")

    @mpi.collective
    def _compute(self, part):
        """Execute the kernel over all members of a MPI-part of the iteration space.

        :arg part: The :class:`SetPartition` to compute over.
        """
        with self._compute_event():
            PETSc.Log.logFlops(part.size*self.num_flops)
            self.global_kernel(self.comm, part.offset, part.offset+part.size, *self.arglist)

    @cached_property
    def num_flops(self):
        return self.global_kernel.num_flops(self.iterset)

    @mpi.collective
    def compute(self):
        # Parloop.compute is an alias for Parloop.__call__
        self()

    @PETSc.Log.EventDecorator("ParLoopExecute")
    @mpi.collective
    def __call__(self):
        """Execute the kernel over all members of the iteration space."""
        self.increment_dat_version()
        self.zero_global_increments()
        orig_lgmaps = self.replace_lgmaps()
        self.global_to_local_begin()
        self._compute(self.iterset.core_part)
        self.global_to_local_end()
        self._compute(self.iterset.owned_part)
        requests = self.reduction_begin()
        self.local_to_global_begin()
        self.update_arg_data_state()
        self.restore_lgmaps(orig_lgmaps)
        self.reduction_end(requests)
        self.finalize_global_increments()
        self.local_to_global_end()

    def increment_dat_version(self):
        """Increment dat versions of :class:`DataCarrier`s in the arguments."""
        for lk_arg, gk_arg, pl_arg in self.zipped_arguments:
            if isinstance(pl_arg, PassthroughParloopArg):
                continue
            assert isinstance(pl_arg.data, DataCarrier)
            if lk_arg.access is not Access.READ:
                if pl_arg.data in self.reduced_globals:
                    self.reduced_globals[pl_arg.data].data.increment_dat_version()
                else:
                    pl_arg.data.increment_dat_version()

    def zero_global_increments(self):
        """Zero any global increments every time the loop is executed."""
        for g in self.reduced_globals.keys():
            g._data[...] = 0

    def replace_lgmaps(self):
        """Swap out any lgmaps for any :class:`MatParloopArg` instances
        if necessary.
        """
        if not self._has_mats:
            return

        orig_lgmaps = []
        for i, (lk_arg, gk_arg, pl_arg) in enumerate(self.zipped_arguments):
            if isinstance(gk_arg, (MatKernelArg, MixedMatKernelArg)):
                new_state = {Access.INC: Mat.ADD_VALUES,
                             Access.WRITE: Mat.INSERT_VALUES}[lk_arg.access]
                for m in pl_arg.data:
                    m.change_assembly_state(new_state)
                pl_arg.data.change_assembly_state(new_state)

                if pl_arg.lgmaps is not None:
                    olgmaps = []
                    for m, lgmaps in zip(pl_arg.data, pl_arg.lgmaps):
                        olgmaps.append(m.handle.getLGMap())
                        m.handle.setLGMap(*lgmaps)
                    orig_lgmaps.append(olgmaps)
        return tuple(orig_lgmaps)

    def restore_lgmaps(self, orig_lgmaps):
        """Restore any swapped lgmaps."""
        if not self._has_mats:
            return

        orig_lgmaps = list(orig_lgmaps)
        for arg, d in reversed(list(zip(self.global_kernel.arguments, self.arguments))):
            if isinstance(arg, (MatKernelArg, MixedMatKernelArg)) and d.lgmaps is not None:
                for m, lgmaps in zip(d.data, orig_lgmaps.pop()):
                    m.handle.setLGMap(*lgmaps)

    @cached_property
    def _has_mats(self):
        return any(isinstance(a, (MatParloopArg, MixedMatParloopArg)) for a in self.arguments)

    @mpi.collective
    def global_to_local_begin(self):
        """Start halo exchanges."""
        for idx, op in self._g2l_begin_ops:
            op(self.arguments[idx].data)

    @mpi.collective
    def global_to_local_end(self):
        """Finish halo exchanges."""
        for idx, op in self._g2l_end_ops:
            op(self.arguments[idx].data)

    @cached_property
    def _g2l_begin_ops(self):
        ops = []
        for idx in self._g2l_idxs:
            op = operator.methodcaller(
                "global_to_local_begin",
                access_mode=self.accesses[idx],
            )
            ops.append((idx, op))
        return tuple(ops)

    @cached_property
    def _g2l_end_ops(self):
        ops = []
        for idx in self._g2l_idxs:
            op = operator.methodcaller(
                "global_to_local_end",
                access_mode=self.accesses[idx],
            )
            ops.append((idx, op))
        return tuple(ops)

    @cached_property
    def _g2l_idxs(self):
        seen = set()
        indices = []
        for i, (lknl_arg, gknl_arg, pl_arg) in enumerate(self.zipped_arguments):
            if (isinstance(gknl_arg, (DatKernelArg, MixedDatKernelArg)) and pl_arg.data not in seen
                    and gknl_arg.is_indirect and lknl_arg.access is not Access.WRITE):
                indices.append(i)
                seen.add(pl_arg.data)
        return tuple(indices)

    @mpi.collective
    def local_to_global_begin(self):
        """Start halo exchanges."""
        for idx, op in self._l2g_begin_ops:
            op(self.arguments[idx].data)

    @mpi.collective
    def local_to_global_end(self):
        """Finish halo exchanges (wait on irecvs)."""
        for idx, op in self._l2g_end_ops:
            op(self.arguments[idx].data)

    @cached_property
    def _l2g_begin_ops(self):
        ops = []
        for idx in self._l2g_idxs:
            op = operator.methodcaller(
                "local_to_global_begin",
                insert_mode=self.accesses[idx],
            )
            ops.append((idx, op))
        return tuple(ops)

    @cached_property
    def _l2g_end_ops(self):
        ops = []
        for idx in self._l2g_idxs:
            op = operator.methodcaller(
                "local_to_global_end",
                insert_mode=self.accesses[idx],
            )
            ops.append((idx, op))
        return tuple(ops)

    @cached_property
    def _l2g_idxs(self):
        seen = set()
        indices = []
        for i, (lknl_arg, gknl_arg, pl_arg) in enumerate(self.zipped_arguments):
            if (isinstance(gknl_arg, (DatKernelArg, MixedDatKernelArg)) and pl_arg.data not in seen
                    and gknl_arg.is_indirect
                    and lknl_arg.access in {Access.INC, Access.MIN, Access.MAX}):
                indices.append(i)
                seen.add(pl_arg.data)
        return tuple(indices)

    @PETSc.Log.EventDecorator("ParLoopRednBegin")
    @mpi.collective
    def reduction_begin(self):
        """Begin reductions."""
        requests = []
        for idx in self._reduction_idxs:
            glob = self.arguments[idx].data
            mpi_op = {Access.INC: mpi.MPI.SUM,
                      Access.MIN: mpi.MPI.MIN,
                      Access.MAX: mpi.MPI.MAX}.get(self.accesses[idx])

            if mpi.MPI.VERSION >= 3:
                requests.append(self.comm.Iallreduce(glob._data, glob._buf, op=mpi_op))
            else:
                self.comm.Allreduce(glob._data, glob._buf, op=mpi_op)
        return tuple(requests)

    @PETSc.Log.EventDecorator("ParLoopRednEnd")
    @mpi.collective
    def reduction_end(self, requests):
        """Finish reductions."""
        if mpi.MPI.VERSION >= 3:
            for idx, req in zip(self._reduction_idxs, requests):
                req.Wait()
                glob = self.arguments[idx].data
                glob._data[:] = glob._buf
        else:
            assert len(requests) == 0

            for idx in self._reduction_idxs:
                glob = self.arguments[idx].data
                glob._data[:] = glob._buf

    @cached_property
    def _reduction_idxs(self):
        return tuple(i for i, arg
                     in enumerate(self.global_kernel.arguments)
                     if isinstance(arg, GlobalKernelArg)
                     and self.accesses[i] in {Access.INC, Access.MIN, Access.MAX})

    def finalize_global_increments(self):
        """Finalise global increments."""
        for tmp, glob in self.reduced_globals.items():
            glob.data._data += tmp._data

    @mpi.collective
    def update_arg_data_state(self):
        r"""Update the state of the :class:`DataCarrier`\s in the arguments to the `par_loop`.

        This marks :class:`Mat`\s that need assembly."""
        for i, (wrapper_arg, d) in enumerate(zip(self.global_kernel.arguments, self.arguments)):
            access = self.accesses[i]
            if access is Access.READ:
                continue
            if isinstance(wrapper_arg, (DatKernelArg, MixedDatKernelArg)):
                d.data.halo_valid = False
            elif isinstance(wrapper_arg, (MatKernelArg, MixedMatKernelArg)):
                state = {Access.WRITE: Mat.INSERT_VALUES,
                         Access.INC: Mat.ADD_VALUES}[access]
                d.data.assembly_state = state

    @classmethod
    def check_iterset(cls, iterset, global_knl, arguments):
        """Check that the iteration set is valid.

        For an explanation of the arguments see :class:`Parloop`.

        :raises MapValueError: If ``iterset`` does not match that of the arguments.
        :raises SetTypeError: If ``iterset`` is of the wrong type.
        """
        if not configuration["type_check"]:
            return

        if not isinstance(iterset, Set):
            raise SetTypeError("Iteration set is of the wrong type")

        if isinstance(iterset, MixedSet):
            raise SetTypeError("Cannot iterate over mixed sets")

        if isinstance(iterset, Subset):
            iterset = iterset.superset

        for i, (lk_arg, gk_arg, pl_arg) in enumerate(cls.zip_arguments(global_knl, arguments)):
            if isinstance(gk_arg, DatKernelArg) and gk_arg.is_direct:
                _iterset = iterset.parent if isinstance(iterset, ExtrudedSet) else iterset
                if pl_arg.data.dataset.set != _iterset:
                    raise MapValueError(f"Iterset of direct arg {i} does not match parloop iterset")

            for j, m in enumerate(pl_arg.maps):
                if m.iterset != iterset and m.iterset not in iterset:
                    raise MapValueError(f"Iterset of arg {i} map {j} does not match parloop iterset")

    @classmethod
    def _check_frozen_access_modes(cls, local_knl, arguments):
        """Check that any frozen :class:`Dat` are getting accessed with the right access mode."""
        for lknl_arg, pl_arg in zip(local_knl.arguments, arguments):
            if isinstance(pl_arg.data, AbstractDat):
                if any(
                    d._halo_frozen and d._frozen_access_mode != lknl_arg.access
                    for d in pl_arg.data
                ):
                    raise RuntimeError(
                        "Dats with frozen halos must always be accessed with the same access mode"
                    )

    def prepare_reduced_globals(self, arguments, global_knl):
        """Swap any :class:`GlobalParloopArg` instances that are INC'd into
        with zeroed replacements.

        This is needed to ensure that successive parloops incrementing into a
        :class:`Global` in parallel produces the right result. The same is not
        needed for MAX and MIN because they commute with the reduction.
        """
        arguments = list(arguments)
        reduced_globals = {}
        for i, (lk_arg, gk_arg, pl_arg) in enumerate(self.zip_arguments(global_knl, arguments)):
            if isinstance(gk_arg, GlobalKernelArg) and lk_arg.access == Access.INC:
                tmp = Global(gk_arg.dim, data=np.zeros_like(pl_arg.data.data_ro), dtype=lk_arg.dtype, comm=self.comm)
                reduced_globals[tmp] = pl_arg
                arguments[i] = GlobalParloopArg(tmp)

        return arguments, reduced_globals

    @staticmethod
    def zip_arguments(global_knl, arguments):
        """Utility method for iterating over the arguments for local kernel,
        global kernel and parloop arguments together.
        """
        return tuple(zip(global_knl.local_kernel.arguments, global_knl.arguments, arguments))


class LegacyArg(abc.ABC):
    """Old-style input to a :func:`parloop` where the codegen-level info is
    passed in alongside any data.
    """

    @property
    @abc.abstractmethod
    def global_kernel_arg(self):
        """Return a corresponding :class:`GlobalKernelArg`."""

    @property
    @abc.abstractmethod
    def parloop_arg(self):
        """Return a corresponding :class:`ParloopArg`."""


@dataclass
class GlobalLegacyArg(LegacyArg):
    """Legacy argument for a :class:`Global`."""

    data: Global
    access: Access

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def global_kernel_arg(self):
        return GlobalKernelArg(self.data.dim)

    @property
    def parloop_arg(self):
        return GlobalParloopArg(self.data)


@dataclass
class DatLegacyArg(LegacyArg):
    """Legacy argument for a :class:`Dat`."""

    data: Dat
    map_: Optional[Map]
    access: Access

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def global_kernel_arg(self):
        map_arg = self.map_._global_kernel_arg if self.map_ is not None else None
        index = self.data.index if isinstance(self.data, DatView) else None
        return DatKernelArg(self.data.dataset.dim, map_arg, index=index)

    @property
    def parloop_arg(self):
        return DatParloopArg(self.data, self.map_)


@dataclass
class MixedDatLegacyArg(LegacyArg):
    """Legacy argument for a :class:`MixedDat`."""

    data: MixedDat
    map_: MixedMap
    access: Access

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def global_kernel_arg(self):
        args = []
        for d, m in zip(self.data, self.map_):
            map_arg = m._global_kernel_arg if m is not None else None
            args.append(DatKernelArg(d.dataset.dim, map_arg))
        return MixedDatKernelArg(tuple(args))

    @property
    def parloop_arg(self):
        return MixedDatParloopArg(self.data, self.map_)


@dataclass
class MatLegacyArg(LegacyArg):
    """Legacy argument for a :class:`Mat`."""

    data: Mat
    maps: Tuple[Map, Map]
    access: Access
    lgmaps: Optional[Tuple[Any, Any]] = None
    needs_unrolling: Optional[bool] = False

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def global_kernel_arg(self):
        map_args = [m._global_kernel_arg for m in self.maps]
        return MatKernelArg(self.data.dims, tuple(map_args), unroll=self.needs_unrolling)

    @property
    def parloop_arg(self):
        return MatParloopArg(self.data, self.maps, self.lgmaps)


@dataclass
class MixedMatLegacyArg(LegacyArg):
    """Legacy argument for a mixed :class:`Mat`."""

    data: Mat
    maps: Tuple[MixedMap, MixedMap]
    access: Access
    lgmaps: Tuple[Any] = None
    needs_unrolling: Optional[bool] = False

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def global_kernel_arg(self):
        nrows, ncols = self.data.sparsity.shape
        mr, mc = self.maps
        mat_args = []
        for i in range(nrows):
            for j in range(ncols):
                mat = self.data[i, j]

                map_args = [m._global_kernel_arg for m in [mr.split[i], mc.split[j]]]
                arg = MatKernelArg(mat.dims, tuple(map_args), unroll=self.needs_unrolling)
                mat_args.append(arg)
        return MixedMatKernelArg(tuple(mat_args), shape=self.data.sparsity.shape)

    @property
    def parloop_arg(self):
        return MixedMatParloopArg(self.data, tuple(self.maps), self.lgmaps)


@dataclass
class PassthroughArg(LegacyArg):
    """Argument that is simply passed to the local kernel without packing.

    :param dtype: The datatype of the argument. This is needed for code generation.
    :param data: A pointer to the data.
    """
    # We don't know what the local kernel is doing with this argument
    access = Access.RW

    dtype: Any
    data: int

    @property
    def global_kernel_arg(self):
        return PassthroughKernelArg()

    @property
    def parloop_arg(self):
        return PassthroughParloopArg(self.data)


def ParLoop(*args, **kwargs):
    return LegacyParloop(*args, **kwargs)


def LegacyParloop(local_knl, iterset, *args, **kwargs):
    """Create a :class:`Parloop` with :class:`LegacyArg` inputs.

    :arg local_knl: The :class:`LocalKernel` to be executed.
    :arg iterset: The iteration :class:`Set` over which the kernel should be executed.
    :*args: Iterable of :class:`LegacyArg` instances representing arguments to the parloop.
    :**kwargs: These will be passed to the :class:`GlobalKernel` constructor.

    :returns: An appropriate :class:`Parloop` instance.
    """
    if not all(isinstance(a, LegacyArg) for a in args):
        raise ValueError("LegacyParloop only expects LegacyArg arguments")

    if not isinstance(iterset, Set):
        raise SetTypeError("Iteration set is of the wrong type")

    # finish building the local kernel
    local_knl.accesses = tuple(a.access for a in args)
    if isinstance(local_knl, CStringLocalKernel):
        local_knl.dtypes = tuple(a.dtype for a in args)

    global_knl_args = tuple(a.global_kernel_arg for a in args)
    extruded = iterset._extruded
    extruded_periodic = iterset._extruded_periodic
    constant_layers = extruded and iterset.constant_layers
    subset = isinstance(iterset, Subset)
    global_knl = GlobalKernel(local_knl, global_knl_args,
                              extruded=extruded,
                              extruded_periodic=extruded_periodic,
                              constant_layers=constant_layers,
                              subset=subset,
                              **kwargs)

    parloop_args = tuple(a.parloop_arg for a in args)
    return Parloop(global_knl, iterset, parloop_args)


def par_loop(*args, **kwargs):
    parloop(*args, **kwargs)


@mpi.collective
def parloop(knl, *args, **kwargs):
    """Construct and execute a :class:`Parloop`.

    For a description of the possible arguments to this function see
    :class:`Parloop` and :func:`LegacyParloop`.
    """
    if isinstance(knl, GlobalKernel):
        Parloop(knl, *args, **kwargs)()
    elif isinstance(knl, LocalKernel):
        LegacyParloop(knl, *args, **kwargs)()
    else:
        raise KernelTypeError


def generate_single_cell_wrapper(iterset, args, forward_args=(),
                                 kernel_name=None, wrapper_name=None):
    """Generates wrapper for a single cell. No iteration loop, but cellwise data is extracted.
    Cell is expected as an argument to the wrapper. For extruded, the numbering of the cells
    is columnwise continuous, bottom to top.

    :param iterset: The iteration set
    :param args: :class:`Arg`s
    :param forward_args: To forward unprocessed arguments to the kernel via the wrapper,
                         give an iterable of strings describing their C types.
    :param kernel_name: Kernel function name
    :param wrapper_name: Wrapper function name

    :return: string containing the C code for the single-cell wrapper
    """
    from pyop2.codegen.builder import WrapperBuilder
    from pyop2.codegen.rep2loopy import generate
    from loopy.types import OpaqueType

    accs = tuple(a.access for a in args)
    dtypes = tuple(a.data.dtype for a in args)
    empty_knl = CStringLocalKernel("", kernel_name, accesses=accs, dtypes=dtypes)

    forward_arg_types = [OpaqueType(fa) for fa in forward_args]
    builder = WrapperBuilder(kernel=empty_knl,
                             subset=isinstance(iterset, Subset),
                             extruded=iterset._extruded,
                             extruded_periodic=iterset._extruded_periodic,
                             constant_layers=iterset._extruded and iterset.constant_layers,
                             single_cell=True,
                             forward_arg_types=forward_arg_types)
    for arg in args:
        builder.add_argument(arg.global_kernel_arg)
    wrapper = generate(builder, wrapper_name)
    code = lp.generate_code_v2(wrapper)

    return code.device_code()
