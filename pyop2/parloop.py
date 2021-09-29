import abc
import collections
import copy
import ctypes
import enum
import itertools
import operator
import os
import types

import loopy as lp
import numpy as np
from petsc4py import PETSc

from . import (
    caching,
    compilation,
    configuration as conf,
    datatypes as dtypes,
    exceptions as ex,
    mpi,
    profiling,
    utils
)
from .kernel import Kernel
from .types import (
    Access,
    Global, Dat, Mat, Map, MixedDat,
    Set, MixedSet, ExtrudedSet, Subset
)


class Arg:

    """An argument to a :func:`pyop2.op2.par_loop`.

    .. warning ::
        User code should not directly instantiate :class:`Arg`.
        Instead, use the call syntax on the :class:`DataCarrier`.
    """

    def __init__(self, data=None, map=None, access=None, lgmaps=None, unroll_map=False):
        """
        :param data: A data-carrying object, either :class:`Dat` or class:`Mat`
        :param map:  A :class:`Map` to access this :class:`Arg` or the default
                     if the identity map is to be used.
        :param access: An access descriptor of type :class:`Access`
        :param lgmaps: For :class:`Mat` objects, a tuple of 2-tuples of local to
            global maps used during assembly.

        Checks that:

        1. the maps used are initialized i.e. have mapping data associated, and
        2. the to Set of the map used to access it matches the Set it is
           defined on.

        A :class:`MapValueError` is raised if these conditions are not met."""
        self.data = data
        self._map = map
        if map is None:
            self.map_tuple = ()
        elif isinstance(map, Map):
            self.map_tuple = (map, )
        else:
            self.map_tuple = tuple(map)

        if data is not None and hasattr(data, "dtype"):
            if data.dtype.kind == "c" and (access == Access.MIN or access == Access.MAX):
                raise ValueError("MIN and MAX access descriptors are undefined on complex data.")
        self._access = access

        self.unroll_map = unroll_map
        self.lgmaps = None
        if self._is_mat and lgmaps is not None:
            self.lgmaps = utils.as_tuple(lgmaps)
            assert len(self.lgmaps) == self.data.nblocks
        else:
            if lgmaps is not None:
                raise ValueError("Local to global maps only for matrices")

        # Check arguments for consistency
        if conf.configuration["type_check"] and not (self._is_global or map is None):
            for j, m in enumerate(map):
                if m.iterset.total_size > 0 and len(m.values_with_halo) == 0:
                    raise ex.MapValueError("%s is not initialized." % map)
                if self._is_mat and m.toset != data.sparsity.dsets[j].set:
                    raise ex.MapValueError(
                        "To set of %s doesn't match the set of %s." % (map, data))
            if self._is_dat and map.toset != data.dataset.set:
                raise ex.MapValueError(
                    "To set of %s doesn't match the set of %s." % (map, data))

    def recreate(self, data=None, map=None, access=None, lgmaps=None, unroll_map=None):
        """Creates a new Dat based on the existing Dat with the changes specified.

        :param data: A data-carrying object, either :class:`Dat` or class:`Mat`
        :param map:  A :class:`Map` to access this :class:`Arg` or the default
                     if the identity map is to be used.
        :param access: An access descriptor of type :class:`Access`
        :param lgmaps: For :class:`Mat` objects, a tuple of 2-tuples of local to
            global maps used during assembly."""
        return type(self)(data=data or self.data,
                          map=map or self.map,
                          access=access or self.access,
                          lgmaps=lgmaps or self.lgmaps,
                          unroll_map=False if unroll_map is None else unroll_map)

    @utils.cached_property
    def _kernel_args_(self):
        return self.data._kernel_args_

    @utils.cached_property
    def _argtypes_(self):
        return self.data._argtypes_

    @utils.cached_property
    def _wrapper_cache_key_(self):
        if self.map is not None:
            map_ = tuple(None if m is None else m._wrapper_cache_key_ for m in self.map)
        else:
            map_ = self.map
        return (type(self), self.access, self.data._wrapper_cache_key_, map_, self.unroll_map)

    @property
    def _key(self):
        return (self.data, self._map, self._access)

    def __eq__(self, other):
        r""":class:`Arg`\s compare equal of they are defined on the same data,
        use the same :class:`Map` with the same index and the same access
        descriptor."""
        return self._key == other._key

    def __ne__(self, other):
        r""":class:`Arg`\s compare equal of they are defined on the same data,
        use the same :class:`Map` with the same index and the same access
        descriptor."""
        return not self.__eq__(other)

    def __str__(self):
        return "OP2 Arg: dat %s, map %s, access %s" % \
            (self.data, self._map, self._access)

    def __repr__(self):
        return "Arg(%r, %r, %r)" % \
            (self.data, self._map, self._access)

    def __iter__(self):
        for arg in self.split:
            yield arg

    @utils.cached_property
    def split(self):
        """Split a mixed argument into a tuple of constituent arguments."""
        if self._is_mixed_dat:
            return tuple(Arg(d, m, self._access)
                         for d, m in zip(self.data, self._map))
        elif self._is_mixed_mat:
            rows, cols = self.data.sparsity.shape
            mr, mc = self.map
            return tuple(Arg(self.data[i, j], (mr.split[i], mc.split[j]), self._access)
                         for i in range(rows) for j in range(cols))
        else:
            return (self,)

    @utils.cached_property
    def name(self):
        """The generated argument name."""
        return "arg%d" % self.position

    @utils.cached_property
    def ctype(self):
        """String representing the C type of the data in this ``Arg``."""
        return self.data.ctype

    @utils.cached_property
    def dtype(self):
        """Numpy datatype of this Arg"""
        return self.data.dtype

    @utils.cached_property
    def map(self):
        """The :class:`Map` via which the data is to be accessed."""
        return self._map

    @utils.cached_property
    def access(self):
        """Access descriptor. One of the constants of type :class:`Access`"""
        return self._access

    @utils.cached_property
    def _is_dat_view(self):
        return isinstance(self.data, types.DatView)

    @utils.cached_property
    def _is_mat(self):
        return isinstance(self.data, Mat)

    @utils.cached_property
    def _is_mixed_mat(self):
        return self._is_mat and self.data.sparsity.shape > (1, 1)

    @utils.cached_property
    def _is_global(self):
        return isinstance(self.data, Global)

    @utils.cached_property
    def _is_global_reduction(self):
        return self._is_global and self._access in {Access.INC, Access.MIN, Access.MAX}

    @utils.cached_property
    def _is_dat(self):
        return isinstance(self.data, Dat)

    @utils.cached_property
    def _is_mixed_dat(self):
        return isinstance(self.data, MixedDat)

    @utils.cached_property
    def _is_mixed(self):
        return self._is_mixed_dat or self._is_mixed_mat

    @utils.cached_property
    def _is_direct(self):
        return isinstance(self.data, Dat) and self.map is None

    @utils.cached_property
    def _is_indirect(self):
        return isinstance(self.data, Dat) and self.map is not None

    @mpi.collective
    def global_to_local_begin(self):
        """Begin halo exchange for the argument if a halo update is required.
        Doing halo exchanges only makes sense for :class:`Dat` objects.
        """
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self._is_direct:
            return
        if self.access is not Access.WRITE:
            self.data.global_to_local_begin(self.access)

    @mpi.collective
    def global_to_local_end(self):
        """Finish halo exchange for the argument if a halo update is required.
        Doing halo exchanges only makes sense for :class:`Dat` objects.
        """
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self._is_direct:
            return
        if self.access is not Access.WRITE:
            self.data.global_to_local_end(self.access)

    @mpi.collective
    def local_to_global_begin(self):
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self._is_direct:
            return
        if self.access in {Access.INC, Access.MIN, Access.MAX}:
            self.data.local_to_global_begin(self.access)

    @mpi.collective
    def local_to_global_end(self):
        assert self._is_dat, "Doing halo exchanges only makes sense for Dats"
        if self._is_direct:
            return
        if self.access in {Access.INC, Access.MIN, Access.MAX}:
            self.data.local_to_global_end(self.access)

    @mpi.collective
    def reduction_begin(self, comm):
        """Begin reduction for the argument if its access is INC, MIN, or MAX.
        Doing a reduction only makes sense for :class:`Global` objects."""
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        if self.access is not Access.READ:
            if self.access is Access.INC:
                op = mpi.MPI.SUM
            elif self.access is Access.MIN:
                op = mpi.MPI.MIN
            elif self.access is Access.MAX:
                op = mpi.MPI.MAX
            if mpi.MPI.VERSION >= 3:
                self._reduction_req = comm.Iallreduce(self.data._data, self.data._buf, op=op)
            else:
                comm.Allreduce(self.data._data, self.data._buf, op=op)

    @mpi.collective
    def reduction_end(self, comm):
        """End reduction for the argument if it is in flight.
        Doing a reduction only makes sense for :class:`Global` objects."""
        assert self._is_global, \
            "Doing global reduction only makes sense for Globals"
        if self.access is not Access.READ:
            if mpi.MPI.VERSION >= 3:
                self._reduction_req.Wait()
                self._reduction_req = None
            self.data._data[:] = self.data._buf[:]


class JITModule(caching.Cached):

    """Cached module encapsulating the generated :class:`ParLoop` stub.

    .. warning::

       Note to implementors.  This object is *cached* and therefore
       should not hold any references to objects you might want to be
       collected (such PyOP2 data objects)."""

    _cppargs = []
    _libraries = []
    _system_headers = []

    _cache = {}

    @classmethod
    def _cache_key(cls, kernel, iterset, *args, **kwargs):
        counter = itertools.count()
        seen = collections.defaultdict(lambda: next(counter))
        key = ((id(mpi.dup_comm(iterset.comm)), ) + kernel._wrapper_cache_key_ + iterset._wrapper_cache_key_
               + (iterset._extruded, (iterset._extruded and iterset.constant_layers), isinstance(iterset, Subset)))

        for arg in args:
            key += arg._wrapper_cache_key_
            for map_ in arg.map_tuple:
                key += (seen[map_],)

        key += (kwargs.get("iterate", None), cls, conf.configuration["simd_width"])

        return key

    def __init__(self, kernel, iterset, *args, **kwargs):
        r"""
        A cached compiled function to execute for a specified par_loop.

        See :func:`~.par_loop` for the description of arguments.

        .. warning ::

           Note to implementors.  This object is *cached*, and therefore
           should not hold any long term references to objects that
           you want to be collected.  In particular, after the
           ``args`` have been inspected to produce the compiled code,
           they **must not** remain part of the object's slots,
           otherwise they (and the :class:`~.Dat`\s, :class:`~.Map`\s
           and :class:`~.Mat`\s they reference) will never be collected.
        """
        # Return early if we were in the cache.
        if self._initialized:
            return
        self.comm = iterset.comm
        self._kernel = kernel
        self._fun = None
        self._iterset = iterset
        self._args = args
        self._iteration_region = kwargs.get('iterate', ALL)
        self._pass_layer_arg = kwargs.get('pass_layer_arg', False)
        # Copy the class variables, so we don't overwrite them
        self._cppargs = copy.deepcopy(type(self)._cppargs)
        self._libraries = copy.deepcopy(type(self)._libraries)
        self._system_headers = copy.deepcopy(type(self)._system_headers)
        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @mpi.collective
    def __call__(self, *args):
        return self._fun(*args)

    @utils.cached_property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @utils.cached_property
    def code_to_compile(self):
        from pyop2.codegen.builder import WrapperBuilder
        from pyop2.codegen.rep2loopy import generate

        builder = WrapperBuilder(kernel=self._kernel,
                                 iterset=self._iterset,
                                 iteration_region=self._iteration_region,
                                 pass_layer_to_kernel=self._pass_layer_arg)
        for arg in self._args:
            builder.add_argument(arg)

        wrapper = generate(builder)
        code = lp.generate_code_v2(wrapper)

        if self._kernel._cpp:
            from loopy.codegen.result import process_preambles
            preamble = "".join(process_preambles(getattr(code, "device_preambles", [])))
            device_code = "\n\n".join(str(dp.ast) for dp in code.device_programs)
            return preamble + "\nextern \"C\" {\n" + device_code + "\n}\n"
        return code.device_code()

    @PETSc.Log.EventDecorator()
    @mpi.collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        compiler = conf.configuration["compiler"]
        extension = "cpp" if self._kernel._cpp else "c"
        cppargs = self._cppargs
        cppargs += ["-I%s/include" % d for d in utils.get_petsc_dir()] + \
                   ["-I%s" % d for d in self._kernel._include_dirs] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        ldargs = ["-L%s/lib" % d for d in utils.get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in utils.get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        ldargs += self._kernel._ldargs

        self._fun = compilation.load(self,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=cppargs,
                                     ldargs=ldargs,
                                     restype=ctypes.c_int,
                                     compiler=compiler,
                                     comm=self.comm)
        # Blow away everything we don't need any more
        del self._args
        del self._kernel
        del self._iterset

    @utils.cached_property
    def argtypes(self):
        index_type = dtypes.as_ctypes(dtypes.IntType)
        argtypes = (index_type, index_type)
        argtypes += self._iterset._argtypes_
        for arg in self._args:
            argtypes += arg._argtypes_
        seen = set()
        for arg in self._args:
            maps = arg.map_tuple
            for map_ in maps:
                for k, t in zip(map_._kernel_args_, map_._argtypes_):
                    if k in seen:
                        continue
                    argtypes += (t,)
                    seen.add(k)
        return argtypes


class IterationRegion(enum.IntEnum):
    BOTTOM = 1
    TOP = 2
    INTERIOR_FACETS = 3
    ALL = 4


ON_BOTTOM = IterationRegion.BOTTOM
"""Iterate over the cells at the bottom of the column in an extruded mesh."""

ON_TOP = IterationRegion.TOP
"""Iterate over the top cells in an extruded mesh."""

ON_INTERIOR_FACETS = IterationRegion.INTERIOR_FACETS
"""Iterate over the interior facets of an extruded mesh."""

ALL = IterationRegion.ALL
"""Iterate over all cells of an extruded mesh."""


class AbstractParLoop(abc.ABC):
    """Represents the kernel, iteration space and arguments of a parallel loop
    invocation.
    .. note ::
        Users should not directly construct :class:`ParLoop` objects, but
        use :func:`pyop2.op2.par_loop` instead.
    An optional keyword argument, ``iterate``, can be used to specify
    which region of an :class:`ExtrudedSet` the parallel loop should
    iterate over.
    """

    @utils.validate_type(('kernel', Kernel, ex.KernelTypeError),
                         ('iterset', Set, ex.SetTypeError))
    def __init__(self, kernel, iterset, *args, **kwargs):
        # INCs into globals need to start with zero and then sum back
        # into the input global at the end.  This has the same number
        # of reductions but means that successive par_loops
        # incrementing into a global get the "right" value in
        # parallel.
        # Don't care about MIN and MAX because they commute with the reduction
        self._reduced_globals = {}
        for i, arg in enumerate(args):
            if arg._is_global_reduction and arg.access == Access.INC:
                glob = arg.data
                tmp = Global(glob.dim, data=np.zeros_like(glob.data_ro), dtype=glob.dtype)
                self._reduced_globals[tmp] = glob
                args[i].data = tmp

        # Always use the current arguments, also when we hit cache
        self._actual_args = args
        self._kernel = kernel
        self._is_layered = iterset._extruded
        self._iteration_region = kwargs.get("iterate", None)
        self._pass_layer_arg = kwargs.get("pass_layer_arg", False)

        check_iterset(self.args, iterset)

        if self._pass_layer_arg:
            if not self._is_layered:
                raise ValueError("Can't request layer arg for non-extruded iteration")

        self.iterset = iterset
        self.comm = iterset.comm

        for i, arg in enumerate(self._actual_args):
            arg.position = i
            arg.indirect_position = i
        for i, arg1 in enumerate(self._actual_args):
            if arg1._is_dat and arg1._is_indirect:
                for arg2 in self._actual_args[i:]:
                    # We have to check for identity here (we really
                    # want these to be the same thing, not just look
                    # the same)
                    if arg2.data is arg1.data and arg2.map is arg1.map:
                        arg2.indirect_position = arg1.indirect_position

        self.arglist = self.prepare_arglist(iterset, *self.args)

    def prepare_arglist(self, iterset, *args):
        """Prepare the argument list for calling generated code.
        :arg iterset: The :class:`Set` iterated over.
        :arg args: A list of :class:`Args`, the argument to the :fn:`par_loop`.
        """
        return ()

    @utils.cached_property
    def num_flops(self):
        iterset = self.iterset
        size = 1
        if iterset._extruded:
            region = self.iteration_region
            layers = np.mean(iterset.layers_array[:, 1] - iterset.layers_array[:, 0])
            if region is ON_INTERIOR_FACETS:
                size = layers - 2
            elif region not in [ON_TOP, ON_BOTTOM]:
                size = layers - 1
        return size * self._kernel.num_flops

    def log_flops(self, flops):
        pass

    @property
    @mpi.collective
    def _jitmodule(self):
        """Return the :class:`JITModule` that encapsulates the compiled par_loop code.
        Return None if the child class should deal with this in another way."""
        return None

    @utils.cached_property
    def _parloop_event(self):
        return profiling.timed_region("ParLoopExecute")

    @mpi.collective
    def compute(self):
        """Executes the kernel over all members of the iteration space."""
        with self._parloop_event:
            orig_lgmaps = []
            for arg in self.args:
                if arg._is_mat:
                    new_state = {Access.INC: Mat.ADD_VALUES,
                                 Access.WRITE: Mat.INSERT_VALUES}[arg.access]
                    for m in arg.data:
                        m.change_assembly_state(new_state)
                    arg.data.change_assembly_state(new_state)
                    # Boundary conditions applied to the matrix appear
                    # as modified lgmaps on the Arg. We set them onto
                    # the matrix so things are correctly dropped in
                    # insertion, and then restore the original lgmaps
                    # afterwards.
                    if arg.lgmaps is not None:
                        olgmaps = []
                        for m, lgmaps in zip(arg.data, arg.lgmaps):
                            olgmaps.append(m.handle.getLGMap())
                            m.handle.setLGMap(*lgmaps)
                        orig_lgmaps.append(olgmaps)
            self.global_to_local_begin()
            iterset = self.iterset
            arglist = self.arglist
            fun = self._jitmodule
            # Need to ensure INC globals are zero on entry to the loop
            # in case it's reused.
            for g in self._reduced_globals.keys():
                g._data[...] = 0
            self._compute(iterset.core_part, fun, *arglist)
            self.global_to_local_end()
            self._compute(iterset.owned_part, fun, *arglist)
            self.reduction_begin()
            self.local_to_global_begin()
            self.update_arg_data_state()
            for arg in reversed(self.args):
                if arg._is_mat and arg.lgmaps is not None:
                    for m, lgmaps in zip(arg.data, orig_lgmaps.pop()):
                        m.handle.setLGMap(*lgmaps)
            self.reduction_end()
            self.local_to_global_end()

    @mpi.collective
    def _compute(self, part, fun, *arglist):
        """Executes the kernel over all members of a MPI-part of the iteration space.
        :arg part: The :class:`SetPartition` to compute over
        :arg fun: The :class:`JITModule` encapsulating the compiled
             code (may be ignored by the backend).
        :arg arglist: The arguments to pass to the compiled code (may
             be ignored by the backend, depending on the exact implementation)"""
        raise RuntimeError("Must select a backend")

    @mpi.collective
    def global_to_local_begin(self):
        """Start halo exchanges."""
        for arg in self.unique_dat_args:
            arg.global_to_local_begin()

    @mpi.collective
    def global_to_local_end(self):
        """Finish halo exchanges"""
        for arg in self.unique_dat_args:
            arg.global_to_local_end()

    @mpi.collective
    def local_to_global_begin(self):
        """Start halo exchanges."""
        for arg in self.unique_dat_args:
            arg.local_to_global_begin()

    @mpi.collective
    def local_to_global_end(self):
        """Finish halo exchanges (wait on irecvs)"""
        for arg in self.unique_dat_args:
            arg.local_to_global_end()

    @utils.cached_property
    def _reduction_event_begin(self):
        return profiling.timed_region("ParLoopRednBegin")

    @utils.cached_property
    def _reduction_event_end(self):
        return profiling.timed_region("ParLoopRednEnd")

    @utils.cached_property
    def _has_reduction(self):
        return len(self.global_reduction_args) > 0

    @mpi.collective
    def reduction_begin(self):
        """Start reductions"""
        if not self._has_reduction:
            return
        with self._reduction_event_begin:
            for arg in self.global_reduction_args:
                arg.reduction_begin(self.comm)

    @mpi.collective
    def reduction_end(self):
        """End reductions"""
        if not self._has_reduction:
            return
        with self._reduction_event_end:
            for arg in self.global_reduction_args:
                arg.reduction_end(self.comm)
            # Finalise global increments
            for tmp, glob in self._reduced_globals.items():
                glob._data += tmp._data

    @mpi.collective
    def update_arg_data_state(self):
        r"""Update the state of the :class:`DataCarrier`\s in the arguments to the `par_loop`.
        This marks :class:`Mat`\s that need assembly."""
        for arg in self.args:
            access = arg.access
            if access is Access.READ:
                continue
            if arg._is_dat:
                arg.data.halo_valid = False
            if arg._is_mat:
                state = {Access.WRITE: Mat.INSERT_VALUES,
                         Access.INC: Mat.ADD_VALUES}[access]
                arg.data.assembly_state = state

    @utils.cached_property
    def dat_args(self):
        return tuple(arg for arg in self.args if arg._is_dat)

    @utils.cached_property
    def unique_dat_args(self):
        seen = {}
        unique = []
        for arg in self.dat_args:
            if arg.data not in seen:
                unique.append(arg)
                seen[arg.data] = arg
            elif arg.access != seen[arg.data].access:
                raise ValueError("Same Dat appears multiple times with different "
                                 "access descriptors")
        return tuple(unique)

    @utils.cached_property
    def global_reduction_args(self):
        return tuple(arg for arg in self.args if arg._is_global_reduction)

    @utils.cached_property
    def kernel(self):
        """Kernel executed by this parallel loop."""
        return self._kernel

    @utils.cached_property
    def args(self):
        """Arguments to this parallel loop."""
        return self._actual_args

    @utils.cached_property
    def is_layered(self):
        """Flag which triggers extrusion"""
        return self._is_layered

    @utils.cached_property
    def iteration_region(self):
        """Specifies the part of the mesh the parallel loop will
        be iterating over. The effect is the loop only iterates over
        a certain part of an extruded mesh, for example on top cells, bottom cells or
        interior facets."""
        return self._iteration_region


class ParLoop(AbstractParLoop):

    def log_flops(self, flops):
        PETSc.Log.logFlops(flops)

    def prepare_arglist(self, iterset, *args):
        arglist = iterset._kernel_args_
        for arg in args:
            arglist += arg._kernel_args_
        seen = set()
        for arg in args:
            maps = arg.map_tuple
            for map_ in maps:
                if map_ is None:
                    continue
                for k in map_._kernel_args_:
                    if k in seen:
                        continue
                    arglist += (k,)
                    seen.add(k)
        return arglist

    @utils.cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.iterset, *self.args,
                         iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg)

    @utils.cached_property
    def _compute_event(self):
        return profiling.timed_region("ParLoop_{0}_{1}".format(self.iterset.name, self._jitmodule._wrapper_name))

    @mpi.collective
    def _compute(self, part, fun, *arglist):
        with self._compute_event:
            self.log_flops(part.size * self.num_flops)
            fun(part.offset, part.offset + part.size, *arglist)


class PyParLoop(AbstractParLoop):
    """A stub implementation of "Python" parallel loops.

    This basically executes a python function over the iteration set,
    feeding it the appropriate data for each set entity.

    Example usage::

    .. code-block:: python

       s = op2.Set(10)
       d = op2.Dat(s)
       d2 = op2.Dat(s**2)

       m = op2.Map(s, s, 2, np.dstack(np.arange(4),
                                      np.roll(np.arange(4), -1)))

       def fn(x, y):
           x[0] = y[0]
           x[1] = y[1]

       d.data[:] = np.arange(4)

       op2.par_loop(fn, s, d2(op2.WRITE), d(op2.READ, m))

       print d2.data
       # [[ 0.  1.]
       #  [ 1.  2.]
       #  [ 2.  3.]
       #  [ 3.  0.]]

      def fn2(x, y):
          x[0] += y[0]
          x[1] += y[0]

      op2.par_loop(fn, s, d2(op2.INC), d(op2.READ, m[1]))

      print d2.data
      # [[ 1.  2.]
      #  [ 3.  4.]
      #  [ 5.  6.]
      #  [ 3.  0.]]
    """
    def __init__(self, kernel, *args, **kwargs):
        if not isinstance(kernel, types.FunctionType):
            raise ValueError("Expecting a python function, not a %r" % type(kernel))
        super().__init__(Kernel(kernel), *args, **kwargs)

    def _compute(self, part, *arglist):
        if part.set._extruded:
            raise NotImplementedError
        subset = isinstance(self.iterset, Subset)

        def arrayview(array, access):
            array = array.view()
            array.setflags(write=(access is not Access.READ))
            return array

        # Just walk over the iteration set
        for e in range(part.offset, part.offset + part.size):
            args = []
            if subset:
                idx = self.iterset._indices[e]
            else:
                idx = e
            for arg in self.args:
                if arg._is_global:
                    args.append(arrayview(arg.data._data, arg.access))
                elif arg._is_direct:
                    args.append(arrayview(arg.data._data[idx, ...], arg.access))
                elif arg._is_indirect:
                    args.append(arrayview(arg.data._data[arg.map.values_with_halo[idx], ...], arg.access))
                elif arg._is_mat:
                    if arg.access not in {Access.INC, Access.WRITE}:
                        raise NotImplementedError
                    if arg._is_mixed_mat:
                        raise ValueError("Mixed Mats must be split before assembly")
                    shape = tuple(map(operator.attrgetter("arity"), arg.map_tuple))
                    args.append(np.zeros(shape, dtype=arg.data.dtype))
                if args[-1].shape == ():
                    args[-1] = args[-1].reshape(1)
            self._kernel(*args)
            for arg, tmp in zip(self.args, args):
                if arg.access is Access.READ:
                    continue
                if arg._is_global:
                    arg.data._data[:] = tmp[:]
                elif arg._is_direct:
                    arg.data._data[idx, ...] = tmp[:]
                elif arg._is_indirect:
                    arg.data._data[arg.map.values_with_halo[idx], ...] = tmp[:]
                elif arg._is_mat:
                    if arg.access is Access.INC:
                        arg.data.addto_values(arg.map[0].values_with_halo[idx],
                                              arg.map[1].values_with_halo[idx],
                                              tmp)
                    elif arg.access is Access.WRITE:
                        arg.data.set_values(arg.map[0].values_with_halo[idx],
                                            arg.map[1].values_with_halo[idx],
                                            tmp)

        for arg in self.args:
            if arg._is_mat and arg.access is not Access.READ:
                # Queue up assembly of matrix
                arg.data.assemble()


def check_iterset(args, iterset):
    """Checks that the iteration set of the :class:`ParLoop` matches the
    iteration set of all its arguments. A :class:`MapValueError` is raised
    if this condition is not met."""

    if isinstance(iterset, Subset):
        _iterset = iterset.superset
    else:
        _iterset = iterset
    if conf.configuration["type_check"]:
        if isinstance(_iterset, MixedSet):
            raise ex.SetTypeError("Cannot iterate over MixedSets")
        for i, arg in enumerate(args):
            if arg._is_global:
                continue
            if arg._is_direct:
                if isinstance(_iterset, ExtrudedSet):
                    if arg.data.dataset.set != _iterset.parent:
                        raise ex.MapValueError(
                            "Iterset of direct arg %s doesn't match ParLoop iterset." % i)
                elif arg.data.dataset.set != _iterset:
                    raise ex.MapValueError(
                        "Iterset of direct arg %s doesn't match ParLoop iterset." % i)
                continue
            for j, m in enumerate(arg._map):
                if isinstance(_iterset, ExtrudedSet):
                    if m.iterset != _iterset and m.iterset not in _iterset:
                        raise ex.MapValueError(
                            "Iterset of arg %s map %s doesn't match ParLoop iterset." % (i, j))
                elif m.iterset != _iterset and m.iterset not in _iterset:
                    raise ex.MapValueError(
                        "Iterset of arg %s map %s doesn't match ParLoop iterset." % (i, j))


@mpi.collective
def par_loop(kernel, iterset, *args, **kwargs):
    r"""Invocation of an OP2 kernel

    :arg kernel: The :class:`Kernel` to be executed.
    :arg iterset: The iteration :class:`Set` over which the kernel should be
                  executed.
    :arg \*args: One or more :class:`base.Arg`\s constructed from a
                 :class:`Global`, :class:`Dat` or :class:`Mat` using the call
                 syntax and passing in an optionally indexed :class:`Map`
                 through which this :class:`base.Arg` is accessed and the
                 :class:`base.Access` descriptor indicating how the
                 :class:`Kernel` is going to access this data (see the example
                 below). These are the global data structures from and to
                 which the kernel will read and write.
    :kwarg iterate: Optionally specify which region of an
            :class:`ExtrudedSet` to iterate over.
            Valid values are:

              - ``ON_BOTTOM``: iterate over the bottom layer of cells.
              - ``ON_TOP`` iterate over the top layer of cells.
              - ``ALL`` iterate over all cells (the default if unspecified)
              - ``ON_INTERIOR_FACETS`` iterate over all the layers
                 except the top layer, accessing data two adjacent (in
                 the extruded direction) cells at a time.

    :kwarg pass_layer_arg: Should the wrapper pass the current layer
        into the kernel (as an ``int``). Only makes sense for
        indirect extruded iteration.

    .. warning ::
        It is the caller's responsibility that the number and type of all
        :class:`base.Arg`\s passed to the :func:`par_loop` match those expected
        by the :class:`Kernel`. No runtime check is performed to ensure this!

    :func:`par_loop` invocation is illustrated by the following example ::

      pyop2.par_loop(mass, elements,
                     mat(pyop2.INC, (elem_node[pyop2.i[0]]), elem_node[pyop2.i[1]]),
                     coords(pyop2.READ, elem_node))

    This example will execute the :class:`Kernel` ``mass`` over the
    :class:`Set` ``elements`` executing 3x3 times for each
    :class:`Set` member, assuming the :class:`Map` ``elem_node`` is of arity 3.
    The :class:`Kernel` takes four arguments, the first is a :class:`Mat` named
    ``mat``, the second is a field named ``coords``. The remaining two arguments
    indicate which local iteration space point the kernel is to execute.

    A :class:`Mat` requires a pair of :class:`Map` objects, one each
    for the row and column spaces. In this case both are the same
    ``elem_node`` map. The row :class:`Map` is indexed by the first
    index in the local iteration space, indicated by the ``0`` index
    to :data:`pyop2.i`, while the column space is indexed by
    the second local index.  The matrix is accessed to increment
    values using the ``pyop2.INC`` access descriptor.

    The ``coords`` :class:`Dat` is also accessed via the ``elem_node``
    :class:`Map`, however no indices are passed so all entries of
    ``elem_node`` for the relevant member of ``elements`` will be
    passed to the kernel as a vector.
    """
    if isinstance(kernel, types.FunctionType):
        return PyParLoop(kernel, iterset, *args, **kwargs).compute()
    return ParLoop(kernel, iterset, *args, **kwargs).compute()


def generate_single_cell_wrapper(iterset, args, forward_args=(), kernel_name=None, wrapper_name=None):
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

    forward_arg_types = [OpaqueType(fa) for fa in forward_args]
    empty_kernel = Kernel("", kernel_name)
    builder = WrapperBuilder(kernel=empty_kernel,
                             iterset=iterset, single_cell=True,
                             forward_arg_types=forward_arg_types)
    for arg in args:
        builder.add_argument(arg)
    wrapper = generate(builder, wrapper_name)
    code = lp.generate_code_v2(wrapper)

    return code.device_code()
