# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""OP2 OpenMP backend for fused/tiled loops."""

from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy as dcopy
import os

from base import _trace, IterationIndex, LazyComputation, Const, IterationSpace, \
    READ, WRITE, RW, MIN, MAX, INC
import host
import compilation
from caching import Cached
from host import Kernel
from profiling import lineprof, timed_region, profile
from logger import warning
from mpi import collective
from op2 import par_loop
from utils import flatten

import coffee
from coffee import base as coffee_ast
from coffee.utils import visit as coffee_ast_visit, \
    ast_update_id as coffee_ast_update_id

import slope_python as slope

# hard coded value to max openmp threads
_max_threads = 32


class Arg(host.Arg):

    def c_kernel_arg_name(self, i, j, idx=None):
        return "p_%s[%s]" % (self.c_arg_name(i, j), idx or 'tid')

    def c_local_tensor_name(self, i, j):
        return self.c_kernel_arg_name(i, j, _max_threads)

    def c_vec_dec(self, is_facet=False):
        cdim = self.data.dataset.cdim if self._flatten else 1
        return ";\n%(type)s *%(vec_name)s[%(arity)s]" % \
            {'type': self.ctype,
             'vec_name': self.c_vec_name(),
             'arity': self.map.arity * cdim * (2 if is_facet else 1)}


# Parallel loop API

class ParLoop(host.ParLoop):

    def __init__(self, kernel, iterset, inspection, *args, **kwargs):
        read_args = [a.data for a in args if a.access in [READ, RW]]
        written_args = [a.data for a in args if a.access in [RW, WRITE, MIN, MAX, INC]]
        LazyComputation.__init__(self, set(read_args) | Const._defs, set(written_args))

        self._kernel = kernel
        self._actual_args = args
        self._inspection = inspection
        self._it_space = self.build_itspace(iterset)

    @collective
    @profile
    def compute(self):
        """Execute the kernel over all members of the iteration space."""
        with timed_region("ParLoopChain: compute"):
            self._compute()

    @collective
    @lineprof
    def _compute(self):
        with timed_region("ParLoopChain: executor"):
            pass

    def build_itspace(self, iterset):
        # Note that the presence of any local iteration space is ignored
        block_shape = None
        return [IterationSpace(i, block_shape) for i in iterset]


# Possible Schedules as produced by an Inspector

class Schedule(object):
    """Represent an execution scheme for a sequence of :class:`ParLoop` objects."""

    def __init__(self, kernels):
        self._kernels = kernels

    def to_par_loop(self, loop_chain):
        """The argument ``loop_chain`` is a list of :class:`ParLoop` objects,
        which is expected to be mapped onto an optimized scheduling.

        In the simplest case, this Schedule's kernels exactly match the :class:`Kernel`
        objects in ``loop_chain``; in this case, the scheduling is given by the
        subsequent execution of the ``par_loops``; that is, resorting to the default
        PyOP2 execution model.

        In other scenarions, this Schedule's kernels could represent the fused
        version, or the tiled version, of the ``par_loops``' kernels in the provided
        ``loop_chain`` argument. In such a case, a sequence of :class:`ParLoop`
        objects using the fused/tiled kernels is returned.
        """
        raise NotImplementedError("Subclass must implement instantiation of ParLoops")


class PlainSchedule(Schedule):

    def __init__(self):
        super(PlainSchedule, self).__init__([])

    def to_par_loop(self, loop_chain):
        return loop_chain


class FusionSchedule(Schedule):
    """Schedule for a sequence of soft/hard fused :class:`ParLoop` objects."""

    def __init__(self, kernels, ranges):
        super(FusionSchedule, self).__init__(kernels)
        self._ranges = ranges

    def to_par_loop(self, loop_chain):
        offset = 0
        fused_par_loops = []
        for kernel, range in zip(self._kernels, self._ranges):
            iterset = loop_chain[offset].it_space.iterset
            args = flatten([loop.args for loop in loop_chain[offset:range]])
            fused_par_loops.append(par_loop(kernel, iterset, *args))
            offset = range
        return fused_par_loops


class TilingSchedule(Schedule):
    """Schedule for a sequence of tiled :class:`ParLoop` objects."""

    def __init__(self, kernels, inspection):
        super(TilingSchedule, self).__init__(kernels)
        self._inspection = inspection

    def _filter_args(self, loop_chain):
        """Uniquify arguments and access modes"""
        args = OrderedDict()
        for loop in loop_chain:
            # 1) Analyze the Args in each loop composing the chain and produce a
            # new sequence of Args for the tiled ParLoop. For example, consider the
            # Arg X and X.DAT be written to in ParLoop_0 (access mode WRITE) and
            # read from in ParLoop_1 (access mode READ); this means that in the
            # tiled ParLoop, X will have access mode RW
            for a in loop.args:
                args[a.data] = args.get(a.data, a)
                if a.access != args[a.data].access:
                    if READ in [a.access, args[a.data].access]:
                        # If a READ and some sort of write (MIN, MAX, RW, WRITE,
                        # INC), then the access mode becomes RW
                        args[a.data] = a.data(RW, a.map, a._flatten)
                    elif WRITE in [a.access, args[a.data].access]:
                        # Can't be a READ, so just stick to WRITE regardless of what
                        # the other access mode is
                        args[a.data] = a.data(WRITE, a.map, a._flatten)
                    else:
                        # Neither READ nor WRITE, so access modes are some
                        # combinations of RW, INC, MIN, MAX. For simplicity,
                        # just make it RW.
                        args[a.data] = a.data(RW, a.map, a._flatten)
        return args.values()

    def _filter_itersets(self, loop_chain):
        return [loop.it_space.iterset for loop in loop_chain]

    def to_par_loop(self, loop_chain):
        args = self._filter_args(loop_chain)
        iterset = self._filter_itersets(loop_chain)
        return [ParLoop(self._kernels, iterset, self._inspection, *args)]


# Loop chain inspection

class Inspector(Cached):
    """An inspector is used to fuse or tile a sequence of :class:`ParLoop` objects.

    The inspector is implemented by the SLOPE library, which the user makes
    visible by setting the environment variable ``SLOPE_DIR`` to the value of
    the root SLOPE directory."""

    _cache = {}
    _modes = ['soft', 'hard', 'tile']

    @classmethod
    def _cache_key(cls, name, loop_chain, tile_size):
        key = (name, tile_size)
        for loop in loop_chain:
            for arg in loop.args:
                if arg._is_global:
                    key += (arg.data.dim, arg.data.dtype, arg.access)
                elif arg._is_dat:
                    if isinstance(arg.idx, IterationIndex):
                        idx = (arg.idx.__class__, arg.idx.index)
                    else:
                        idx = arg.idx
                    map_arity = arg.map.arity if arg.map else None
                    key += (arg.data.dim, arg.data.dtype, map_arity, idx, arg.access)
                elif arg._is_mat:
                    idxs = (arg.idx[0].__class__, arg.idx[0].index,
                            arg.idx[1].index)
                    map_arities = (arg.map[0].arity, arg.map[1].arity)
                    key += (arg.data.dims, arg.data.dtype, idxs, map_arities, arg.access)
        return key

    def __init__(self, name, loop_chain, tile_size):
        if self._initialized:
            return
        if not hasattr(self, '_inspected'):
            # Initialization can occur more than once, but only the first time
            # this attribute should be set
            self._inspected = 0
        self._name = name
        self._tile_size = tile_size
        self._loop_chain = loop_chain

    def inspect(self, mode):
        """Inspect this Inspector's loop chain and produce a Schedule object.

        :param mode: can take any of the values in ``Inspector._modes``, namely
                     ``soft``, ``hard``, and ``tile``. If ``soft`` is specified,
                     only soft fusion takes place; that is, only consecutive loops
                     over the same iteration set that do not present RAW or WAR
                     dependencies through indirections are fused. If ``hard`` is
                     specified, then first ``soft`` is applied, followed by fusion
                     of loops over different iteration sets, provided that RAW or
                     WAR dependencies are not present. If ``tile`` is specified,
                     than tiling through the SLOPE library takes place just after
                     ``soft`` and ``hard`` fusion.
        """
        self._inspected += 1
        if self._heuristic_skip_inspection():
            # Heuristically skip this inspection if there is a suspicion the
            # overhead is going to be too much; for example, when the loop
            # chain could potentially be execution only once or a few time.
            # Blow away everything we don't need any more
            del self._name
            del self._loop_chain
            del self._tile_size
            return PlainSchedule()
        elif hasattr(self, '_schedule'):
            # An inspection plan is in cache.
            # It should not be possible to pull a jit module out of the cache
            # /with/ the loop chain
            if hasattr(self, '_loop_chain'):
                raise RuntimeError("Inspector is holding onto loop_chain, memory leaks!")
            # The fusion mode was recorded, and must match the one provided for
            # this inspection
            if self.mode != mode:
                raise RuntimeError("Cached Inspector's mode doesn't match")
            return self._schedule
        elif not hasattr(self, '_loop_chain'):
            # The inspection should be executed /now/. We weren't in the cache,
            # so we /must/ have a loop chain
            raise RuntimeError("Inspector must have a loop chain associated with it")
        # Finally, we check the legality of `mode`
        if mode not in Inspector._modes:
            raise TypeError("Inspection accepts only %s fusion modes",
                            str(Inspector._modes))
        self._mode = mode
        mode = Inspector._modes.index(mode)

        with timed_region("ParLoopChain `%s`: inspector" % self._name):
            self._soft_fuse()
            if mode > 0:
                self._tile()

        # A schedule has been computed by any of /_soft_fuse/, /_hard_fuse/ or
        # or /_tile/; therefore, consider this Inspector initialized, and
        # retrievable from cache in subsequent calls to inspect().
        self._initialized = True

        # Blow away everything we don't need any more
        del self._name
        del self._loop_chain
        del self._tile_size
        return self._schedule

    def _heuristic_skip_inspection(self):
        """Decide heuristically whether to run an inspection or not."""
        # At the moment, a simple heuristic is used: if the inspection is
        # requested more than once, then it is performed
        if self._inspected < 2:
            return True
        return False

    def _soft_fuse(self):
        """Fuse consecutive loops over the same iteration set by concatenating
        kernel bodies and creating new :class:`ParLoop` objects representing
        the fused sequence.

        The conditions under which two loops over the same iteration set are
        hardly fused are:

            * They are both direct, OR
            * One is direct and the other indirect

        This is detailed in the paper::

            "Mesh Independent Loop Fusion for Unstructured Mesh Applications"

        from C. Bertolli et al.
        """

        def do_fuse(loop_a, loop_b, unique_id):
            """Fuse ``loop_b`` into ``loop_a``. All symbols identifiers in
            ``loop_b`` are modified appending the suffix ``unique_id``."""
            kernel_a, kernel_b = loop_a.kernel, loop_b.kernel

            # 1) Name and additional parameters of the fused kernel
            name = 'fused_%s_%s' % (kernel_a._name, kernel_b._name)
            opts = dict(kernel_a._opts.items() + kernel_b._opts.items())
            include_dirs = kernel_a._include_dirs + kernel_b._include_dirs
            headers = kernel_a._headers + kernel_b._headers
            user_code = "\n".join([kernel_a._user_code, kernel_b._user_code])

            # 2) Fuse the ASTs
            fused_ast, ast_b = dcopy(kernel_a._ast), dcopy(kernel_b._ast)
            fused_ast.name = name
            # 2-A) Concatenate the arguments in the signature
            fused_ast.args.extend(ast_b.args)
            # 2-B) Uniquify symbols identifiers
            ast_b_info = coffee_ast_visit(ast_b, None)
            ast_b_decls = ast_b_info['decls']
            ast_b_symbols = ast_b_info['symbols']
            for str_sym, decl in ast_b_decls.items():
                for symbol in ast_b_symbols.keys():
                    coffee_ast_update_id(symbol, str_sym, unique_id)
            # 2-C) Concatenate kernels' bodies
            marker_ast_node = coffee_ast.FlatBlock("\n\n// Begin of fused kernel\n\n")
            fused_ast.children[0].children.extend([marker_ast_node] + ast_b.children)

            args = loop_a.args + loop_b.args
            kernel = Kernel(fused_ast, name, opts, include_dirs, headers, user_code)
            return par_loop(kernel, loop_a.it_space.iterset, *args)

        # In the process of soft fusion, temporary "fake" ParLoops are constructed
        # to simplify tracking of data dependencies.
        # In the following, the word "range" indicates an offset in the original
        # loop chain to represent of slice of original ParLoops that have been fused
        fused_loops_ranges, fusing_loop_range = [], []
        base_loop = self._loop_chain[0]
        for i, loop in enumerate(self._loop_chain[1:], 1):
            if base_loop.it_space != loop.it_space or \
                    (base_loop.is_indirect and loop.is_indirect):
                # Fusion not legal
                fused_loops_ranges.append((base_loop, i))
                base_loop = loop
                fusing_loop_range = []
                continue
            elif base_loop.is_direct and loop.is_direct:
                base_loop = do_fuse(base_loop, loop, i)
            elif base_loop.is_direct and loop.is_indirect:
                base_loop = do_fuse(loop, base_loop, i)
            elif base_loop.is_indirect and loop.is_direct:
                base_loop = do_fuse(base_loop, loop, i)
            fusing_loop_range = [(base_loop, i+1)]
        fused_loops_ranges.extend(fusing_loop_range)

        fused_loop_chain, ranges = zip(*fused_loops_ranges)
        fused_kernels = [loop.kernel for loop in fused_loop_chain]

        self._loop_chain = fused_loop_chain
        self._schedule = FusionSchedule(fused_kernels, ranges)

    def _tile(self):
        """Tile consecutive loops over different iteration sets characterized
        by RAW and WAR dependencies. This requires interfacing with the SLOPE
        library."""
        inspector = slope.Inspector('OMP', self._tile_size)

        # Build arguments types and values
        arguments = []
        sets, maps, loops = set(), {}, []
        for loop in self._loop_chain:
            slope_desc = set()
            # Add sets
            sets.add((loop.it_space.name, loop.it_space.core_size))
            for a in loop.args:
                map = a.map
                # Add map
                if map:
                    maps[map.name] = (map.name, map.iterset.name,
                                      map.toset.name, map.values)
                # Track descriptors
                desc_name = "DIRECT" if not a.map else a.map.name
                desc_access = a.access._mode  # Note: same syntax as SLOPE
                slope_desc.add((desc_name, desc_access))
            # Add loop
            loops.append((loop.kernel.name, loop.it_space.name, list(slope_desc)))
        # Provide structure of loop chain to the SLOPE's inspector
        inspector.add_sets(sets)
        arguments.extend([inspector.add_maps(maps.values())])
        inspector.add_loops(loops)
        # Get type and value of any additional arguments that the SLOPE's inspector
        # expects
        arguments.extend(inspector.set_external_dats())

        # Arguments types and values
        argtypes, argvalues = zip(*arguments)

        # Generate inspector C code
        src = inspector.generate_code()

        # Return type of the inspector
        rettype = slope.Executor._ctype

        # Compiler and linker options
        slope_dir = os.environ['SLOPE_DIR']
        compiler = coffee.plan.compiler.get('name')
        cppargs = slope.get_compile_opts(compiler)
        cppargs += ['-I%s/sparsetiling/include' % slope_dir]
        ldargs = ['-L%s/lib' % slope_dir, '-l%s' % slope.get_lib_name()]

        # Compile and run inspector
        fun = compilation.load(src, "cpp", "inspector", cppargs, ldargs,
                               argtypes, rettype, compiler)
        inspection = fun(*argvalues)

        executor = slope.Executor(inspector)

        # Generate executor C code
        src = executor.generate_code()

        # Create the Kernel object, which contains the executor code
        kernel = Kernel(src, "executor")
        self._schedule = TilingSchedule(kernel, inspection)

    @property
    def mode(self):
        return self._mode


# Interface for triggering loop fusion

def reschedule_loops(name, loop_chain, tile_size, mode='tile'):
    """Given a list of :class:`ParLoop` in ``loop_chain``, return a list of new
    :class:`ParLoop` objects implementing an optimized scheduling of the loop chain.

    .. note:: The unmodified loop chain is instead returned if any of these
    conditions verify:

        * a global reduction is present;
        * at least one loop iterates over an extruded set
    """
    # Loop fusion is performed through the SLOPE library, which must be accessible
    # by reading the environment variable SLOPE_DIR
    try:
        os.environ['SLOPE_DIR']
    except KeyError:
        warning("Set the env variable SLOPE_DIR to the location of SLOPE")
        warning("Loops won't be fused, and plain pyop2.ParLoops will be executed")
        return loop_chain

    # If there are global reduction or extruded sets are present, return
    if any([l._reduced_globals for l in loop_chain]) or \
            any([l.is_layered for l in loop_chain]):
        return loop_chain

    # Get an inspector for fusing this loop_chain, possibly retrieving it from
    # the cache, and obtain the fused ParLoops through the schedule it produces
    inspector = Inspector(name, loop_chain, tile_size)
    schedule = inspector.inspect(mode)
    return schedule.to_par_loop(loop_chain)


@contextmanager
def loop_chain(name, time_unroll=1, tile_size=0):
    """Analyze the sub-trace of loops lazily evaluated in this contextmanager ::

        [loop_0, loop_1, ..., loop_n-1]

    and produce a new sub-trace (``m <= n``) ::

        [fused_loops_0, fused_loops_1, ..., fused_loops_m-1, peel_loops]

    which is eventually inserted in the global trace of :class:`ParLoop` objects.

    That is, sub-sequences of :class:`ParLoop` objects are potentially replaced by
    new :class:`ParLoop` objects representing the fusion or the tiling of the
    original trace slice.

    :param name: identifier of the loop chain
    :param time_unroll: in a time stepping loop, the length of the loop chain
                        is given by ``num_loops * time_unroll``, where ``num_loops``
                        is the number of loops per time loop iteration. Therefore,
                        setting this value to a number greater than 1 enables
                        fusing/tiling longer loop chains (optional, defaults to 1).
    :param tile_size: suggest a tile size in case loop tiling is used (optional).
    """
    trace = _trace._trace
    stamp = trace[-1:]

    yield

    if time_unroll < 1:
        return

    start_point = trace.index(stamp[0])+1 if stamp else 0
    extracted_loop_chain = trace[start_point:]

    # Unroll the loop chain ``time_unroll`` times before fusion/tiling
    total_loop_chain = loop_chain.unrolled_loop_chain + extracted_loop_chain
    if len(total_loop_chain) / len(extracted_loop_chain) == time_unroll:
        trace[start_point:] = reschedule_loops(name, total_loop_chain, tile_size)
        loop_chain.unrolled_loop_chain = []
    else:
        unrolled_loop_chain.extend(total_loop_chain)
loop_chain.unrolled_loop_chain = []
