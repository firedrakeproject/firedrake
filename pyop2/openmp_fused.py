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
import os

from base import LazyComputation, Const, _trace, \
    READ, WRITE, RW, INC, MIN, MAX
import host
import compilation
from host import Kernel  # noqa: for inheritance
from openmp import _detect_openmp_flags
from profiling import lineprof, timed_region, profile
from logger import warning
from mpi import collective

import slope_python as slope

# hard coded value to max openmp threads
_max_threads = 32
# cache of inspectors for all of the loop chains encountered in the execution
_inspectors = {}
# track the loop chain in a time stepping loop which is being unrolled
# this is a 2-tuple: (loop_chain_name, loops)
_active_loop_chain = ()


class LoopChain(object):
    """Define a loop chain through a set of information:

        * loops: a list of loops crossed
        * time_unroll: an integer indicating how many times the loop chain was
                       unrolled in the time stepping loop embedding it
    """

    def __init__(self, loops, time_unroll):
        self.loops = loops
        self.time_unroll = time_unroll


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


class Inspector(object):
    """Represent the inspector for the fused sequence of :class:`ParLoop`.

    The inspector is implemented by the SLOPE library, which the user makes
    visible by setting the environment variable ``SLOPE_DIR`` to the value of
    the root SLOPE directory."""

    def __init__(self, it_spaces, args):
        self._it_spaces = it_spaces
        self._args = args

    def compile(self):
        slope_dir = os.environ['SLOPE_DIR']
        cppargs = slope.get_compile_opts()
        cppargs += ["-I%s/sparsetiling/include" % slope_dir]
        ldargs = ["-L%s/lib" % slope_dir, "-l%s" % slope.get_lib_name()]

        inspector = slope.Inspector()

        # Build arguments values
        argvalues = []
        # - Sets
        argvalues += [inspector.add_sets([(s.name, s.core_size) for s
                                          in set(self._it_spaces)])]

        # Build arguments types
        argtypes = inspector.get_arg_types()

        # Generate inspector C code
        src = inspector.generate_code()

        fun = compilation.load(src, "cpp", "inspector", cppargs, ldargs,
                               argtypes, None, "intel")
        fun(*argvalues, argtypes=argtypes, restype=None)


# Parallel loop API


class JITModule(host.JITModule):
    """Represent the executor code for the fused sequence of :class:`ParLoop`"""

    ompflag, omplib = _detect_openmp_flags()
    _cppargs = [os.environ.get('OMP_CXX_FLAGS') or ompflag]
    _libraries = [ompflag] + [os.environ.get('OMP_LIBS') or omplib]
    _system_headers = ['#include <omp.h>']

    _wrapper = """
"""

    def generate_code(self):

        # Bits of the code to generate are the same as that for sequential
        code_dict = super(JITModule, self).generate_code()

        return code_dict


class ParLoop(host.ParLoop):

    def __init__(self, name, loop_chain, it_spaces, args):
        LazyComputation.__init__(self,
                                 set([a.data for a in args if a.access in [READ, RW]]) | Const._defs,
                                 set([a.data for a in args if a.access in [RW, WRITE, MIN, MAX, INC]]))
        self._name = name
        self._loop_chain = loop_chain
        self._actual_args = args
        self._it_spaces = it_spaces
        self._inspector = None

    @collective
    @profile
    def compute(self):
        """Execute the kernel over all members of the iteration space."""
        with timed_region("ParLoopChain `%s`: compute" % self.name):
            self._compute()

    @collective
    @lineprof
    def _compute(self):
        self._get_plan()

        with timed_region("ParLoopChain `%s`: executor" % self.name):
            pass

    def _get_plan(self):
        """Retrieve an execution plan by generating, jit-compiling and running
        an inspection scheme implemented through calls to the SLOPE library.

        The result is saved in the global variable ``_inspectors``, so inspection
        needs be executed at most once."""

        global _inspectors

        if _inspectors.get(self._name):
            return _inspectors[self._name]

        inspector = Inspector(self.it_space, self.args)
        with timed_region("ParLoopChain `%s`: inspector" % self.name):
            inspector.compile()
        # Cache the inspection output
        _inspectors[self._name] = inspector

    @property
    def it_space(self):
        return self._it_spaces

    @property
    def inspector(self):
        return self._inspector

    @property
    def loop_chain(self):
        return self._loop_chain

    @property
    def name(self):
        return self._name


def fuse_loops(name, loop_chain):
    """Given a list of :class:`openmp.ParLoop`, return a :class:`fused_openmp.ParLoop`
    object representing the fusion of the loop chain. The original list is instead
    returned if ``loop_chain`` presents one of the following non currently supported
    features:

        * a global reduction;
        * iteration over extruded sets
    """

    # Loop fusion is performed through the SLOPE library, which must be accessible
    # by reading the environment variable SLOPE_DIR
    try:
        os.environ['SLOPE_DIR']
    except KeyError:
        warning("Set the env variable SLOPE_DIR to the location of SLOPE")
        warning("Loops won't be fused, and plain pyop2.ParLoops will be executed")
        return loop_chain

    # If there are global reduction, return
    if any([l._reduced_globals for l in loop_chain]) or \
            any([l.is_layered for l in loop_chain]):
        return loop_chain

    # Analyze the Args in each loop composing the chain and produce a new sequence
    # of Args for the fused ParLoop. For example, consider the Arg X and X.DAT be
    # written to in ParLoop_0 (access mode WRITE) and read from in ParLoop_1 (access
    # mode READ); this means that in the fused ParLoop, X will have access mode RW
    args = OrderedDict()
    for l in loop_chain:
        for a in l.args:
            args[a.data] = args.get(a.data, a)
            if a.access != args[a.data].access:
                if READ in [a.access, args[a.data].access]:
                    # If a READ and some sort of write (MIN, MAX, RW, WRITE, INC),
                    # then the access mode becomes RW
                    args[a.data] = a.data(RW, a.map, a._flatten)
                elif WRITE in [a.access, args[a.data].access]:
                    # Can't be a READ, so just stick to WRITE regardless of what
                    # the other access mode is
                    args[a.data] = a.data(WRITE, a.map, a._flatten)
                else:
                    # Neither READ nor WRITE, so access modes are some combinations
                    # of RW, INC, MIN, MAX. For simplicity, just make it RW
                    args[a.data] = a.data(RW, a.map, a._flatten)

    # The iteration space of the fused loop is the union of the iteration spaces
    # of the individual loops composing the chain
    it_spaces = [l.it_space for l in loop_chain]

    return ParLoop(name, loop_chain, it_spaces, args.values())


@contextmanager
def loop_chain(name, time_unroll=0, tile_size=0):
    """Analyze the trace of lazily evaluated loops ::

        [loop_0, loop_1, ..., loop_n-1]

    and produce a new trace ::

        [fused_loopchain_0, fused_loopchain_1, ..., fused_loopchain_n-1, peel_loop_i]

    where sequences of loops of length ``_max_loop_chain_length`` (which is a global
    variable) are replaced by openmp_fused.ParLoop instances, plus a trailing
    sequence of loops in case ``n`` is greater than and does not divide equally
    ``_max_loop_chain_length``.

    :param name: identifier of the loop chain
    :param time_unroll: if in a time stepping loop, the length of the loop chain
                        will be ``num_loops * time_unroll``, where ``num_loops``
                        is the number of loops in the time stepping loop. By
                        setting this parameter to a value greater than 0, the runtime
                        system is informed that the loop chain should be extracted
                        from a time stepping loop, which can results in better
                        fusion (by 1- descarding the first loop chain iteration,
                        in which some time-independent loops may be evaluated
                        and stored in temporaries for later retrieval, and 2-
                        allowing tiling through inspection/execution).
                        If the value of this parameter is greater than zero, but
                        the loop chain is not actually in a time stepping loop,
                        the behaviour is undefined.
    :param tile_size: suggest a tile size in case loop fusion can only be achieved
                      trough tiling within a time stepping loop.
    """

    global _active_loop_chain
    trace, new_trace = _trace._trace, []

    # Mark the last loop out of the loop chain
    pre_loop_chain = trace[-1:]
    yield
    start_point = trace.index(pre_loop_chain[0])+1 if pre_loop_chain else 0
    loop_chain = trace[start_point:]

    if time_unroll == 0:
        # If *not* in a time stepping loop, just replace the loops in the trace
        # with a fused version
        trace[start_point:] = [fuse_loops(name, loop_chain)]
        _active_loop_chain = ()
        return
    if not _active_loop_chain or _active_loop_chain[0] != name:
        # In a time stepping loop; open a new context and discard first iteration
        # by returning immediately, since the first iteration may be characterized
        # by the computation of time-independent loops (i.e., loops that are
        # executed only once and accessed in read-only mode successively)
        _active_loop_chain = (name, [])
        return
    else:
        # In a time stepping loop; unroll the loop chain ``time_unroll`` times
        # before replacing with the fused version
        unrolled_loop_chain = _active_loop_chain[1]
        current_loop_chain = unrolled_loop_chain + loop_chain
        if len(current_loop_chain) / len(loop_chain) == time_unroll:
            trace[start_point:] = [fuse_loops(name, current_loop_chain)]
        else:
            unrolled_loop_chain.extend(loop_chain)
