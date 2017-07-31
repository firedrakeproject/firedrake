# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2016, Imperial College London and
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

"""Interface for loop fusion. Some functions will be called from within PyOP2
itself, whereas others directly from application code."""


import os
from contextlib import contextmanager

from pyop2.base import _LazyMatOp
from pyop2.mpi import MPI
from pyop2.logger import warning, debug
from pyop2.utils import flatten

try:
    from pyslope import slope
    backend = os.environ.get('SLOPE_BACKEND')
    if backend not in ['SEQUENTIAL', 'OMP']:
        backend = 'SEQUENTIAL'
    if MPI.COMM_WORLD.size > 1:
        if backend == 'SEQUENTIAL':
            backend = 'ONLY_MPI'
        if backend == 'OMP':
            backend = 'OMP_MPI'
    slope.set_exec_mode(backend)
    debug("SLOPE backend set to %s" % backend)
except ImportError:
    slope = None

lazy_trace_name = 'lazy_trace'
"""The default name for sequences of lazily evaluated :class:`ParLoop`s."""

from pyop2.fusion.transformer import Inspector
from pyop2.fusion import extended


def fuse(name, loop_chain, **kwargs):
    """Apply fusion (and possibly tiling) to an iterator of :class:`ParLoop`
    obecjts, which we refer to as ``loop_chain``. Return an iterator of
    :class:`ParLoop` objects, in which some loops may have been fused or tiled.
    If fusion could not be applied, return the unmodified ``loop_chain``.

    .. note::
       At the moment, the following features are not supported, in which
       case the unmodified ``loop_chain`` is returned.

        * mixed ``Datasets`` and ``Maps``;
        * extruded ``Sets``

    .. note::
       Tiling cannot be applied if any of the following conditions verifies:

        * a global reduction/write occurs in ``loop_chain``
    """
    # If there is nothing to fuse, just return
    if len(loop_chain) in [0, 1]:
        return loop_chain

    # Are there _LazyMatOp objects (i.e., synch points) preventing fusion?
    remainder = []
    synch_points = [l for l in loop_chain if isinstance(l, _LazyMatOp)]
    if synch_points:
        # Fuse only the sub-sequence before the first synch point
        synch_point = loop_chain.index(synch_points[0])
        remainder, loop_chain = loop_chain[synch_point:], loop_chain[:synch_point]

    # Return if there is nothing to fuse (e.g. only _LazyMatOp objects were present)
    if len(loop_chain) in [0, 1]:
        return loop_chain + remainder

    # Get an inspector for fusing this /loop_chain/. If there's a cache hit,
    # return the fused par loops straight away. Otherwise, try to run an inspection.
    options = {
        'log': kwargs.get('log', False),
        'mode': kwargs.get('mode', 'hard'),
        'ignore_war': kwargs.get('ignore_war', False),
        'use_glb_maps': kwargs.get('use_glb_maps', False),
        'use_prefetch': kwargs.get('use_prefetch', 0),
        'tile_size': kwargs.get('tile_size', 1),
        'seed_loop': kwargs.get('seed_loop', 0),
        'extra_halo': kwargs.get('extra_halo', False),
        'coloring': kwargs.get('coloring', 'default')
    }
    inspector = Inspector(name, loop_chain, **options)
    if inspector._initialized:
        return inspector.schedule(loop_chain) + remainder

    # Otherwise, is the inspection legal ?
    mode = kwargs.get('mode', 'hard')
    force_glb = kwargs.get('force_glb', False)

    # Skip if loops in /loop_chain/ are already /fusion/ objects: this could happen
    # when loops had already been fused in a /loop_chain/ context
    if any(isinstance(l, extended.ParLoop) for l in loop_chain):
        return loop_chain + remainder

    # Global reductions are dangerous for correctness, so avoid fusion unless the
    # user is forcing it
    if not force_glb and any(l._reduced_globals for l in loop_chain):
        return loop_chain + remainder

    # Loop fusion requires modifying kernels, so ASTs must be available
    if not mode == 'only_tile':
        if any(not l.kernel._ast or l.kernel._attached_info['flatblocks'] for l in loop_chain):
            return loop_chain + remainder

    # Mixed still not supported
    if any(a._is_mixed for a in flatten([l.args for l in loop_chain])):
        return loop_chain + remainder

    # Extrusion still not supported
    if any(l.is_layered for l in loop_chain):
        return loop_chain + remainder

    # If tiling is requested, SLOPE must be visible
    if mode in ['tile', 'only_tile'] and not slope:
        warning("Couldn't locate SLOPE. Falling back to plain op2.ParLoops.")
        return loop_chain + remainder

    schedule = inspector.inspect()
    return schedule(loop_chain) + remainder


@contextmanager
def loop_chain(name, **kwargs):
    """Analyze the sub-trace of loops lazily evaluated in this contextmanager ::

        [loop_0, loop_1, ..., loop_n-1]

    and produce a new sub-trace (``m <= n``) ::

        [fused_loops_0, fused_loops_1, ..., fused_loops_m-1, peel_loops]

    which is eventually inserted in the global trace of :class:`ParLoop` objects.

    That is, sub-sequences of :class:`ParLoop` objects are potentially replaced by
    new :class:`ParLoop` objects representing the fusion or the tiling of the
    original trace slice.

    :arg name: identifier of the loop chain
    :arg kwargs:
        * mode (default='hard'): the fusion/tiling mode (accepted: soft, hard,
            tile, only_tile, only_omp): ::
            * soft: consecutive loops over the same iteration set that do
                not present RAW or WAR dependencies through indirections
                are fused.
            * hard: fuse consecutive loops presenting inc-after-inc
                dependencies, on top of soft fusion.
            * tile: apply tiling through the SLOPE library, on top of soft
                and hard fusion.
            * only_tile: apply tiling through the SLOPE library, but do not
                apply soft or hard fusion
            * only_omp: ompize individual parloops through the SLOPE library
                (i.e., no fusion takes place)
        * tile_size: (default=1) suggest a starting average tile size.
        * num_unroll (default=1): in a time stepping loop, the length of the loop
            chain is given by ``num_loops * num_unroll``, where ``num_loops`` is the
            number of loops per time loop iteration. Setting this value to something
            greater than 1 may enable fusing longer chains.
        * seed_loop (default=0): the seed loop from which tiles are derived. Ignored
            in case of MPI execution, in which case the seed loop is enforced to 0.
        * force_glb (default=False): force tiling even in presence of global
            reductions. In this case, the user becomes responsible of semantic
            correctness.
        * coloring (default='default'): set a coloring scheme for tiling. The ``default``
            coloring should be used because it ensures correctness by construction,
            based on the execution mode (sequential, openmp, mpi, mixed). So this
            should be changed only if totally confident with what is going on.
            Possible values are default, rand, omp; these are documented in detail
            in the documentation of the SLOPE library.
        * explicit (default=None): an iterator of 3-tuples (f, l, ts), each 3-tuple
            indicating a sub-sequence of loops to be inspected. ``f`` and ``l``
            represent, respectively, the first and last loop index of the sequence;
            ``ts`` is the tile size for the sequence.
        * ignore_war: (default=False) inform SLOPE that inspection doesn't need
            to care about write-after-read dependencies.
        * log (default=False): output inspector and loop chain info to a file.
        * use_glb_maps (default=False): when tiling, use the global maps provided by
            PyOP2, rather than the ones constructed by SLOPE.
        * use_prefetch (default=False): when tiling, try to prefetch the next iteration.
    """
    assert name != lazy_trace_name, "Loop chain name must differ from %s" % lazy_trace_name

    num_unroll = kwargs.setdefault('num_unroll', 1)
    tile_size = kwargs.setdefault('tile_size', 1)
    kwargs.setdefault('seed_loop', 0)
    kwargs.setdefault('use_glb_maps', False)
    kwargs.setdefault('use_prefetch', 0)
    kwargs.setdefault('coloring', 'default')
    kwargs.setdefault('ignore_war', False)
    explicit = kwargs.pop('explicit', None)

    # Get a snapshot of the trace before new par loops are added within this
    # context manager
    from pyop2.base import _trace
    stamp = list(_trace._trace)

    yield

    trace = _trace._trace
    if trace == stamp:
        return

    # What's the first item /B/ that appeared in the trace /before/ entering the
    # context manager and that still has to be executed ?
    # The loop chain will be (B, end_of_current_trace]
    bottom = 0
    for i in reversed(stamp):
        if i in trace:
            bottom = trace.index(i) + 1
            break
    extracted_trace = trace[bottom:]

    # Three possibilities:
    if num_unroll < 1:
        # 1) No tiling requested, but the openmp backend was set, so we still try to
        # omp-ize the loops with SLOPE
        if slope and slope.get_exec_mode() in ['OMP', 'OMP_MPI'] and tile_size > 0:
            block_size = tile_size    # This is rather a 'block' size (no tiling)
            options = {'mode': 'only_omp',
                       'tile_size': block_size}
            new_trace = [Inspector(name, [loop], **options).inspect()([loop])
                         for loop in extracted_trace]
            trace[bottom:] = list(flatten(new_trace))
            _trace.evaluate_all()
    elif explicit:
        # 2) Tile over subsets of loops in the loop chain, as specified
        # by the user through the /explicit/ list
        prev_last = 0
        transformed = []
        for i, (first, last, tile_size) in enumerate(explicit):
            sub_name = "%s_sub%d" % (name, i)
            kwargs['tile_size'] = tile_size
            transformed.extend(extracted_trace[prev_last:first])
            transformed.extend(fuse(sub_name, extracted_trace[first:last+1], **kwargs))
            prev_last = last + 1
        transformed.extend(extracted_trace[prev_last:])
        trace[bottom:] = transformed
        _trace.evaluate_all()
    else:
        # 3) Tile over the entire loop chain, possibly unrolled as by user
        # request of a factor equals to /num_unroll/
        total_loop_chain = loop_chain.unrolled_loop_chain + extracted_trace
        if len(total_loop_chain) / len(extracted_trace) == num_unroll:
            bottom = trace.index(total_loop_chain[0])
            trace[bottom:] = fuse(name, total_loop_chain, **kwargs)
            loop_chain.unrolled_loop_chain = []
            _trace.evaluate_all()
        else:
            loop_chain.unrolled_loop_chain.extend(extracted_trace)


loop_chain.unrolled_loop_chain = []
