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

"""This module implements scheduling functions as special classes. Scheduling
functions are composable. For example, given a sequence of loops L = [L0, L1, L2, L3]
and two scheduling functions S1 and S2, one can compute L' = S2(S1(L)), with S1(L)
returning, for example, [L0, L1',L3] and L' = S2([L0, L1', L3]) = [L0, L1''].
Different scheduling functions may implement different loop fusion strategies."""


from copy import deepcopy as dcopy, copy as scopy
import numpy as np

from pyop2.base import Dat, RW, _make_object
from pyop2.utils import flatten

from .extended import FusionArg, FusionParLoop, \
    TilingArg, TilingIterationSpace, TilingParLoop
from .filters import Filter, WeakFilter


__all__ = ['Schedule', 'PlainSchedule', 'FusionSchedule',
           'HardFusionSchedule', 'TilingSchedule']


class Schedule(object):

    """Represent an execution scheme for a sequence of :class:`ParLoop` objects."""

    def __init__(self, insp_name, schedule=None):
        self._insp_name = insp_name
        self._schedule = schedule

    def __call__(self, loop_chain):
        """Given an iterator of :class:`ParLoop` objects (``loop_chain``),
        return an iterator of new :class:`ParLoop` objects. The input parloops
        are "scheduled" according to the strategy of this Schedule. The Schedule
        itself was produced by an Inspector.

        In the simplest case, the returned value is identical to the input
        ``loop_chain``. That is, the Inspector that created this Schedule could
        not apply any fusion or tiling.

        In general, the Schedule could fuse or tile the loops in ``loop_chain``.
        A sequence of  :class:`fusion.ParLoop` objects would then be returned.
        """
        return loop_chain

    def _filter(self, loops):
        return list(Filter().loop_args(loops).values())


class PlainSchedule(Schedule):

    def __init__(self, insp_name, kernels):
        super(PlainSchedule, self).__init__(insp_name)
        self._kernel = kernels

    def __call__(self, loop_chain):
        for loop in loop_chain:
            for arg in loop.args:
                arg.gather = None
                arg.c_index = False
        return loop_chain


class FusionSchedule(Schedule):

    """Schedule an iterator of :class:`ParLoop` objects applying soft fusion."""

    def __init__(self, insp_name, schedule, kernels, offsets):
        super(FusionSchedule, self).__init__(insp_name, schedule)
        self._kernel = list(kernels)

        # Track the /ParLoop/s in the loop chain that each fused kernel maps to
        offsets = [0] + list(offsets)
        loop_indices = [list(range(offsets[i], o)) for i, o in enumerate(offsets[1:])]
        self._info = [{'loop_indices': li} for li in loop_indices]

    def _combine(self, loop_chain):
        fused_loops = []
        for kernel, info in zip(self._kernel, self._info):
            loop_indices = info['loop_indices']
            extra_args = info.get('extra_args', [])
            # Create the ParLoop arguments. Note that both the iteration set
            # and the iteration region correspond to the /base/ loop's
            iterregion = loop_chain[loop_indices[0]].iteration_region
            it_space = loop_chain[loop_indices[0]].it_space
            args = self._filter([loop_chain[i] for i in loop_indices])
            # Create any ParLoop additional arguments
            extra_args = [Dat(*d)(*a) for d, a in extra_args]
            args += extra_args
            # Remove now incorrect cached properties:
            for a in args:
                a.__dict__.pop('name', None)
            # Create the actual ParLoop, resulting from the fusion of some kernels
            fused_loops.append(self._make(kernel, it_space, iterregion, args, info))
        return fused_loops

    def _make(self, kernel, it_space, iterregion, args, info):
        return _make_object('ParLoop', kernel, it_space.iterset, *args,
                            iterate=iterregion, insp_name=self._insp_name)

    def __call__(self, loop_chain):
        return self._combine(self._schedule(loop_chain))


class HardFusionSchedule(FusionSchedule, Schedule):

    """Schedule an iterator of :class:`ParLoop` objects applying hard fusion
    on top of soft fusion."""

    def __init__(self, insp_name, schedule, fused):
        Schedule.__init__(self, insp_name, schedule)
        self._fused = fused

        # Set proper loop_indices for this schedule
        self._info = dcopy(schedule._info)
        for i, info in enumerate(schedule._info):
            for k, v in info.items():
                self._info[i][k] = [i] if k == 'loop_indices' else v

        # Update the input schedule to make use of hard fusion kernels
        kernel = scopy(schedule._kernel)
        for ofs, (fused_kernel, fused_map, fargs) in enumerate(fused):
            # Find the position of the /fused/ kernel in the new loop chain.
            base, fuse = fused_kernel._kernels
            base_idx, fuse_idx = kernel.index(base), kernel.index(fuse)
            pos = min(base_idx, fuse_idx)
            self._info[pos]['loop_indices'] = [base_idx + ofs, fuse_idx + ofs]
            # A bitmap indicates whether the i-th iteration in /fuse/ has been executed
            self._info[pos]['extra_args'] = [((fused_map.toset, None, np.int32),
                                              (RW, fused_map))]
            # Keep track of the arguments needing a postponed gather
            self._info[pos]['fargs'] = fargs
            # Now we can modify the kernel sequence
            kernel.insert(pos, fused_kernel)
            kernel.pop(pos+1)
            pos = max(base_idx, fuse_idx)
            self._info.pop(pos)
            kernel.pop(pos)
        self._kernel = kernel

    def __call__(self, loop_chain, only_hard=False):
        if not only_hard:
            loop_chain = self._schedule(loop_chain)
        return self._combine(loop_chain)

    def _make(self, kernel, it_space, iterregion, args, info):
        fargs = info.get('fargs', {})
        args = tuple(FusionArg(arg, *fargs[j]) if j in fargs else arg
                     for j, arg in enumerate(args))
        return FusionParLoop(kernel, it_space.iterset, *args, it_space=it_space,
                             iterate=iterregion, insp_name=self._insp_name)

    def _filter(self, loops):
        return list(WeakFilter().loop_args(loops).values())


class TilingSchedule(Schedule):

    """Schedule an iterator of :class:`ParLoop` objects applying tiling, possibly on
    top of hard fusion and soft fusion."""

    def __init__(self, insp_name, schedule, kernel, inspection, executor, **options):
        super(TilingSchedule, self).__init__(insp_name, schedule)
        self._inspection = inspection
        self._executor = executor
        self._kernel = kernel
        # Schedule's optimizations
        self._opt_glb_maps = options.get('use_glb_maps', False)
        self._opt_prefetch = options.get('use_prefetch', 0)

    def __call__(self, loop_chain):
        loop_chain = self._schedule(loop_chain)
        # Track the individual kernels, and the args of each kernel
        all_itspaces = tuple(loop.it_space for loop in loop_chain)
        all_args = []
        for i, (loop, gtl_maps) in enumerate(zip(loop_chain, self._executor.gtl_maps)):
            all_args.append([TilingArg(arg, i, None if self._opt_glb_maps else gtl_maps)
                             for arg in loop.args])
        all_args = tuple(all_args)
        # Data for the actual ParLoop
        it_space = TilingIterationSpace(all_itspaces)
        args = self._filter(loop_chain)
        reduced_globals = [loop._reduced_globals for loop in loop_chain]
        read_args = set(flatten([loop.reads for loop in loop_chain]))
        written_args = set(flatten([loop.writes for loop in loop_chain]))
        inc_args = set(flatten([loop.incs for loop in loop_chain]))
        kwargs = {
            'all_kernels': self._kernel._kernels,
            'all_itspaces': all_itspaces,
            'all_args': all_args,
            'read_args': read_args,
            'written_args': written_args,
            'reduced_globals': reduced_globals,
            'inc_args': inc_args,
            'insp_name': self._insp_name,
            'use_glb_maps': self._opt_glb_maps,
            'use_prefetch': self._opt_prefetch,
            'inspection': self._inspection,
            'executor': self._executor
        }
        return [TilingParLoop(self._kernel, it_space, *args, **kwargs)]
