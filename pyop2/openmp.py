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

"""OP2 OpenMP backend."""

import ctypes
import math
import numpy as np
import os
from subprocess import Popen, PIPE

from base import ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS
from exceptions import *
import device
import host
from host import Kernel  # noqa: for inheritance
from logger import warning
import plan as _plan
from petsc_base import *
from profiling import lineprof
from utils import *

# hard coded value to max openmp threads
_max_threads = 32
# cache line padding
_padding = 8


def _detect_openmp_flags():
    p = Popen(['mpicc', '--version'], stdout=PIPE, shell=False)
    _version, _ = p.communicate()
    if _version.find('Free Software Foundation') != -1:
        return '-fopenmp', '-lgomp'
    elif _version.find('Intel Corporation') != -1:
        return '-openmp', '-liomp5'
    else:
        warning('Unknown mpicc version:\n%s' % _version)
        return '', ''


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

    def padding(self):
        return int(_padding * (self.data.cdim / _padding + 1)) * \
            (_padding / self.data.dtype.itemsize)

    def c_reduction_dec(self):
        return "%(type)s %(name)s_l[%(max_threads)s][%(dim)s]" % \
            {'type': self.ctype,
             'name': self.c_arg_name(),
             'dim': self.padding(),
             # Ensure different threads are on different cache lines
             'max_threads': _max_threads}

    def c_reduction_init(self):
        if self.access == INC:
            init = "(%(type)s)0" % {'type': self.ctype}
        else:
            init = "%(name)s[i]" % {'name': self.c_arg_name()}
        return "for ( int i = 0; i < %(dim)s; i++ ) %(name)s_l[tid][i] = %(init)s" % \
            {'dim': self.padding(),
             'name': self.c_arg_name(),
             'init': init}

    def c_reduction_finalisation(self):
        d = {'gbl': self.c_arg_name(),
             'local': "%s_l[thread][i]" % self.c_arg_name()}
        if self.access == INC:
            combine = "%(gbl)s[i] += %(local)s" % d
        elif self.access == MIN:
            combine = "%(gbl)s[i] = %(gbl)s[i] < %(local)s ? %(gbl)s[i] : %(local)s" % d
        elif self.access == MAX:
            combine = "%(gbl)s[i] = %(gbl)s[i] > %(local)s ? %(gbl)s[i] : %(local)s" % d
        return """
        for ( int thread = 0; thread < nthread; thread++ ) {
            for ( int i = 0; i < %(dim)s; i++ ) %(combine)s;
        }""" % {'combine': combine,
                'dim': self.data.cdim}

    def c_global_reduction_name(self, count=None):
        return "%(name)s_l%(count)d[0]" % {
            'name': self.c_arg_name(),
            'count': count}

# Parallel loop API


class JITModule(host.JITModule):

    ompflag, omplib = _detect_openmp_flags()
    _cppargs = [os.environ.get('OMP_CXX_FLAGS') or ompflag]
    _libraries = [ompflag] + [os.environ.get('OMP_LIBS') or omplib]
    _system_headers = ['#include <omp.h>']

    _wrapper = """
void %(wrapper_name)s(int boffset,
                      int nblocks,
                      int *blkmap,
                      int *offset,
                      int *nelems,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s
                      %(layer_arg)s) {
  %(user_code)s
  %(wrapper_decs)s;
  %(const_inits)s;
  #pragma omp parallel shared(boffset, nblocks, nelems, blkmap)
  {
    %(map_decl)s
    int tid = omp_get_thread_num();
    %(interm_globals_decl)s;
    %(interm_globals_init)s;
    %(vec_decs)s;

    #pragma omp for schedule(static)
    for ( int __b = boffset; __b < boffset + nblocks; __b++ )
    {
      int bid = blkmap[__b];
      int nelem = nelems[bid];
      int efirst = offset[bid];
      for (int n = efirst; n < efirst+ nelem; n++ )
      {
        int i = %(index_expr)s;
        %(vec_inits)s;
        %(map_init)s;
        %(extr_loop)s
        %(map_bcs_m)s;
        %(buffer_decl)s;
        %(buffer_gather)s
        %(kernel_name)s(%(kernel_args)s);
        %(itset_loop_body)s;
        %(map_bcs_p)s;
        %(apply_offset)s;
        %(extr_loop_close)s
      }
    }
    %(interm_globals_writeback)s;
  }
}
"""

    def set_argtypes(self, iterset, *args):
        """Set the ctypes argument types for the JITModule.

        :arg iterset: The iteration :class:`Set`
        :arg args: A list of :class:`Arg`\s, the arguments to the :fn:`.par_loop`.
        """
        argtypes = [ctypes.c_int, ctypes.c_int,                      # start end
                    ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp]  # plan args
        if isinstance(iterset, Subset):
            argtypes.append(iterset._argtype)
        for arg in args:
            if arg._is_mat:
                argtypes.append(arg.data._argtype)
            else:
                for d in arg.data:
                    argtypes.append(d._argtype)
            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    for m in map:
                        argtypes.append(m._argtype)

        for c in Const._definitions():
            argtypes.append(c._argtype)

        if iterset._extruded:
            argtypes.append(ctypes.c_int)
            argtypes.append(ctypes.c_int)

        self._argtypes = argtypes

    def generate_code(self):

        # Most of the code to generate is the same as that for sequential
        code_dict = super(JITModule, self).generate_code()

        _reduction_decs = ';\n'.join([arg.c_reduction_dec()
                                     for arg in self._args if arg._is_global_reduction])
        _reduction_inits = ';\n'.join([arg.c_reduction_init()
                                      for arg in self._args if arg._is_global_reduction])
        _reduction_finalisations = '\n'.join(
            [arg.c_reduction_finalisation() for arg in self._args
             if arg._is_global_reduction])

        code_dict.update({'reduction_decs': _reduction_decs,
                          'reduction_inits': _reduction_inits,
                          'reduction_finalisations': _reduction_finalisations})
        return code_dict


class ParLoop(device.ParLoop, host.ParLoop):

    def prepare_arglist(self, iterset, *args):
        arglist = []

        if isinstance(iterset, Subset):
            arglist.append(iterset._indices.ctypes.data)
        for arg in self.args:
            if arg._is_mat:
                arglist.append(arg.data.handle.handle)
            else:
                for d in arg.data:
                    arglist.append(d._data.ctypes.data)
            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    for m in map:
                        arglist.append(m._values.ctypes.data)
        for c in Const._definitions():
            arglist.append(c._data.ctypes.data)

        if iterset._extruded:
            region = self.iteration_region
            # Set up appropriate layer iteration bounds
            if region is ON_BOTTOM:
                arglist.append(0)
                arglist.append(1)
                arglist.append(iterset.layers - 1)
            elif region is ON_TOP:
                arglist.append(iterset.layers - 2)
                arglist.append(iterset.layers - 1)
                arglist.append(iterset.layers - 1)
            elif region is ON_INTERIOR_FACETS:
                arglist.append(0)
                arglist.append(iterset.layers - 2)
                arglist.append(iterset.layers - 2)
            else:
                arglist.append(0)
                arglist.append(iterset.layers - 1)
                arglist.append(iterset.layers - 1)

        return arglist

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.it_space, *self.args,
                         direct=self.is_direct, iterate=self.iteration_region)

    @collective
    @lineprof
    def _compute(self, part, fun, *arglist):
        if part.size > 0:
            # TODO: compute partition size
            plan = self._get_plan(part, 1024)
            blkmap = plan.blkmap.ctypes.data
            offset = plan.offset.ctypes.data
            nelems = plan.nelems.ctypes.data
            boffset = 0
            for c in range(plan.ncolors):
                nblocks = plan.ncolblk[c]
                with timed_region("ParLoop kernel"):
                    fun(boffset, nblocks, blkmap, offset, nelems, *arglist)
                boffset += nblocks

    def _get_plan(self, part, part_size):
        if self._is_indirect:
            plan = _plan.Plan(part,
                              *self._unwound_args,
                              partition_size=part_size,
                              matrix_coloring=True,
                              staging=False,
                              thread_coloring=False)
        else:
            # TODO:
            # Create the fake plan according to the number of cores available
            class FakePlan(object):

                def __init__(self, part, partition_size):
                    self.nblocks = int(math.ceil(part.size / float(partition_size)))
                    self.ncolors = 1
                    self.ncolblk = np.array([self.nblocks], dtype=np.int32)
                    self.blkmap = np.arange(self.nblocks, dtype=np.int32)
                    self.nelems = np.array([min(partition_size, part.size - i * partition_size) for i in range(self.nblocks)],
                                           dtype=np.int32)
                    self.offset = np.arange(part.offset, part.offset + part.size, partition_size, dtype=np.int32)

            plan = FakePlan(part, part_size)
        return plan

    @property
    def _requires_matrix_coloring(self):
        """Direct code generation to follow colored execution for global
        matrix insertion."""
        return True


def _setup():
    pass
