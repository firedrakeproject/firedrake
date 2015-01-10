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
from numpy.ctypeslib import ndpointer
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
                      %(off_args)s
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
        %(layout_decl)s;
        %(layout_loop)s
            %(layout_assign)s;
        %(layout_loop_close)s
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

    @collective
    @lineprof
    def _compute(self, part):
        fun = JITModule(self.kernel, self.it_space, *self.args, direct=self.is_direct, iterate=self.iteration_region)
        if not hasattr(self, '_jit_args'):
            self._jit_args = [None] * 5
            self._argtypes = [None] * 5
            self._argtypes[0] = ctypes.c_int
            self._argtypes[1] = ctypes.c_int
            if isinstance(self._it_space._iterset, Subset):
                self._argtypes.append(self._it_space._iterset._argtype)
                self._jit_args.append(self._it_space._iterset._indices)
            for arg in self.args:
                if arg._is_mat:
                    self._argtypes.append(arg.data._argtype)
                    self._jit_args.append(arg.data.handle.handle)
                else:
                    for d in arg.data:
                        # Cannot access a property of the Dat or we will force
                        # evaluation of the trace
                        self._argtypes.append(d._argtype)
                        self._jit_args.append(d._data)

                if arg._is_indirect or arg._is_mat:
                    maps = as_tuple(arg.map, Map)
                    for map in maps:
                        for m in map:
                            self._argtypes.append(m._argtype)
                            self._jit_args.append(m.values_with_halo)

            for c in Const._definitions():
                self._argtypes.append(c._argtype)
                self._jit_args.append(c.data)

            # offset_args returns an empty list if there are none
            for a in self.offset_args:
                self._argtypes.append(ndpointer(a.dtype, shape=a.shape))
                self._jit_args.append(a)

            if self.iteration_region in [ON_BOTTOM]:
                self._argtypes.append(ctypes.c_int)
                self._argtypes.append(ctypes.c_int)
                self._jit_args.append(0)
                self._jit_args.append(1)
            if self.iteration_region in [ON_TOP]:
                self._argtypes.append(ctypes.c_int)
                self._argtypes.append(ctypes.c_int)
                self._jit_args.append(self._it_space.layers - 2)
                self._jit_args.append(self._it_space.layers - 1)
            elif self.iteration_region in [ON_INTERIOR_FACETS]:
                self._argtypes.append(ctypes.c_int)
                self._argtypes.append(ctypes.c_int)
                self._jit_args.append(0)
                self._jit_args.append(self._it_space.layers - 2)
            elif self._it_space._extruded:
                self._argtypes.append(ctypes.c_int)
                self._argtypes.append(ctypes.c_int)
                self._jit_args.append(0)
                self._jit_args.append(self._it_space.layers - 1)

        if part.size > 0:
            # TODO: compute partition size
            plan = self._get_plan(part, 1024)
            self._argtypes[2] = ndpointer(plan.blkmap.dtype, shape=plan.blkmap.shape)
            self._jit_args[2] = plan.blkmap
            self._argtypes[3] = ndpointer(plan.offset.dtype, shape=plan.offset.shape)
            self._jit_args[3] = plan.offset
            self._argtypes[4] = ndpointer(plan.nelems.dtype, shape=plan.nelems.shape)
            self._jit_args[4] = plan.nelems
            # Must call compile on all processes even if partition size is
            # zero since compilation is collective.
            fun = fun.compile(argtypes=self._argtypes, restype=None)

            boffset = 0
            for c in range(plan.ncolors):
                nblocks = plan.ncolblk[c]
                self._jit_args[0] = boffset
                self._jit_args[1] = nblocks
                with timed_region("ParLoop kernel"):
                    fun(*self._jit_args)
                boffset += nblocks
        else:
            # Fake types for arguments so that ctypes doesn't complain
            self._argtypes[2] = ndpointer(np.int32, shape=(0, ))
            self._argtypes[3] = ndpointer(np.int32, shape=(0, ))
            self._argtypes[4] = ndpointer(np.int32, shape=(0, ))
            # No need to actually call function since partition size
            # is zero, however we must compile it because compilation
            # is collective
            fun.compile(argtypes=self._argtypes, restype=None)

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
