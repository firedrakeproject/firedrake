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

import os
import numpy as np
import math

from exceptions import *
from find_op2 import *
from utils import *
import op_lib_core as core
import petsc_base
from petsc_base import *
import host
import device

# hard coded value to max openmp threads
_max_threads = 32

def _detect_openmp_flags():
    import subprocess
    _version = subprocess.check_output(['mpicc', '--version'], shell=False)
    if _version.find('Free Software Foundation') != -1:
        return '-fopenmp'
    elif _version.find('Intel Corporation') != -1:
        return '-openmp'
    else:
        from warnings import warn
        warn('Unknown mpicc version:\n%s' % _version)
        return ''

_cppargs = os.environ.get('OMP_CXX_FLAGS') or _detect_openmp_flags()

class Arg(host.Arg):

    def c_vec_name(self, idx=None):
        return self.c_arg_name() + "_vec[%s]" % (idx or 'tid')

    def c_kernel_arg_name(self):
        return "p_%s[tid]" % self.c_arg_name()

    def c_global_reduction_name(self):
        return "%s_l[tid]" % self.c_arg_name()

    def c_vec_dec(self):
        return ";\n%(type)s *%(vec_name)s[%(dim)s]" % \
                   {'type' : self.ctype,
                    'vec_name' : self.c_vec_name(str(_max_threads)),
                    'dim' : self.map.dim}

    def c_addto_scalar_field(self):
        name = self.c_arg_name()
        p_data = 'p_%s[tid]' % name
        maps = as_tuple(self.map, Map)
        nrows = maps[0].dim
        ncols = maps[1].dim

        return 'addto_vector(%(mat)s, %(vals)s, %(nrows)s, %(rows)s, %(ncols)s, %(cols)s, %(insert)d)' % \
            {'mat' : name,
             'vals' : p_data,
             'nrows' : nrows,
             'ncols' : ncols,
             'rows' : "%s + i * %s" % (self.c_map_name(), nrows),
             'cols' : "%s2 + i * %s" % (self.c_map_name(), ncols),
             'insert' : self.access == WRITE }

    def c_addto_vector_field(self):
        name = self.c_arg_name()
        p_data = 'p_%s[tid]' % name
        maps = as_tuple(self.map, Map)
        nrows = maps[0].dim
        ncols = maps[1].dim
        dims = self.data.sparsity.dims
        rmult = dims[0]
        cmult = dims[1]
        s = []
        for i in xrange(rmult):
            for j in xrange(cmult):
                idx = '[%d][%d]' % (i, j)
                val = "&%s%s" % (p_data, idx)
                row = "%(m)s * %(map)s[i * %(dim)s + i_0] + %(i)s" % \
                      {'m' : rmult,
                       'map' : self.c_map_name(),
                       'dim' : nrows,
                       'i' : i }
                col = "%(m)s * %(map)s2[i * %(dim)s + i_1] + %(j)s" % \
                      {'m' : cmult,
                       'map' : self.c_map_name(),
                       'dim' : ncols,
                       'j' : j }

                s.append('addto_scalar(%s, %s, %s, %s, %d)' \
                        % (name, val, row, col, self.access == WRITE))
        return ';\n'.join(s)

    def c_assemble(self):
        name = self.c_arg_name()
        return "assemble_mat(%s)" % name

    def tmp_decl(self, extents):
        t = self.data.ctype
        if self.data._is_scalar_field:
            dims = ''.join(["[%d]" % d for d in extents])
        elif self.data._is_vector_field:
            dims = ''.join(["[%d]" % d for d in self.data.dims])
        else:
            raise RuntimeError("Don't know how to declare temp array for %s" % self)
        return "%s p_%s[%s]%s" % (t, self.c_arg_name(), _max_threads, dims)

    def c_zero_tmp(self):
        name = "p_" + self.c_arg_name()
        t = self.ctype
        if self.data._is_scalar_field:
            idx = ''.join(["[i_%d]" % i for i,_ in enumerate(self.data.dims)])
            return "%(name)s[tid]%(idx)s = (%(t)s)0" % \
                {'name' : name, 't' : t, 'idx' : idx}
        elif self.data._is_vector_field:
            size = np.prod(self.data.dims)
            return "memset(%(name)s[tid], 0, sizeof(%(t)s) * %(size)s)" % \
                {'name' : name, 't' : t, 'size' : size}
        else:
            raise RuntimeError("Don't know how to zero temp array for %s" % self)

    def c_reduction_dec(self):
        return "%(type)s %(name)s_l[%(max_threads)s][%(dim)s]" % \
          {'type' : self.ctype,
           'name' : self.c_arg_name(),
           'dim' : self.data.cdim,
           # Ensure different threads are on different cache lines
           'max_threads' : _max_threads}

    def c_reduction_init(self):
        if self.access == INC:
            init = "(%(type)s)0" % {'type' : self.ctype}
        else:
            init = "%(name)s[i]" % {'name' : self.c_arg_name()}
        return "for ( int i = 0; i < %(dim)s; i++ ) %(name)s_l[tid][i] = %(init)s" % \
          {'dim' : self.data.cdim,
           'name' : self.c_arg_name(),
           'init' : init}

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
        }""" % {'combine' : combine,
                'dim' : self.data.cdim}

class Mat(petsc_base.Mat):
    # This is needed for the test harness to check that two Mats on
    # the same Sparsity share data.
    @property
    def _colidx(self):
        return self._sparsity._colidx

    @property
    def _rowptr(self):
        return self._sparsity._rowptr

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    ParLoop(kernel, it_space, *args).compute()

class ParLoop(device.ParLoop):
    def compute(self):
        _fun = self.generate_code()
        _args = [self._it_space.size]
        for arg in self.args:
            if arg._is_mat:
                _args.append(arg.data.handle.handle)
            else:
                _args.append(arg.data._data)

            if arg._is_dat:
                maybe_setflags(arg.data._data, write=False)

            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    _args.append(map.values)

        for c in Const._definitions():
            _args.append(c.data)

        part_size = 1024  #TODO: compute partition size

        # Create a plan, for colored execution
        if [arg for arg in self.args if arg._is_indirect or arg._is_mat]:
            plan = device.Plan(self._kernel, self._it_space.iterset,
                               *self._unwound_args,
                               partition_size=part_size,
                               matrix_coloring=True,
                               staging=False,
                               thread_coloring=False)

        else:
            # Create a fake plan for direct loops.
            # Make the fake plan according to the number of cores available
            # to OpenMP
            class FakePlan:
                def __init__(self, iset, part_size):
                    nblocks = int(math.ceil(iset.size / float(part_size)))
                    self.ncolors = 1
                    self.ncolblk = np.array([nblocks], dtype=np.int32)
                    self.blkmap = np.arange(nblocks, dtype=np.int32)
                    self.nelems = np.array([min(part_size, iset.size - i * part_size) for i in range(nblocks)],
                                           dtype=np.int32)

            plan = FakePlan(self._it_space.iterset, part_size)

        _args.append(part_size)
        _args.append(plan.ncolors)
        _args.append(plan.blkmap)
        _args.append(plan.ncolblk)
        _args.append(plan.nelems)

        _fun(*_args)

    def generate_code(self):

        key = self._cache_key
        _fun = petsc_base._parloop_cache.get(key)

        if _fun is not None:
            return _fun

        from instant import inline_with_numpy

        def itspace_loop(i, d):
            return "for (int i_%d=0; i_%d<%d; ++i_%d){" % (i, i, d, i)

        def c_const_arg(c):
            return 'PyObject *_%s' % c.name

        def c_const_init(c):
            d = {'name' : c.name,
                 'type' : c.ctype}
            if c.cdim == 1:
                return '%(name)s = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[0]' % d
            tmp = '%(name)s[%%(i)s] = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[%%(i)s]' % d
            return ';\n'.join([tmp % {'i' : i} for i in range(c.cdim)])

        args = self.args
        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in args])

        _tmp_decs = ';\n'.join([arg.tmp_decl(self._it_space.extents) for arg in args if arg._is_mat])
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in args])

        _const_decs = '\n'.join([const._format_declaration() for const in Const._definitions()]) + '\n'

        _kernel_user_args = [arg.c_kernel_arg() for arg in args]
        _kernel_it_args   = ["i_%d" % d for d in range(len(self._it_space.extents))]
        _kernel_args = ', '.join(_kernel_user_args + _kernel_it_args)
        _vec_inits = ';\n'.join([arg.c_vec_init() for arg in args \
                                 if not arg._is_mat and arg._is_vec_map])

        _itspace_loops = '\n'.join([itspace_loop(i,e) for i, e in zip(range(len(self._it_space.extents)), self._it_space.extents)])
        _itspace_loop_close = '}'*len(self._it_space.extents)

        _addtos_vector_field = ';\n'.join([arg.c_addto_vector_field() for arg in args \
                                           if arg._is_mat and arg.data._is_vector_field])
        _addtos_scalar_field = ';\n'.join([arg.c_addto_scalar_field() for arg in args \
                                           if arg._is_mat and arg.data._is_scalar_field])

        _assembles = ';\n'.join([arg.c_assemble() for arg in args if arg._is_mat])

        _zero_tmps = ';\n'.join([arg.c_zero_tmp() for arg in args if arg._is_mat])

        _set_size_wrapper = 'PyObject *_%(set)s_size' % {'set' : self._it_space.name}
        _set_size_dec = 'int %(set)s_size = (int)PyInt_AsLong(_%(set)s_size);' % {'set' : self._it_space.name}
        _set_size = '%(set)s_size' % {'set' : self._it_space.name}

        _reduction_decs = ';\n'.join([arg.c_reduction_dec() for arg in args if arg._is_global_reduction])
        _reduction_inits = ';\n'.join([arg.c_reduction_init() for arg in args if arg._is_global_reduction])
        _reduction_finalisations = '\n'.join([arg.c_reduction_finalisation() for arg in args if arg._is_global_reduction])

        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])
        wrapper = """
            void wrap_%(kernel_name)s__(%(set_size_wrapper)s, %(wrapper_args)s %(const_args)s, PyObject* _part_size, PyObject* _ncolors, PyObject* _blkmap, PyObject* _ncolblk, PyObject* _nelems) {

            int part_size = (int)PyInt_AsLong(_part_size);
            int ncolors = (int)PyInt_AsLong(_ncolors);
            int* blkmap = (int *)(((PyArrayObject *)_blkmap)->data);
            int* ncolblk = (int *)(((PyArrayObject *)_ncolblk)->data);
            int* nelems = (int *)(((PyArrayObject *)_nelems)->data);

            %(set_size_dec)s;
            %(wrapper_decs)s;
            %(const_inits)s;
            %(tmp_decs)s;

            #ifdef _OPENMP
            int nthread = omp_get_max_threads();
            #else
            int nthread = 1;
            #endif

            %(reduction_decs)s;

            #pragma omp parallel default(shared)
            {
              int tid = omp_get_thread_num();
              %(reduction_inits)s;
            }

            int boffset = 0;
            for ( int __col  = 0; __col < ncolors; __col++ ) {
              int nblocks = ncolblk[__col];

              #pragma omp parallel default(shared)
              {
                int tid = omp_get_thread_num();

                #pragma omp for schedule(static)
                for ( int __b = boffset; __b < (boffset + nblocks); __b++ ) {
                  int bid = blkmap[__b];
                  int nelem = nelems[bid];
                  int efirst = bid * part_size;
                  for (int i = efirst; i < (efirst + nelem); i++ ) {
                    %(vec_inits)s;
                    %(itspace_loops)s
                    %(zero_tmps)s;
                    %(kernel_name)s(%(kernel_args)s);
                    %(addtos_vector_field)s;
                    %(itspace_loop_close)s
                    %(addtos_scalar_field)s;
                  }
                }
              }
              %(reduction_finalisations)s
              boffset += nblocks;
            }
            %(assembles)s;
          }"""

        if any(arg._is_soa for arg in args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            inline %(code)s
            #undef OP2_STRIDE
            """ % {'code' : self._kernel.code}
        else:
            kernel_code = """
            inline %(code)s
            """ % {'code' : self._kernel.code }

        code_to_compile =  wrapper % { 'kernel_name' : self._kernel.name,
                                       'wrapper_args' : _wrapper_args,
                                       'wrapper_decs' : _wrapper_decs,
                                       'const_args' : _const_args,
                                       'const_inits' : _const_inits,
                                       'tmp_decs' : _tmp_decs,
                                       'set_size' : _set_size,
                                       'set_size_dec' : _set_size_dec,
                                       'set_size_wrapper' : _set_size_wrapper,
                                       'itspace_loops' : _itspace_loops,
                                       'itspace_loop_close' : _itspace_loop_close,
                                       'vec_inits' : _vec_inits,
                                       'zero_tmps' : _zero_tmps,
                                       'kernel_args' : _kernel_args,
                                       'addtos_vector_field' : _addtos_vector_field,
                                       'addtos_scalar_field' : _addtos_scalar_field,
                                       'assembles' : _assembles,
                                       'reduction_decs' : _reduction_decs,
                                       'reduction_inits' : _reduction_inits,
                                       'reduction_finalisations' : _reduction_finalisations}

        # We need to build with mpicc since that's required by PETSc
        cc = os.environ.get('CC')
        os.environ['CC'] = 'mpicc'
        _fun = inline_with_numpy(code_to_compile, additional_declarations = kernel_code,
                                 additional_definitions = _const_decs + kernel_code,
                                 include_dirs=[OP2_INC, get_petsc_dir()+'/include'],
                                 source_directory=os.path.dirname(os.path.abspath(__file__)),
                                 wrap_headers=["mat_utils.h"],
                                 library_dirs=[OP2_LIB, get_petsc_dir()+'/lib'],
                                 libraries=['op2_seq', 'petsc'],
                                 sources=["mat_utils.cxx"],
                                 cppargs=_cppargs,
                                 system_headers=['omp.h'])
        if cc:
            os.environ['CC'] = cc
        else:
            os.environ.pop('CC')

        petsc_base._parloop_cache[key] = _fun
        return _fun

    @property
    def _requires_matrix_coloring(self):
        """Direct code generation to follow colored execution for global matrix insertion."""
        return True

def _setup():
    pass
