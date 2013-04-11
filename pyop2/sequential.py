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

"""OP2 sequential backend."""

import os
import numpy as np

from exceptions import *
from utils import as_tuple
import op_lib_core as core
import petsc_base
from petsc_base import *
import host
from host import Arg

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    ParLoop(kernel, it_space, *args).compute()

class ParLoop(host.ParLoop):

    wrapper = """
              void wrap_%(kernel_name)s__(PyObject *_start, PyObject *_end, %(wrapper_args)s %(const_args)s) {
                int start = (int)PyInt_AsLong(_start);
                int end = (int)PyInt_AsLong(_end);
                %(wrapper_decs)s;
                %(tmp_decs)s;
                %(const_inits)s;
                for ( int i = start; i < end; i++ ) {
                  %(vec_inits)s;
                  %(itspace_loops)s
                  %(ind)s%(zero_tmps)s;
                  %(ind)s%(kernel_name)s(%(kernel_args)s);
                  %(ind)s%(addtos_vector_field)s;
                  %(itspace_loop_close)s
                  %(addtos_scalar_field)s;
                }
              }
              """

    def compute(self):
        _fun = self.build()
        _args = [0, 0]          # start, stop
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

        # kick off halo exchanges
        self.halo_exchange_begin()
        # compute over core set elements
        _args[0] = 0
        _args[1] = self.it_space.core_size
        _fun(*_args)
        # wait for halo exchanges to complete
        self.halo_exchange_end()
        # compute over remaining owned set elements
        _args[0] = self.it_space.core_size
        _args[1] = self.it_space.size
        _fun(*_args)
        # By splitting the reduction here we get two advantages:
        # - we don't double count contributions in halo elements
        # - once our MPI supports the asynchronous collectives in
        #   MPI-3, we can do more comp/comms overlap
        self.reduction_begin()
        if self.needs_exec_halo:
            _args[0] = self.it_space.size
            _args[1] = self.it_space.exec_size
            _fun(*_args)
        self.reduction_end()
        self.maybe_set_halo_update_needed()
        for arg in self.args:
            if arg._is_mat:
                arg.data._assemble()

    def generate_code(self):

        def itspace_loop(i, d):
            return "for (int i_%d=0; i_%d<%d; ++i_%d) {" % (i, i, d, i)

        def c_const_arg(c):
            return 'PyObject *_%s' % c.name

        def c_const_init(c):
            d = {'name' : c.name,
                 'type' : c.ctype}
            if c.cdim == 1:
                return '%(name)s = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[0]' % d
            tmp = '%(name)s[%%(i)s] = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[%%(i)s]' % d
            return ';\n'.join([tmp % {'i' : i} for i in range(c.cdim)])

        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in self.args])

        _tmp_decs = ';\n'.join([arg.tmp_decl(self._it_space.extents) for arg in self.args if arg._is_mat])
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in self.args])

        _kernel_user_args = [arg.c_kernel_arg() for arg in self.args]
        _kernel_it_args   = ["i_%d" % d for d in range(len(self._it_space.extents))]
        _kernel_args = ', '.join(_kernel_user_args + _kernel_it_args)
        _vec_inits = ';\n'.join([arg.c_vec_init() for arg in self.args \
                                 if not arg._is_mat and arg._is_vec_map])

        nloops = len(self._it_space.extents)
        _itspace_loops = '\n'.join(['  ' * i + itspace_loop(i,e) for i, e in enumerate(self._it_space.extents)])
        _itspace_loop_close = '\n'.join('  ' * i + '}' for i in range(nloops - 1, -1, -1))

        _addtos_vector_field = ';\n'.join([arg.c_addto_vector_field() for arg in self.args \
                                           if arg._is_mat and arg.data._is_vector_field])
        _addtos_scalar_field = ';\n'.join([arg.c_addto_scalar_field() for arg in self.args \
                                           if arg._is_mat and arg.data._is_scalar_field])

        _zero_tmps = ';\n'.join([arg.c_zero_tmp() for arg in self.args if arg._is_mat])

        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])

        indent = lambda t, i: ('\n' + '  ' * i).join(t.split('\n'))
        return {'ind': '  ' * nloops,
                'kernel_name': self._kernel.name,
                'wrapper_args': _wrapper_args,
                'wrapper_decs': indent(_wrapper_decs, 1),
                'const_args': _const_args,
                'const_inits': indent(_const_inits, 1),
                'tmp_decs': indent(_tmp_decs, 1),
                'itspace_loops': indent(_itspace_loops, 2),
                'itspace_loop_close': indent(_itspace_loop_close, 2),
                'vec_inits': indent(_vec_inits, 2),
                'zero_tmps': indent(_zero_tmps, 2 + nloops),
                'kernel_args': _kernel_args,
                'addtos_vector_field': indent(_addtos_vector_field, 2 + nloops),
                'addtos_scalar_field': indent(_addtos_scalar_field, 2)}

def _setup():
    pass
