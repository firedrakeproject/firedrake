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

from exceptions import *
from utils import as_tuple
from mpi import collective
from petsc_base import *
import host
import ctypes
from numpy.ctypeslib import ndpointer
from host import Kernel, Arg  # noqa: needed by BackendSelector
from base import ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS

# Parallel loop API


class JITModule(host.JITModule):

    _wrapper = """
void %(wrapper_name)s(int start, int end,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s
                      %(off_args)s
                      %(layer_arg)s) {
  %(wrapper_decs)s;
  %(const_inits)s;
  %(map_decl)s
  for ( int n = start; n < end; n++ ) {
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
    %(itset_loop_body)s
    %(map_bcs_p)s;
    %(apply_offset)s;
    %(extr_loop_close)s
  }
}
"""


class ParLoop(host.ParLoop):

    def __init__(self, *args, **kwargs):
        host.ParLoop.__init__(self, *args, **kwargs)

    @collective
    def _compute(self, part):
        fun = JITModule(self.kernel, self.it_space, *self.args, direct=self.is_direct, iterate=self.iteration_region)
        if not hasattr(self, '_jit_args'):
            self._argtypes = [ctypes.c_int, ctypes.c_int]
            self._jit_args = [0, 0]
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

        self._jit_args[0] = part.offset
        self._jit_args[1] = part.offset + part.size
        # Must call fun on all processes since this may trigger
        # compilation.
        fun(*self._jit_args, argtypes=self._argtypes, restype=None)


def _setup():
    pass
