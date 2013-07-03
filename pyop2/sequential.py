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
from mpi import collective
from utils import as_tuple
from petsc_base import *
import host
from host import Arg  # noqa: needed by BackendSelector

# Parallel loop API


@collective
def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    ParLoop(kernel, it_space, *args)


class JITModule(host.JITModule):

    _wrapper = """
void wrap_%(kernel_name)s__(PyObject *_start, PyObject *_end,
                            %(wrapper_args)s %(const_args)s %(off_args)s) {
  int start = (int)PyInt_AsLong(_start);
  int end = (int)PyInt_AsLong(_end);
  %(wrapper_decs)s;
  %(local_tensor_decs)s;
  %(const_inits)s;
  %(off_inits)s;
  for ( int i = start; i < end; i++ ) {
    %(vec_inits)s;
    %(itspace_loops)s
    %(extr_loop)s
    %(ind)s%(zero_tmps)s;
    %(ind)s%(kernel_name)s(%(kernel_args)s);
    %(ind)s%(addtos_vector_field)s;
    %(apply_offset)s
    %(extr_loop_close)s
    %(itspace_loop_close)s
    %(addtos_scalar_field)s;
  }
}
"""


class ParLoop(host.ParLoop):

    def __init__(self, *args, **kwargs):
        host.ParLoop.__init__(self, *args, **kwargs)

    def _compute(self, part):
        fun = JITModule(self.kernel, self.it_space, *self.args)
        if not hasattr(self, '_jit_args'):
            self._jit_args = [0, 0]
            for arg in self.args:
                if arg._is_mat:
                    self._jit_args.append(arg.data.handle.handle)
                else:
                    self._jit_args.append(arg.data._data)

                if arg._is_indirect or arg._is_mat:
                    maps = as_tuple(arg.map, Map)
                    for map in maps:
                        self._jit_args.append(map.values)

            for c in Const._definitions():
                self._jit_args.append(c.data)

            self._jit_args.extend(self.offset_args())

        if part.size > 0:
            self._jit_args[0] = part.offset
            self._jit_args[1] = part.offset + part.size
            fun(*self._jit_args)


def _setup():
    pass
