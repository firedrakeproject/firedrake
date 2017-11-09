# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012-2014, Imperial College London and
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

"""A stub implementation of "Python" parallel loops.

This basically executes a python function over the iteration set,
feeding it the appropriate data for each set entity.

Example usage::

.. code-block:: python

   s = op2.Set(10)
   d = op2.Dat(s)
   d2 = op2.Dat(s**2)

   m = op2.Map(s, s, 2, np.dstack(np.arange(4),
                                  np.roll(np.arange(4), -1)))

   def fn(x, y):
       x[0] = y[0]
       x[1] = y[1]

   d.data[:] = np.arange(4)

   op2.par_loop(fn, s, d2(op2.WRITE), d(op2.READ, m))

   print d2.data
   # [[ 0.  1.]
   #  [ 1.  2.]
   #  [ 2.  3.]
   #  [ 3.  0.]]

  def fn2(x, y):
      x[0] += y[0]
      x[1] += y[0]

  op2.par_loop(fn, s, d2(op2.INC), d(op2.READ, m[1]))

  print d2.data
  # [[ 1.  2.]
  #  [ 3.  4.]
  #  [ 5.  6.]
  #  [ 3.  0.]]
"""

import numpy as np
from pyop2 import base


# Fake kernel for type checking
class Kernel(base.Kernel):
    @classmethod
    def _cache_key(cls, *args, **kwargs):
        return None

    def __init__(self, code, name=None, **kwargs):
        self._func = code
        self._name = name
        self._attached_info = {'fundecl': None, 'attached': False}

    def __getattr__(self, attr):
        """Return None on unrecognised attributes"""
        return None

    def __call__(self, *args):
        return self._func(*args)

    def __repr__(self):
        return 'Kernel("""%s""", %r)' % (self._func, self._name)


# Inherit from parloop for type checking and init
class ParLoop(base.ParLoop):

    def _compute(self, part, *arglist):
        if part.set._extruded:
            raise NotImplementedError
        subset = isinstance(self.iterset, base.Subset)

        def arrayview(array, access):
            array = array.view()
            array.setflags(write=(access is not base.READ))
            return array

        # Just walk over the iteration set
        for e in range(part.offset, part.offset + part.size):
            args = []
            if subset:
                idx = self.iterset._indices[e]
            else:
                idx = e
            for arg in self.args:
                if arg._is_global:
                    args.append(arrayview(arg.data._data, arg.access))
                elif arg._is_direct:
                    args.append(arrayview(arg.data._data[idx, ...], arg.access))
                elif arg._is_indirect:
                    args.append(arrayview(arg.data._data[arg.map.values_with_halo[idx], ...], arg.access))
                elif arg._is_mat:
                    if arg.access not in [base.INC, base.WRITE]:
                        raise NotImplementedError
                    if arg._is_mixed_mat:
                        raise ValueError("Mixed Mats must be split before assembly")
                    args.append(np.zeros(arg._block_shape[0][0], dtype=arg.data.dtype))
                if args[-1].shape == ():
                    args[-1] = args[-1].reshape(1)
            self._kernel(*args)
            for arg, tmp in zip(self.args, args):
                if arg.access is base.READ:
                    continue
                if arg._is_global:
                    arg.data._data[:] = tmp[:]
                elif arg._is_direct:
                    arg.data._data[idx, ...] = tmp[:]
                elif arg._is_indirect:
                    arg.data._data[arg.map.values_with_halo[idx], ...] = tmp[:]
                elif arg._is_mat:
                    if arg.access is base.INC:
                        arg.data.addto_values(arg.map[0].values_with_halo[idx],
                                              arg.map[1].values_with_halo[idx],
                                              tmp)
                    elif arg.access is base.WRITE:
                        arg.data.set_values(arg.map[0].values_with_halo[idx],
                                            arg.map[1].values_with_halo[idx],
                                            tmp)

        for arg in self.args:
            if arg._is_mat and arg.access is not base.READ:
                # Queue up assembly of matrix
                arg.data.assemble()
                # Now force the evaluation of everything.  Python
                # parloops are not performance critical, so this is
                # fine.
                # We need to do this because the
                # set_values/addto_values calls are lazily evaluated,
                # and the parloop is already lazily evaluated so this
                # lazily spawns lazy computation and getting
                # everything to execute in the right order is
                # otherwise madness.
                arg.data._force_evaluation(read=True, write=False)
