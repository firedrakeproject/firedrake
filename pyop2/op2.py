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

"""The PyOP2 API specification."""

import backends
import configuration as cfg
import op_lib_core as core
import base
from base import READ, WRITE, RW, INC, MIN, MAX, IdentityMap, i


def init(**kwargs):
    """Initialise OP2: select the backend."""
    cfg.configure(**kwargs)
    backends.set_backend(cfg.backend)
    core.op_init(args=None, diags=0)

def exit():
    """Exit OP2 and clean up"""
    cfg.reset()
    if backends.get_backend() != 'pyop2.void':
        core.op_exit()
        backends.unset_backend()

class IterationSpace(base.IterationSpace):
    __metaclass__ = backends._BackendSelector

class Kernel(base.Kernel):
    __metaclass__ = backends._BackendSelector

class Set(base.Set):
    __metaclass__ = backends._BackendSelectorWithH5

class Dat(base.Dat):
    __metaclass__ = backends._BackendSelectorWithH5

class Mat(base.Mat):
    __metaclass__ = backends._BackendSelector

class Const(base.Const):
    __metaclass__ = backends._BackendSelectorWithH5

class Global(base.Global):
    __metaclass__ = backends._BackendSelector

class Map(base.Map):
    __metaclass__ = backends._BackendSelectorWithH5

class Sparsity(base.Sparsity):
    __metaclass__ = backends._BackendSelector

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel"""
    return backends.par_loop(kernel, it_space, *args)

def solve(M, x, b):
    """Invocation of an OP2 solve"""
    return backends.solve(M, x, b)
