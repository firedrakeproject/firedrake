# This file is part of PyOP2.
#
# PyOP2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PyOP2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PyOP2.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.

"""The PyOP2 API specification."""

import backends
import op_lib_core as core
import sequential
from sequential import READ, WRITE, RW, INC, MIN, MAX, IdentityMap


def init(backend='sequential', diags=2):
    """Initialise OP2: select the backend."""
    core.op_init(args=None, diags=diags)
    backends.set_backend(backend)

class IterationSpace(sequential.IterationSpace):
    __metaclass__ = backends.BackendSelector

class Kernel(sequential.Kernel):
    __metaclass__ = backends.BackendSelector

class Set(sequential.Set):
    __metaclass__ = backends.BackendSelector

class Dat(sequential.Dat):
    __metaclass__ = backends.BackendSelector

class Mat(sequential.Mat):
    __metaclass__ = backends.BackendSelector

class Const(sequential.Const):
    __metaclass__ = backends.BackendSelector

class Global(sequential.Global):
    __metaclass__ = backends.BackendSelector

class Map(sequential.Map):
    __metaclass__ = backends.BackendSelector

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel"""
    return backends.par_loop(kernel, it_space, *args)
