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

from backends import void
_backend = void
IdentityMap = None
READ = None
WRITE = None
RW = None
INC = None
MIN = None
MAX = None

# Kernel API

def Access(mode):
    """OP2 access type."""
    return _backend.Access(mode)

def IterationSpace(iterset, dims):
    """OP2 iteration space type."""
    return _backend.IterationSpace(iterset, dims)

def Kernel(code, name=None):
    """OP2 kernel type."""
    return _backend.Kernel(code, name)

# Data API

def Set(size, name=None):
    """OP2 set."""
    return _backend.Set(size, name)

def Dat(dataset, dim, datatype=None, data=None, name=None):
    """OP2 vector data. A Dat holds a value for every member of a set."""
    return _backend.Dat(dataset, dim, datatype, data, name)

def Mat(datasets, dim, datatype=None, name=None):
    """OP2 matrix data. A Mat is defined on the cartesian product of two Sets
    and holds a value for each element in the product."""
    return _backend.Mat(datatype, dim, datatype, name)

def Const(dim, value, name=None):
    """Data that is constant for any element of any set."""
    return _backend.Const(dim, value, name)

def Global(dim, value, name=None):
    """OP2 global value."""
    return _backend.Global(dim, value, name)

def Map(iterset, dataset, dim, values, name=None):
    """OP2 map, a relation between two Sets."""
    return _backend.Map(iterset, dataset, dim, values, name)

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    _backend.par_loop(kernel, it_space, *args)

def init(backend='void'):
    #TODO: make backend selector code
    global _backend
    global IdentityMap
    global READ, WRITE, RW, INC, MIN, MAX
    if backend == 'cuda':
        from backends import cuda
        _backend = cuda
    elif backend == 'opencl':
        from backends import opencl
        _backend = opencl
    IdentityMap = _backend.IdentityMap
    READ = _backend.READ
    WRITE = _backend.WRITE
    RW = _backend.RW
    INC = _backend.INC
    MIN = _backend.MIN
    MAX = _backend.MAX
