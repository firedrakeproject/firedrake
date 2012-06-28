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

# Kernel API

class Access(object):
    """OP2 access type."""

    def __new__(klass, mode):
        return _backend.Access(mode)

class IterationSpace(object):
    """OP2 iteration space type."""

    def __new__(klass, iterset, dims):
        return _backend.IterationSpace(iterset, dims)

class Kernel(object):
    """OP2 kernel type."""

    def __new__(klass, code, name=None):
        return _backend.Kernel(code, name)

    def compile(self):
        pass

    def handle(self):
        pass

# Data API

class Set(object):
    """OP2 set."""

    def __new__(klass, size, name=None):
        return _backend.Set(size, name)

    @property
    def size(self):
        pass

class DataCarrier(object):
    """Abstract base class for OP2 data."""

    pass

class Dat(DataCarrier):
    """OP2 vector data. A Dat holds a value for every member of a set."""

    def __new__(klass, dataset, dim, datatype=None, data=None, name=None):
        return _backend.Dat(dataset, dim, datatype, data, name)

class Mat(DataCarrier):
    """OP2 matrix data. A Mat is defined on the cartesian product of two Sets
    and holds a value for each element in the product."""

    def __new__(klass, datasets, dim, datatype=None, name=None):
        return _backend.Mat(datatype, dim, datatype, name)

class Const(DataCarrier):
    """Data that is constant for any element of any set."""

    def __new__(klass, dim, value, name=None):
        return _backend.Const(dim, value, name)

class Global(DataCarrier):
    """OP2 global value."""

    def __new__(klass, dim, value, name=None):
        return _backend.Global(dim, value, name)

    @property
    def value(self):
        pass

class Map(object):
    """OP2 map, a relation between two Sets."""

    def __new__(klass, iterset, dataset, dim, values, name=None):
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
    IdentityMap = Map(Set(0), Set(0), 1, [], 'identity')
    READ = _backend.READ
    WRITE = _backend.WRITE
    RW = _backend.RW
    INC = _backend.INC
    MIN = _backend.MIN
    MAX = _backend.MAX
