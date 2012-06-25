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

from copy import copy
import numpy as np

def as_tuple(item, type=None, length=None):
    # Empty list if we get passed None
    if item is None:
        t = []
    else:
        # Convert iterable to list...
        try:
            t = tuple(item)
        # ... or create a list of a single item
        except TypeError:
            t = (item,)*(length or 1)
    if length:
        assert len(t) == length, "Tuple needs to be of length %d" % length
    if type:
        assert all(isinstance(i, type) for i in t), \
                "Items need to be of %s" % type
    return t

# Kernel API

class Access(object):
    """OP2 access type."""

    _modes = ["READ", "WRITE", "RW", "INC"]

    def __init__(self, mode):
        assert mode in self._modes, "Mode needs to be one of %s" % self._modes
        self._mode = mode

    def __str__(self):
        return "OP2 Access: %s" % self._mode

    def __repr__(self):
        return "Access('%s')" % self._mode

READ  = Access("READ")
WRITE = Access("WRITE")
RW    = Access("RW")
INC   = Access("INC")

class IterationSpace(object):
    """OP2 iteration space type."""

    def __init__(self, iterset, dims):
        assert isinstance(iterset, Set), "Iteration set needs to be of type Set"
        self._iterset = iterset
        self._dims = as_tuple(dims, int)

    def __str__(self):
        return "OP2 Iteration Space: %s and extra dimensions %s" % self._dims

    def __repr__(self):
        return "IterationSpace(%r, %r)" % (self._iterset, self._dims)

class Kernel(object):
    """OP2 kernel type."""

    _globalcount = 0

    def __init__(self, code, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        self._name = name or "kernel_%d" % Kernel._globalcount
        self._code = code
        Kernel._globalcount += 1

    def compile():
        pass

    def handle():
        pass

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("""%s""", "%s")' % (self._code, self._name)

# Data API

class Set(object):
    """OP2 set."""

    _globalcount = 0

    def __init__(self, size, name=None):
        assert isinstance(size, int), "Size must be of type int"
        assert not name or isinstance(name, str), "Name must be of type str"
        self._size = size
        self._name = name or "set_%d" % Set._globalcount
        Set._globalcount += 1

    @property
    def size(self):
        return self._size

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "Set(%s, '%s')" % (self._size, self._name)

class DataCarrier(object):
    """Abstract base class for OP2 data."""

    pass

class Dat(DataCarrier):
    """OP2 vector data. A Dat holds a value for every member of a set."""

    _globalcount = 0
    _modes = [READ, WRITE, RW, INC]

    def __init__(self, dataset, dim, datatype=None, data=None, name=None):
        assert isinstance(dataset, Set), "Data set must be of type Set"
        assert not name or isinstance(name, str), "Name must be of type str"

        t = np.dtype(datatype)
        # If both data and datatype are given make sure they agree
        if datatype is not None and data is not None:
            assert t == np.asarray(data).dtype, \
                    "data is of type %s not of requested type %s" \
                    % (np.asarray(data).dtype, t)

        self._dataset = dataset
        self._dim = as_tuple(dim, int)
        try:
            self._data = np.asarray(data, dtype=t).reshape((dataset.size,)+self._dim)
        except ValueError:
            raise ValueError("Invalid data: expected %d values, got %d" % \
                    (dataset.size*np.prod(dim), np.asarray(data).size))
        self._name = name or "dat_%d" % Dat._globalcount
        self._map = None
        self._access = None
        Dat._globalcount += 1

    def __call__(self, map, access):
        assert access in self._modes, \
                "Acess descriptor must be one of %s" % self._modes
        assert map == IdentityMap or map._dataset == self._dataset, \
                "Invalid data set for map %s (is %s, should be %s)" \
                % (map._name, map._dataset._name, self._dataset._name)
        arg = copy(self)
        arg._map = map
        arg._access = access
        return arg

    def __str__(self):
        call = " associated with (%s) in mode %s" % (self._map, self._access) \
                if self._map and self._access else ""
        return "OP2 Dat: %s on (%s) with dim %s and datatype %s%s" \
               % (self._name, self._dataset, self._dim, self._data.dtype.name, call)

    def __repr__(self):
        call = "(%r, %r)" % (self._map, self._access) \
                if self._map and self._access else ""
        return "Dat(%r, %s, '%s', None, '%s')%s" \
               % (self._dataset, self._dim, self._data.dtype, self._name, call)

class Mat(DataCarrier):
    """OP2 matrix data. A Mat is defined on the cartesian product of two Sets
    and holds a value for each element in the product."""

    _globalcount = 0
    _modes = [READ, WRITE, RW, INC]

    def __init__(self, datasets, dim, datatype=None, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        self._datasets = as_tuple(datasets, Set, 2)
        self._dim = as_tuple(dim, int)
        self._datatype = np.dtype(datatype)
        self._name = name or "mat_%d" % Mat._globalcount
        self._maps = None
        self._access = None
        Mat._globalcount += 1

    def __call__(self, maps, access):
        assert access in self._modes, \
                "Acess descriptor must be one of %s" % self._modes
        for map, dataset in zip(maps, self._datasets):
            assert map._dataset == dataset, \
                    "Invalid data set for map %s (is %s, should be %s)" \
                    % (map._name, map._dataset._name, dataset._name)
        arg = copy(self)
        arg._maps = maps
        arg._access = access
        return arg

    def __str__(self):
        call = " associated with (%s, %s) in mode %s" % (self._maps[0], self._maps[1], self._access) \
                if self._maps and self._access else ""
        return "OP2 Mat: %s, row set (%s), col set (%s), dimension %s, datatype %s%s" \
               % (self._name, self._datasets[0], self._datasets[1], self._dim, self._datatype.name, call)

    def __repr__(self):
        call = "(%r, %r)" % (self._maps, self._access) \
                if self._maps and self._access else ""
        return "Mat(%r, %s, '%s', '%s')%s" \
               % (self._datasets, self._dim, self._datatype, self._name, call)

class Const(DataCarrier):
    """Data that is constant for any element of any set."""

    _globalcount = 0
    _modes = [READ]

    def __init__(self, dim, value, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        self._dim = as_tuple(dim, int)
        try:
            self._value = np.asarray(value).reshape(dim)
        except ValueError:
            raise ValueError("Invalid value: expected %d values, got %d" % \
                    (np.prod(dim), np.asarray(value).size))
        self._name = name or "const_%d" % Const._globalcount
        self._access = READ
        Const._globalcount += 1

    def __str__(self):
        return "OP2 Const: %s of dim %s and type %s with value %s" \
               % (self._name, self._dim, self._value.dtype.name, self._value)

    def __repr__(self):
        return "Const(%s, %s, '%s')" \
               % (self._dim, self._value, self._name)

class Global(DataCarrier):
    """OP2 global value."""

    _globalcount = 0
    _modes = [READ, INC]

    def __init__(self, dim, value, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        self._dim = as_tuple(dim, int)
        self._value = np.asarray(value).reshape(dim)
        self._name = name or "global_%d" % Global._globalcount
        self._access = None
        Global._globalcount += 1

    def __call__(self, access):
        assert access in self._modes, \
                "Acess descriptor must be one of %s" % self._modes
        arg = copy(self)
        arg._access = access
        return arg

    def __str__(self):
        call = " in mode %s" % self._access if self._access else ""
        return "OP2 Global Argument: %s with dim %s and value %s%s" \
                % (self._name, self._dim, self._value, call)

    def __repr__(self):
        call = "(%r)" % self._access if self._access else ""
        return "Global('%s', %r, %r)%s" % (self._name, self._dim, self._value, call)

    @property
    def value(self):
        return self._value

class Map(object):
    """OP2 map, a relation between two Sets."""

    _globalcount = 0

    def __init__(self, iterset, dataset, dim, values, name=None):
        assert isinstance(iterset, Set), "Iteration set must be of type Set"
        assert isinstance(dataset, Set), "Data set must be of type Set"
        assert isinstance(dim, int), "dim must be a scalar integer"
        assert len(values) == iterset.size*dim, \
                "Invalid data: expected %d values, got %d" % \
                (iterset.size*dim, np.asarray(values).size)
        assert not name or isinstance(name, str), "Name must be of type str"
        self._iterset = iterset
        self._dataset = dataset
        self._dim = dim
        self._values = np.asarray(values, dtype=np.int64)
        self._name = name or "map_%d" % Map._globalcount
        self._index = None
        Map._globalcount += 1

    def __call__(self, index):
        assert isinstance(index, int), "Only integer indices are allowed"
        return self.indexed(index)

    def indexed(self, index):
        # Check we haven't already been indexed
        assert self._index is None, "Map has already been indexed"
        assert 0 <= index < self._dim, \
                "Index must be in interval [0,%d]" % (self._dim-1)
        indexed = copy(self)
        indexed._index = index
        return indexed

    def __str__(self):
        indexed = " and component %s" % self._index if self._index else ""
        return "OP2 Map: %s from (%s) to (%s) with dim %s%s" \
               % (self._name, self._iterset, self._dataset, self._dim, indexed)

    def __repr__(self):
        indexed = "(%s)" % self._index if self._index else ""
        return "Map(%r, %r, %s, None, '%s')%s" \
               % (self._iterset, self._dataset, self._dim, self._name, indexed)

IdentityMap = Map(Set(0), Set(0), 1, [], 'identity')

# Parallel loop API

def par_loop(self, kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""

    pass
