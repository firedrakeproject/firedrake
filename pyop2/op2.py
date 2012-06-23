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

"""Example of the PyOP2 API specification. An implementation is pending subject
to the API being finalised."""

from copy import copy

# Kernel API

class Access(object):
    """Represents an OP2 access type."""
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return "OP2 Access: %s" % self._name

    def __repr__(self):
        return "Access('%s')" % self._name

READ  = Access("read")
WRITE = Access("write")
INC   = Access("inc")
RW    = Access("rw")

class IterationSpace(object):

    def __init__(self, iterset, *dims):
        self._iterset = iterset
        self._dims = dims

    def __str__(self):
        return "OP2 Iteration Space: %s and extra dimensions %s" % self._dims

    def __repr__(self):
        return "IterationSpace(%s,%s)" % (self._set, self._dims)

class Kernel(object):

    def __init__(self, name, code):
        self._name = name
        self._code = code

    def compile():
        pass

    def handle():
        pass

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("%s","""%s""")' % (self._name, self._code)

# Data API

class DataSet(object):
    """Represents an OP2 Set on which a DataCarrier is defined."""

    def __init__(self, size, name):
        self._size = size
        self._name = name

    def size(self):
        return self._size

    def __str__(self):
        return "OP2 DataSet: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "DataSet(%s,'%s')" % (self._size, self._name)

class IterationSet(DataSet):
    """Represents an OP2 Set on which a Kernel is defined."""

    def __str__(self):
        return "OP2 IterationSet: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "IterationSet(%s,'%s')" % (self._size, self._name)

class DataCarrier(object):
    """Abstract base class for OP2 data."""

    pass

class Dat(DataCarrier):
    """Represents OP2 vector data. A Dat holds a value for every member of a
    set."""

    def __init__(self, dataset, dim, datatype, data, name):
        self._dataset = dataset
        self._dim = dim
        self._datatype = datatype
        self._data = data
        self._name = name

    def __str__(self):
        return "OP2 Dat: %s on DataSet %s with dim %s and datatype %s" \
               % (self._name, self._dataset, self._dim, self._datatype)

    def __repr__(self):
        return "Dat(%s, %s,'%s',None,'%s')" \
               % (self._dataset, self._dim, self._datatype, self._name)

class Mat(DataCarrier):
    """Represents OP2 matrix data. A Mat is defined on the cartesian product
    of two DataSets, and holds an value for each element in the product"""

    def __init__(self, row_set, col_set, dim, datatype, name):
        self._row_set = row_set
        self._col_set = col_set
        self._dim = dim
        self._datatype = datatype
        self._name = name

    def __str__(self):
        return "OP2 Mat: %s, row set %s, col set %s, dimension %s, datatype %s" \
               % (self._name, self._row_set, self._col_set, self._dim, self._datatype)

    def __repr__(self):
        return "Mat(%s,%s,%s,'%s','%s')" \
               % (self._row_set, self._col_set, self._dim, self._datatype, self._name)

class Const(DataCarrier):
    """Represents a value that is constant for all elements of all sets."""

    def __init__(self, dim, datatype, value, name):
        self._dim = dim
        self._datatype = datatype
        self._data = value
        self._name = name

    def __str__(self):
        return "OP2 Const value: %s of dim %s and type %s, value %s" \
               % (self._name, self._dim, self._datatype, self._value)

    def __repr__(self):
        return "Const(%s,'%s',%s,'%s')" \
               % (self._dim, self._datatype, self._value, self._name)

class Global(DataCarrier):
    """Represents an OP2 global value."""

    def __init__(self, name, val=0):
        self._val = val
        self._name = name

    def __str__(self):
        return "OP2 Global Argument: %s with value %s"

    def __repr__(self):
        return "Global('%s')"

    def val(self):
        return self._val

class Map(object):
    """Represents an OP2 map. A map is a relation between an IterationSet and
    a DataSet."""

    def __init__(self, iterset, dataset, dim, values, name):
        self._iterset = iterset
        self._dataset = dataset
        self._dim = dim
        self._values = values
        self._name = name
        self._index = None

    def __getitem__(self, index): # x[y] <-> x.__getitem__(y)
        # Indexing with [:] is a no-op (OP_ALL semantics)
        if index == slice(None, None, None):
            return self

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
        return "OP2 Map: %s from %s to %s, dim %s " \
               % (self._name, self._iterset, self._dataset, self.dim)

    def __repr__(self):
        return "Map(%s,%s,%s,None,'%s')" \
               % (self._iterset, self._dataset, self._dim, self._name)

# Parallel loop API

class ParLoop(object):
    """Represents an invocation of an OP2 kernel with an access descriptor"""
    def __init__(self, kernel, it_space, *args):
        self._kernel = kernel
        self._it_space = it_space
        self._args = args
