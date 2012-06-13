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

# Kernel API

class Access(object):
    """Represents an OP2 access type."""
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return "OP2 Access: %s" % self._name

    def __repr__(self):
        return "Access('%s')" % name

read  = Access("read")
write = Access("write")
inc   = Access("inc")
rw    = Access("rw")

class Index(object):
    """Represents the index into a Map through which a Dat is accessed in the
    argument list."""
    def __init__(self, index):
        self._index = index

idx_all = Index("all")

class Arg(object):
    """Represents a single argument passed to a par_loop"""
    def __init__(self, dat, index, map, access):
        self._dat = dat
        self._index = index
        self._map = map
        self._access = access

class ArgDat(Arg):
    """Represents a single Dat argument passed to a par_loop"""
    def __init__(self, dat, index, map, access):
        super(ArgDat, self).__init__(dat, index, map, access)

    def __str__(self):
        return "OP2 Dat Argument: %s accessed through index %s of %s, operation %s" \
               % (self._dat, self._index, self._map, self._access)

    def __repr__(self):
        return "ArgDat(%s,%s,%s,%s)" % (self._dat, self._index, self._map, self._access)

class ArgMat(Arg):
    """Represents a single Mat argument passed to a par_loop"""
    def __init__(self, mat, row_idx, row_map, col_idx, col_map, dim, access):
        super(ArgMat, self).__init__(dat, row_idx, row_map, dim, access)
        self._index2 = col_idx
        self._map2 = col_map

    def __str__(self):
        return "OP2 Mat Argument: %s, rows accessed through index %s of %s, " \
               "columns accessed through index %s of %s,  operation %s" \
               % (self._dat, self._index, self._map, self._index2, self._map2, self._access)

    def __repr__(self):
        return "ArgMat(%s,%s,%s,%s,%s,%s)" \
                % (self._dat, self._index, self._map, self._index2, self._map2, self._access)

class ArgGbl(Arg):
    """Represents a single global argument passed to a par_loop"""
    def __init__(self, var, access):
        self._var = var
        self._access = access

    def str(self):
        return "OP2 Global Argument: %s accessed with %s" % (self._var, self._access)

    def __repr__(self):
        return "ArgGbl(%s,%s)" % (self._var, self._access)

class IterationSpace(object):

    def __init__(self, set, *dims):
        self._set = set
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

class Set(object):
    """Represents an OP2 Set."""
    def __init__(self, size, name):
        self._size = size
        self._name = name

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "Set(%s,'%s')" % (self._size, self._name)

    def size(self):
        return self._size

class Dat(object):
    """Represents an OP2 dataset. A dataset holds a value for every member of
    a set."""
    def __init__(self, set, dim, type, data, name):
        self._set = set
        self._dim = dim
        self._type = type
        self._data = data
        self._name = name

    def __str__(self):
        return "OP2 dataset: %s on set %s with dim %s and type %s" \
               % (self._name, self._set, self._dim, self._type)

    def __repr__(self):
        return "Dat(%s,%s,'%s',None,'%s')" \
               % (self._set, self._dim, self._type, self._name)


class Mat(Dat):
    """Represents an OP2 matrix. A matrix is defined as the product of two
    sets, and holds an value for each element in the product"""
    def __init__(self, row_set, col_set, dim, type, name):
        self._row_set = row_set
        self._col_set = col_set
        self._dim = dim
        self._type = type
        self._name = name

    def __str__(self):
        return "OP2 Matrix: %s, row set %s, col set %s, dimension %s, type %s" \
               % (self._name, self._row_set, self._col_set, self._dim, self._type)

    def __repr__(self):
        return "Mat(%s,%s,%s,'%s','%s')" \
               % (self._row_set, self._col_set, self._dim, self._type, self._name)

class Global(object):
    """Represents an OP2 global argument."""
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
    """Represents an OP2 map. A map is a relation between two sets."""
    def __init__(self, frm, to, dim, values, name):
        self._from = frm
        self._to = to
        self._dim = dim
        self._values = values
        self._name = name

    def __str__(self):
        return "OP2 Map: %s from %s to %s, dim %s " \
               % (self._name, self._from, self._to, self.dim)

    def __repr__(self):
        return "Map(%s,%s,%s,None,'%s')" \
               % (self._from, self._to, self._dim, self._name)

class Const(object):
    """Represents a value that is constant for all elements of all sets."""
    def __init__(self, dim, type, value, name):
        self._dim = dim
        self._type = type
        self._data = value
        self._name = name

    def __str__(self):
        return "OP2 Const value: %s of dim %s and type %s, value %s" \
               % (self._name, self._dim, self._type, self._value)

    def __repr__(self):
        return "Const(%s,'%s',%s,'%s')" \
               % (self._dim, self._type, self._value, self._name)

# Parallel loop API

def par_loop(kernel, it_space, *args):
    pass
