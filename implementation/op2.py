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

class Index(object):
    """Represents the index into a Map through which a Dat is accessed in the
    argument list."""
    def __init__(self, index):
        self._index = idx

class Arg(object):
    """Represents a single argument passed to a par_loop"""
    def __init__(self, dat, index, map, access):
        self._dat = dat
        self._index = index
        self._map = map
        self._dim = dim
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

class IterationSpace(object):

    def __init__(self, set, *dims):
        self._set = set
        self._dims = dims

class Kernel(object):

    def __init__(self, code):
        self._code = code

    def compile():
        pass

    def handle():
        pass

# Data API

class Dat(object):
    pass

class Mat(Dat):
    pass

class Set(object):
    pass

class Map(object):
    pass

class Const(object):
    pass

