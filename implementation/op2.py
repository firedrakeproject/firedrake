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

class Access(Object):

class Arg(Object):
    """Represents a single argument passed to a par_loop"""
    def __init__(self, dat, index, map, dim, access):
        self._dat = dat
        self._index = index
        self._map = map
        self._dim = dim
        self._access = access

class ArgDat(Arg):
    """Represents a single dat argument pass to a par_loop"""
    def __init__(self, dat, index, map, dim, access):
        super(ArgDat, self).__init__(dat, index, map, dim, access)

class IterationSpace(Object):

    def __init__(self, set, *dims):
        self._set = set
        self._dims = dims

class Kernel(Object):

    def __init__(self, code)
        self._code = code

    def compile():
        pass

# Data API

class Dat(Object):
    pass

class Mat(Dat):
    pass

class Set(Object):
    pass

class Map(Object):
    pass

class Const(Object):
    pass

