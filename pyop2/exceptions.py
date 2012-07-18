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

"""OP2 exception types"""

class DataTypeError(TypeError):
    """Invalid type for data."""

class DimTypeError(TypeError):
    """Invalid type for dimension."""

class IndexTypeError(TypeError):
    """Invalid type for index."""

class NameTypeError(TypeError):
    """Invalid type for name."""

class SetTypeError(TypeError):
    """Invalid type for Set."""

class SizeTypeError(TypeError):
    """Invalid type for size."""

class DataValueError(ValueError):
    """Illegal value for data."""

class IndexValueError(ValueError):
    """Illegal value for index."""

class ModeValueError(ValueError):
    """Illegal value for mode."""

class SetValueError(ValueError):
    """Illegal value for Set."""
