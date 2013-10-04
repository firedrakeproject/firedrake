#!/usr/bin/env python

#    This file is part of Diamond.
#
#    Diamond is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Diamond is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Diamond.  If not, see <http://www.gnu.org/licenses/>.

import plist

def print_type(datatype, bracket = True):
  """
  Create a string to be displayed in place of empty data / attributes.
  """

  def familiar_type(type_as_printable):
    """
    Convert some type names to more familiar equivalents.
    """

    if type_as_printable == "decim":
      return "float"
    elif type_as_printable == "int":
      return "integer"
    elif type_as_printable == "str":
      return "string"
    else:
      return type_as_printable

  def type_name(datatype):
    """
    Return a human readable version of datatype.
    """

    datatype_string = str(datatype)

    if datatype_string[:7] == "<type '" and datatype_string[len(datatype_string) - 2:] == "'>":
      value_type_split = datatype_string.split("'")
      return familiar_type(value_type_split[1])

    value_type_split1 = datatype_string.split(".")
    value_type_split2 = value_type_split1[len(value_type_split1) - 1].split(" ")
    if len(value_type_split2) == 1:
      return familiar_type(value_type_split2[0][0:len(value_type_split2[0]) - 6])
    else:
      return familiar_type(value_type_split2[0])

  #print_type

  if isinstance(datatype, plist.List):
    if (isinstance(datatype.cardinality, int) and datatype.cardinality == 1) or datatype.cardinality == "":
      type_as_printable = type_name(datatype.datatype).lower()
    else:
      type_as_printable = type_name(datatype).lower() + " of "
      list_type_as_printable = type_name(datatype.datatype).lower()
      if isinstance(datatype.cardinality, int):
        type_as_printable += str(datatype.cardinality) + " " + list_type_as_printable + "s"
      else:
        type_as_printable += list_type_as_printable + "s"
  else:
    type_as_printable = type_name(datatype).lower()

  if bracket:
    type_as_printable = "(" + type_as_printable + ")"

  return type_as_printable
