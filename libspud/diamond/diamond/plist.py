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

class List:
  def __init__(self, datatype, cardinality=''):
    self.datatype = datatype
    self.cardinality = cardinality

  # The input to call is a string containing a list of elements seperated by "," or " ". It
  # returns a string, containing the elements in val seperated by " ".
  def __call__(self, val):
    val = val.strip()
    if "," in val:
      x = val.split(",")
    else:
      x = val.split(" ")

    # Perform some checks on the list cardinality, for oneOrMore elements ('+')
    # and compulsory elements ('').
    if self.cardinality == '+':
      assert len(x) > 0
    elif self.cardinality == '':
      assert len(x) == 1

    # Check the list cardinality (as an integer) matches up with the number of elements
    # in the seperated val string.
    try:
      assert len(x) == int(self.cardinality)
    except ValueError:
      pass # The int conversion may fail (if cardinality is '+' or ''), so just ignore it.

    # Make sure each element can be converted to type 'self.datatype'. An exception will be
    # thrown if this is not possible.
    for y in x:
      z = self.datatype(y)

    # Return a string of the elements in val, seperated by " ".
    return " ".join(x)

  def __str__(self):
    return "list of " + str(self.datatype) + " of cardinality: " + self.cardinality

  def __repr__(self):
    return "list of " + str(self.datatype) + " of cardinality: " + self.cardinality
