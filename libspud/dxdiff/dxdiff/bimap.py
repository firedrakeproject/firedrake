#!/usr/bin/env python

#    This file is part of dxdiff.
#
#    dxdiff is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    dxdiff is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Diamond.  If not, see <http://www.gnu.org/licenses/>.


class Bimap:
  """
  Bimap is a simple wrapper class over two dicts,
  it behaves as a bi-directional map.
  """

  def __init__(self):
    self.left = {}
    self.right = {}

  def __len__(self):
    return len(self.left)

  def __iter__(self):
    # we iter over the left dict so that the left item is
    # on the left side of the tuple returned
    for item in self.left.iteritems():
      yield item

  def __contains__(self, item):
    # check that the left dict contains left and points to right
    try:
      left, right = item
      return self.left[left] == right
    except KeyError:
      return False

  def add(self, item):
    x, y = item
    self.left[x] = y
    self.right[y] = x
