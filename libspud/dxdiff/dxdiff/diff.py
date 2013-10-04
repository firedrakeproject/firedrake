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

import fmes

def diff(xmlold, xmlnew):
  """
  Compares two xml trees.
  Returns an editscript to transform old into new.
  """

  return fmes.diff(xmlold, xmlnew)
