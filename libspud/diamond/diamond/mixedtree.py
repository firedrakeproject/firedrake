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

class MixedTree:
  def __init__(self, parent, child):
    """
    The .doc and .attrs comes from parent, and the .data comes from child. This
    is used to hide integer_value etc. from the left hand side, but for its data
    entry to show up on the right.
    """

    self.parent = parent
    self.child = child

    self.name = parent.name
    self.schemaname = parent.schemaname

    excluded_attrs = ["shape"]
    self.attrs = dict(self.parent.attrs.items() + [x for x in self.child.attrs.items() if x[0] not in excluded_attrs])

    self.children = parent.children
    self.datatype = child.datatype
    self.data = child.data
    self.doc = parent.doc
    self.active = parent.active

    return

  def set_attr(self, attr, val):
    if attr in self.parent.attrs:
      self.parent.set_attr(attr, val)
    elif attr in self.child.attrs:
      self.child.set_attr(attr, val)
    else:
      raise Exception, "Attribute not present in either parent or child!"

    return

  def get_attr(self, attr):
    if attr in self.parent.attrs:
      self.parent.get_attr(attr)
    elif attr in self.child.attrs:
      self.child.get_attr(attr)
    else:
      raise Exception, "Attribute not present in either parent or child!"

  def set_data(self, data):
    self.child.set_data(data)
    self.datatype = self.child.datatype
    self.data = self.child.data

    return

  def valid_data(self, datatype, data):
    return self.child.valid_data(datatype, data)

  def validity_check(self, datatype, data):
    return self.child.validity_check(datatype, data)

  def matches(self, text, case_sensitive = False):
    old_parent_data = self.parent.data
    self.parent.data = None
    parent_matches = self.parent.matches(text, case_sensitive)
    self.parent.data = old_parent_data

    if parent_matches:
      return True

    if case_sensitive:
      text_re = re.compile(text)
    else:
      text_re = re.compile(text, re.IGNORECASE)

    if not self.child.data is None and not text_re.search(self.child.data) is None:
      return True
    else:
      return False

  def is_comment(self):
    return self.parent.is_comment()

  def get_comment(self):
    return self.parent.get_comment()

  def is_tensor(self, geometry_dim_tree):
    """
    Perform a series of tests on the current MixedTree, to determine if
    it is intended to be used to store tensor or vector data.
    """

    # Check that a geometry is defined
    if geometry_dim_tree is None:
      return False

    # Check that this element has calculable and positive dimensions
      try:
        dim1, dim2 = self.tensor_shape(geometry_dim_tree)
        assert dim1 > 0
        assert dim2 > 0
      except AssertionError:
        return False

    # The element must have dim1, rank and shape attributes
    if "dim1" not in self.child.attrs.keys() \
       or "rank" not in self.child.attrs.keys() \
       or "shape" not in self.child.attrs.keys():
      return False

    # The dim1 and rank attributes must be of fixed type
    if self.child.attrs["dim1"][0] != "fixed" or self.child.attrs["rank"][0] != "fixed":
      return False

    if "dim2" in self.child.attrs.keys():
      # If a dim2 attribute is specified, it must be of fixed type and the rank must be 2
      # Also, the shape attribute must be a list of integers with cardinality equal to the rank
      if self.child.attrs["dim2"][0] != "fixed" \
         or self.child.attrs["rank"][1] != "2" \
         or not isinstance(self.child.attrs["shape"][0], plist.List) \
         or self.child.attrs["shape"][0].datatype is not int \
         or str(self.child.attrs["shape"][0].cardinality) != self.child.attrs["rank"][1]:
        return False

    # Otherwise, the rank must be one and the shape an integer
    elif self.child.attrs["rank"][1] != "1" or self.child.attrs["shape"][0] is not int:
      return False

    # The data for the element must be a list of one or more
    if not isinstance(self.datatype, plist.List) or self.datatype.cardinality != "+":
      return False

    # If the shape has been set, check that it has a valid value
    if self.child.attrs["shape"][1] != None:
      if geometry_dim_tree.data is None:
        return False

      dim1, dim2 = self.tensor_shape(geometry_dim_tree)
      if "dim2" in self.child.attrs.keys():
        if self.child.attrs["shape"][1] != str(dim1) + " " + str(dim2):
          return False
      elif self.child.attrs["shape"][1] != str(dim1):
        return False

    return True

  def tensor_shape(self, dimension):
    """
    Read the tensor shape for tensor or vector data in the current MixedTree.
    """

    if not isinstance(dimension, int):
      dimension = int(dimension.data)

    dim1 = 1
    dim2 = 1
    if "dim1" in self.child.attrs.keys():
      dim1 = int(eval(self.child.attrs["dim1"][1], {"dim":dimension}))
      if "dim2" in self.child.attrs.keys():
        dim2 = int(eval(self.child.attrs["dim2"][1], {"dim":dimension}))

    return (dim1, dim2)

  def is_symmetric_tensor(self, geometry):
    """
    Read if the tensor data in the current MixedTree is symmetric.
    """

    dim1, dim2 = self.tensor_shape(geometry)

    return dim1 == dim2 and "symmetric" in self.child.attrs.keys() and self.child.attrs["symmetric"][1] == "true"

  def is_code(self):
    """
    Perform a series of tests on the current MixedTree, to determine if
    it is intended to be used to store code data.
    """
    try:
      lang = self.get_attr("language")
      if lang == "python":
        return True
    except:
      pass

    if self.datatype is not str:
      return False

    if "type" in self.child.attrs.keys():
      return self.child.attrs["type"][1] == "code"

    return False

  def get_code_language(self):
    """
    Assuming this is a code snippet return the language.
    """

    if not self.is_code():
      return "python"

    try:
      lang = self.child.get_attr("language")
      return lang
    except:
      try:
        lang = self.get_attr("language")
        return lang
      except:
        return "python"

  def get_name_path(self, leaf = True):
    return self.parent.get_name_path(leaf)

  def is_sliceable(self):
    return True

  def all_attrs_fixed(self):
    for attr in self.attrs:
      if self.attrs[attr][0] != "fixed":
        return False

    return True
