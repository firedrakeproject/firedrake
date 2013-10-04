#/usr/bin/env python

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

import base64
import bz2
import copy
import cPickle as pickle
import cStringIO as StringIO
import re
import zlib
from lxml import etree
import sys

import gobject

import debug
import choice
import mixedtree

class Tree(gobject.GObject):
  """This class maps pretty much 1-to-1 with an xml tree.
     It is used to represent the options in-core."""

  __gsignals__ = { "on-set-data" : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, (str,)),
                   "on-set-attr"  : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, (str, str))}

  def __init__(self, name="", schemaname="", attrs={}, children=None, cardinality='', datatype=None, doc=None):
    gobject.GObject.__init__(self)

    # name: the element name in the options XML
    # e.g. "fluidity_options"
    self.name = name

    # schemaname: the label given to it in the Xvif parsing of the schema
    # this is necessary to walk the tree to see what possible valid
    # children this node could have
    # e.g. "0:elt" for the root node.
    self.schemaname = schemaname

    # Any children?
    if children is None:
      self.children = []
    else:
      self.children = children

    # The cardinality of a node is
    # how many you must/can have, e.g.
    # "exactly one", "zero or one", "any amount", etc.
    # This is set by Schema.valid_children for candidate
    # nodes in the tree, you see.
    # Possible choices: '' '?' '*' '+'
    # with the usual regex meanings.
    self.cardinality = cardinality

    # Used for Optional or ZeroOrMore
    # trees. False means it is present but inactive.
    # must be set if cardinality is changed!
    self.set_default_active()

    # Any documentation associated with this node?
    self.doc = doc

    # What is the parent of this tree?
    # None means the root node.
    self.parent = None

    # Does this node require attention from the user?
    self.valid = False

    # The datatype that this tree stores and the data stored
    if isinstance(datatype, tuple) and len(datatype) == 1:
      self.datatype = "fixed"
      self.data = datatype[0]
    else:
      self.datatype = datatype
      self.data = None

    # The attributes of the tree
    self.attrs = {}
    for key in attrs.keys():
      if isinstance(attrs[key][0], tuple) and len(attrs[key][0]) == 1:
        self.attrs[key] = ("fixed", attrs[key][0][0])
      else:
        self.attrs[key] = attrs[key]

    self.recompute_validity()

  def set_attr(self, attr, val):
    """Set an attribute."""
    (datatype, curval) = self.attrs[attr]
    (invalid, newdata) = self.valid_data(datatype, val)
    if invalid:
      raise Exception, "invalid data: (%s, %s)" % (datatype, val)
    self.attrs[attr] = (datatype, newdata)
    self.recompute_validity()
    self.emit("on-set-attr", attr, val)

  def get_attr(self, attr):
    """Get an attribute."""
    (datatype, curval) = self.attrs[attr]
    return curval

  def get_attrs(self):
    """Get all attributes"""
    return self.attrs

  def set_data(self, data):
    (invalid, data) = self.valid_data(self.datatype, data)
    if invalid:
      raise Exception, "invalid data: (%s, %s)" % (str(self.datatype), data)
    self.data = data
    self.recompute_validity()
    self.emit("on-set-data", data)

  def valid_data(self, datatype, data):
    if datatype is None:
      raise Exception, "datatype is None!"

    elif datatype == "fixed":
      raise Exception, "datatype is fixed!"

    datatypes_to_check = []

    if isinstance(datatype, tuple):
      if isinstance(datatype[0], tuple):
        fixed_values = datatype[0]
      else:
        fixed_values = datatype
      if data in fixed_values:
        return (False, data)
      else:
        if not isinstance(datatype[0], tuple):
          return (True, data)
        datatypes_to_check = list(datatype[1:])
    else:
      datatypes_to_check = [datatype]

    for datatype in datatypes_to_check:
      try:
        tempval = datatype(data)
        if isinstance(tempval, str):
          data = tempval
        return (False, data)
      except:
        pass

    return (True, data)

  def validity_check(self, datatype, data):
    """
    Check to see if the supplied data with supplied type can be stored in a
    tree.Tree.
    """

    (invalid, new_data) = self.valid_data(datatype, data)
    if not invalid and isinstance(new_data, str) and new_data != "":
      if new_data != data and self.validity_check(datatype, new_data) is None:
        return None
      else:
        return new_data
    else:
      return None

  def copy(self):
    new_copy = Tree()
    for attr in ["attrs", "name", "schemaname", "doc", "cardinality", "datatype", "data", "active", "valid"]:
      setattr(new_copy, attr, copy.copy(getattr(self, attr)))

    new_copy.parent = self.parent
    new_copy.children = []

    return new_copy

  def recompute_validity(self):

    new_valid = True

    # if any children are invalid,
    # we are invalid too
    for child in self.children:
      if child.active is False: continue

      if child.__class__ is choice.Choice:
        child = child.get_current_tree()

      if child.valid is False:
        new_valid = False

    # if any attributes are unset,
    # we are invalid.
    for attr in self.attrs.keys():
      (datatype, val) = self.attrs[attr]
      if not datatype is None and val is None:
        new_valid = False

    # if we're supposed to have data and don't,
    # we are invalid.
    if self.datatype is not None:
      if not hasattr(self, "data"):
        new_valid = False

      if self.data is None:
        new_valid = False

    # so we are valid.
    # in either case, let's let the parent know.
    self.valid = new_valid

    if self.parent is not None:
      self.parent.recompute_validity()

  def find_or_add(self, treelist):
    """Append a child node to this node in the tree.
       If it already exists, make tree point to it."""

    outlist = []

    for tree in treelist:
      new_tree = None
      found = False
      for t in self.children:
        if t.schemaname == tree.schemaname:
          tree = t
          found = True
          break
      if not found:
        tree.set_parent(self)

        self.children.append(tree)
        tree.recompute_validity()

      outlist.append(tree)

      for tree in outlist:
        if tree.cardinality in ['+', '*']:
          inactive_list = [x for x in outlist if x.schemaname == tree.schemaname and x.active is False]
          if len(inactive_list) > 0: continue
          else:
            new_tree = self.add_inactive_instance(tree)
            outlist.insert(outlist.index(tree)+1, new_tree)

    return outlist

  def write(self, filename):
    if isinstance(filename, str):
      file = open(filename, "w")
    else:
      file = filename

    xmlTree=etree.tostring(self.write_core(None), pretty_print = True, xml_declaration = True, encoding="utf-8")

    file.write(xmlTree)

  def write_core(self, parent):
    """Write to XML; this is the part that recurses"""

    sub_tree=etree.Element(self.name)

    for key in self.attrs:
      val = self.attrs[key]
      output_val = val[1]
      if output_val is not None:
        sub_tree.set(unicode(key), unicode(output_val))

    for child in self.children:
      if child.active is True:
        child.write_core(sub_tree)

    if self.data is not None:
      sub_tree.text = unicode(self.data)

    if parent is not None:
      parent.append(sub_tree)

    return sub_tree

  def pickle(self):
    if hasattr(self, "xmlnode"):
      del self.xmlnode

    return base64.b64encode(bz2.compress(pickle.dumps(self)))

  def unpickle(self, pick):
    return pickle.loads(bz2.decompress(base64.b64decode(pick)))

  def print_str(self):
    s = "name: %s at %s\n" % (self.name, hex(id(self)))
    s = s + "schemaname: %s\n" % self.schemaname
    s = s + "attrs: %s\n" % self.attrs
    s = s + "children: %s\n" % self.children
    if self.parent is not None:
      s = s + "parent: %s %s at %s\n" % (self.parent.__class__, self.parent.name, hex(id(self.parent)))
    else:
      s = s + "parent: %s at %s\n" % (self.parent.__class__, hex(id(self.parent)))
    s = s + "datatype: %s\n" % str(self.datatype)
    s = s + "data: %s\n" % str(self.data)
    s = s + "cardinality: %s\n" % self.cardinality
    s = s + "active: %s\n" % self.active
    s = s + "valid: %s\n" % self.valid
    return s

  def set_default_active(self):
    self.active = True
    if self.cardinality == '?' or self.cardinality == '*':
      self.active = False

  def count_children_by_schemaname(self, schemaname):
    count = len(filter(lambda x: x.schemaname == schemaname, self.children))
    return count

  def get_children_by_schemaname(self, schemaname):
    return filter(lambda x: x.schemaname == schemaname, self.children)

  def delete_child_by_ref(self, ref):
    self.children.remove(ref)

  def add_inactive_instance(self, tree):
    for t in self.children:
      if t.schemaname == tree.schemaname and t.active is False:
        return t

    new_tree = tree.copy()
    new_tree.active = False
    if new_tree.__class__ is Tree:
      new_tree.children = []
    new_tree.parent = tree.parent
    self.children.insert(self.children.index(tree)+1, new_tree)
    return new_tree

  def print_recursively(self, indent=""):
    s = self.__str__()
    debug.dprint(indent + ' ' + s.replace('\n', '\n' + indent + ' '), 0, newline = False)
    debug.dprint("", 0)
    for i in range(len(self.children)):
      if isinstance(self.children[i], Tree):
        self.children[i].print_recursively(indent + ">>")
      elif isinstance(self.children[i], choice.Choice):
        ref = self.children[i].get_current_tree()
        ref.print_recursively(indent + ">>")
      if i < len(self.children) - 1:
        debug.dprint("", 0)

    return

  def add_children(self, schema):
    l = schema.valid_children(self)
    l = self.find_or_add(l)
    for child in self.children:
      child.add_children(schema)

  def matches(self, text, case_sensitive = False):
    if case_sensitive:
      text_re = re.compile(text)
    else:
      text_re = re.compile(text, re.IGNORECASE)

    if not text_re.search(self.name) is None:
      return True

    if not self.doc is None:
      if not text_re.search(self.doc) is None:
        return True

    for key in self.attrs:
      if not text_re.search(key) is None:
        return True
      if not self.get_attr(key) is None:
        if not text_re.search(self.get_attr(key)) is None:
          return True

    if not self.data is None:
      if not text_re.search(self.data) is None:
        return True

    return False

  def get_current_tree(self):
    return self

  def get_possible_names(self):
    return [self.name]

  def set_parent(self, parent):
    self.parent = parent

  def find_tree(self, name):
    if name == self.name:
      return self
    else:
      raise Exception, "ban the bomb"

  def choices(self):
    return [self]

  def is_comment(self):
    """
    Test whether the given node is a comment node.
    """

    if not self.name == "comment":
      return False

    if not self.attrs == {}:
      return False

    if not self.children == []:
      return False

    if not self.datatype is str:
      return False

    if not self.cardinality == "?":
      return False

    return True

  def get_comment(self):
    """
    Return the first comment found as a child of the supplied node, or None if
    none found.
    """

    for child in self.children:
      if child.is_comment():
        return child

    return None

  def is_tensor(self, geometry_dim_tree):
    return False

  def is_code(self):
    """
    Perform a series of tests on the current Tree, to determine if
    it is intended to be used to store code data.
    """

    try:
      lang = self.get_attr("language")
      if lang == "python":
        return True
    except:
      pass

    return False

  def get_code_language(self):
    """
    Assuming this is a code snippet return the language.
    """

    if not self.is_code():
      return "python"

    try:
      lang = self.get_attr("language")
      return lang
    except:
      return "python"

  def get_display_name(self):
    """
    This is a fluidity hack, allowing the name displayed in the treeview on the
    left to be different to the element name. If it has an attribute name="xxx",
    element_tag (xxx) is displayed.
    """

    name = self.get_name()

    if name is None:
      return self.name
    else:
      return self.name + " (" + name + ")"

  def get_name(self):
    if "name" in self.attrs:
      name = self.attrs["name"][1]
      return name

    return None

  def get_children(self):
    return self.children

  def get_choices(self):
    return [self]

  def is_hidden(self):
    """
    Tests whether the supplied tree should be hidden in view.
    """
    return self.is_comment() or self.name in ["integer_value", "real_value", "string_value", "logical_value"]

  def get_name_path(self, leaf = True):
    name = self.get_display_name() if leaf else self.get_name()

    if self.parent is None:
      return name
    else:

      pname = self.parent.get_name_path(False)

      if name is None:
        return pname
      elif pname is None:
        return name
      else:
        return pname + "/" + name

  def get_mixed_data(self):
    integers = [child for child in self.children if child.name == "integer_value"]
    reals    = [child for child in self.children if child.name == "real_value"]
    logicals = [child for child in self.children if child.name == "logical_value"]
    strings  = [child for child in self.children if child.name == "string_value"]

    child = None
    if len(integers) > 0:
      child = integers[0]
    if len(reals) > 0:
      child = reals[0]
    if len(logicals) > 0:
      child = logicals[0]
    if len(strings) > 0:
      child = strings[0]

    if child is None:
      return self
    else:
      return mixedtree.MixedTree(self, child)

  def is_sliceable(self):
    mixed = self.get_mixed_data()
    if isinstance(mixed, mixedtree.MixedTree):
      return True

    return (self.datatype is not None and self.datatype != "fixed") or self.attrs

  def __str__(self):
    return self.get_display_name()

  def __repr__(self):
    return self.get_name_path()

  def all_attrs_fixed(self):
    for attr in self.attrs:
      if self.attrs[attr][0] != "fixed":
        return False

    return True

gobject.type_register(Tree)

