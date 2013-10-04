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

import base64
import bz2
import copy
import StringIO
from lxml import etree


import gobject

import tree

class Choice(gobject.GObject):

  __gsignals__ = { "on-set-data" : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, (str,)),
                   "on-set-attr"  : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, (str, str))}

  def __init__(self, choices, schemaname="", cardinality=''):
    gobject.GObject.__init__(self)

    self.choices = choices
    if choices == []:
      raise Exception
    self.index = 0
    name = ""
    for choice in choices:
      assert choice.__class__ is tree.Tree
      choice.connect("on-set-data", self._on_set_data)
      choice.connect("on-set-attr", self._on_set_attr)

    name = ":".join(choice.name for choice in choices)

    self.name = name
    self.schemaname = schemaname
    self.cardinality = cardinality
    self.parent = None
    self.set_default_active()

  def _on_set_data(self, node, data):
    self.emit("on-set-data", data)

  def _on_set_attr(self, node, attr, value):
    self.emit("on-set-attr", attr, value)

  def get_attrs(self):
    """Get all attributes"""
    return self.get_current_tree().get_attrs()

  def set_default_active(self):
    self.active = True
    if self.cardinality == '?' or self.cardinality == '*':
      self.active = False

  def choose(self, i):
    self.index = i

  def find_tree(self, name):
    for t in self.choices:
      if t.name == name:
        return t

    debug.deprint("self.name == %s" % self.name, 0)
    for choice in self.choices:
      debug.deprint("choice.name == %s" % choice.name, 0)
    raise Exception, "No such choice name: %s" % name

  def set_active_choice_by_name(self, name):
    matched = False
    for t in self.choices:
      if t.name == name.strip():
        matched = True
        self.index = self.choices.index(t)

    if not matched:
      raise Exception, "no such name %s found" % name

    self.recompute_validity()

  def set_active_choice_by_ref(self, ref):
    self.index = self.choices.index(ref)
    self.recompute_validity()

  def get_current_tree(self):
    return self.choices[self.index]

  def add_children(self, schema):
    return self.get_current_tree().add_children(schema)

  def pickle(self):
    return base64.b64encode(bz2.compress(pickle.dumps(self)))

  def recompute_validity(self):
    self.get_current_tree().recompute_validity()

  def copy(self):
    new_choices = []
    for choice in self.choices:
      new_choices.append(choice.copy())

    new_choice = Choice(new_choices)
    for attr in ["index", "name", "schemaname", "cardinality", "active"]:
      setattr(new_choice, attr, copy.copy(getattr(self, attr)))

    new_choice.set_parent(self.parent)
    for choice in new_choice.choices:
      choice.children = copy.copy([])

    return new_choice

  def get_possible_names(self):
    return [x.name for x in self.choices]

  def set_parent(self, parent):
    self.parent = parent
    for choice in self.choices:
      choice.parent = parent

  def write_core(self, parent):
    self.choices[self.index].write_core(parent)
    return parent

  def is_comment(self):
    return False

  def get_comment(self):
    return None

  def get_display_name(self):
    """
    This is a fluidity hack, allowing the name displayed in the treeview on the
    left to be different to the element name. If it has an attribute name="xxx",
    element_tag (xxx) is displayed.
    """

    return self.get_current_tree().get_display_name()

  def get_name(self):
    return self.get_current_tree().get_name()

  def get_children(self):
    return [self.get_current_tree()]

  def get_choices(self):
    return self.choices

  def is_hidden(self):
    """
    Tests whether the supplied choice should be hidden in view.
    """
    return False

  def get_name_path(self, leaf = True):
    return self.get_current_tree().get_name_path(leaf)

  def get_mixed_data(self):
    return self

  def is_sliceable(self):
    return self.get_current_tree().is_sliceable()

  def __str__(self):
    """
    Returns the display name of the selected tree.
    """
    return self.get_display_name()

  def __repr__(self):
    return self.get_name_path() + "[" + self.name + "]"

gobject.type_register(Choice)
