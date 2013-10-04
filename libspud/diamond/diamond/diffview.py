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

import os
import os.path
import sys
import cStringIO as StringIO

import gobject
import gtk
import threading

from lxml import etree

import attributewidget
import databuttonswidget
import datawidget
import mixedtree

from config import config
import dxdiff.diff as xmldiff

class DiffView(gtk.Window):

  def __init__(self, path, tree):
    gtk.Window.__init__(self)

    self.__add_controls()

    tree1 = etree.parse(path)
    tree2 = etree.ElementTree(tree.write_core(None))

    self.__update(tree1, tree2)

    self.show_all()

  def __add_controls(self):
    self.set_default_size(800, 600)
    self.set_title("Diff View")

    mainvbox = gtk.VBox()

    menubar = gtk.MenuBar()
    edititem = gtk.MenuItem("_Edit")
    menubar.append(edititem)

    agr = gtk.AccelGroup()
    self.add_accel_group(agr)

    self.popup = editmenu = gtk.Menu()
    edititem.set_submenu(editmenu)
    copyitem = gtk.MenuItem("Copy")
    copyitem.connect("activate", self.on_copy)
    key, mod = gtk.accelerator_parse("<Control>C")
    copyitem.add_accelerator("activate", agr, key, mod, gtk.ACCEL_VISIBLE)
    editmenu.append(copyitem)

    mainvbox.pack_start(menubar, expand = False)

    hpane = gtk.HPaned()
    scrolledwindow = gtk.ScrolledWindow()
    scrolledwindow.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)

    self.treeview = gtk.TreeView()

    self.treeview.get_selection().set_mode(gtk.SELECTION_SINGLE)
    self.treeview.get_selection().connect("changed", self.on_select_row)

    # Node column
    celltext = gtk.CellRendererText()
    column = gtk.TreeViewColumn("Node", celltext)
    column.set_cell_data_func(celltext, self.set_celltext)

    self.treeview.append_column(column)

    # 0: The node tag
    # 1: The attributes dict
    # 2: The value of the node if any
    # 3: The old value of the node
    # 4: "insert", "delete", "update",  ""
    self.treestore = gtk.TreeStore(gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT)
    self.treeview.set_enable_search(False)
    self.treeview.connect("button_press_event", self.on_treeview_button_press)
    self.treeview.connect("popup_menu", self.on_treeview_popup)
    scrolledwindow.add(self.treeview)
    hpane.pack1(scrolledwindow, True)

    vpane = gtk.VPaned()
    frame = gtk.Frame()
    label = gtk.Label()
    label.set_markup("<b>Attributes</b>")
    frame.set_label_widget(label)
    frame.set_shadow_type(gtk.SHADOW_NONE)

    self.attribview = gtk.TreeView()

    celltext = gtk.CellRendererText()
    keycolumn = gtk.TreeViewColumn("Key", celltext)
    keycolumn.set_cell_data_func(celltext, self.set_cellkey)

    self.attribview.append_column(keycolumn)

    celltext = gtk.CellRendererText()
    valuecolumn = gtk.TreeViewColumn("Value", celltext)
    valuecolumn.set_cell_data_func(celltext, self.set_cellvalue)

    self.attribview.append_column(valuecolumn)

    frame.add(self.attribview)
    vpane.pack1(frame)

    frame = gtk.Frame()
    label = gtk.Label()
    label.set_markup("<b>Data</b>")
    frame.set_label_widget(label)
    frame.set_shadow_type(gtk.SHADOW_NONE)

    self.dataview = gtk.TextView()
    self.dataview.set_cursor_visible(False)
    self.dataview.set_editable(False)
    self.__create_tags(self.dataview.get_buffer())

    frame.add(self.dataview)
    vpane.pack2(frame)

    hpane.pack2(vpane)
    mainvbox.pack_start(hpane)
    self.add(mainvbox)

  def on_treeview_button_press(self, treeview, event):
    pathinfo = treeview.get_path_at_pos(int(event.x), int(event.y))
    if event.button == 3:
      if pathinfo is not None:
        treeview.get_selection().select_path(pathinfo[0])
        self.show_popup(None, event.button, event.time)
        return True

  def popup_location(self, widget, user_data):
    column = self.treeview.get_column(0)
    path = self.treeview.get_selection().get_selected()[1]
    area = self.treeview.get_cell_area(path, column)
    tx, ty = area.x, area.y
    x, y = self.treeview.tree_to_widget_coords(tx, ty)
    return (x, y, True)

  def on_treeview_popup(self, treeview):
    self.show_popup(None, self.popup_location, gtk.get_current_event_time())
    return

  def show_popup(self, func, button, time):
    self.popup.popup( None, None, func, button, time)
    return

  def __update(self, tree1, tree2):

    def async_diff(self, tree1, tree2):
      editscript = xmldiff.diff(tree1, tree2)
      self.__set_treestore(tree1.getroot())
      self.__parse_editscript(editscript)
      self.__floodfill(self.treestore.get_iter_root())
      gtk.idle_add(self.treeview.set_model, self.treestore)

    t = threading.Thread(target = async_diff, args = (self, tree1, tree2))
    t.start()

  def __set_treestore(self, tree, iter = None):

    attrib = {}
    for key, value in tree.attrib.iteritems():
      #             (new,   old,   edit)
      attrib[key] = (value, value, None)

    child_iter = self.treestore.append(iter, [tree.tag, attrib, tree.text, tree.text, None])
    for child in tree:
      self.__set_treestore(child, child_iter)

  def __parse_editscript(self, editscript):
    for edit in editscript:
      iter, key = self.__get_iter(edit["location"])
      if key:
        attribs = self.treestore.get_value(iter, 1)
        old = attribs[key][1]
        if edit["type"] == "delete":
          attribs[key] = (None, old, "delete")
        elif edit["type"] == "update":
          attribs[key] = (edit["value"], old, "update")
        elif edit["type"] == "move":
          attribs[key] = (None, old, "delete")
          self.__insert(self.__get_iter(edit["value"])[0], key + " " + old, 0)

      else:

        if edit["type"] == "insert":
          self.__insert(iter, edit["value"], int(edit["index"]))
        elif edit["type"] == "delete":
          self.treestore.set(iter, 2, None)
          self.treestore.set(iter, 4, "delete")
        elif edit["type"] == "update":
          self.treestore.set(iter, 2, edit["value"])
        elif edit["type"] == "move":
          self.__move(iter, edit["value"], int(edit["index"]))

  def __floodfill(self, iter, parentedit = None):
    """
    Floodfill the tree with the correct edit types.
    If something has changed below you, "subupdate"
    If your value or attrs has changed "update"
    If insert, all below insert
    If delete, all below delete
    """
    attribs, new, old, edit = self.treestore.get(iter, 1, 2, 3, 4)

    if parentedit == "insert":
      edit = "insert"
    elif parentedit == "delete":
      edit = "delete"

    if edit == "insert" or edit == "delete":
      for key, (valuenew, valueold, valueedit) in attribs.iteritems():
        attribs[key] = (valuenew, valueold, edit)

      child = self.treestore.iter_children(iter)
      while child is not None:
        self.__floodfill(child, edit)
        child = self.treestore.iter_next(child)

      self.treestore.set(iter, 4, edit)
    else:
      update = False
      for key in attribs:
        # edit value
        if attribs[key][2] is not None:
          update = True
          break
      if new != old:
        update = True

      if update:
        self.treestore.set(iter, 4, "update")
      else:
        child = self.treestore.iter_children(iter)
        while child is not None:
          change = self.__floodfill(child, edit)
          if change is not None:
            self.treestore.set(iter, 4, "subupdate")
          child = self.treestore.iter_next(child)

    return self.treestore.get_value(iter, 4)

  def __insert(self, iter, value, index):
    if " " in value:
      key, value = value.split(" ")
      attrib = self.treestore.get_value(iter, 1)
      attrib[key] = (value, None, "insert")
    else:
      before = self.__iter_nth_child(iter, index - 1)
      if before:
        self.treestore.insert_before(iter, before, [value, {}, None, None, "insert"])
      else:
        self.treestore.append(iter, [value, {}, None, None, "insert"])

  def __move(self, iter, value, index):
    """
    Copy the entire subtree at iter to the path at value[index],
    mark all of iter as deleted, and all of the copy inserted.
    """
    tag, attrib, text = self.treestore.get(iter, 0, 1, 2)
    self.treestore.set(iter, 2, None)
    self.treestore.set(iter, 4, "delete")

    destiter = self.__get_iter(value)[0]

    before = self.__iter_nth_child(destiter, index - 1)
    if before:
      destiter = self.treestore.insert_before(destiter, before, [tag, attrib, text, None, "insert"])
    else:
      destiter = self.treestore.append(destiter, [tag, attrib, text, None, "insert"])

    def move(iterfrom, iterto):
      for childfrom in self.__iter_children(iterfrom):
        tag, attrib, text = self.treestore.get(childfrom, 0, 1, 2)
        self.treestore.set(childfrom, 2, None)
        self.treestore.set(childfrom, 4, "delete")

        childto = self.treestore.append(iterto, [tag, attrib, text, None, "insert"])
        move(childfrom, childto)

    move(iter, destiter)

  def __iter_children(self, iter):
    child = self.treestore.iter_children(iter)

    while child:
      active = self.treestore.get_value(child, 4) != "delete"
      if active:
        yield child
      child = self.treestore.iter_next(child)

  def __iter_nth_child(self, iter, n):

    for child in self.__iter_children(iter):
      if n == 0:
        return child
      else:
        n -= 1
    return None

  def __get_iter(self, path, iter = None):
    """
    Convert the given XML path to an iter into the treestore.
    """

    if iter is None:
      iter = self.treestore.get_iter_first()

    tag, edit = self.treestore.get(iter, 0, 4)
    if edit == "delete":
      return None # don't search deleted paths

    parentiter = self.treestore.iter_parent(iter)
    if parentiter:
      siblings = []
      for siblingiter in self.__iter_children(parentiter):
        siblingtag = self.treestore.get_value(siblingiter, 0)
        if siblingtag == tag:
          siblings.append(self.treestore.get_path(siblingiter))

      if len(siblings) != 1:
        index = "[" + str(siblings.index(self.treestore.get_path(iter)) + 1) + "]"
      else:
        index = ""

      tag = "/" + tag + index
    else:
      tag = "/" + tag

    index = path.find("/", 1)
    if index == -1:
      index = len(path)

    root = path[:index]
    path = path[index:]

    #check we match root
    if root != tag:
      return None

    if path:
      # check for text()
      if path == "/text()":
        return (iter, None)

      # check attributes
      if path.startswith("/@"):
        attrib = self.treestore.get_value(iter, 1)
        for key in attrib:
          if path == "/@" + key:
            return (iter, key)
        return None

      # check children
      for iter in self.__iter_children(iter):
        edit = self.treestore.get_value(iter, 4)
        if edit != "delete":
          result = self.__get_iter(path, iter)
          if result:
            return result

      return None
    else:
      # must be us
      return (iter, None)

  def on_select_row(self, selection):
    """
    Called when a row is selected.
    """
    (model, row) = selection.get_selected()
    if row is None:
      return

    attrib, new, old, edit = model.get(row, 1, 2, 3, 4)

    databuffer = self.dataview.get_buffer()
    tag = databuffer.get_tag_table().lookup("tag")
    if new or old:
      self.__set_textdiff(self.dataview.get_buffer(), old, new)
      tag.set_property("background-set", False)
      tag.set_property("foreground", "black")
    else:
      databuffer.set_text("No data")
      self.__set_cell_property(tag, None)
      tag.set_property("foreground", "grey")

    bounds = databuffer.get_bounds()
    databuffer.apply_tag(tag, bounds[0], bounds[1])

    # 0: Key
    # 1: Value
    # 2: Old value
    # 3: "insert", "delete", "update",  ""

    attribstore = gtk.TreeStore(gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT)

    for key, (new, old, diff) in attrib.iteritems():
      attribstore.append(None, [key, new, old, diff])

    self.attribview.set_model(attribstore)

  def __set_textdiff(self, databuffer, old, new):
    text1 = old.splitlines() if old else []
    text2 = new.splitlines() if new else []

    from difflib import Differ
    differ = Differ()
    result = differ.compare(text1, text2)
    result = [line + "\n" for line in result if not line.startswith("? ")]

    databuffer.set_text("")
    for line in result:
      iter = databuffer.get_end_iter()
      if line.startswith("  "):
        databuffer.insert(iter, line)
      elif line.startswith("+ "):
        databuffer.insert_with_tags_by_name(iter, line, "add")
      elif line.startswith("- "):
        databuffer.insert_with_tags_by_name(iter, line, "rem")


  def __create_tags(self, databuffer):
    databuffer.create_tag("tag")
    add = databuffer.create_tag("add")
    rem = databuffer.create_tag("rem")

    add.set_property("background", config.get("colour", "diffadd"))
    rem.set_property("background", config.get("colour", "diffsub"))

  def __set_cell_property(self, cell, edit):
    if edit is None:
      cell.set_property("foreground", config.get("colour", "normal"))
    elif edit == "insert":
      cell.set_property("foreground", config.get("colour", "insert"))
    elif edit == "delete":
      cell.set_property("foreground", config.get("colour", "delete"))
    elif edit == "update":
      cell.set_property("foreground", config.get("colour", "update"))
    elif edit == "subupdate":
      cell.set_property("foreground", config.get("colour", "subupdate"))

  def set_celltext(self, column, cell, model, iter):

    tag, text, edit = model.get(iter, 0, 2, 4)

    cell.set_property("text", tag)
    self.__set_cell_property(cell, edit)

  def set_cellkey(self, column, cell, model, iter):

    key, edit = model.get(iter, 0, 3)
    cell.set_property("text", key)
    self.__set_cell_property(cell, edit)

  def set_cellvalue(self, column, cell, model, iter):

    new, old, edit = model.get(iter, 1, 2, 3)
    if edit == "delete":
      cell.set_property("text", old)
    else:
      cell.set_property("text", new)
    self.__set_cell_property(cell, edit)

  def _get_focus_widget(self, parent):
    """
    Gets the widget that is a child of parent with the focus.
    """
    focus = parent.get_focus_child()
    if focus is None or (focus.flags() & gtk.HAS_FOCUS):
      return focus
    else:
      return self._get_focus_widget(focus)

  def _handle_clipboard(self, widget, signal):
    """
    This finds the currently focused widget.
    If no widget is focused or the focused widget doesn't support
    the given clipboard operation use the treeview (False), otherwise
    signal the widget to handel the clipboard operation (True).
    """
    widget = self._get_focus_widget(self)

    if widget is None or widget is self.treeview:
      return False

    if gobject.signal_lookup(signal + "-clipboard", widget):
      widget.emit(signal + "-clipboard")
      return True
    else:
      return False

  def __get_treestore(self, iter):

    tag, attrib, text = self.treestore.get(iter, 0, 1, 2)

    tree = etree.Element(tag)

    for key, (newvalue, oldvalue, edit) in attrib.iteritems():
      tree.attrib[key] = newvalue

    child_iter = self.treestore.iter_children(iter)
    while child_iter:
      child = self.__get_treestore(child_iter)
      tree.append(child)
      child_iter = self.treestore.iter_next(child_iter)

    return tree

  def on_copy(self, widget=None):
    if self._handle_clipboard(widget, "copy"):
      return

    (model, row) = self.treeview.get_selection().get_selected()
    if row is None:
      return

    tree = etree.ElementTree(self.__get_treestore(row))

    ios = StringIO.StringIO()
    tree.write(ios, pretty_print = True, xml_declaration = False, encoding = "utf-8")

    clipboard = gtk.clipboard_get()
    clipboard.set_text(ios.getvalue())
    clipboard.store()

    ios.close()
