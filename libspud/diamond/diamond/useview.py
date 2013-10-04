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

import gobject
import gtk
import os
import threading

import schemausage

RELAXNGNS = "http://relaxng.org/ns/structure/1.0"
RELAXNG = "{" + RELAXNGNS + "}"

class UseView(gtk.Window):
  def __init__(self, schema, suffix, folder = None):
    gtk.Window.__init__(self)
    self.__add_controls()

    if folder is None:
      dialog = gtk.FileChooserDialog(title = "Input directory",
                                   action = gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER,
                                   buttons = (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))

      response = dialog.run()
      if response != gtk.RESPONSE_OK:
        dialog.destroy()
        return

      folder = os.path.abspath(dialog.get_filename())
      dialog.destroy()
    #endif

    paths = []
    for dirpath, dirnames, filenames in os.walk(folder):
      paths.extend([os.path.join(dirpath, filename) for filename in filenames if filename.endswith(suffix)])

    self.__update(schema, paths)
    self.show_all()

  def __add_controls(self):
    self.set_title("Unused schema entries")
    self.set_default_size(800, 600)

    vbox = gtk.VBox()

    scrolledwindow = gtk.ScrolledWindow()
    scrolledwindow.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)

    self.treeview = gtk.TreeView()

    self.treeview.get_selection().set_mode(gtk.SELECTION_SINGLE)

    # Node column
    celltext = gtk.CellRendererText()
    column = gtk.TreeViewColumn("Node", celltext)
    column.set_cell_data_func(celltext, self.set_celltext)

    self.treeview.append_column(column)

    # 0: The node tag
    # 1: Used (0 == Not used, 1 = Child not used, 2 = Used)
    self.treestore = gtk.TreeStore(gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT)
    self.treeview.set_enable_search(False)

    scrolledwindow.add(self.treeview)
    vbox.pack_start(scrolledwindow)

    self.statusbar = gtk.Statusbar()
    vbox.pack_end(self.statusbar, expand = False)
    self.add(vbox)

  def __set_treestore(self, node):
    def set_treestore(node, iter, type):
      if node.tag == RELAXNG + "element":
        name = schemausage.node_name(node)
        if name == "comment":
          return #early out to skip comment nodes

        tag = name + (type if type else "")
        child_iter = self.treestore.append(iter, [tag, 2])
        self.mapping[self.tree.getpath(node)] = self.treestore.get_path(child_iter)
        type = None
      elif node.tag == RELAXNG + "choice" and all(n.tag != RELAXNG + "value" for n in node):
        tag = "choice" + (type if type else "")
        child_iter = self.treestore.append(iter, [tag, 2])
        self.mapping[self.tree.getpath(node)] = self.treestore.get_path(child_iter)
        type = None
      elif node.tag == RELAXNG + "optional":
        child_iter = iter
        type = " ?"
      elif node.tag == RELAXNG + "oneOrMore":
        child_iter = iter
        type = " +"
      elif node.tag == RELAXNG + "zeroOrMore":
        child_iter = iter
        type = " *"
      elif node.tag == RELAXNG + "ref":
        query = '/t:grammar/t:define[@name="' + node.get("name") + '"]'
        if query not in cache:
          cache[query] = self.tree.xpath(query, namespaces={'t': RELAXNGNS})[0]
        node = cache[query]
        child_iter = iter
      elif node.tag == RELAXNG + "group" or node.tag == RELAXNG + "interleave":
        child_iter = iter
      else:
        return

      for child in node:
        set_treestore(child, child_iter, type)

    cache = {}
    set_treestore(node, None, None)

  def __set_useage(self, useage):
    for xpath in useage:
      try:
        iter = self.treestore.get_iter(self.mapping[xpath])
        self.treestore.set_value(iter, 1, 0)
      except KeyError:
        pass #probably a comment node

  def __floodfill(self, iter, parent = 2):
    """
    Floodfill the tree with the correct useage.
    """
    if parent == 0: #parent is not used
      self.treestore.set_value(iter, 1, 0) #color us not used

    useage = self.treestore.get_value(iter, 1)

    child = self.treestore.iter_children(iter)
    while child is not None:
      change = self.__floodfill(child, useage)
      if change != 2 and useage == 2:
        self.treestore.set(iter, 1, 1)
      child = self.treestore.iter_next(child)

    return self.treestore.get_value(iter, 1)


  def __update(self, schema, paths):
    self.tree = schema.tree
    start = self.tree.xpath('/t:grammar/t:start', namespaces={'t': RELAXNGNS})[0]
    self.mapping = {}

    def async_update(self, start, schema, paths, context):
      gtk.idle_add(self.statusbar.push, context, "Parsing schema")
      self.__set_treestore(start[0])
      gtk.idle_add(self.statusbar.push, context, "Schema parsed... finding usage")
      self.__set_useage(schemausage.find_unusedset(schema, paths))
      gtk.idle_add(self.statusbar.push, context, "Usage found")
      self.__floodfill(self.treestore.get_iter_root())
      gtk.idle_add(self.statusbar.push, context, "")
      gtk.idle_add(self.treeview.set_model, self.treestore)

    t = threading.Thread(target = async_update, args = (self, start, schema, paths, self.statusbar.get_context_id("update")))
    t.start()

  def set_celltext(self, column, cell, model, iter):
    tag, useage = model.get(iter, 0, 1)
    cell.set_property("text", tag)

    if useage == 0:
      cell.set_property("foreground", "red")
    elif useage == 1:
      cell.set_property("foreground", "indianred")
    else:
      cell.set_property("foreground", "black")
