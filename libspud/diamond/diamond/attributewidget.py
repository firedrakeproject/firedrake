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

import datatype
import dialogs

class AttributeWidget(gtk.Frame):

  __gsignals__ = { "on-store" : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, ()),
                   "update-name"  : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, ())}

  def __init__(self):
    gtk.Frame.__init__(self)

    scrolledWindow = gtk.ScrolledWindow()
    scrolledWindow.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)

    treeview = self.treeview = gtk.TreeView()

    model = gtk.ListStore(gobject.TYPE_STRING, gobject.TYPE_STRING, gobject.TYPE_PYOBJECT)
    treeview.set_model(model)
    treeview.connect("motion-notify-event", self.treeview_mouse_over)

    key_renderer = gtk.CellRendererText()
    key_renderer.set_property("editable", False)

    column1 = gtk.TreeViewColumn("Name", key_renderer, text = 0)
    column1.set_cell_data_func(key_renderer, self.key_data_func)
    column1.set_property("min-width", 75)

    entry_renderer = gtk.CellRendererText()
    entry_renderer.connect("edited", self.entry_edited)
    entry_renderer.connect("editing-started", self.entry_edit_start)

    combo_renderer = gtk.CellRendererCombo()
    combo_renderer.set_property("text-column", 0)
    combo_renderer.connect("edited", self.combo_selected)
    combo_renderer.connect("editing-started", self.combo_edit_start)

    column2 = gtk.TreeViewColumn("Value", entry_renderer, text = 1)
    column2.pack_start(combo_renderer)
    column2.set_attributes(combo_renderer, text = 1)
    column2.set_cell_data_func(entry_renderer, self.entry_data_func)
    column2.set_cell_data_func(combo_renderer, self.combo_data_func)
    column2.set_property("expand", True)
    column2.set_property("min-width", 75)

    icon_renderer = gtk.CellRendererPixbuf()

    column3 = gtk.TreeViewColumn("", icon_renderer)
    column3.set_cell_data_func(icon_renderer, self.icon_data_func)

    treeview.append_column(column1)
    treeview.append_column(column2)
    treeview.append_column(column3)

    scrolledWindow.add(treeview)

    label = gtk.Label()
    label.set_markup("<b>Attributes</b>")

    self.set_label_widget(label)
    self.set_shadow_type(gtk.SHADOW_NONE)
    self.add(scrolledWindow)

  def update(self, node):

    self.treeview.get_model().clear()

    self.node = node

    if node is None or node.all_attrs_fixed():
      self.set_property("visible", False)
    else:
      self.set_property("visible", True)

      for key in node.attrs.keys():
        model = self.treeview.get_model()
        cell_model = gtk.ListStore(gobject.TYPE_STRING)

        iter = model.append()
        model.set_value(iter, 0, key)
        model.set_value(iter, 2, cell_model)

        if isinstance(node.attrs[key][0], tuple):
          if node.attrs[key][1] is None:
            if isinstance(node.attrs[key][0][0], tuple):
              model.set_value(iter, 1, "Select " + datatype.print_type(node.attrs[key][0][1]) + "...")
            else:
              model.set_value(iter, 1, "Select...")
          else:
            model.set_value(iter, 1, node.attrs[key][1])

          if isinstance(node.attrs[key][0][0], tuple):
            opts = node.attrs[key][0][0]
          else:
            opts = node.attrs[key][0]

          for opt in opts:
            cell_iter = cell_model.append()
            cell_model.set_value(cell_iter, 0, opt)

          self.treeview.get_column(2).set_property("visible", True)
        elif node.attrs[key][0] is None:
          model.set_value(iter, 1, "No data")
        elif node.attrs[key][1] is None:
          model.set_value(iter, 1, datatype.print_type(node.attrs[key][0]))
        else:
          model.set_value(iter, 1, node.attrs[key][1])

      self.treeview.queue_resize()

    return

  def treeview_mouse_over(self, widget, event):
    """
    Called when the mouse moves over the attributes widget. Sets the
    appropriate attribute widget tooltip.
    """

    path_info = self.treeview.get_path_at_pos(int(event.x), int(event.y))
    if path_info is None:
      try:
        self.treeview.set_tooltip_text("")
        self.treeview.set_property("has-tooltip", False)
      except:
        pass
      return

    path = path_info[0]
    col = path_info[1]
    if col is not self.treeview.get_column(1):
      try:
        self.treeview.set_tooltip_text("")
        self.treeview.set_property("has-tooltip", False)
      except:
        pass
      return

    iter = self.treeview.get_model().get_iter(path)
    iter_key = self.treeview.get_model().get_value(iter, 0)

    return

  def key_data_func(self, col, cell_renderer, model, iter):
    """
    Attribute name data function. Sets the cell renderer text colours.
    """

    iter_key = model.get_value(iter, 0)

    if not self.node.active or self.node.attrs[iter_key][0] is None or self.node.attrs[iter_key][0] == "fixed":
      cell_renderer.set_property("foreground", "grey")
    elif self.node.attrs[iter_key][1] is None:
      cell_renderer.set_property("foreground", "blue")
    else:
      cell_renderer.set_property("foreground", "black")

    return

  def entry_data_func(self, col, cell_renderer, model, iter):
    """
    Attribute text data function. Hides the renderer if a combo box is required,
    and sets colours and editability otherwise.
    """

    iter_key = model.get_value(iter, 0)

    if not self.node.active or self.node.attrs[iter_key][0] is None or self.node.attrs[iter_key][0] == "fixed":
      cell_renderer.set_property("editable", False)
      cell_renderer.set_property("foreground", "grey")
      cell_renderer.set_property("visible", True)
    elif not isinstance(self.node.attrs[iter_key][0], tuple):
      cell_renderer.set_property("editable", True)
      cell_renderer.set_property("visible", True)
      if self.node.attrs[iter_key][1] is None:
        cell_renderer.set_property("foreground", "blue")
      else:
        cell_renderer.set_property("foreground", "black")
    else:
      cell_renderer.set_property("editable", False)
      cell_renderer.set_property("visible", False)

    return

  def combo_data_func(self, col, cell_renderer, model, iter):
    """
    Attribute combo box data function. Hides the renderer if a combo box is not
    required, and sets the combo box options otherwise. Adds an entry if required.
    """

    iter_key = model.get_value(iter, 0)

    if self.node.active and isinstance(self.node.attrs[iter_key][0], tuple):
      cell_renderer.set_property("editable", True)
      cell_renderer.set_property("visible", True)
      if isinstance(self.node.attrs[iter_key][0][0], tuple):
        cell_renderer.set_property("has-entry", True)
      else:
        cell_renderer.set_property("has-entry", False)
      if self.node.attrs[iter_key][1] is None:
        cell_renderer.set_property("foreground", "blue")
      else:
        cell_renderer.set_property("foreground", "black")
    else:
      cell_renderer.set_property("visible", False)
      cell_renderer.set_property("editable", False)
    cell_renderer.set_property("model", model.get_value(iter, 2))

    return

  def icon_data_func(self, col, cell_renderer, model, iter):
    """
    Attribute icon data function. Used to add downward pointing arrows for combo
    attributes, for consistency with the LHS.
    """

    iter_key = model.get_value(iter, 0)

    if self.node.active and isinstance(self.node.attrs[iter_key][0], tuple):
      cell_renderer.set_property("stock-id", gtk.STOCK_GO_DOWN)
    else:
      cell_renderer.set_property("stock-id", None)

    return

  def entry_edit_start(self, cell_renderer, editable, path):
    """
    Called when editing is started on an attribute text cell. Used to delete the
    printable_type placeholder.
    """

    iter = self.treeview.get_model().get_iter(path)
    iter_key = self.treeview.get_model().get_value(iter, 0)

    if self.node.attrs[iter_key][1] is None:
      editable.set_text("")

    return

  def combo_edit_start(self, cell_renderer, editable, path):
    """
    Called when editing is started on an attribute combo cell. Used to delete the
    select placeholder for mixed entry / combo attributes.
    """

    iter = self.treeview.get_model().get_iter(path)
    iter_key = self.treeview.get_model().get_value(iter, 0)

    if isinstance(self.node.attrs[iter_key][0][0], tuple) and self.node.attrs[iter_key][1] is None:
      editable.child.set_text("")

    return

  def entry_edited(self, cell_renderer, path, new_text):
    """
    Called when editing is finished on an attribute text cell. Updates data in the
    treestore.
    """

    iter = self.treeview.get_model().get_iter(path)
    iter_key = self.treeview.get_model().get_value(iter, 0)

    if self.node.get_attr(iter_key) is None and new_text == "":
      return

    value_check = self.node.validity_check(self.node.attrs[iter_key][0], new_text)

    if value_check is not None and value_check != self.node.attrs[iter_key][1]:
      if iter_key == "name" and not self._name_check(value_check):
        return

      self.treeview.get_model().set_value(iter, 1, value_check)
      self.node.set_attr(iter_key, value_check)
      if iter_key == "name":
        self.emit("update-name")

      self.emit("on-store")

    return

  def combo_selected(self, cell_renderer, path, new_text):
    """
    Called when an attribute combo box element is selected, or combo box entry
    element entry is edited. Updates data in the treestore.
    """

    iter = self.treeview.get_model().get_iter(path)
    iter_key = self.treeview.get_model().get_value(iter, 0)

    if new_text is None:
      return

    if isinstance(self.node.attrs[iter_key][0][0], tuple) and new_text not in self.node.attrs[iter_key][0][0]:
      if self.node.get_attr(iter_key) is None and new_text == "":
        return

      new_text = self.node.validity_check(self.node.attrs[iter_key][0][1], new_text)
      if iter_key == "name" and not self._name_check(new_text):
        return False
    if new_text != self.node.attrs[iter_key][1]:
      self.treeview.get_model().set_value(iter, 1, new_text)
      self.node.set_attr(iter_key, new_text)
      if iter_key == "name":
        self.emit("update-name")

      self.emit("on-store")

    return

  def _name_check(self, value):
    """
    Check to see if the supplied data is a valid tree name.
    """

    valid_chars = "_:[]1234567890qwertyuioplkjhgfdsazxcvbnmMNBVCXZASDFGHJKLPOIUYTREWQ"
    for char in value:
      if char not in valid_chars:
        dialogs.error(None, "Invalid value entered")
        return False

    return True

gobject.type_register(AttributeWidget)
