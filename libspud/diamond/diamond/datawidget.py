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
import pango

import dialogs
import datatype
import mixedtree
import plist

class DataWidget(gtk.VBox):

  __gsignals__ = { "on-store"  : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, ())}

  def __init__(self):
    gtk.VBox.__init__(self)

    frame = self.frame = gtk.Frame()

    label = gtk.Label()
    label.set_markup("<b>Data</b>")

    frame.set_label_widget(label)
    frame.set_shadow_type(gtk.SHADOW_NONE)

    self.pack_start(frame)
    self.buttons = None
    return

  def set_buttons(self, buttons):
    self.buttons = buttons
    buttons.connect("revert", self.revert)
    buttons.connect("store", self.store)
    buttons.show_all()

  def update(self, node):

    self.node = node

    if not self.is_node_editable():
      self.set_data_fixed()
    elif node.is_tensor(self.geometry_dim_tree):
      self.set_data_tensor()
    elif isinstance(node.datatype, tuple):
      self.set_data_combo()
    else:
      self.set_data_entry()

    return

  def revert(self, button = None):
    """
    "Revert Data" button click signal handler. Reverts data in the data frame.
    """

    self.update(self.node)

    return

  def store(self, button = None):
    """
    "Store Data" button click signal handler. Stores data from the data frame
    in the treestore.
    """

    if not self.is_node_editable():
      return True
    elif self.node.is_tensor(self.geometry_dim_tree):
      return self.data_tensor_store()
    elif isinstance(self.node.datatype, tuple):
      return self.data_combo_store()
    else:
      return self.data_entry_store()

# This would be nice, look at it later
#    if self.scherror.errlist_is_open():
#      if self.scherror.errlist_type == 0:
#         self.scherror.on_validate_schematron()
#      else:
#         self.scherror.on_validate()

  def is_node_editable(self):
    return (self.node is not None
       and self.node.active
       and self.node.datatype is not None
       and self.node.datatype != "fixed"
       and (not self.node.is_tensor(self.geometry_dim_tree) or self.geometry_dim_tree.data is not None))
       # not A or B == A implies B
       # A B T
       # 0 0 1
       # 0 1 1
       # 1 0 0
       # 1 1 1

  def add_scrolled_window(self):
    scrolledWindow = gtk.ScrolledWindow()
    scrolledWindow.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
    self.frame.add(scrolledWindow)
    scrolledWindow.show()
    return scrolledWindow

  def add_text_view(self):
    scrolledWindow = self.add_scrolled_window()
    self.set_child_packing(self.frame, True, True, 0, gtk.PACK_START)

    try:
      import gtksourceview2
      buf = gtksourceview2.Buffer()
      lang_manager = gtksourceview2.LanguageManager()
      buf.set_highlight_matching_brackets(True)
      if self.node is not None and self.node.is_code():
        codelanguage = self.node.get_code_language();
        if codelanguage in lang_manager.get_language_ids():
          language = lang_manager.get_language(codelanguage)
        else:
          language = lang_manager.get_language("python")
        buf.set_language(language)
        buf.set_highlight_syntax(True)
      textview = gtksourceview2.View(buffer=buf)
      textview.set_auto_indent(True)
      textview.set_insert_spaces_instead_of_tabs(True)
      textview.set_tab_width(2)
      if self.node is not None and self.node.is_code():
        textview.set_show_line_numbers(True)
        font_desc = pango.FontDescription("monospace")
        if font_desc:
          textview.modify_font(font_desc)
    except ImportError:
      textview = gtk.TextView()

    textview.set_pixels_above_lines(2)
    textview.set_pixels_below_lines(2)
    textview.set_wrap_mode(gtk.WRAP_WORD)
    textview.connect("focus-in-event", self.entry_focus_in)

    scrolledWindow.add(textview)
    textview.show()
    return textview

  def set_data_empty(self):
    """
    Empty the data frame.
    """

    if self.frame.child is not None:
      if isinstance(self.data, gtk.TextView):
        self.data.handler_block_by_func(self.entry_focus_in)
      elif isinstance(self.data, gtk.ComboBox):
        self.data.handler_block_by_func(self.combo_focus_child)

      self.frame.remove(self.frame.child)

    self.interacted = False

    return

  def set_data_fixed(self):
    """
    Create a non-editable text view to show help or fixed data.
    """

    self.set_data_empty()

    self.data = self.add_text_view()

    self.data.get_buffer().create_tag("tag")
    text_tag = self.data.get_buffer().get_tag_table().lookup("tag")

    self.data.set_cursor_visible(False)
    self.data.set_editable(False)
    self.buttons.hide()
    text_tag.set_property("foreground", "grey")

    if self.node is None:
      self.data.get_buffer().set_text("")
    elif not self.node.active:
      self.data.get_buffer().set_text("Inactive node")
    elif self.node.datatype is None:
      self.data.get_buffer().set_text("No data")
    elif self.node.is_tensor(self.geometry_dim_tree):
      self.data.get_buffer().set_text("Dimension not set")
    else: # self.node.datatype == "fixed":
      self.data.get_buffer().set_text(self.node.data)

    buffer_bounds = self.data.get_buffer().get_bounds()
    self.data.get_buffer().apply_tag(text_tag, buffer_bounds[0], buffer_bounds[1])

    return

  def set_data_entry(self):
    """
    Create a text view for data entry in the data frame.
    """

    self.set_data_empty()

    self.data = self.add_text_view()

    self.data.get_buffer().create_tag("tag")
    text_tag = self.data.get_buffer().get_tag_table().lookup("tag")

    self.data.set_cursor_visible(True)
    self.data.set_editable(True)
    self.buttons.show()

    if self.node.data is None:
      self.data.get_buffer().set_text(datatype.print_type(self.node.datatype))
      text_tag.set_property("foreground", "blue")
    else:
      self.data.get_buffer().set_text(self.node.data)

    buffer_bounds = self.data.get_buffer().get_bounds()
    self.data.get_buffer().apply_tag(text_tag, buffer_bounds[0], buffer_bounds[1])

    return

  def set_data_tensor(self):
    """
    Create a table container packed with appropriate widgets for tensor data entry
    in the node data frame.
    """

    self.set_data_empty()

    scrolledWindow = self.add_scrolled_window()

    dim1, dim2 = self.node.tensor_shape(self.geometry_dim_tree)
    self.data = gtk.Table(dim1, dim2)
    scrolledWindow.add_with_viewport(self.data)
    scrolledWindow.child.set_property("shadow-type", gtk.SHADOW_NONE)

    self.set_child_packing(self.frame, True, True, 0, gtk.PACK_START)

    self.show_all()
    self.buttons.show()

    is_symmetric = self.node.is_symmetric_tensor(self.geometry_dim_tree)
    for i in range(dim1):
      for j in range(dim2):
        iindex = dim1 - i - 1
        jindex = dim2 - j - 1

        entry = gtk.Entry()
        self.data.attach(entry, jindex, jindex + 1, iindex, iindex + 1)

        if not is_symmetric or i >= j:
          entry.show()
          entry.connect("focus-in-event", self.tensor_element_focus_in, jindex, iindex)

          if self.node.data is None:
            entry.set_text(datatype.print_type(self.node.datatype.datatype))
            entry.modify_text(gtk.STATE_NORMAL, gtk.gdk.color_parse("blue"))
          else:
            entry.set_text(self.node.data.split(" ")[jindex + iindex * dim2])

    self.interacted = [False for i in range(dim1 * dim2)]

    return

  def set_data_combo(self):
    """
    Create a combo box for node data selection in the node data frame. Add an
    entry if required.
    """

    self.set_data_empty()

    if isinstance(self.node.datatype[0], tuple):
      self.data = gtk.combo_box_entry_new_text()
    else:
      self.data = gtk.combo_box_new_text()

    self.frame.add(self.data)
    self.data.show()

    self.data.connect("set-focus-child", self.combo_focus_child)

    self.set_child_packing(self.frame, False, False, 0, gtk.PACK_START)

    if isinstance(self.node.datatype[0], tuple):
      self.buttons.show()
    else:
      self.buttons.hide()

    if self.node.data is None:
      if isinstance(self.node.datatype[0], tuple):
        self.data.child.set_text("Select " + datatype.print_type(self.node.datatype[1]) + "...")
      else:
        self.data.append_text("Select...")
        self.data.set_active(0)
      self.data.child.modify_text(gtk.STATE_NORMAL, gtk.gdk.color_parse("blue"))
      self.data.child.modify_text(gtk.STATE_PRELIGHT, gtk.gdk.color_parse("blue"))

    if isinstance(self.node.datatype[0], tuple):
      options = self.node.datatype[0]
    else:
      options = self.node.datatype

    for (i, opt) in enumerate(options):
      self.data.append_text(opt)
      if self.node.data == opt:
        self.data.set_active(i)

    if (isinstance(self.node.datatype[0], tuple)
       and self.node.data is not None
       and self.node.data not in self.node.datatype[0]):
      self.data.child.set_text(self.node.data)

    self.data.connect("changed", self.combo_changed)

    return

  def data_entry_store(self):
    """
    Attempt to store data read from a textview packed in the data frame.
    """

    new_data = self.data.get_buffer().get_text(*self.data.get_buffer().get_bounds())

    if new_data == "":
      return True

    if self.node.data is None and not self.interacted:
      return True
    else:
      value_check = self.node.validity_check(self.node.datatype, new_data)
      if value_check is None:
        dialogs.error(None, "Invalid value entered")
        return False
      elif value_check != self.node.data:
        self.node.set_data(value_check)
        if (isinstance(self.node, mixedtree.MixedTree)
           and "shape" in self.node.child.attrs.keys()
           and self.node.child.attrs["shape"][0] is int
           and isinstance(self.node.datatype, plist.List)
           and self.node.datatype.cardinality == "+"):
          self.node.child.set_attr("shape", str(len(value_check.split(" "))))

        self.emit("on-store")
        self.interacted = False

    return True

  def data_tensor_store(self):
    """
    Attempt to store data read from tensor data entry widgets packed in the
    data frame.
    """

    dim1, dim2 = self.node.tensor_shape(self.geometry_dim_tree)
    is_symmetric = self.node.is_symmetric_tensor(self.geometry_dim_tree)

    if True not in self.interacted:
      return True

    entry_values = []
    for i in range(dim1):
      for j in range(dim2):
        if is_symmetric and i > j:
          entry_values.append(self.data.get_children()[i + j * dim1].get_text())
        else:
          entry_values.append(self.data.get_children()[j + i * dim2].get_text())

    changed = False
    for i in range(dim1):
      for j in range(dim2):
        if (self.interacted[j + i * dim2]
           and entry_values[j + i * dim2] != ""
           and (self.node.data is None
                or self.node.data.split(" ")[j + i * dim2] != entry_values[j + i * dim2])):
          changed = True
    if not changed:
      return True
    elif (self.node.data is None and False in self.interacted) or "" in entry_values:
      dialogs.error(None, "Invalid value entered")
      return False

    new_data = ""
    for i in range(dim1):
      for j in range(dim2):
        new_data += " " + entry_values[j + i * dim2]

    value_check = self.node.validity_check(self.node.datatype, new_data)
    if value_check is None:
      return False
    elif not value_check == self.node.data:
      self.node.set_data(value_check)

      dim1, dim2 = self.node.tensor_shape(self.geometry_dim_tree)
      if int(self.node.child.attrs["rank"][1]) == 1:
        self.node.child.set_attr("shape", str(dim1))
      else:
        self.node.child.set_attr("shape", str(dim1) + " " + str(dim2))

      self.emit("on-store")
      self.interacted = [False for i in range(dim1 * dim2)]

    return True

  def data_combo_store(self):
    """
    Attempt to store data read from a combo box entry packed in the node data.
    """

    if not isinstance(self.node.datatype[0], tuple):
      return True

    new_data = self.data.get_text()

    if self.node.data is None and not self.interacted:
      return True
    elif not new_data in self.node.datatype[0]:
      new_data = self.node.validity_check(self.node.datatype[1], new_data)
      if new_data is None:
        return False

    if not new_data == self.node.data:
      self.node.set_data(new_data)
      self.emit("on-store")
      self.interacted = False

    return True

  def entry_focus_in(self, widget, event):
    """
    Called when a text view data entry widget gains focus. Used to delete the
    printable_type placeholder.
    """

    if (self.node is not None
       and self.node.datatype is not None
       and not self.node.is_tensor(self.geometry_dim_tree)
       and self.node.data is None
       and not self.interacted):
      self.data.get_buffer().set_text("")

    self.interacted = True

    return

  def tensor_element_focus_in(self, widget, event, row, col):
    """
    Called when a tensor data entry widget gains focus. Used to delete the
    printable_type placeholder.
    """

    dim1, dim2 = self.node.tensor_shape(self.geometry_dim_tree)
    if not self.interacted[col + row * dim2]:
      self.interacted[col + row * dim2] = True
      if self.node.is_symmetric_tensor(self.geometry_dim_tree):
        self.interacted[row + col * dim1] = True
      if self.node.data is None:
        widget.set_text("")
        widget.modify_text(gtk.STATE_NORMAL, gtk.gdk.color_parse("black"))

    return

  def combo_focus_child(self, container, widget):
    """
    Called when a data selection widget gains focus. Used to delete the select
    placeholder.
    """
    if not self.interacted:
      self.interacted = True
      if self.node.data is None:
        self.data.handler_block_by_func(self.combo_changed)
        if isinstance(self.node.datatype[0], tuple):
          self.data.child.set_text("")
        else:
          self.data.remove_text(0)

        self.data.child.modify_text(gtk.STATE_NORMAL, gtk.gdk.color_parse("black"))
        self.data.child.modify_text(gtk.STATE_PRELIGHT, gtk.gdk.color_parse("black"))
        self.data.handler_unblock_by_func(self.combo_changed)

    return

  def combo_changed(self, combo_box):
    """
    Called when a data combo box element is selected. Updates data in the
    treestore.
    """

    if not isinstance(self.node.datatype[0], tuple):
      text = self.data.get_active_text()
      if text is None:
        return

      self.node.set_data(text)
      self.emit("on-store")
      self.interacted = False
    return

gobject.type_register(DataWidget)
