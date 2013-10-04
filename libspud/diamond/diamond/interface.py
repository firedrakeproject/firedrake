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
import re
import time
import sys
import tempfile
import cStringIO as StringIO

import pango
import gobject
import gtk
import gtk.glade

import choice
import config
import datatype
import debug
import dialogs
import mixedtree
import plist
import plugins
import schema
import scherror
import tree

import StringIO
import TextBufferMarkup

import attributewidget
import commentwidget
import descriptionwidget
import databuttonswidget
import datawidget
import diffview
import sliceview
import useview

from lxml import etree

try:
  gtk.Tooltip()
except:
  debug.deprint("Interface warning: Unable to use GTK tooltips")

"""
Here are some notes about the code:

Important fields:
  file_path: the directory containing the current file (the working directory or directory of last opened / saved file if no file is open)
  filename: output filename of the current file
  data_paths: paths (from the root node) to important Diamond data (e.g. geometry dimension)
  geometry_dim_tree: MixedTree, with parent equal to the geometry dimension tree and child equal to the geometry dimension data subtree
  gladefile: input Glade file
  gui: GUI GladeXML object
  logofile: the GUI logo file
  main_window: GUI toplevel window
  node_attrs: RHS attributes entry widget
  description: RHS description widget
  data = RHS data widget
  comment: RHS comment entry widget
  node_data: RHS data entry widget
  node_data_buttons_hbox: container for "Revert Data" and "Store Data" buttons
  node_data_interacted: used to determine if a node data widget has been interacted with without data being stored
  node_data_frame: frame containing data entry widgets
  options_tree_select_func_enabled: boolean, true if the options tree select function is enabled (used to overcome a nasty clash with the treeview clicked signal) - re-enabled on next options_tree_select_func call
  selected_node: a tree.Tree or MixedTree containing data to be displayed on the RHS
  selected_iter: last iter set by on_select_row
  s: current schema
  saved: boolean, false if the current file has been edited
  schemafile: the current RNG schema file
  schemafile_path: the directory containing the current schema file (the working directory if no schema is open)
  signals: dictionary containing signal handlers for signals set up in the Glade file
  statusbar: GUI status bar
  tree: LHS tree root
  treestore: the LHS tree model
  treeview: the LHS tree widget

Important routines:
  cellcombo_changed: called when a choice is selected on the left-hand pane
  init_treemodel: set up the treemodel and treeview
  on_treeview_clicked: when a row is clicked, process the consequences (e.g. activate inactive instance)
  set_treestore: stuff the treestore with a given tree.Tree
  on_find_find_button & search_treestore: the find functionality
  on_select_row: when a row is selected, update the options frame
  update_options_frame: paint the right-hand side

If there are bugs in reading in, see schema.read.
If there are bugs in writing out, see tree.write.
"""

class Diamond:
  def __init__(self, gladefile, schemafile = None, schematron_file = None, logofile = None, input_filename = None,
      dim_path = "/geometry/dimension", suffix=None):
    self.gladefile = gladefile
    self.gui = gtk.glade.XML(self.gladefile)

    self.statusbar = DiamondStatusBar(self.gui.get_widget("statusBar"))
    self.find      = DiamondFindDialog(self, gladefile)
    self.popup = self.gui.get_widget("popupmenu")

    self.add_custom_widgets()

    self.plugin_buttonbox = self.gui.get_widget("plugin_buttonbox")
    self.plugin_buttonbox.set_layout(gtk.BUTTONBOX_START)
    self.plugin_buttonbox.show()
    self.plugin_buttons = []

    self.scherror  = scherror.DiamondSchemaError(self, gladefile, schemafile, schematron_file)

    signals     =  {"on_new": self.on_new,
                    "on_quit": self.on_quit,
                    "on_open": self.on_open,
                    "on_open_schema": self.on_open_schema,
                    "on_save": self.on_save,
                    "on_save_as": self.on_save_as,
                    "on_validate": self.scherror.on_validate,
                    "on_validate_schematron": self.scherror.on_validate_schematron,
                    "on_expand_all": self.on_expand_all,
                    "on_collapse_all": self.on_collapse_all,
                    "on_find": self.find.on_find,
                    "on_go_to_node": self.on_go_to_node,
                    "on_console": self.on_console,
                    "on_display_properties_toggled": self.on_display_properties_toggled,
                    "on_about": self.on_about,
                    "on_copy_spud_path": self.on_copy_spud_path,
                    "on_copy": self.on_copy,
                    "on_paste": self.on_paste,
                    "on_slice": self.on_slice,
                    "on_diff": self.on_diff,
                    "on_diffsave": self.on_diffsave,
                    "on_finduseage": self.on_finduseage,
                    "on_group": self.on_group,
                    "on_ungroup": self.on_ungroup}

    self.gui.signal_autoconnect(signals)

    self.main_window = self.gui.get_widget("mainWindow")
    self.main_window.connect("delete_event", self.on_delete)

    self.logofile = logofile
    if self.logofile is not None:
      gtk.window_set_default_icon_from_file(self.logofile)

    self.init_treemodel()

    self.data_paths = {}
    self.data_paths["dim"] = dim_path

    self.suffix = suffix

    self.selected_node = None
    self.selected_iter = None
    self.update_options_frame()

    self.file_path = os.getcwd()
    self.schemafile_path = os.getcwd()
    self.filename = None
    self.schemafile = None
    self.init_datatree()
    self.set_saved(True)
    self.open_file(schemafile = None, filename = None)

    self.main_window.show()

    if schemafile is not None:
      self.open_file(schemafile = schemafile, filename = input_filename)

    # Hide specific menu items
    menu = self.gui.get_widget("menu")

    # Disable Find
    menu.get_children()[1].get_submenu().get_children()[0].set_property("visible", False)

    if schematron_file is None:
      # Disable Validate Schematron
      menu.get_children()[3].get_submenu().get_children()[1].set_property("sensitive", False)

    return

  def program_exists(self, name):
    ret = os.system("which %s > /dev/null" % name)
    return ret == 0

  ### MENU ###

  def update_title(self):
    """
    Update the Diamond title based on the save status of the currently open file.
    """

    title = "Diamond: "
    if not self.saved:
      title += "*"
    if self.filename is None:
      title += "(Unsaved)"
    else:
      title += os.path.basename(self.filename)
      if len(os.path.dirname(self.filename)) > 0:
        title += " (%s)" % os.path.dirname(self.filename)

    self.main_window.set_title(title)

    return

  def set_saved(self, saved, filename = ""):
    """
    Change the save status of the current file.
    """

    self.saved = saved
    if filename != "":
      self.filename = filename
      if filename is not None:
        self.file_path = os.path.dirname(filename) + os.path.sep
    self.update_title()

    return

  def close_schema(self):
    if self.schemafile is None:
      return

    # clear the schema.
    self.s = None
    self.schemafile = None
    self.schemafile_path = None
    self.scherror.schema_file = None

    return

  def load_schema(self, schemafile):
    # so, if the schemafile has already been opened, then ..
    if schemafile == self.schemafile:
      self.statusbar.set_statusbar('Schema ' + schemafile + ' already loaded')
      return

    # if we aren't using a http schema, and we're passed a relative filename, we
    # need to absolut-ify it.
    if 'http' not in schemafile:
      schemafile = os.path.abspath(schemafile)

    self.statusbar.set_statusbar('Loading schema from ' + schemafile)

    # now, let's try and read the schema.
    try:
      s_read = schema.Schema(schemafile)
      self.s = s_read
      self.statusbar.set_statusbar('Loaded schema from ' + schemafile)
    except:
      dialogs.error_tb(self.main_window, "Unable to open schema file \"" + schemafile + "\"")
      self.statusbar.clear_statusbar()
      return

    self.schemafile = schemafile
    self.schemafile_path = os.path.dirname(schemafile) + os.path.sep
    self.scherror.schema_file = schemafile

    self.remove_children(None)
    self.init_datatree()

    return

  def close_file(self):
    self.remove_children(None)
    self.init_datatree()

    self.filename = None

    return

  def display_validation_errors(self, errors):
    # Extract and display validation errors
    saved = True
    lost_eles, added_eles, lost_attrs, added_attrs = errors
    if len(lost_eles) > 0 or len(added_eles) > 0 or len(lost_attrs) > 0 or len(added_attrs) > 0:
      saved = False
      msg = ""
      if len(lost_eles) > 0:
        msg += "Warning: lost xml elements:\n"
        for ele in lost_eles:
          msg += ele + "\n"
      if len(added_eles) > 0:
        msg += "Warning: added xml elements:\n"
        for ele in added_eles:
          msg += ele + "\n"
      if len(lost_attrs) > 0:
        msg += "Warning: lost xml attributes:\n"
        for ele in lost_attrs:
          msg += ele + "\n"
      if len(added_attrs) > 0:
        msg += "Warning: added xml attributes:\n"
        for ele in added_attrs:
          msg += ele + "\n"

      dialogs.long_message(self.main_window, msg)
    return saved

  def load_file(self, filename):
    # if we have a relative path, make it absolute
    filename = os.path.abspath(filename)

    try:
      os.stat(filename)
    except OSError:
      self.filename = filename
      self.set_saved(False)

      self.remove_children(None)
      self.init_datatree()

      return

    try:
      tree_read = self.s.read(filename)

      if tree_read is None:
        dialogs.error_tb(self.main_window, "Unable to open file \"" + filename + "\"")
        return

      saved = self.display_validation_errors(self.s.read_errors())

      self.tree = tree_read
      self.filename = filename
    except:
      dialogs.error_tb(self.main_window, "Unable to open file \"" + filename + "\"")
      return

    self.set_saved(saved, filename)

    return

  def open_file(self, schemafile = "", filename = ""):
    """
    Handle opening or clearing of the current file and / or schema.
    """

    self.find.on_find_close_button()
    if schemafile is None:
      self.close_schema()
    elif schemafile != "":
      self.load_schema(schemafile)
    if filename is None:
      self.close_file()
    elif filename != "":
      self.load_file(filename)

    self.treeview.freeze_child_notify()
    self.treeview.set_model(None)
    self.signals = {}
    self.set_treestore(None, [self.tree], True)
    self.treeview.set_model(self.treestore)
    self.treeview.thaw_child_notify()

    self.set_geometry_dim_tree()

    self.treeview.grab_focus()
    self.treeview.get_selection().select_path(0)

    self.selected_node = None
    self.update_options_frame()

    self.scherror.destroy_error_list()

    return

  def save_continue(self):

    if not self.saved:
      prompt_response = dialogs.prompt(self.main_window,
        "Unsaved data. Do you want to save the current document before continuing?", gtk.MESSAGE_WARNING, True)

      if prompt_response == gtk.RESPONSE_YES:
        if self.filename is None:
          return self.on_save_as()
        else:
          return self.on_save()
      elif prompt_response == gtk.RESPONSE_CANCEL:
        return False

    return True

  def on_new(self, widget=None):
    """
    Called when new is clicked. Clear the treestore and reset the datatree.
    """

    if not self.save_continue():
      return

    self.open_file(filename = None)
    self.filename = None

    return

  def on_open(self, widget=None):
    """
    Called when open is clicked. Open a user supplied file.
    """

    if not self.save_continue():
      return

    filter_names_and_patterns = {}
    if self.suffix is None:
      for xmlname in config.schemata:
        filter_names_and_patterns[config.schemata[xmlname][0]] = "*." + xmlname
    elif self.suffix in config.schemata.keys():
      filter_names_and_patterns[config.schemata[self.suffix][0]] = "*." + self.suffix
    else:
      filter_names_and_patterns[self.suffix] = "*." + self.suffix

    filename = dialogs.get_filename(title = "Open XML file", action = gtk.FILE_CHOOSER_ACTION_OPEN, filter_names_and_patterns = filter_names_and_patterns, folder_uri = self.file_path)

    if filename is not None:
      self.open_file(filename = filename)

    return

  def on_open_schema(self, widget=None):
    """
    Called when open schema is clicked. Clear the treestore and reset the schema.
    """

    if not self.save_continue():
      return

    filename = dialogs.get_filename(title = "Open RELAX NG schema", action = gtk.FILE_CHOOSER_ACTION_OPEN, filter_names_and_patterns = {"RNG files":"*.rng"}, folder_uri = self.schemafile_path)
    if filename is not None:
      self.open_file(schemafile = filename)

    return

  def on_save(self, widget=None):
    """
    Write out to XML. If we don't already have a filename, open a dialog to get
    one.
    """

    self.data.store()

    if self.filename is None:
      return self.on_save_as(widget)
    else:
      self.statusbar.set_statusbar("Saving ...")
      self.main_window.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.WATCH))
      try:
        self.tree.write(self.filename)
      except:
        dialogs.error_tb(self.main_window, "Saving to \"" + self.filename + "\" failed")
        self.statusbar.clear_statusbar()
        self.main_window.window.set_cursor(None)
        return False

      self.set_saved(True)

      self.statusbar.clear_statusbar()
      self.main_window.window.set_cursor(None)
      return True

    return False

  def on_save_as(self, widget=None):
    """
    Write out the XML to a file.
    """

    if self.schemafile is None:
      dialogs.error(self.main_window, "No schema file open")
      return False

    filter_names_and_patterns = {}
    if self.suffix is None:
      for xmlname in config.schemata:
        filter_names_and_patterns[config.schemata[xmlname][0]] = "*." + xmlname
    elif self.suffix in config.schemata.keys():
      filter_names_and_patterns[config.schemata[self.suffix][0]] = "*." + self.suffix
    else:
      filter_names_and_patterns[self.suffix] = "*." + self.suffix

    filename = dialogs.get_filename(title = "Save XML file", action = gtk.FILE_CHOOSER_ACTION_SAVE, filter_names_and_patterns = filter_names_and_patterns, folder_uri = self.file_path)

    if filename is not None:
      # Check that the selected file has a file extension. If not, add a .xml extension.
      if len(filename.split(".")) <= 1:
        filename += ".xml"

      # Save the file
      self.statusbar.set_statusbar("Saving ...")
      self.main_window.window.set_cursor(gtk.gdk.Cursor(gtk.gdk.WATCH))
      self.tree.write(filename)
      self.set_saved(True, filename)
      self.statusbar.clear_statusbar()
      self.main_window.window.set_cursor(None)
      return True

    return False

  def on_delete(self, widget, event):
    """
    Called when the main window is deleted. Return "True" to prevent the deletion
    of the main window (deletion is handled by "on_quit").
    """

    self.on_quit(widget, event)

    return True

  def on_quit(self, widget, event = None):
    """
    Quit the program. Prompt the user to save data if the current file has been
    changed.
    """

    if not self.save_continue():
      return

    self.destroy()

    return

  def destroy(self):
    """
    End the program.
    """

    try:
      gtk.main_quit()
    except:
      debug.dprint("Failed to quit - already quit?")

    return

  def on_display_properties_toggled(self, widget=None):
    optionsFrame = self.gui.get_widget("optionsFrame")
    optionsFrame.set_property("visible", not optionsFrame.get_property("visible"))
    return

  def on_go_to_node(self, widget=None):
   """
   Go to a node, identified by an XPath
   """

   dialog = dialogs.GoToDialog(self)
   spudpath = dialog.run()

   return

  def on_expand_all(self, widget=None):
    """
    Show the whole tree.
    """

    self.treeview.expand_all()

    return

  def on_collapse_all(self, widget=None):
    """
    Collapse the whole tree.
    """

    self.treeview.collapse_all()

    return

  def on_console(self, widget = None):
    """
    Launch a python console
    """

    # Construct the dictionary of locals that will be used by the interpreter
    locals = {}
    locals["interface"] = globals()
    locals["diamond_gui"] = self

    dialogs.console(self.main_window, locals)

    return

  def on_about(self, widget=None):
    """
    Tell the user how fecking great we are.
    """

    about = gtk.AboutDialog()
    about.set_name("Diamond")
    about.set_copyright("GPLv3")
    about.set_comments("A RELAX-NG-aware XML editor")
    about.set_authors(["Patrick E. Farrell", "James R. Maddison", "Matthew T. Whitworth", "Fraser J. Waters"])
    about.set_license("Diamond is free software: you can redistribute it and/or modify\n"+
                      "it under the terms of the GNU General Public License as published by\n"+
                      "the Free Software Foundation, either version 3 of the License, or\n"+
                      "(at your option) any later version.\n"+
                      "\n"+
                      "Diamond is distributed in the hope that it will be useful,\n"+
                      "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"+
                      "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"+
                      "GNU General Public License for more details.\n"+
                      "You should have received a copy of the GNU General Public License\n"+
                      "along with Diamond.  If not, see http://www.gnu.org/licenses/.")

    if self.logofile is not None:
      logo = gtk.gdk.pixbuf_new_from_file(self.logofile)
      about.set_logo(logo)

    try:
      image = about.get_children()[0].get_children()[0].get_children()[0]
      image.set_tooltip_text("Diamond: it's clearer than GEM")
    except:
      pass

    about.show()

    return


  def on_copy_spud_path(self, widget=None):
    path = self.get_selected_row(self.treeview.get_selection())
    if path is None:
      debug.deprint("No selection.")
      return
    iter = self.treestore.get_iter(path)
    active_tree = self.treestore.get_value(iter, 1)
    name = self.get_spudpath(active_tree)
    clipboard = gtk.clipboard_get()
    clipboard.set_text(name)
    clipboard.store()

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
    widget = self._get_focus_widget(self.main_window)

    if widget is None or widget is self.treeview:
      return False

    if gobject.signal_lookup(signal + "-clipboard", widget):
      widget.emit(signal + "-clipboard")
      return True
    else:
      return False

  def on_copy(self, widget=None):
    if self._handle_clipboard(widget, "copy"):
      return

    if isinstance(self.selected_node, mixedtree.MixedTree):
      node = self.selected_node.parent
    else:
      node = self.selected_node

    if node != None and node.active:
      ios = StringIO.StringIO()
      node.write(ios)

      clipboard = gtk.clipboard_get()
      clipboard.set_text(ios.getvalue())
      clipboard.store()

      ios.close()
    return

  def on_paste(self, widget=None):
    if self._handle_clipboard(widget, "paste"):
      return

    clipboard = gtk.clipboard_get()
    ios = StringIO.StringIO(clipboard.wait_for_text())

    if self.selected_iter is not None:
      node = self.treestore.get_value(self.selected_iter, 0)

    if node != None:

      expand = not node.active
      if expand:
        self.expand_tree(self.selected_iter)

      newnode = self.s.read(ios, node)

      if newnode is None:
        if expand:
          self.collapse_tree(self.selected_iter, False)
        self.statusbar.set_statusbar("Trying to paste invalid XML.")
        return

      if node.parent is not None:
        newnode.set_parent(node.parent)
        children = node.parent.get_children()
        children.insert(children.index(node), newnode)
        children.remove(node)

      self.treeview.freeze_child_notify()
      iter = self.set_treestore(self.selected_iter, [newnode], True, True)
      newnode.recompute_validity()
      self.treeview.thaw_child_notify()

      self.treeview.get_selection().select_iter(iter)

      self.display_validation_errors(self.s.read_errors())

      self.set_saved(False)

    return

  def __diff(self, path):
    def run_diff(self, path):
      start = time.clock()
      diffview.DiffView(path, self.tree)
      seconds = time.clock() - start
      self.statusbar.set_statusbar("Diff calculated (took " + str(seconds) + " seconds)")
      return False

    if path and os.path.isfile(path):
      filename = path
    else:
      dialog = gtk.FileChooserDialog(
                                     title = "Diff against",
                                     action = gtk.FILE_CHOOSER_ACTION_OPEN,
                                     buttons = (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL, gtk.STOCK_OPEN, gtk.RESPONSE_OK))

      if path:
        dialog.set_current_folder(path)

      response = dialog.run()
      if response != gtk.RESPONSE_OK:
        dialog.destroy()
        return

      filename = dialog.get_filename()
      dialog.destroy()

    self.statusbar.set_statusbar("Calculating diff... (this may take a while)")
    gobject.idle_add(run_diff, self, filename)

  def on_diff(self, widget = None, path = None):
    if path is None:
      path = os.path.dirname(self.filename) if self.filename else None
    self.__diff(path)

  def on_diffsave(self, widget = None):
    if self.filename:
      self.__diff(self.filename)
    else:
      dialogs.error(self.main_window, "No save to diff against.")

  def on_finduseage(self, widget = None):
    useview.UseView(self.s, self.suffix)

  def on_slice(self, widget = None):
    if not self.selected_node.is_sliceable():
      self.statusbar.set_statusbar("Cannot slice on this element.")
      return

    window = sliceview.SliceView(self.main_window)
    window.geometry_dim_tree = self.geometry_dim_tree
    window.update(self.selected_node, self.tree)
    window.connect("destroy", self._slice_destroy)
    return

  def _slice_destroy(self, widget):
    self.on_select_row()


  groupmode = False

  def on_group(self, widget = None):
    """
    Clears the treeview and then fills it with nodes
    with the same type as the selected node.
    """

    if self.selected_node == self.tree or not self.selected_iter:
      self.statusbar.set_statusbar("Cannot group on this element.")
      return #Group on the entire tree... ie don't group or nothing selected

    self.gui.get_widget("menuitemUngroup").show()
    self.gui.get_widget("popupmenuitemUngroup").show()

    self.groupmode = True
    node, tree = self.treestore.get(self.selected_iter, 0, 1)

    self.treeview.freeze_child_notify()
    self.treeview.set_model(None)

    def get_nodes(node, tree):
      nodes = []

      if isinstance(tree, choice.Choice):
        child = tree.get_current_tree()
        if child.name == node.name:
          nodes.append(tree)
        nodes += get_nodes(node, child)
      else:
        for child in tree.get_children():
          if child.name == node.name:
            nodes.append(child)
          nodes += get_nodes(node, child)

      return nodes

    self.set_treestore(None, get_nodes(tree, self.tree), True)

    self.treeview.set_model(self.treestore)
    self.treeview.thaw_child_notify()

    path = self.get_treestore_path_from_node(node)
    self.treeview.get_selection().select_path(path)

    return

  def on_ungroup(self, widget = None):
    """
    Restores the treeview to normal.
    """

    self.gui.get_widget("menuitemUngroup").hide()
    self.gui.get_widget("popupmenuitemUngroup").hide()

    self.groupmode = False
    node = self.treestore.get_value(self.selected_iter, 0)

    self.treeview.freeze_child_notify()
    self.treeview.set_model(None)

    self.set_treestore(None, [self.tree], True)

    self.treeview.set_model(self.treestore)
    self.treeview.thaw_child_notify()

    path = self.get_treestore_path_from_node(node)
    self.treeview.expand_to_path(path)
    self.treeview.scroll_to_cell(path)
    self.treeview.get_selection().select_path(path)

    return

  ## LHS ###

  def init_datatree(self):
    """
    Add the root node of the XML tree to the treestore, and its children.
    """

    if self.schemafile is None:
      self.set_treestore(None, [])
      self.tree = None
    else:
      l = self.s.valid_children(":start")

      self.tree = l[0]
      self.signals = {}
      self.set_treestore(None, l)

    root_iter = self.treestore.get_iter_first()
    self.treeview.freeze_child_notify()
    self.treeview.set_model(None)
    self.expand_treestore(root_iter)
    self.treeview.set_model(self.treestore)
    self.treeview.thaw_child_notify()

    return

  def init_treemodel(self):
    """
    Set up the treestore and treeview.
    """

    self.treeview = optionsTree = self.gui.get_widget("optionsTree")
    self.treeview.connect("key_press_event", self.on_treeview_key_press)
    self.treeview.connect("button_press_event", self.on_treeview_button_press)
    self.treeview.connect("row-activated", self.on_activate_row)
    self.treeview.connect("popup_menu", self.on_treeview_popup)

    self.treeview.set_property("rules-hint", True)

    self.treeview.get_selection().set_mode(gtk.SELECTION_SINGLE)
    self.treeview.get_selection().connect("changed", self.on_select_row)
    self.treeview.get_selection().set_select_function(self.options_tree_select_func)
    self.options_tree_select_func_enabled = True

    # Node column
    column = gtk.TreeViewColumn("Node")
    column.set_property("expand", True)
    column.set_resizable(True)

    self.cellcombo = cellCombo = gtk.CellRendererCombo()
    cellCombo.set_property("text-column", 0)
    cellCombo.set_property("editable", True)
    cellCombo.set_property("has-entry", False)
    cellCombo.connect("changed", self.cellcombo_changed)
    column.pack_start(cellCombo)
    column.set_cell_data_func(cellCombo, self.set_combobox_liststore)

    self.choicecell = choiceCell = gtk.CellRendererPixbuf()
    column.pack_end(choiceCell, expand=False)
    column.set_cell_data_func(choiceCell, self.set_cellpicture_choice)

    optionsTree.append_column(column)

    self.imgcell = cellPicture = gtk.CellRendererPixbuf()
    self.imgcolumn = imgcolumn = gtk.TreeViewColumn("", cellPicture)
    imgcolumn.set_property("expand", False)
    imgcolumn.set_property("fixed-width", 20)
    imgcolumn.set_property("sizing", gtk.TREE_VIEW_COLUMN_FIXED)
    imgcolumn.set_cell_data_func(cellPicture, self.set_cellpicture_cardinality)
    optionsTree.append_column(imgcolumn)

    # 0: pointer to node in self.tree -- a choice or a tree
    # 1: pointer to currently active tree
    # 2: gtk.ListStore containing the display names of possible choices

    self.treestore = gtk.TreeStore(gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT, gobject.TYPE_PYOBJECT)
    self.treeview.set_model(self.treestore)
    self.treeview.set_enable_search(False)

    return

  def create_liststore(self, choice_or_tree):
    """
    Given a list of possible choices, create the liststore for the
    gtk.CellRendererCombo that contains the names of possible choices.
    """

    liststore = gtk.ListStore(str, gobject.TYPE_PYOBJECT)

    for t in choice_or_tree.get_choices():
      liststore.append([str(t), t])

    return liststore

  def set_treestore(self, iter=None, new_tree=[], recurse=False, replace=False):
    """
    Given a list of children of a node in a treestore, stuff them in the treestore.
    """

    if replace:
      replacediter = iter
      iter = self.treestore.iter_parent(replacediter)
    else:
      self.remove_children(iter)

    for t in new_tree:
      if t is not None and t in self.signals:
        t.disconnect(self.signals[t][0])
        t.disconnect(self.signals[t][1])

      if isinstance(t, tree.Tree):
        if t.is_hidden():
          attrid = t.connect("on-set-attr", self.on_set_attr, self.treestore.get_path(iter))
          dataid = t.connect("on-set-data", self.on_set_data, self.treestore.get_path(iter))
          self.signals[t] = (attrid, dataid)
          continue

        liststore = self.create_liststore(t)

        if replace:
          child_iter = self.treestore.insert_before(iter, replacediter, [t, t, liststore])
        else:
          child_iter = self.treestore.append(iter, [t, t, liststore])

        attrid = t.connect("on-set-attr", self.on_set_attr, self.treestore.get_path(child_iter))
        dataid = t.connect("on-set-data", self.on_set_data, self.treestore.get_path(child_iter))
        self.signals[t] = (attrid, dataid)

        if recurse and t.active: self.set_treestore(child_iter, t.children, recurse)

      elif isinstance(t, choice.Choice):
        liststore = self.create_liststore(t)
        ts_choice = t.get_current_tree()
        if ts_choice.is_hidden():
          attrid = t.connect("on-set-attr", self.on_set_attr, self.treestore.get_path(iter))
          dataid = t.connect("on-set-data", self.on_set_data, self.treestore.get_path(iter))
          self.signals[t] = (attrid, dataid)
          continue

        if replace:
          child_iter = self.treestore.insert_before(iter, replacediter, [t, ts_choice, liststore])
        else:
          child_iter = self.treestore.append(iter, [t, ts_choice, liststore])

        attrid = t.connect("on-set-attr", self.on_set_attr, self.treestore.get_path(child_iter))
        dataid = t.connect("on-set-data", self.on_set_data, self.treestore.get_path(child_iter))
        self.signals[t] = (attrid, dataid)

        if recurse and t.active: self.set_treestore(child_iter, ts_choice.children, recurse)

    if replace:
      self.treestore.remove(replacediter)
      return child_iter

    return iter

  def expand_choice_or_tree(self, choice_or_tree):
    """
    Query the schema for what valid children can live under this node, and add
    them to the choice or tree. This recurses.
    """

    if isinstance(choice_or_tree, choice.Choice):
      for opt in choice_or_tree.choices():
        self.expand_choice_or_tree(opt)
    else:
      l = self.s.valid_children(choice_or_tree.schemaname)
      l = choice_or_tree.find_or_add(l)
      for opt in l:
        self.expand_choice_or_tree(opt)

    return

  def expand_treestore(self, iter = None):
    """
    Query the schema for what valid children can live under this node, then set the
    treestore appropriately. This recurses.
    """

    if iter is None:
      iter = self.treestore.get_iter_first()
      if iter is None:
        self.set_treestore(iter, [])
        return

    choice_or_tree, active_tree = self.treestore.get(iter, 0, 1)
    if active_tree.active is False or choice_or_tree.active is False:
      return

    l = self.s.valid_children(active_tree.schemaname)
    l = active_tree.find_or_add(l)
    self.set_treestore(iter, l)

    child_iter = self.treestore.iter_children(iter)
    while child_iter is not None:
      # fix for recursive schemata!
      child_active_tree = self.treestore.get_value(child_iter, 1)
      if child_active_tree.schemaname == active_tree.schemaname:
        debug.deprint("Warning: recursive schema elements not supported: %s" % active_tree.name)
        child_iter = self.treestore.iter_next(child_iter)
        if child_iter is None: break

      self.expand_treestore(child_iter)
      child_iter = self.treestore.iter_next(child_iter)

    return

  def remove_children(self, iter):
    """
    Delete the children of iter in the treestore.
    """

    childiter = self.treestore.iter_children(iter)
    if childiter is None: return

    result = True

    while result is True:
      result = self.treestore.remove(childiter)

    return

  def set_combobox_liststore(self, column, cellCombo, treemodel, iter, user_data=None):
    """
    This hook function sets the properties of the gtk.CellRendererCombo for each
    row. It sets up the cellcombo to use the correct liststore for its choices,
    decides whether the cellcombo should be editable or not, and sets the
    foreground colour.
    """

    choice_or_tree, active_tree, liststore = treemodel.get(iter, 0, 1, 2)

    if self.groupmode and treemodel.iter_parent(iter) is None:
      cellCombo.set_property("text", choice_or_tree.get_name_path())
    else:
      cellCombo.set_property("text", str(choice_or_tree))


    cellCombo.set_property("model", liststore)

    # set the properties: colour, etc.
    if isinstance(choice_or_tree, tree.Tree):
      cellCombo.set_property("editable", False)
    elif isinstance(choice_or_tree, choice.Choice):
      cellCombo.set_property("editable", True)

    if self.iter_is_active(treemodel, iter):
      if active_tree.valid is True:
        cellCombo.set_property("foreground", "black")
      else:
        cellCombo.set_property("foreground", "blue")
    else:
        cellCombo.set_property("foreground", "gray")

    return

  def set_cellpicture_choice(self, column, cell, treemodel, iter):
    """
    This hook function sets up the other gtk.CellRendererPixbuf, the one that gives
    the clue to the user whether this is a choice or not.
    """

    choice_or_tree = treemodel.get_value(iter, 0)
    if isinstance(choice_or_tree, tree.Tree):
      cell.set_property("stock-id", None)
    elif isinstance(choice_or_tree, choice.Choice):
      cell.set_property("stock-id", gtk.STOCK_GO_DOWN)

    return

  def set_cellpicture_cardinality(self, column, cell, treemodel, iter):
    """
    This hook function sets up the gtk.CellRendererPixbuf on the extreme right-hand
    side for each row; this paints a plus or minus or nothing depending on whether
    something can be added or removed or has to be there.
    """

    choice_or_tree = treemodel.get_value(iter, 0)
    if choice_or_tree.cardinality == "":
      cell.set_property("stock-id", None)
    elif choice_or_tree.cardinality == "?" or choice_or_tree.cardinality == "*":
      if choice_or_tree.active:
        cell.set_property("stock-id", gtk.STOCK_REMOVE)
      else:
        cell.set_property("stock-id", gtk.STOCK_ADD)
    elif choice_or_tree.cardinality == "+":
      parent_tree = choice_or_tree.parent
      count = parent_tree.count_children_by_schemaname(choice_or_tree.schemaname)

      if choice_or_tree.active and count == 2: # one active, one inactive
        cell.set_property("stock-id", None)
      elif choice_or_tree.active:
        cell.set_property("stock-id", gtk.STOCK_REMOVE)
      else:
        cell.set_property("stock-id", gtk.STOCK_ADD)

    return

  def toggle_tree(self, iter):
    """
    Toggles the state of part of the tree.
    """

    choice_or_tree = self.treestore.get_value(iter, 0)

    if choice_or_tree.active:
      self.collapse_tree(iter)
    else:
      self.expand_tree(iter)

    self.on_select_row()

    return

  def collapse_tree(self, iter, confirm = True):
    """
    Collapses part of the tree.
    """

    choice_or_tree, = self.treestore.get(iter, 0)
    parent_tree = choice_or_tree.parent

    if not choice_or_tree.active:
      return

    if choice_or_tree.cardinality == "":
      return

    if choice_or_tree.cardinality == "?":
      choice_or_tree.active = False
      self.set_saved(False)
      self.remove_children(iter)

    elif choice_or_tree.cardinality == "*":
      # If this is the only one, just make it inactive.
      # Otherwise, just delete it.
      count = parent_tree.count_children_by_schemaname(choice_or_tree.schemaname)
      if count == 1:
        choice_or_tree.active = False
        self.set_saved(False)
        self.remove_children(iter)
      else:
        self.delete_tree(iter, confirm)

    elif choice_or_tree.cardinality == "+":
      count = parent_tree.count_children_by_schemaname(choice_or_tree.schemaname)
      if count == 2: # one active, one inactive
        # do nothing
        return
      else: # count > 2
        self.delete_tree(iter, confirm)

    parent_tree.recompute_validity()
    self.treeview.queue_draw()
    return

  def delete_tree(self, iter, confirm):
    choice_or_tree, = self.treestore.get(iter, 0)
    parent_tree = choice_or_tree.parent
    isSelected = self.treeview.get_selection().iter_is_selected(iter)
    sibling = self.treestore.iter_next(iter)

    if confirm:
      response = dialogs.prompt(self.main_window, "Are you sure you want to delete this node?")

    # not A or B == A implies B
    if not confirm or response == gtk.RESPONSE_YES:
      parent_tree.delete_child_by_ref(choice_or_tree)
      self.remove_children(iter)
      self.treestore.remove(iter)
      self.set_saved(False)

      if isSelected and sibling:
        self.treeview.get_selection().select_iter(sibling)
    return

  def expand_tree(self, iter):
    """
    Expands part of the tree.
    """

    choice_or_tree, active_tree = self.treestore.get(iter, 0, 1)
    parent_tree = choice_or_tree.parent

    if choice_or_tree.active:
      return

    if choice_or_tree.cardinality == "":
      return

    elif choice_or_tree.cardinality == "?":
      choice_or_tree.active = True
      self.set_saved(False)
      self.expand_treestore(iter)

    elif choice_or_tree.cardinality == "*" or choice_or_tree.cardinality == "+":
      # Make this active, and add a new inactive instance
      choice_or_tree.active = True
      new_tree = parent_tree.add_inactive_instance(choice_or_tree)
      liststore = self.create_liststore(new_tree)
      self.expand_treestore(iter)
      iter = self.treestore.insert_after(
        None, iter, [new_tree, new_tree.get_current_tree(), liststore])
      attrid = new_tree.connect("on-set-attr", self.on_set_attr, self.treestore.get_path(iter))
      dataid = new_tree.connect("on-set-data", self.on_set_data, self.treestore.get_path(iter))
      self.signals[new_tree] = (attrid, dataid)
      self.set_saved(False)

    parent_tree.recompute_validity()
    return

  def options_tree_select_func(self, info = None):
    """
    Called when the user selected a new item in the treeview. Prevents changing of
    node and attempts to save data if appropriate.
    """

    if not self.options_tree_select_func_enabled:
      self.options_tree_select_func_enabled = True
      return False

    if not self.data.store():
      return False

    if (isinstance(self.selected_node, mixedtree.MixedTree)
       and self.geometry_dim_tree is not None
       and self.selected_node.parent is self.geometry_dim_tree.parent
       and self.selected_node.data is not None):
      self.geometry_dim_tree.set_data(self.selected_node.data)

    return True

  def on_treeview_key_press(self, treeview, event):
    """
    Called when treeview intercepts a key press. Collapse and expand rows.
    """

    if event.keyval == gtk.keysyms.Right:
      self.treeview.expand_row(self.get_selected_row(), open_all = False)

    if event.keyval == gtk.keysyms.Left:
      self.treeview.collapse_row(self.get_selected_row())

    if event.keyval == gtk.keysyms.Delete:
       self.collapse_tree(self.treestore.get_iter(self.get_selected_row()))
       self.on_select_row()

    return

  def on_treeview_button_press(self, treeview, event):
    """
    This routine is called every time the mouse is clicked on the treeview on the
    left-hand side. It processes the "buttons" gtk.STOCK_ADD and gtk.STOCK_REMOVE
    in the right-hand column, activating, adding and removing tree nodes as
    necessary.
    """
    pathinfo = treeview.get_path_at_pos(int(event.x), int(event.y))

    if event.button == 1:

      if pathinfo is not None:
        path = pathinfo[0]
        col = pathinfo[1]

        if col is self.imgcolumn:
          iter = self.treestore.get_iter(path)
          self.toggle_tree(iter)

    elif event.button == 3:
      if pathinfo is not None:
        treeview.get_selection().select_path(pathinfo[0])
        self.show_popup(None, event.button, event.time)
        return True

  def popup_location(self, widget):
    column = self.treeview.get_column(0)
    treemodel, treeiter = self.treeview.get_selection().get_selected()
    treepath = treemodel.get_path(treeiter)
    area = self.treeview.get_cell_area(treepath, column)
    sx, sy = self.popup.size_request()
    tx, ty = area.x, area.y + int(sy * 1.25)
    x, y = self.treeview.tree_to_widget_coords(tx, ty)
    return (x, y, True)

  def on_treeview_popup(self, treeview):
    self.show_popup(self.popup_location, 0, gtk.get_current_event_time())
    return

  def show_popup(self, func, button, time):
    self.popup.popup(None, None, func, button, time)
    return

  def on_select_row(self, selection=None):
    """
    Called when a row is selected. Update the options frame.
    """

    if isinstance(selection, str) or isinstance(selection, tuple):
      path = selection
    else:
      path = self.get_selected_row(selection)

    if path is None:
      return

    self.selected_iter = iter = self.treestore.get_iter(path)
    choice_or_tree, active_tree = self.treestore.get(iter, 0, 1)

    debug.dprint(active_tree)

    self.selected_node = self.get_painted_tree(iter)
    self.update_options_frame()

    name = self.get_spudpath(active_tree)
    self.statusbar.set_statusbar(name)
    self.current_spudpath = name
    self.current_xpath = self.get_xpath(active_tree)

    self.clear_plugin_buttons()

    for plugin in plugins.plugins:
      if plugin.matches(name):
        self.add_plugin_button(plugin)

    return

  def get_spudpath(self, active_tree):
    # get the name to paint on the statusbar
    name_tree = active_tree
    name = ""
    while name_tree is not None:
      if "name" in name_tree.attrs and name_tree.attrs["name"][1] is not None:
        used_name = name_tree.name + '::%s' % name_tree.attrs["name"][1]
      elif name_tree.parent is not None and name_tree.parent.count_children_by_schemaname(name_tree.schemaname) > 1:
        siblings = [x for x in name_tree.parent.children if x.schemaname == name_tree.schemaname]
        i = 0
        for sibling in siblings:
          if sibling is name_tree:
            break
          else:
            i = i + 1
        used_name = name_tree.name + "[%s]" % i
      else:
        used_name = name_tree.name

      name = "/" + used_name + name
      name_tree = name_tree.parent

    # and split off the root name:
    name = '/' + '/'.join(name.split('/')[2:])
    return name

  def get_xpath(self, active_tree):
    # get the name to paint on the statusbar
    name_tree = active_tree
    name = ""
    while name_tree is not None:
      if "name" in name_tree.attrs and name_tree.attrs["name"][1] is not None:
        used_name = name_tree.name + '[@name="%s"]' % name_tree.attrs["name"][1]
      elif name_tree.parent is not None and name_tree.parent.count_children_by_schemaname(name_tree.schemaname) > 1:
        siblings = [x for x in name_tree.parent.children if x.schemaname == name_tree.schemaname]
        i = 0
        for sibling in siblings:
          if sibling is name_tree:
            break
          else:
            i = i + 1
        used_name = name_tree.name + "[%s]" % i
      else:
        used_name = name_tree.name

      name = "/" + used_name + name
      name_tree = name_tree.parent

    return name

  def clear_plugin_buttons(self):
    for button in self.plugin_buttons:
      self.plugin_buttonbox.remove(button)

    self.plugin_buttons = []

  def add_plugin_button(self, plugin):
    button = gtk.Button(label=plugin.name)
    button.connect('clicked', self.plugin_handler, plugin)
    button.show()

    self.plugin_buttons.append(button)
    self.plugin_buttonbox.add(button)

  def plugin_handler(self, widget, plugin):
    f = StringIO.StringIO()
    self.tree.write(f)
    xml = f.getvalue()
    xml = plugin.execute(xml, self.current_xpath)
    if xml:
      ios = StringIO.StringIO(xml)

      try:
        tree_read = self.s.read(filename)

        if tree_read is None:
          self.statusbar.set_statusbar("Unable to read plugin result")
          return

        self.display_validation_errors(self.s.read_errors())
        self.tree = tree_read
      except:
        dialogs.error_tb(self.main_window, "Unable to read plugin result")
        return

      path = self.treestore.get_path(self.selected_iter)

      self.display_validation_errors(self.s.read_errors())

      self.set_saved(False)

      self.treeview.freeze_child_notify()
      self.treeview.set_model(None)
      self.signals = {}
      self.set_treestore(None, [self.tree], True)
      self.treeview.set_model(self.treestore)
      self.treeview.thaw_child_notify()

      self.set_geometry_dim_tree()

      self.treeview.get_selection().select_path(path)

      self.scherror.destroy_error_list()


  def get_selected_row(self, selection=None):
    """
    Get the iter to the selected row.
    """

    if (selection == None):
        selection = self.treeview.get_selection()

    (model, paths) = selection.get_selected_rows()
    if paths:
      return paths[0]
    else:
      return None

  def on_activate_row(self, treeview, path, view_column):
    """
    Called when you double click or press Enter on a row.
    """

    iter = self.treestore.get_iter(path)

    self.expand_tree(iter)
    self.on_select_row()

    if path is None:
      return

    if treeview.row_expanded(path):
      treeview.collapse_row(path)
    else:
      treeview.expand_row(path, False)

    return

  def cellcombo_changed(self, combo, tree_path, combo_iter):
    """
    This is called when a cellcombo on the left-hand treeview is edited,
    i.e. the user chooses between more than one possible choice.
    """

    tree_iter = self.treestore.get_iter(tree_path)
    choice = self.treestore.get_value(tree_iter, 0)

    # get the ref to the new active choice
    liststore = self.treestore.get_value(tree_iter, 2)
    ref = liststore.get_value(combo_iter, 1)

    # record the choice in the datatree
    choice.set_active_choice_by_ref(ref)
    new_active_tree = choice.get_current_tree()

    name = self.get_spudpath(new_active_tree)
    self.statusbar.set_statusbar(name)
    self.treestore.set(tree_iter, 1, new_active_tree)
    self.current_spudpath = name
    xpath = self.get_xpath(new_active_tree)
    self.current_xpath = xpath

    self.clear_plugin_buttons()

    for plugin in plugins.plugins:
      if plugin.matches(xpath):
        self.add_plugin_button(plugin)

    self.remove_children(tree_iter)
    self.expand_treestore(tree_iter)
    self.treeview.expand_row(tree_path, False)

    self.set_saved(False)
    self.selected_node = self.get_painted_tree(tree_iter)
    self.update_options_frame()

    return

  def get_treeview_iter(self, selection):
    """
    Get a treeview iterator object, given a selection.
    """

    path = self.get_selected_row(selection)
    if path is None:
      return self.get_treestore_iter_from_xmlpath(self.current_xpath)

    return self.treestore.get_iter(path)

  def on_set_data(self, node, data, path):
    self.set_saved(False)
    self.treeview.queue_draw()

  def on_set_attr(self, node, attr, value, path):
    self.set_saved(False)
    self.treeview.queue_draw()

  def get_painted_tree(self, iter_or_tree, lock_geometry_dim = True):
    """
    Check if the given tree, or the active tree at the given iter in the treestore,
    have any children of the form *_value. If so, we need to make the node painted
    by the options tree a mix of the two: the documentation and attributes come from
    the parent, and the data from the child.

    Also check if it is the geometry node, validity of any tuple data, and, if an
    iter is supplied, check that the node is active.
    """

    if isinstance(iter_or_tree, tree.Tree):
      active_tree = iter_or_tree
    else:
      active_tree = self.treestore.get_value(iter_or_tree, 1)

    painted_tree = active_tree.get_mixed_data()

    if not isinstance(iter_or_tree, tree.Tree) and not self.iter_is_active(self.treestore, iter_or_tree):
      painted_tree = tree.Tree(painted_tree.name, painted_tree.schemaname, painted_tree.attrs, doc = painted_tree.doc)
      painted_tree.active = False
    elif lock_geometry_dim and self.geometry_dim_tree is not None and self.geometry_dim_tree.data is not None:
      if active_tree is self.geometry_dim_tree:
        data_tree = tree.Tree(painted_tree.name, painted_tree.schemaname, datatype = "fixed")
        data_tree.data = painted_tree.data
        painted_tree = MixedTree(painted_tree, data_tree)
      elif isinstance(self.geometry_dim_tree, mixedtree.MixedTree) and active_tree is self.geometry_dim_tree.parent:
        data_tree = tree.Tree(painted_tree.child.name, painted_tree.child.schemaname, datatype = "fixed")
        data_tree.data = painted_tree.data
        painted_tree = mixedtree.MixedTree(painted_tree, data_tree)

    return painted_tree

  def get_treestore_iter_from_xmlpath(self, xmlpath):
    """
    Convert the given XML path to an iter into the treestore. For children of a
    single parent with the same names, only the first child is considered.
    """

    names = xmlpath.split("/")

    iter = self.treestore.get_iter_first()
    if iter is None:
      return None
    for name in names[1:-1]:
      while str(self.treestore.get_value(iter, 0)) != name:
        iter = self.treestore.iter_next(iter)
        if iter is None:
          return None
      iter = self.treestore.iter_children(iter)
      if iter is None:
        return None

    return iter

  def get_treestore_path_from_node(self, node):
    """
    Look for the path for the given node.
    """

    def search(iter, node, indent=""):
      while iter:
        if self.treestore.get_value(iter, 0) is node:
          return iter
        else:
          child = search(self.treestore.iter_children(iter), node, indent + "  ")
          if child: return child

          iter = self.treestore.iter_next(iter)
      return iter

    iter = search(self.treestore.get_iter_first(), node)
    return self.treestore.get_path(iter) if iter else None

  def set_geometry_dim_tree(self):
    """
    Find the iter into the treestore corresponding to the geometry dimension, and
    perform checks to test that the geometry dimension node is valid.
    """

    self.geometry_dim_tree = self.data.geometry_dim_tree = None
    # The tree must exist
    if self.tree is None:
      return

    # A geometry dimension element must exist
    iter = self.get_treestore_iter_from_xmlpath("/" + self.tree.name + self.data_paths["dim"])
    if iter is None:
      return

    painted_tree = self.get_painted_tree(iter, False)
    if isinstance(painted_tree, mixedtree.MixedTree):
       # If the geometry dimension element has a hidden data element, it must
       # have datatype tuple or fixed
       if not isinstance(painted_tree.datatype, tuple) and painted_tree.datatype != "fixed":
         self.geometry_dim_tree = None
         return
    elif painted_tree.datatype != "fixed":
      # Otherwise, only fixed datatype is permitted
      return

    # All parents of the geometry dimension element must have cardinality ""
    # (i.e. not ?, * or +).
    parent = painted_tree.parent
    while parent is not None:
      if parent.cardinality != "":
        self.geometry_dim_tree = None
        return
      parent = parent.parent

    # All possible geometry dimensions must be positive integers
    if isinstance(painted_tree.datatype, tuple):
      possible_dims = painted_tree.datatype
    elif painted_tree.datatype == "fixed":
      possible_dims = [painted_tree.data]
    else:
      return
    for opt in possible_dims:
      try:
        test = int(opt)
        assert test > 0
      except:
        return

    # A valid geometry dimension element has been located
    self.geometry_dim_tree = self.data.geometry_dim_tree = painted_tree

    return

  def iter_is_active(self, treemodel, iter):
    """
    Test whether the node at the given iter in the LHS treestore is active.
    """

    while iter is not None:
      choice_or_tree = treemodel.get_value(iter, 0)
      active_tree = treemodel.get_value(iter, 1)
      if not choice_or_tree.active or not active_tree.active:
        return False
      iter = treemodel.iter_parent(iter)

    return True

  def choice_or_tree_matches(self, text, choice_or_tree, recurse, search_active_subtrees = False):
    """
    See if the supplied node matches a given piece of text. If recurse is True,
    the node is deemed to match if any of its children match or, if the node is a
    choice and search_active_subtrees is True, any of the available trees in the
    choice match.
    """

    if choice_or_tree.is_hidden():
      return False
    elif isinstance(choice_or_tree, choice.Choice):
      if self.choice_or_tree_matches(text, choice_or_tree.get_current_tree(), False):
        return True
      elif recurse and self.find.search_gui.get_widget("searchInactiveChoiceSubtreesCheckButton").get_active():
        for opt in choice_or_tree.choices():
          if not search_active_subtrees and opt is choice_or_tree.get_current_tree():
            continue
          if opt.children == []:
            self.expand_choice_or_tree(opt)
          if self.choice_or_tree_matches(text, opt, recurse, True):
            return True
    else:
      if self.get_painted_tree(choice_or_tree).matches(text, self.find.search_gui.get_widget("caseSensitiveCheckButton").get_active()):
        return True
      else:
        if self.find.search_gui.get_widget("caseSensitiveCheckButton").get_active():
          text_re = re.compile(text)
        else:
          text_re = re.compile(text, re.IGNORECASE)
        comment = choice_or_tree.get_comment()
        if comment is not None and comment.data is not None and text_re.search(comment.data) is not None:
          return True
        elif recurse:
          for opt in choice_or_tree.children:
            if self.choice_or_tree_matches(text, opt, recurse, True):
              return True

      return False

  def search_treestore(self, text, iter = None):
    """
    Recursively search the tree for a node that matches a given piece of text.
    MixedTree.matches and choice_or_tree_matches decide what is a match (using
    tree.Matches).

    This uses lazy evaluation to only search as far as necessary; I love
    Python generators. If you don't know what a Python generator is and need to
    understand this, see PEP 255.
    """

    if iter is None:
      iter = self.treestore.get_iter_first()
    if iter is None:
      yield None
    choice_or_tree = self.treestore.get_value(iter, 0)

    if self.choice_or_tree_matches(text, choice_or_tree, isinstance(choice_or_tree, choice.Choice)):
      yield iter

    child_iter = self.treestore.iter_children(iter)
    while child_iter is not None:
      for iter in self.search_treestore(text, child_iter):
        yield iter
      child_iter = self.treestore.iter_next(child_iter)

    return

  ### RHS ###

  def add_custom_widgets(self):
    """
    Adds custom python widgets that aren't easily handeled by glade.
    """

    optionsFrame = self.gui.get_widget("optionsFrame")

    vpane1 = gtk.VPaned()
    vpane2 = gtk.VPaned()
    vbox = gtk.VBox()

    vpane1.pack2(vpane2, True, False)
    vpane2.pack1(vbox, True, False)
    optionsFrame.add(vpane1)

    self.description = descriptionwidget.DescriptionWidget()
    vpane1.pack1(self.description, True, False)

    self.attributes = attributewidget.AttributeWidget()
    vbox.pack_start(self.attributes, True, True)

    databuttons = databuttonswidget.DataButtonsWidget()
    vbox.pack_end(databuttons, False)

    self.data = datawidget.DataWidget()
    self.data.set_buttons(databuttons)
    vbox.pack_end(self.data, True, True)

    self.comment = commentwidget.CommentWidget()
    vpane2.pack2(self.comment, True, False)

    optionsFrame.show_all()
    return

  def update_options_frame(self):
    """
    Update the RHS.
    """

    self.description.update(self.selected_node)

    self.attributes.update(self.selected_node)

    self.data.update(self.selected_node)

    self.comment.update(self.selected_node)

    self.gui.get_widget("optionsFrame").queue_resize()

    return

class DiamondFindDialog:
  def __init__(self, parent, gladefile):
    self.parent = parent
    self.gladefile = gladefile
    self.search_dialog = None

    return

  def on_find(self, widget=None):
    """
    Open up the find dialog. It has to be created each time from the glade file.
    """

    if not self.search_dialog is None:
      return

    signals =      {"on_find_dialog_close": self.on_find_close_button,
                    "on_close_clicked": self.on_find_close_button,
                    "on_find_clicked": self.on_find_find_button}

    self.search_gui = gtk.glade.XML(self.gladefile, root="find_dialog")
    self.search_dialog = self.search_gui.get_widget("find_dialog")
    self.search_gui.signal_autoconnect(signals)
    search_entry = self.search_gui.get_widget("search_entry")
    search_entry.connect("activate", self.on_find_find_button)

    # reset the search parameters
    self.search_generator = None
    self.search_text = ""
    self.search_count = 0
    self.search_dialog.show()
    self.parent.statusbar.set_statusbar("")

    return

  def on_find_find_button(self, button):
    """
    Search. Each time "Find" is clicked, we compare the stored search text to the
    text in the entry box. If it's the same, we find next; if it's different, we
    start a new search. self.search_treestore does the heavy lifting.
    """

    search_entry = self.search_gui.get_widget("search_entry")

    self.parent.statusbar.clear_statusbar()

    text = search_entry.get_text()
    if text == "":
      self.parent.statusbar.set_statusbar("No text")
      return

    # check if we've started a new search
    if text != self.search_text:
      # started a new search
      self.search_generator = None
      self.search_generator = self.parent.search_treestore(text)
      self.search_text = text
      self.search_count = 0

    try:
      # get the iter of the next tree that matches
      iter = self.search_generator.next()
      path = self.parent.treestore.get_path(iter)
      # scroll down to it, expand it, and select it
      self.parent.treeview.expand_to_path(path)
      self.parent.treeview.get_selection().select_iter(iter)
      self.parent.treeview.scroll_to_cell(path, use_align=True, col_align=0.5)
      # count how many hits we've had
      self.search_count = self.search_count + 1
    except StopIteration:
      # reset the search and cycle
      self.search_text = ""
      # if something was found, go through again
      if self.search_count > 0:
        self.on_find_find_button(button)
      else:
        self.parent.statusbar.set_statusbar("No results")

    return

  def on_find_close_button(self, button = None):
    """
    Close the search widget.
    """

    if not self.search_dialog is None:
      self.search_dialog.hide()
      self.search_dialog = None
    self.parent.statusbar.clear_statusbar()

    return

class DiamondStatusBar:
  def __init__(self, statusbar):
    self.statusbar = statusbar
    self.context_id = statusbar.get_context_id("Messages")

    return

  def set_statusbar(self, msg):
    """
    Set the status bar message.
    """

    self.statusbar.push(self.context_id, msg)

    return

  def clear_statusbar(self):
    """
    Clear the status bar.
    """

    self.statusbar.push(self.context_id, "")

    return
