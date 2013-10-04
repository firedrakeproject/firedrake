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

import debug
import pango
import gobject
import gtk
import gtk.glade
import sys
import tempfile
import os
import re

import dialogs
import interface

class DiamondSchemaError:
  def __init__(self, parent, gladefile, schema_file, schematron_file):
    self.parent = parent
    self.gladefile = gladefile
    self.schema_file = schema_file
    self.schematron_file = schematron_file
    self.errlist_type = 0
    self.listview = None

    self.listwindow = self.parent.gui.get_widget("errListWindow")
    self.listview = self.parent.gui.get_widget("sch_err_list")

    if self.listwindow is None or self.listview is None:
      raise Exception("Could not find the error list widgets")

    self.model = gtk.ListStore(gobject.TYPE_STRING, gobject.TYPE_STRING)
    self.cellcombo = gtk.CellRendererCombo()
    self.cellcombo.set_property("model", self.model)
    self.cellcombo.set_property("editable", False)

    column_renderer = gtk.CellRendererText()
    column_renderer.set_property("editable", False)
    column_xpath = gtk.TreeViewColumn("Element", column_renderer, text = 0)
    column_error = gtk.TreeViewColumn("Error", column_renderer, text = 1)

    self.listview.append_column(column_xpath)
    self.listview.append_column(column_error)

    column_xpath.set_property("expand", True)
    column_xpath.set_resizable(True)
    column_xpath.set_property("min-width", 75)

    column_error.set_property("expand", True)
    column_error.set_resizable(True)
    column_error.set_property("min-width", 75)

    # Set adjustment parameters for the list view
    self.listview.set_model(self.model)
    self.listview.connect("row_activated", self.on_xml_row_activate)

    # User can only select one error at a time to go to.
    self.listview.get_selection().set_mode(gtk.SELECTION_SINGLE)

    self.listview.hide()
    self.listwindow.hide()

  # Create the error list, model and tree view columns if the list does not exist.
  def create_error_list(self):
    if self.listview is not None:
      return

  # Destroy the error list widgets
  def destroy_error_list(self):
    if self.listview is not None:
      self.listview.hide()
      self.listwindow.hide()

  def on_validate(self, widget=None):
    """
    Tools > Validate XML. This writes out the XML to a temporary file, then calls
    xmllint on it. (I didn't use the Xvif validation capabilities because the error
    messages are worse than useless; xmllint actually gives line numbers and such).
    """

    if self.schema_file is None:
      dialogs.error(self.parent.main_window, "No schema file open")
      return

    self.tmp = tempfile.NamedTemporaryFile()
    self.parent.tree.write(self.tmp.name)

    std_input, std_output, std_error = os.popen3("xmllint --relaxng %s %s --noout --path \".\"" % (self.schema_file, self.tmp.name))
    output = std_error.read()

    output = output.replace("%s fails to validate" % self.tmp.name, "")

    if output.strip() == "%s validates" % self.tmp.name:
      # No errors. Close the error list (if it is open) and display a congratulatory message box.
      self.destroy_error_list()
      dialogs.message_box(self.parent.main_window, "XML file validated successfully", "Validation successful")
    else:
      self.create_error_list()

      self.errlist_type = 1
      self.model.clear()

      lines = output.split("\n")

      # Read the temporary output file, split it by lines and use it to convert
      # line numbers to xpaths for the column data.
      f = file(self.tmp.name)
      data = f.read()
      output_lines = data.split("\n")

      for line in lines:
        if len(line) == 0:
          continue

        tokens = line.split(" ")

	# Parse each line of the xmllint --relaxng output.
	# FORMAT:
	# ELEMENT:LINE: element NAME:  Relax-NG validity error : ERROR MESSAGE
	# (Capitals denotes variable data). Update the following code if the output changes.
        sub_tokens = tokens[0].split(":")

        message = " ".join(tokens[7:])

        line = sub_tokens[1]
        xpath = self.add_to_xpath(output_lines, "", long(line)-1)

        self.model.append([ xpath, message ])

      self.listview.set_property("visible", True)
      self.listwindow.set_property("visible", True)

      # Update the status bar. Inform the user of the number of errors.
      self.parent.statusbar.set_statusbar("There are %u Relax-NG errors in the document" % len(lines))

    return

  def on_validate_schematron(self, widget=None):
    """
    Tools > Validate Schematron. This uses the etree.Schematron API, if it exists, to
    validate the document tree against a supplied schematron file.
    """

    if self.schematron_file is None:
      dialogs.error(self.parent.main_window, "No schematron file supplied")
      return

    # Write the current version out to file.
    tmp = tempfile.NamedTemporaryFile()
    self.parent.tree.write(tmp.name)
    std_input, std_output, err_output = os.popen3("xmllint --schematron %s %s --noout" % (self.schematron_file, tmp.name))
    output = err_output.read()

    output = output.replace("%s fails to validate" % tmp.name, "")
    output = output.strip()

    if output == "%s validates" % tmp.name:
      self.destroy_error_list()
      dialogs.message_box(self.parent.main_window, "XML file validated successfully against %s" % self.schematron_file, "XML validation successful")

    # Clear the list, as there may still be errors from a previous schematron validation.
    self.model.clear()
    self.errlist_type = 0

    # Add each line of output as a new row.
    lines = output.split("\n")

    for line in lines:
      if len(line) == 0:
        continue

    tokens = line.split(" ")
    self.model.append([ tokens[0], " ".join(tokens[3:]), 0 ])

    # Finish set up of list view widget.
    self.listview.set_property("visible", True)
    self.listwindow.set_property("visible", True)
    self.parent.statusbar.set_statusbar("There are %u Schematron errors in the document" % len(lines))

  def errlist_is_open(self):
    return self.listview.get_property("visible")

  def get_elem_name(self, line):
    xml = re.compile("<([/?!]?\w+)", re.VERBOSE)
    matches = xml.findall(line)
 #   print "get_elem_name", line, "=", matches[0]
    return matches[0]

  def add_to_xpath(self, lines, xpath, line_no):

    line = lines[line_no].strip()
    name = self.get_elem_name(line)

    if name not in ["string_value", "integer_value", "real_value"]:
      xpath = "/" + name + xpath

    # Are we at the start of the document?
    if line_no == 0 or line_no == 1:
      return xpath

    # Find the parent node.
    while 1:
      line_no = line_no - 1
      line = lines[line_no].strip()

#      print "MAIN LOOP: ", line

      # Skip past comments
      if line.find("-->") != -1:
        while line.find("<!--") == -1:
          line_no = line_no - 1
          line = lines[line_no].strip()
        continue

      if line.startswith("</"):
        name = line.strip("<>/")
        while line.find("<%s" % (name)) == -1:
#          print "BACK LOOP: ", line
          line_no = line_no - 1
          line = lines[line_no].strip()
        continue

      name = self.get_elem_name(line)
      if name.find("<%s />" % (name)) == -1 and name.find("</ %s>" % (name)):
        return self.add_to_xpath(lines, xpath, line_no)

  def scroll_to_xpath(self, xpath):
    iter = self.parent.get_treestore_iter_from_xmlpath(xpath)
    path = self.parent.treestore.get_path(iter)
    self.parent.treeview.expand_to_path(path)
    self.parent.treeview.get_selection().select_iter(iter)
    self.parent.treeview.scroll_to_cell(path, use_align=True, col_align=0.5)

  # Signal handlers
  def on_xml_row_activate(self, treeview, path, view_column):

    # Get the selected row.
    selection = treeview.get_selection()
    (model, iter) = selection.get_selected()

    # Get the line number
    xpath = model.get_value(iter, 0)

    # Use the in-memory schema representation to get the XPath from the name.
    self.scroll_to_xpath(xpath)
