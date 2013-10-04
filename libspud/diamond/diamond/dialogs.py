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
import sys
import traceback

import gtk

import pygtkconsole

def prompt(parent, message, type = gtk.MESSAGE_QUESTION, has_cancel = False):
  """
  Display a simple Yes / No dialog. Returns one of gtk.RESPONSE_{YES,NO,CANCEL}.
  """

  prompt_dialog = gtk.MessageDialog(parent, 0, type, gtk.BUTTONS_NONE, message)
  prompt_dialog.add_buttons(gtk.STOCK_YES, gtk.RESPONSE_YES, gtk.STOCK_NO, gtk.RESPONSE_NO)
  if has_cancel:
    prompt_dialog.add_buttons(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
  prompt_dialog.connect("response", prompt_response)

  prompt_dialog.run()

  return prompt_response.response

def long_message(parent, message):
  """
  Display a message prompt, with the message contained within a scrolled window.
  """

  message_dialog = gtk.Dialog(parent = parent, buttons = (gtk.STOCK_OK, gtk.RESPONSE_ACCEPT))
  message_dialog.set_default_size(400, 300)
  message_dialog.connect("response", close_dialog)

  scrolled_window = gtk.ScrolledWindow()
  message_dialog.vbox.add(scrolled_window)
  scrolled_window.show()

  scrolled_window.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_ALWAYS)

  text_view = gtk.TextView()
  scrolled_window.add(text_view)
  text_view.show()

  text_view.get_buffer().set_text(message)
  text_view.set_cursor_visible(False)
  text_view.set_property("editable", False)
  text_view.set_property("height-request", 180)
  text_view.set_property("width-request", 240)

  message_dialog.run()

  return

def error(parent, message):
  """
  Display an error message.
  """

  error_dialog = gtk.MessageDialog(parent, 0, gtk.MESSAGE_WARNING, gtk.BUTTONS_OK, message)
  error_dialog.connect("response", close_dialog)
  error_dialog.run()

  return

def error_tb(parent, message):
  """
  Display an error message, together with the last traceback.
  """

  tb = traceback.format_exception(sys.exc_info()[0] ,sys.exc_info()[1], sys.exc_info()[2])
  tb_msg = ""
  for tbline in tb:
    tb_msg += tbline
  long_message(parent, tb_msg + "\n" + message)

  return

def get_filename(title, action, filter_names_and_patterns = {}, folder_uri = None):
  """
  Utility function to get a filename.
  """

  if action == gtk.FILE_CHOOSER_ACTION_SAVE:
    buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_SAVE,gtk.RESPONSE_OK)
  elif action == gtk.FILE_CHOOSER_ACTION_CREATE_FOLDER:
    buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_NEW,gtk.RESPONSE_OK)
  else:
    buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,gtk.STOCK_OPEN,gtk.RESPONSE_OK)
  filew = gtk.FileChooserDialog(title=title, action=action, buttons=buttons)
  filew.set_default_response(gtk.RESPONSE_OK)

  if not folder_uri is None:
    filew.set_current_folder_uri("file://" + os.path.abspath(folder_uri))

  for filtername in filter_names_and_patterns:
    filter = gtk.FileFilter()
    filter.set_name(filtername)
    filter.add_pattern(filter_names_and_patterns[filtername])
    filew.add_filter(filter)

  allfilter = gtk.FileFilter()
  allfilter.set_name("All known files")
  for filtername in filter_names_and_patterns:
    allfilter.add_pattern(filter_names_and_patterns[filtername])
  filew.add_filter(allfilter)

  filter = gtk.FileFilter()
  filter.set_name("All files")
  filter.add_pattern("*")
  filew.add_filter(filter)

  result = filew.run()

  if result == gtk.RESPONSE_OK:
    filename = filew.get_filename()
    filtername = filew.get_filter().get_name()
    filew.destroy()
    return filename
  else:
    filew.destroy()
    return None

def console(parent, locals = None):
  """
  Launch a python console.
  """

  console_dialog = gtk.Dialog(parent = parent, buttons = (gtk.STOCK_QUIT, gtk.RESPONSE_ACCEPT))
  console_dialog.set_default_size(400, 300)
  console_dialog.connect("response", close_dialog)

  stdout = sys.stdout
  stderr = sys.stderr

  console_widget = pygtkconsole.GTKInterpreterConsole(locals)
  console_dialog.vbox.add(console_widget)
  console_widget.show()

  console_dialog.run()

  sys.stdout = stdout
  sys.stderr = stderr

  return

def prompt_response(dialog, response_id):
  """
  Signal handler for dialog response signals. Stores the dialog response in the
  function namespace, to allow response return in other functions.
  """

  if response_id == gtk.RESPONSE_DELETE_EVENT:
    response_id = gtk.RESPONSE_CANCEL

  prompt_response.response = response_id
  close_dialog(dialog, response_id)

  return

def close_dialog(dialog, response_id = None):
  """
  Signal handler for dialog reponse or destroy signals. Closes the dialog.
  """

  dialog.destroy()

  return

def radio_dialog(title, message, choices, logo):
  r = RadioDialog(title, message, choices, logo)
  return r.data

def message_box(window, title, message):
  dialog = gtk.MessageDialog(window, 0, gtk.MESSAGE_INFO, gtk.BUTTONS_OK, message)
  dialog.set_title(title)
  dialog.connect("response", close_dialog)
  dialog.run()

class RadioDialog:
  def __init__(self, title, message, choices, logo):
    self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
    self.window.connect("delete_event", self.cleanup)
    self.window.connect("key_press_event", self.key_press)
    self.window.set_title(title)
    self.window.set_position(gtk.WIN_POS_CENTER)
    if not logo is None:
      self.window.set_icon_from_file(logo)
    self.window.show()

    #swindow = gtk.ScrolledWindow()
    #swindow.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
    #self.window.add(swindow)
    #swindow.show()

    main_box = gtk.VBox(False, 0)
    self.window.add(main_box)
    main_box.show()

    if not logo is None:
      image = gtk.Image()
      image.set_from_file(logo)
      main_box.pack_start(image, True, True, 0)
      image.show()

    label = gtk.Label(message)
    main_box.add(label)
    label.show()

    radio_box = gtk.VBox(False, 10)
    main_box.pack_start(radio_box, True, True, 0)
    radio_box.show()

    separator = gtk.HSeparator()
    main_box.pack_start(separator, False, True, 0)
    separator.show()

    close = gtk.Button(stock=gtk.STOCK_OK)
    close.connect("clicked", self.cleanup)
    main_box.pack_start(close, False, False, 0)
    close.show()

    prev_radio = None
    for choice in choices:
      radio = gtk.RadioButton(prev_radio, choice)
      radio.connect("toggled", self.radio_callback, choice)
      radio_box.pack_start(radio, False, False, 0)
      radio.show()
      if prev_radio is None:
        radio.set_active(True)
      prev_radio = radio

    self.data = choices[0]

    gtk.main()

  def cleanup(self, widget, data=None):
    self.window.destroy()
    gtk.main_quit()

  def key_press(self, widget, event):
    if event.keyval == gtk.keysyms.Return:
      self.cleanup(None)

  def radio_callback(self, widget, data):
    self.data = data

class GoToDialog:
  def __init__(self, parent):
    self.goto_gui = gtk.glade.XML(parent.gladefile, root="GoToDialog")
    self.dialog_box = self.goto_gui.get_widget("GoToDialog")
    self.dialog_box.set_modal(True)

  def run(self):
    signals =      {"goto_activate": self.on_goto_activate,
                    "cancel_activate": self.on_cancel_activate}

    self.goto_gui.signal_autoconnect(signals)

    self.dialog_box.show()
    return ""

  def on_goto_activate(self, widget=None):
    print "goto"

  def on_cancel_activate(self, widget=None):
    print "cancel"

