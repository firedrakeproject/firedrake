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

class CommentWidget(gtk.Frame):

  __gsignals__ = { "on-store"  : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, ())}

  def __init__(self):
    gtk.Frame.__init__(self)

    scrolledWindow = gtk.ScrolledWindow()
    scrolledWindow.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)

    textView = self.textView = gtk.TextView()
    textView.set_editable(False)
    textView.set_wrap_mode(gtk.WRAP_WORD)
    textView.set_cursor_visible(False)
    textView.connect("focus-in-event", self.focus_in)
    textView.connect("focus-out-event", self.focus_out)
    textView.get_buffer().create_tag("tag")

    scrolledWindow.add(textView)

    label = gtk.Label()
    label.set_markup("<b>Comment</b>")

    self.set_shadow_type(gtk.SHADOW_NONE)
    self.set_label_widget(label)
    self.add(scrolledWindow)

    self.comment_tree = None
    return

  def update(self, node):
    """
    Update the widget with the given node
    """

    #before updateing store the old
    self.store()

    if node is None or not node.active:
      self.textView.get_buffer().set_text("")
      self.textView.set_cursor_visible(False)
      self.textView.set_editable(False)
      try:
        self.textView.set_tooltip_text("")
        self.textView.set_property("has-tooltip", False)
      except:
        pass

      return

    self.comment_tree = comment_tree = node.get_comment()
    text_tag = self.textView.get_buffer().get_tag_table().lookup("tag")
    if comment_tree is None:
      self.textView.get_buffer().set_text("No comment")
      self.textView.set_cursor_visible(False)
      self.textView.set_editable(False)
      text_tag.set_property("foreground", "grey")
      try:
        self.textView.set_tooltip_text("")
        self.textView.set_property("has-tooltip", False)
      except:
        pass
    else:
      if comment_tree.data is None:
        self.textView.get_buffer().set_text("(string)")
      else:
        self.textView.get_buffer().set_text(comment_tree.data)
      if node.active:
        self.textView.set_cursor_visible(True)
        self.textView.set_editable(True)
        text_tag.set_property("foreground", "black")
      else:
        self.textView.set_cursor_visible(False)
        self.textView.set_editable(False)
        text_tag.set_property("foreground", "grey")

    buffer_bounds = self.textView.get_buffer().get_bounds()
    self.textView.get_buffer().apply_tag(text_tag, buffer_bounds[0], buffer_bounds[1])

    self.interacted = False

    return

  def store(self):
    """
    Store data in the node comment.
    """

    comment_tree = self.comment_tree
    if comment_tree is None or not self.interacted:
      return

    data_buffer_bounds = self.textView.get_buffer().get_bounds()
    new_comment = self.textView.get_buffer().get_text(data_buffer_bounds[0], data_buffer_bounds[1])

    if new_comment != comment_tree.data:
      if new_comment == "":
        comment_tree.data = None
        comment_tree.active = False
      else:
        comment_tree.set_data(new_comment)
        comment_tree.active = True
        self.emit("on-store")
    return

  def focus_in(self, widget, event):
    """
    Called when the comment widget gains focus. Removes the printable_type
    placeholder.
    """

    comment_tree = self.comment_tree
    if not comment_tree is None and not self.interacted:
      self.interacted = True
      if comment_tree.data is None:
        self.textView.get_buffer().set_text("")

    return

  def focus_out(self, widget, event):
    """"
    Called when the comment widget loses focus. Stores the comment.
    """
    self.store()

gobject.type_register(CommentWidget)
