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

import gtk
import re
import TextBufferMarkup
import webbrowser

class DescriptionWidget(gtk.Frame):
  def __init__(self):
    gtk.Frame.__init__(self)

    scrolledWindow = gtk.ScrolledWindow()
    scrolledWindow.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)

    textView = self.textView = gtk.TextView()
    textView.set_editable(False)
    textView.set_wrap_mode(gtk.WRAP_WORD)
    textView.set_cursor_visible(False)

    textView.set_buffer(TextBufferMarkup.PangoBuffer())
    textView.connect("button-release-event", self.mouse_button_release)
    textView.connect("motion-notify-event", self.mouse_over)

    scrolledWindow.add(textView)

    label = gtk.Label()
    label.set_markup("<b>Description</b>")

    self.set_shadow_type(gtk.SHADOW_NONE)
    self.set_label_widget(label)
    self.add(scrolledWindow)

    return

  def update(self, node):
    if node is None:
      self.set_description("<span foreground=\"grey\">No node selected</span>")
    elif node.doc is None:
      self.set_description("<span foreground=\"red\">No documentation</span>")
    else:
      self.set_description(node.doc)

    return

  def set_description(self, text):
    """
    Set the node description.
    """

    #text saved in self.text without markup
    self.text = text = self.render_whitespace(text)
    link_bounds = self.link_bounds = self.get_link_bounds(text)

    if link_bounds:
      new_text = []
      index = 0
      for bounds in link_bounds:
        new_text.append(text[index:bounds[0]])
        new_text.append("<span foreground=\"blue\" underline=\"single\">")
        new_text.append(text[bounds[0]:bounds[1]])
        new_text.append("</span>")
        index = bounds[1]

      new_text.append(text[index:])
      text = ''.join(new_text)

    self.textView.get_buffer().set_text(text)

    return

  def get_link_bounds(self, text):
    """
    Return a list of tuples corresponding to the start and end points of links in
    the supplied string.
    """
    text = text.lower()
    bounds = []

    for match in re.finditer(r"\b(" #start at beginging of word
                                +r"(?:https?://|www\.)" #http:// https:// or www.
                                +r"(?:[a-z0-9][a-z0-9_\-]*[a-z0-9]\.)*" #Domains
                                +r"(?:[a-z][a-z0-9\-]*[a-z0-9])" #TLD
                                +r"(?:/([a-z0-9$_.+\\*'(),;:@&=\-]|%[0-9a-f]{2})*)*" #path
                                +r"(?:\?([a-z0-9$_.+!*'(),;:@&=\-]|%[0-9a-f]{2})*)?" #query
                                +r")", text):
      bounds.append(match.span())

    return bounds

  def render_whitespace(self, desc):
    ''' Render the line wrapping in desc as follows:

    * Newlines followed by 0-1 spaces are ignored, and 1 space is used.
    * Blank lines start new paragraphs.
    * Newlines followed by more than 1 space are honoured.
    '''

    text = []
    para = False
    literal = False

    for line in desc.split("\n"):

      if line == "" or line.isspace(): #Blank line, start paragraph
        if literal: #if following a literal line add newlines
          text.append("\n")
        para = True

      elif line[0] == " " and line[1] == " ": # >1 space, treat literaly
        text.append("\n")
        text.append(line)
        para = False
        literal = True

      else: #normal case
        if para: #add if starting a new paragraph
          text.append("\n   ")
        if not line.startswith(" "):
          text.append(" ")
        text.append(line)
        para = False
        literal = False

    return ''.join(text)

  def get_hyperlink(self, x, y):
    """
    Given an x and y window position (eg from a mouse click) return the hyperlink
    at that position if there is one. Else return None.
    """
    if self.text is None:
      return None

    buffer_pos = self.textView.window_to_buffer_coords(gtk.TEXT_WINDOW_TEXT, x, y)
    char_offset = self.textView.get_iter_at_location(buffer_pos[0], buffer_pos[1]).get_offset()

    for bounds in self.link_bounds:
      if char_offset >= bounds[0] and char_offset <= bounds[1]:
        return self.text[bounds[0]:bounds[1]]

    return None

  def mouse_over(self, widget, event):
    """
    Called when the mouse moves over the node description widget. Sets the cursor
    to a hand if the mouse hovers over a link.

    Based on code from HyperTextDemo class in hypertext.py from PyGTK 2.12 demos
    """

    if self.get_hyperlink(int(event.x), int(event.y)) is not None:
      self.textView.get_window(gtk.TEXT_WINDOW_TEXT).set_cursor(gtk.gdk.Cursor(gtk.gdk.HAND2))
    else:
      self.textView.get_window(gtk.TEXT_WINDOW_TEXT).set_cursor(gtk.gdk.Cursor(gtk.gdk.XTERM))

    return

  def mouse_button_release(self, widget, event):
    """
    Called when a mouse button is released over the node description widget.
    Launches a browser if the mouse release was over a link, the left mouse button
    was released and no text was selected.

    Based on code from HyperTextDemo class in hypertext.py from PyGTK 2.12 demos
    """

    if not event.button == 1:
      return

    if self.textView.get_buffer().get_selection_bounds():
      return

    hyperlink = self.get_hyperlink(int(event.x), int(event.y))
    if hyperlink is not None:
      webbrowser.open(hyperlink)

    return
