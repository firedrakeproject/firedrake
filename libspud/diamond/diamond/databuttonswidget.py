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

class DataButtonsWidget(gtk.HBox):

  __gsignals__ = { "revert" : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, ()),
                   "store"  : (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE, ())}

  def __init__(self):
    gtk.HBox.__gobject_init__(self)
    revertButton = gtk.Button()
    revertButton.set_label("Revert data")
    revertButton.connect("clicked", self._revert)

    storeButton = gtk.Button()
    storeButton.set_label("Store data")
    storeButton.connect("clicked", self._store)

    self.pack_start(revertButton)
    self.pack_end(storeButton)

    return

  def _revert(self, widget = None):
    self.emit("revert")

  def _store(self, widget = None):
    self.emit("store")

gobject.type_register(DataButtonsWidget)
