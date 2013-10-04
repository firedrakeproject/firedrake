#!/usr/bin/env python

#    This file is part of dxdiff.
#
#    dxdiff is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    dxdiff is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Diamond.  If not, see <http://www.gnu.org/licenses/>.

from lxml import etree

class EditScript:

  def __init__(self):
    self.script = []

  def __str__(self):
    return etree.tostring(self.to_xml(), pretty_print = True)

  def __len__(self):
    return len(self.script)

  def __getitem__(self, key):
    return self.script[key]

  def __iter__(self):
    return self.script.__iter__()

  def update(self, path, value, userdata = None):
    self.script.append({ "type": "update",
                         "location": path,
                         "value": value,
                         "userdata": userdata })

  def insert(self, path, index, tag, value = None, userdata = None):
    self.script.append({ "type": "insert",
                         "location": path,
                         "index": index,
                         "value": tag + (" " + value if value is not None else ""),
                         "userdata": userdata})

  def delete(self, path, userdata = None):
    self.script.append({ "type": "delete",
                         "location": path,
                         "userdata": userdata})

  def move(self, path, destination, index, userdata = None):
    self.script.append({ "type": "move",
                         "location": path,
                         "index": index,
                         "value": destination,
                         "userdata": userdata })

  def to_xml(self):
    tree = etree.Element("xmldiff")

    for edit in self.script:
      node = etree.Element(edit["type"], location = edit["location"])
      if "index" in edit:
        node.attrib["index"] = edit["index"]
      if edit["userdata"] is not None:
        node.attrib["userdata"] = edit["userdata"]

      if "value" in edit:
        node.text = edit["value"]
      tree.append(node)

    return etree.ElementTree(tree)

  def write(self, path):
    self.to_xml().write(path, pretty_print = True, xml_declaration = True, encoding = "utf-8")
