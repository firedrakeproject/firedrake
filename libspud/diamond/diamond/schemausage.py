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

from cStringIO import StringIO
from lxml import etree

from schema import Schema

RELAXNGNS = "http://relaxng.org/ns/structure/1.0"
RELAXNG = "{" + RELAXNGNS + "}"

def find_fullset(tree):
  """
  Given a schema tree pulls out xpaths for every element.
  """

  cache = set()

  def traverse(node):

    if node.tag == RELAXNG + "element" or (node.tag == RELAXNG + "choice" and all(n.tag != RELAXNG + "value" for n in node)):
      fullset.add(tree.getpath(node))

    elif node.tag == RELAXNG + "ref":
      query = '/t:grammar/t:define[@name="' + node.get("name") + '"]'
      if query in cache:
        return # we only need to add the reference once
      cache.add(query)
      node = tree.xpath('/t:grammar/t:define[@name="' + node.get("name") + '"]', namespaces={'t': RELAXNGNS})[0]

    for child in node:
      traverse(child)

  start = tree.xpath("/t:grammar/t:start", namespaces={'t': RELAXNGNS})[0]

  root = start[0]

  fullset = set()
  traverse(root)
  return fullset

def find_useset(tree):
  """
  Given a diamond xml tree pulls out scehama paths for every element and attribute.
  """

  def traverse(node):
    if node.active:
      useset.add(node.schemaname)

      for child in node.get_children():
        traverse(child)

  useset = set()
  traverse(tree)
  return useset

def find_unusedset(schema, paths):
  """
  Given the a diamond schema and a list of paths to xml files
  find the unused xpaths.
  """
  def traverse(node):
    if node.active:
      unusedset.discard(node.schemaname)

      for child in node.get_children():
        traverse(child)

  unusedset = find_fullset(schema.tree)

  for path in paths:
    try:
      tree = schema.read(path)
      traverse(tree)
    except IOError:
      pass

  return unusedset

def strip(tag):
  return tag[tag.index("}") + 1:]

def node_name(node):
  """
  Returns a name for this node.
  """
  tagname = node.get("name") if "name" in node.keys() else strip(node.tag)
  name = None

  for child in node:
    if child.tag == RELAXNG + "attribute":
      if "name" in child.keys() and child.get("name") == "name":
        for grandchild in child:
          if grandchild.tag == RELAXNG + "value":
            name = " (" + grandchild.text + ")"
            break
    if name:
      break
  return tagname + (name if name else "")
