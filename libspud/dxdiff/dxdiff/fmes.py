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
"""
Diff xml trees using a modification of FMES [http://infolab.stanford.edu/pub/papers/tdiff3-8.ps]
"""

from lxml import etree
from collections import deque
from bimap import Bimap
from editscript import EditScript

import lcs
import utils

class Dom:
  def __init__(self, tag, value, parent, attribute = False):
    self.tag = tag
    self.value = value
    self.parent = parent
    self.children = []

    if value is None:
      self.typetag = "/Element"
    elif attribute:
      self.typetag = "/Attribute"
    else:
      self.typetag = "/Text"

    if parent:
      self.depth = parent.depth + 1
    else:
      self.depth = 0

    self.inorder = False

  def elements(self):
    return [child for child in self.children if child.is_element()]

  def attributes(self):
    return [child for child in self.children if child.is_attribute()]

  def text(self):
    return [child for child in self.children if child.is_text()]

  def is_element(self):
    return self.typetag == "/Element"

  def is_text(self):
    return self.typetag == "/Text"

  def is_attribute(self):
    return self.typetag == "/Attribute"

  def __repr__(self):
    return "<" + self.label() + ">" + (self.value or "")

  def __str__(self, indent = ""):
    title = indent + "<" + self.path() + ">" + (self.value or "") + "\n" + indent
    children = ("\n" + indent).join(child.__str__(indent + "  ") for child in self.children)
    return title + children

  def path(self):
    """
    Finds the path of this element.
    """
    if self.is_text():
      return self.parent.path() + "/text()"

    if self.is_attribute():
      return self.parent.path() + "/@" + self.tag

    if self.parent:
      siblings = [sibling for sibling in self.parent.elements() if sibling.tag == self.tag]
      if len(siblings) != 1:
        index = "[" + str(siblings.index(self) + 1) + "]"
      else:
        index = ""
      return self.parent.path() + "/" + self.tag + index
    else:
      return "/" + self.tag


  def find(self, path):

    if self.is_text():
      if path == "/text()":
        return self
      else: return None

    if self.is_attribute():
      if path == "/@" + self.tag:
        return self
      else: return None

    index = path.find("/", 1)
    if index == -1:
      index = len(path)

    root = path[:index]
    path = path[index:]

    if self.parent:
      siblings = [sibling for sibling in self.parent.elements() if sibling.tag == self.tag]
      if len(siblings) != 1:
        index = "[" + str(siblings.index(self) + 1) + "]"
      else:
        index = ""

      if root != "/" + self.tag + index:
        return None
    else:
      if root != "/" + self.tag:
        return None

    if path:
      for child in self.children:
        result = child.find(path)
        if result:
          return result
    else:
      return self

  def _real_index(self, parent, index):
    if index == 0:
      return 0

    elements = parent.elements()
    if len(elements) < index:
      return len(parent.children)
    return parent.children.index(elements[index - 1])

  def insert(self, tag, tagtype, value, path, index):
    parent = self.find(path)

    node = Dom(tag, value, parent, tagtype == "/Attribute")
    parent.children.insert(self._real_index(parent, index), node)
    return node

  def update(self, path, value):
    node = self.find(path)

    node.value = value
    return node

  def move(self, from_path, to_path, index):
    node = self.find(from_path)
    node.parent.children.remove(node)

    parent = self.find(to_path)
    parent.children.insert(self._real_index(parent, index), node)
    node.parent = parent

  def delete(self, path):
    node = self.find(path)
    node.parent.children.remove(node)
    node.parent = None

def _get_text(tree):
  """
  Returns the text and child tails.
  """
  return "".join([tree.text or ""] + [child.tail or "" for child in tree]).strip()

def dom(tree = None, parent = None):
  node = Dom(tree.tag, None, parent)

  text = _get_text(tree)
  if text:
    text = Dom(tree.tag, text, node)
    node.children.append(text)

  for key, value in tree.items():
    attr = Dom(key, value, node, True)
    node.children.append(attr)

  for child in tree:
    node.children.append(dom(child, node))

  return node

def get_leaf_nodes(tree):
  """
  Gets all the leaf nodes of an xml tree.
  """
  if tree.children:
    return utils.flatten([get_leaf_nodes(child) for child in tree.children])
  else:
    return [tree]

def get_parent_nodes(tree):
  """
  Returns all the non leaf nodes of an xml tree.
  """

  if tree.children:
    return utils.flatten([get_parent_nodes(child) for child in tree.children]) + [tree]
  else:
    return []

def get_depth(tree):
  """
  Returns the maximum depth of a tree.
  """
  depth = tree.depth
  for child in tree.children:
    depth = max(depth, get_depth(child))
  return depth

def get_depth_nodes(tree, depth):
  """
  Gets all the nodes of a certain depth of an xml tree.
  """
  if tree.depth == depth:
    return [tree]
  else:
    if tree.children:
      return utils.flatten([get_depth_nodes(child, depth) for child in tree.children])
    else:
      return []

def get_chain(nodes, label):
  return [node for node in nodes if node.label == label]

def compare_value(value1, value2):
  if value1 is None and value2 is None:
    return 0.0
  if value1 is None or value2 is None:
    return 1.0
  return 1.0 - (float(len(lcs.lcs(lcs.path(value1, value2)))) / max(len(value1), len(value2)))

def leaf_equal(f, M, l1, l2):
  return compare_value(l1.value, l2.value) <= f

def common(children1, children2, M):
  return [(x, y) for (x, y) in M if x in children1 and y in children2]

def compare_children(children1, children2, M):
  return (float(len(common(children1, children2, M))) / max(len(children1), len(children2)))

def node_equal(t, M, n1, n2):
  return compare_children(n1.children, n2.children, M) > t

def depth_equal(f, t, M, n1, n2):
  if n1.label != n2.label:
    return False #labels must match

  if "[" not in n1.path() and "[" not in n2.path():
    return True #if paths are not indexed and match then match the nodes

  if n1.children or n2.children:
    return node_equal(t, M, n1, n2)
  else:
    return leaf_equal(f, M, n1, n2)

def _match(nodes1, nodes2, M, equal):
  nodes = nodes1 + nodes2
  for label in utils.nub([node.label for node in nodes]):

    s1 = get_chain(nodes1, label)
    s2 = get_chain(nodes2, label)

    path = lcs.lcs(lcs.path(s1, s2, equal))

    for x, y in path:
      M.add((s1[x], s2[y]))
    for x, y in reversed(path):
      s1.pop(x)
      s2.pop(y)

    for x in range(len(s1)):
      for y in range(len(s2)):
        if equal(s1[x], s2[y]):
          M.add((s1[x], s2[y]))
          s2.pop(y)
          break

def label(tree):
  """
  Strips out indexers from the paths.
  """

  for node in depth_iter(tree):
    path = node.path()
    while True:
      lindex = path.find("[")
      if lindex == -1:
        break
      rindex = path.find("]", lindex)
      path = path[:lindex] + path[rindex + 1:]
    node.label = path

def fastmatch(t1, t2):
  """
  Calculates a match between t1 and t2.
  See figure 10 in reference.
  """
  M = Bimap()

  label(t1)
  label(t2)
  depth = max(get_depth(t1), get_depth(t2))

  while 0 <= depth:
    nodes1 = get_depth_nodes(t1, depth)
    nodes2 = get_depth_nodes(t2, depth)

    equal = utils.partial(depth_equal, 0.6, 0.5, M)

    _match(nodes1, nodes2, M, equal)

    depth -= 1

  return M

def depth_iter(tree):
  Q = []
  Q.append(tree)
  while Q:
    t = Q.pop()
    for child in t.children:
      Q.append(child)
    yield t

def breadth_iter(tree):
  Q = deque()
  Q.append(tree)
  while Q:
    t = Q.popleft()
    if t is not tree:
      yield t
    if t.parent is not None or t is tree: #check we haven't deleted it
      for child in t.children:
        Q.append(child)

def postorder_iter(tree):
  S = []
  O = []
  S.append(tree)
  while S:
    t = S.pop()
    O.append(t)
    for child in t.children:
      S.append(child)
  while O:
    t = O.pop()
    if t is not tree:
      yield t

def editscript(t1, t2):
  """
  Finds an editscript between t1 and t2.
  See figure 8 in reference.
  """

  E = EditScript()
  M = fastmatch(t1, t2)

  M.add((t1, t2))
  alignchildren(t1, t2, M, E, t1, t2)

  for x in breadth_iter(t2):
    y = x.parent
    z = M.right[y]

    if x not in M.right:
      if x.typetag == "/Text": #Can't insert Text, do an update
        E.update(z.path(), x.value)
        w = t1.insert(x.tag, x.typetag, x.value, z.path(), 0)
        M.add((w, x))
      else:
        x.inorder = True
        k = findpos(M, x)
        E.insert(z.path(), str(k), x.tag, x.value)
        w = t1.insert(x.tag, x.typetag, x.value, z.path(), k)
        M.add((w, x))
    else: # y is not None:
      w = M.right[x]
      v = w.parent
      if w.value != x.value:
        E.update(w.path(), x.value)
        t1.update(w.path(), x.value)
      if (v, y) not in M:
        x.inorder = True
        k = findpos(M, x)
        E.move(w.path(), z.path(), str(k))
        t1.move(w.path(), z.path(), k)

    alignchildren(t1, t2, M, E, w, x)

  for w in breadth_iter(t1):
    if w not in M.left:
      if w.typetag == "/Text": #Can't delete Text, do an update
        E.update(w.path(), "")
        t1.update(w.path(), "")
      else:
        E.delete(w.path())
        t1.delete(w.path())

  return E

def alignchildren(t1, t2, M, E, w, x):
  """
  See figure 9 in reference.
  """

  for c in w.elements():
    c.inorder = False
  for c in x.elements():
    c.inorder = False

  s1 = [child for child in w.elements() if child in M.left and M.left[child].parent == x]
  s2 = [child for child in x.elements() if child in M.right and M.right[child].parent == w]

  def equal(a, b):
    return (a, b) in M

  S = [(s1[x], s2[y]) for x, y in lcs.lcs(lcs.path(s1, s2, equal))]
  for (a, b) in S:
    a.inorder = b.inorder = True

  for a in s1:
    for b in s2:
      if (a, b) in M and (a, b) not in S:
        k = findpos(M, b)
        E.move(a.path(), w.path(), k)
        t1.move(a.path(), w.path(), str(k))
        a.inorder = b.inorder = True

def findpos(M, x):
  """
  See figure 9 in reference.
  """
  if x.is_text() or x.is_attribute():
    return 0

  y = x.parent
  children = y.elements()

  #find the rightmost inorder node left of x (v)
  index = children.index(x)
  v = None
  for i in range(index):
    c = children[i]
    if c.inorder:
      v = c

  if v is None:
    return 1

  u = M.right[v]
  children = u.parent.elements()
  index = children.index(u) + 1
  return index + 1

def diff(tree1, tree2):
  t1 = dom(tree1.getroot())
  t2 = dom(tree2.getroot())
  E = editscript(t1, t2)
  return E
