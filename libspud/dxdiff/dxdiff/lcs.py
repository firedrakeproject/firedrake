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
Find the LCS (Longest Common Subsequence). Uses [http://www.xmailserver.org/diff2.pdf]
"""

from utils import irange

def __path(V, D, k):
  if D == 0:
    return [(xy, xy) for xy in irange(V[0][0])]

  tx = V[D][k]

  if k == -D or (k != D and V[D][k - 1] < V[D][k + 1]):
    x = V[D][k + 1]
    y = x - (k + 1)
    k = k + 1
    y = y + 1
  else:
    x = V[D][k - 1]
    y = x - (k - 1)
    k = k - 1
    x = x + 1

  return __path(V, D - 1, k) + [(x + d, y + d) for d in irange(tx - x)]

def __eq(a, b): return a == b

def path(a, b, eq = __eq):
  """
  Finds the path through the match grid of sequence a and b,
  using the function eq to determine equality.
  Returns path
  """

  m = len(a)
  n = len(b)
  mn = m + n

  if mn == 0: # two empty sequences
    return [(0, 0)]

  Vd = []
  V = {1: 0}

  for D in irange(mn):
    for k in irange(-D, D, 2):
      if k == -D or (k != D and V[k - 1] < V[k + 1]):
        x = V[k + 1]
      else:
        x = V[k - 1] + 1
      y = x - k

      while x < m and y < n and eq(a[x], b[y]):
        x += 1
        y += 1

      V[k] = x

      if x >= m and y >= n:
        Vd.append(V.copy())
        return __path(Vd, D, k)

    Vd.append(V.copy())

  raise Exception("lcs should not reach here")

def lcs(path):
  """
  Given an edit script path returns the longest common subseqence.
  """

  result = []

  for i in range(1, len(path)):
    x, y = path[i]
    px, py = path[i - 1]
    dx, dy = x - px, y - py
    if dx == 1 and dy == 1:
      result.append((px, py))

  return result

def ses(path, b):
  """
  Returns an edit script for a given match grid path.
  The edit script transforms sequence A of the match grid
  into sequence B via deletions ("D", index) and inserations
  ("I", A index, B value).
  """

  patch = []
  for i in range(len(path) - 1):
    x, y = path[i]
    nx, ny = path[i + 1]
    dx, dy = nx - x, ny - y
    if dx == 1 and dy == 1:
      pass #match
    elif dx == 1:
      patch.append(("D", x))
    else: #dy == 1:
      patch.append(("I", x, b[y]))

  return patch

def patch(patch, a):
  """
  Given a sequence and a patch from the ses function transforms a into b
  """

  seq = type(a)
  result = seq()
  i = 0

  for op in patch:
    while i < op[1]:
      result += seq(a[i])
      i += 1

    if op[0] == "D":
      i += 1
    else:
      result += seq(op[2])

  while i < len(a):
    result += seq(a[i])
    i += 1

  return result

##################
### Unit Tests ###
##################

import unittest

class __Test_lcs(unittest.TestCase):
  def test_zero(self):
    self.assertEqual(path("", ""), [(0, 0)])
    self.assertEqual(path("", "a"), [(0, 0), (0, 1)])
    self.assertEqual(path("a", ""), [(0, 0), (1, 0)])

  def test_single(self):
    self.assertEqual(path("a", "a"), [(0, 0), (1, 1)])
    self.assertEqual(path("a", "b"), [(0, 0), (1, 0), (1, 1)])

  def test_short(self):
    self.assertEqual(path("ab", "ab"), [(0, 0), (1, 1), (2, 2)])
    self.assertEqual(path("ab", "ac"), [(0, 0), (1, 1), (2, 1), (2, 2)])
    self.assertEqual(path("abcabba", "cbabac"), [(0, 0), (1, 0), (2, 0), (3, 1), (3, 2), (4, 3), (5, 4), (6, 4), (7, 5), (7, 6)])
    self.assertEqual(path("hello", "help me"), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7)])

  def test_long(self):
    self.assertEqual(path("hello", "night night"),
                        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 4),
                         (3, 4), (4, 4), (5, 4), (5, 5), (5, 6), (5, 7),
                         (5, 8), (5, 9), (5, 10), (5, 11)])

    self.assertEqual(path([
			"This part of the",
			"document has stayed the",
			"same from version to",
			"version.  It shouldn't",
			"be shown if it doesn't",
			"change.  Otherwise, that",
			"would not be helping to",
			"compress the size of the",
			"changes.",
                        "",
			"This paragraph contains",
			"text that is outdated.",
			"It will be deleted in the",
			"near future.",
                        "",
			"It is important to spell",
			"check this dokument. On",
			"the other hand, a",
			"misspelled word isn't",
			"the end of the world.",
			"Nothing in the rest of",
			"this paragraph needs to",
			"be changed. Things can",
			"be added after it."
			], [
			"This is an important",
			"notice! It should",
			"therefore be located at",
			"the beginning of this",
			"document!",
                        "",
			"This part of the",
			"document has stayed the",
			"same from version to",
			"version.  It shouldn't",
			"be shown if it doesn't",
			"change.  Otherwise, that",
			"would not be helping to",
			"compress anything.",
                        "",
			"It is important to spell",
			"check this document. On",
			"the other hand, a",
			"misspelled word isn't",
			"the end of the world.",
			"Nothing in the rest of",
			"this paragraph needs to",
			"be changed. Things can",
			"be added after it.",
                        "",
			"This paragraph contains",
			"important new additions",
			"to this document.",
			]), [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                             (0, 5), (0, 6), (1, 7), (2, 8), (3, 9),
                             (4, 10), (5, 11), (6, 12), (7, 13), (8, 13),
                             (9, 13), (9, 14), (10, 15), (11, 15), (12, 15),
                             (13, 15), (14, 15), (15, 15), (16, 16), (17, 16),
                             (17, 17), (18, 18), (19, 19), (20, 20), (21, 21),
                             (22, 22), (23, 23), (24, 24), (24, 25), (24, 26),
                             (24, 27), (24, 28)])

class __Test_diff(unittest.TestCase):
  def test_zero(self):
    p = path("", "")

    self.assertEqual(ses(p, ""), [])

  def test_delete(self):
    p = path("a", "")
    self.assertEqual(ses(p, ""), [("D", 0)])

    p = path("abcd", "")
    self.assertEqual(ses(p, ""), [("D", 0), ("D", 1), ("D", 2), ("D", 3)])

    p = path("abcd", "cd")
    self.assertEqual(ses(p, "cd"), [("D", 0), ("D", 1)])

    p = path("abcd", "ab")
    self.assertEqual(ses(p, "ab"), [("D", 2), ("D", 3)])

    p = path("abcd", "bc")
    self.assertEqual(ses(p, "bc"), [("D", 0), ("D", 3)])

  def test_insert(self):
    p = path("", "a")
    self.assertEqual(ses(p, "a"), [("I", 0, "a")])

    p = path("", "abcd")
    self.assertEqual(ses(p, "abcd"), [("I", 0, "a"), ("I", 0, "b"), ("I", 0, "c"), ("I", 0, "d")])

  def test_delins(self):
    p = path("abcd", "abef")
    self.assertEqual(ses(p, "abef"), [("D", 2), ("D", 3), ("I", 4, "e"), ("I", 4, "f")])

class __Test_patch(unittest.TestCase):
  def do_patch(self, a, b):
    self.assertEqual(patch(ses(path(a, b), b), a), b)

  def test_patch(self):
    self.do_patch("", "hello")
    self.do_patch("hello", "")
    self.do_patch("hello", "hello")
    self.do_patch("hello", "night night")
    self.do_patch("hello", "help me")
    self.do_patch("test bob", "kill bob")
    self.do_patch("test bill", "test jane")

if __name__ == "__main__":
  unittest.main()
