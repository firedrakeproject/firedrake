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

def flatten(l):
  """
  Flattens a list of lists into a list.
  """
  return [item for sublist in l for item in sublist]

def nub(l, reverse=False):
  """
  Removes duplicates from a list.
  If reverse is true keeps the last duplicate item
  as opposed to the first.
  """
  if reverse:
    seen = {}
    result = []
    for item in reversed(l):
      if item in seen: continue
      seen[item] = 1
      result.append(item)
    return reversed(result)
  else:
    seen = {}
    result = []
    for item in l:
      if item in seen: continue
      seen[item] = 1
      result.append(item)
    return result

def partial(fn, *cargs, **ckwargs):
  """
  Partial function application, taken from PEP 309.
  """
  ckwargs = ckwargs.copy()
  def call_fn(*fargs, **fkwargs):
    d = ckwargs
    d.update(fkwargs)
    return fn(*(cargs + fargs), **d)
  return call_fn

def irange(*args):
  """
  Similar to range but stop is an inclusive upper bound.
  """
  if len(args) == 0:
    raise TypeError("irange expected at least 1 arguments, got 0")
  elif len(args) == 1:
    stop = args[0]
    start = 0
    step = 1
  elif len(args) == 2:
    start, stop = args
    step = 1
  elif len(args) == 3:
    start, stop, step = args
  else:
    raise TypeError("irange expected at most 3 arguments, got " + str(len(args)))

  if step == 0:
    raise ValueError("irange() step argument must not be zero")

  stop = stop + 1 if step > 0 else stop - 1
  return range(start, stop, step)

##################
### Unit Tests ###
##################

import unittest

class __Test_flatten(unittest.TestCase):
  def test_type(self):
    self.assertRaises(TypeError, flatten, 1, 2, 3)

  def test_zero(self):
    self.assertEqual(flatten([]), [])
    self.assertEqual(flatten([[]]), [])
    self.assertEqual(flatten([[], []]), [])

  def test_one(self):
    self.assertEqual(flatten([[1]]), [1])
    self.assertEqual(flatten([[1, 2], [3, 4]]), [1, 2, 3, 4])

  def test_two(self):
    self.assertEqual(flatten([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), [[1, 2], [3, 4], [5, 6], [7, 8]])

class __Test_nub(unittest.TestCase):
  def test_zero(self):
    self.assertEqual(nub([]), [])

  def test_nodups(self):
    self.assertEqual(nub([1, 2, 3, 4]), [1, 2, 3, 4])

  def test_dups(self):
    self.assertEqual(nub([1, 1, 2, 3, 4, 2]), [1, 2, 3, 4])

class __Test_partial(unittest.TestCase):
  def printer(*args, **kargs):
    result = []
    for arg in args:
      result.append(str(args))
    for k, v in kargs.items():
      result.append(str(k) + ": " + str(v))
    return ' '.join(result)

  def test_zero(self):
    self.assertEqual(partial(self.printer)(), self.printer())

  def test_args(self):
    self.assertEqual(partial(self.printer, 1)(), self.printer(1))
    self.assertEqual(partial(self.printer)(1), self.printer(1))

  def test_kargs(self):
    self.assertEqual(partial(self.printer, a = 0)(), self.printer(a = 0))
    self.assertEqual(partial(self.printer)(a = 0), self.printer(a = 0))

class __Test_irange(unittest.TestCase):
  def test_type(self):
    self.assertRaises(TypeError, irange)
    self.assertRaises(TypeError, irange, 0, 1, 2, 3)
    self.assertRaises(TypeError, irange, 0, 1, 2, 3, 4)

  def test_zerostep(self):
    self.assertRaises(ValueError, irange, 0, 0, 0)
    self.assertRaises(ValueError, irange, 1, 2, 0)

  def test_zero(self):
    self.assertEqual(irange(0), [0])

  def test_stop(self):
    self.assertEqual(irange(1), [0, 1])
    self.assertEqual(irange(2), [0, 1, 2])

  def test_start(self):
    self.assertEqual(irange(5, 10), [5, 6, 7, 8, 9, 10])

  def test_step(self):
    self.assertEqual(irange(0, 4, 2), [0, 2, 4])
    self.assertEqual(irange(0, 3, 2), [0, 2])

  def test_negative(self):
    self.assertEqual(irange(-10, -5), [-10, -9, -8, -7, -6, -5])
    self.assertEqual(irange(-5, 5), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

  def test_negstep(self):
    self.assertEqual(irange(5, 0, -1), [5, 4, 3, 2, 1, 0])
    self.assertEqual(irange(2, -2, -1), [2, 1, 0, -1, -2])

  def test_norange(self):
    self.assertEqual(irange(5, 0), [])
    self.assertEqual(irange(1, -1), [])
    self.assertEqual(irange(0, 5, -1), [])

if __name__ == "__main__":
  unittest.main()
