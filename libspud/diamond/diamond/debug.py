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

"""
Debugging module. Stores the program debug level and provides debugging output
functions. Only debugging output with level less than or equal to the current
debug level is output. Note that debugging output with level greater than the
current maximum debug level is treated as having a level equal to the current
maximum debug level, and that debugging output with level less than zero is
treated as having a level equal to zero.
"""

import sys

class DebugLevel:
  """
  Class used to store a debug level.
  """

  def __init__(self, level = 1, maxLevel = 3):
    """
    Initialise a debug level.
    """

    self.SetLevel(level)
    self.SetMaxLevel(maxLevel)

    return

  def GetLevel(self):
    """
    Get the debug level.
    """

    return self._level

  def SetLevel(self, level = 1):
    """
    Set the debug level.
    """

    level = max(level, 0)

    try:
      self._level = min(level, self.GetMaxLevel())
    except:
      self._level = level

    return

  def GetMaxLevel(self):
    """
    Get the maximum debug level.
    """

    return self._maxLevel

  def SetMaxLevel(self, maxLevel = 3):
    """
    Set the maximum debug level.
    """

    maxLevel = max(maxLevel, 0)

    self._maxLevel = maxLevel
    self.SetLevel(self._level)

    return

# Stores the current module debug level
_debugLevel = DebugLevel()

def GetDebugLevel():
  """
  Get the current debug level.
  """

  return _debugLevel.GetLevel()

def SetDebugLevel(level = 1):
  """
  Set the current debug level.
  """

  _debugLevel.SetLevel(level)

  return

def GetMaxDebugLevel():
  """
  Get the current maximum debug level.
  """

  return _debugLevel.GetMaxLevel()

def SetMaxDebugLevel(level = 3):
  """
  Set the current maximum debug level.
  """

  _debugLevel.SetMaxLevel(level)

  return

def dprint(msg, level = 1, newline = True, flush = True):
  """
  Print a debug message to standard output with supplied debug level.
  """

  dwrite(sys.stdout, msg, level, newline, flush)

  return

def deprint(msg, level = 1, newline = True, flush = True):
  """
  Print a debug message to standard error with supplied debug level.
  """

  dwrite(sys.stderr, msg, level, newline, flush)

  return

def dwrite(stream, msg, level = 1, newline = True, flush = True):
  """
  Print a debug message to the supplied file stream with supplied debug level.
  """

  level = max(level, 0)
  level = min(level, GetMaxDebugLevel())

  if level <= GetDebugLevel():
    stream.write(str(msg))
    if newline:
      stream.write("\n")
    if flush:
      stream.flush()

  return
