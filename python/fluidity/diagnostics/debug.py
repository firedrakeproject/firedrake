#!/usr/bin/env python

# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301  USA

import sys
import unittest

"""
Debugging output and error routines
"""

_debugLevel = 2


def GetDebugLevel():
    """
    Get the current debug level
    """

    global _debugLevel

    return _debugLevel


def SetDebugLevel(level):
    """
    Set the current debug level
    """

    global _debugLevel

    _debugLevel = max(min(level, 3), 0)

    return


def dprint(msg, level=1, newline=True, flush=True):
    """Write a message to standard output if the supplied level is below or
    equal to the current debug level."""

    dwrite(sys.stdout, msg, level, newline, flush)

    return


def deprint(msg, level=1, newline=True, flush=True):
    """Write a message to standard error if the supplied level is below or
    equal to the current debug level."""

    dwrite(sys.stderr, msg, level, newline, flush)

    return


def dwrite(stream, msg, level=1, newline=True, flush=True):
    """
    Write a message to the supplied stream if the supplied level is below or
    equal to the current debug level
    """

    global _debugLevel

    if level <= max(min(_debugLevel, 3), 0):
        stream.write(str(msg))
        if newline:
            stream.write("\n")
        if flush:
            stream.flush()

    return


def FatalError(message):
    """Send an error message to standard error and terminate with a non-zero
    return value."""

    deprint(message, 0)

    sys.exit(1)


class debugUnittests(unittest.TestCase):

    def testGetDebugLevel(self):
        global _debugLevel

        self.assertEquals(GetDebugLevel(), _debugLevel)

        return

    def testSetDebugLevel(self):
        oldDebugLevel = GetDebugLevel()
        SetDebugLevel(-1)
        self.assertEquals(GetDebugLevel(), 0)
        SetDebugLevel(1)
        self.assertEquals(GetDebugLevel(), 1)
        SetDebugLevel(10)
        self.assertEquals(GetDebugLevel(), 3)
        SetDebugLevel(oldDebugLevel)

        return

    def testDwrite(self):
        class DummyStream:

            def __init__(self):
                self._written = False

                return

            def flush(self, *args):
                return

            def write(self, *args):
                self._written = True

                return

            def Reset(self):
                self._written = False

                return

            def Written(self):
                return self._written

        oldDebugLevel = GetDebugLevel()
        SetDebugLevel(1)

        stream = DummyStream()
        dwrite(stream, "", 0)
        self.assertTrue(stream.Written())

        stream.Reset()
        dwrite(stream, "", 1)
        self.assertTrue(stream.Written())

        stream.Reset()
        dwrite(stream, "", 2)
        self.assertFalse(stream.Written())

        stream.Reset()
        dwrite(stream, "", 3)
        self.assertFalse(stream.Written())

        SetDebugLevel(oldDebugLevel)

        return
