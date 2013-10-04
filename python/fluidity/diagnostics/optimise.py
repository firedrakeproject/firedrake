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

import unittest

import fluidity.diagnostics.debug as debug

_debugging = False


def EnableDebugging():
    global _debugging

    _debugging = True

    debug.dprint("Enabled debugging")

    return


def DisableDebugging():
    global _debugging

    _debugging = False

    debug.dprint("Disabled debugging")

    return


def DebuggingEnabled():
    global _debugging

    return _debugging


def PsycoSupport():
    try:
        import psyco  # noqa
        return True
    except ImportError:
        return False


def EnablePsyco():
    if PsycoSupport():
        psyco.full()  # noqa
        debug.dprint("Enabled psyco specialising compiler")
    return


def EnableAll():
    EnablePsyco()
    DisableDebugging()
    return


class optimiseUnittests(unittest.TestCase):

    def testPsycoSupport(self):
        try:
            import ctypes
        except ImportError:
            return

        if ctypes.sizeof(ctypes.c_voidp) == 4:
            import psyco  # noqa: testing
            self.assertTrue(PsycoSupport())

        return

    def testEnableDebugging(self):
        global _debugging

        debugging = _debugging
        EnableDebugging()
        self.assertTrue(_debugging)
        if debugging:
            EnableDebugging()
        else:
            DisableDebugging()

        return

    def testDisableDebugging(self):
        global _debugging

        debugging = _debugging
        DisableDebugging()
        self.assertFalse(_debugging)
        if debugging:
            EnableDebugging()
        else:
            DisableDebugging()

        return
