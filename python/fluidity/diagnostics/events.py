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

"""
Event handling tools
"""

import unittest


class Evented:

    """
    A base class defining an object which has events
    """

    def __init__(self, eventNames):
        self._handlers = {}
        for name in eventNames:
            self._handlers[name] = []

        return

    def RegisterEventHandler(self, name, handler):
        self._handlers[name].append(handler)

        return

    def UnregisterEventHandler(self, name, handler):
        self._handlers[name].remove(handler)

        return

    def _RaiseEvent(self, name, *args, **namedArgs):
        for handler in self._handlers[name]:
            handler(*args, **namedArgs)

        return


class eventsUnittests(unittest.TestCase):

    def testEvented(self):
        class TestEvented(Evented):

            def __init__(self):
                Evented.__init__(self, ["event1", "event2"])

                self.Reset()

                self.RegisterEventHandler("event1", self.OnEvent1)
                self.RegisterEventHandler("event2", self.OnEvent2)

                return

            def OnEvent1(self):
                self._event1Handled = True

                return

            def OnEvent2(self, arg):
                self._event2Handled = True
                self._event2Arg = arg

                return

            def RaiseEvent1(self):
                self._RaiseEvent("event1")

                return

            def RaiseEvent2(self, arg):
                self._RaiseEvent("event2", arg)

                return

            def Event1Handled(self):
                return self._event1Handled

            def Event2Handled(self):
                return self._event2Handled

            def GetEvent2Arg(self):
                return self._event2Arg

            def Reset(self):
                self._event1Handled = False
                self._event2Handled = False
                self._event2Arg = None

                return

        test = TestEvented()
        self.assertFalse(test.Event1Handled())
        self.assertFalse(test.Event2Handled())
        self.assertEquals(test.GetEvent2Arg(), None)

        test.RaiseEvent1()
        self.assertTrue(test.Event1Handled())
        self.assertFalse(test.Event2Handled())
        self.assertEquals(test.GetEvent2Arg(), None)

        test.RaiseEvent2(0)
        self.assertTrue(test.Event1Handled())
        self.assertTrue(test.Event2Handled())
        self.assertEquals(test.GetEvent2Arg(), 0)

        test.RaiseEvent2(1)
        self.assertTrue(test.Event1Handled())
        self.assertTrue(test.Event2Handled())
        self.assertEquals(test.GetEvent2Arg(), 1)

        test.Reset()
        self.assertFalse(test.Event1Handled())
        self.assertFalse(test.Event2Handled())
        self.assertEquals(test.GetEvent2Arg(), None)

        test.UnregisterEventHandler("event1", test.OnEvent1)
        test.RaiseEvent1()
        self.assertFalse(test.Event1Handled())
        self.assertFalse(test.Event2Handled())
        self.assertEquals(test.GetEvent2Arg(), None)

        test.RaiseEvent2(0)
        self.assertFalse(test.Event1Handled())
        self.assertTrue(test.Event2Handled())
        self.assertEquals(test.GetEvent2Arg(), 0)

        return
