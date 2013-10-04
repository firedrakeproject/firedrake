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

# Bibliography:
#
# Python and XML: An Introduction
# http://www.boddie.org.uk/python/XML_intro.html

"""
Finite element halo classes
"""

import os
import tempfile
import unittest

import fluidity.diagnostics.debug as debug

try:
    import xml.dom.ext
except ImportError:
    pass
try:
    import xml.dom.minidom
except ImportError:
    debug.deprint("Warning: Failed to import xml.dom.minidom module")

import fluidity.diagnostics.filehandling as filehandling
import fluidity.diagnostics.utils as utils


def XmlSupport():
    try:
        import xml.dom.minidom  # noqa: testing
        return True
    except ImportError:
        return False


def XmlExtSupport():
    try:
        import xml.dom.ext  # noqa: testing
        return True
    except ImportError:
        return False


def HaloIOSupport():
    return XmlSupport()


class Halo:

    """
    A halo.
    """

    def __init__(self, process, nProcesses, nOwnedNodes=None, sends=None,
                 receives=None):
        assert(process >= 0)
        assert(nProcesses > 0)

        self._process = process
        self._nProcesses = nProcesses
        self._nOwnedNodes = None

        if not nOwnedNodes is None:
            self.SetNOwnedNodes(nOwnedNodes)
        if not sends is None:
            self.SetSends(sends)
        else:
            self._sends = [[] for i in range(nProcesses)]
        if not receives is None:
            self.SetReceives(receives)
        else:
            self._receives = [[] for i in range(nProcesses)]

        return

    def GetProcess(self):
        return self._process

    def GetNProcesses(self):
        return self._nProcesses

    def HasNOwnedNodes(self):
        return not self._nOwnedNodes is None

    def GetNOwnedNodes(self):
        assert(self.HasNOwnedNodes())

        return self._nOwnedNodes

    def SetNOwnedNodes(self, nOwnedNodes):
        assert(nOwnedNodes >= 0)

        self._nOwnedNodes = nOwnedNodes

        return

    def SendCount(self, process):
        return len(self._sends[process])

    def ReceiveCount(self, process):
        return len(self._receives[process])

    def GetSend(self, process, index):
        return self._sends[process][index]

    def GetReceive(self, process, index):
        return self._receives[process][index]

    def GetSends(self, process=None):
        if process is None:
            return self._sends
        else:
            return self._sends[process]

    def GetReceives(self, process=None):
        if process is None:
            return self._receives
        else:
            return self._receives[process]

    def AddSend(self, process, send):
        assert(send >= 0)

        self._sends[process].append(send)

    def AddReceive(self, process, receive):
        assert(receive >= 0)

        self._receives[process].append(receive)

    def SetSend(self, process, index, send):
        assert(send >= 0)

        self._sends[process][index] = send

        return

    def SetReceive(self, process, index, receive):
        assert(receive >= 0)

        self._receives[process][index] = receive

        return

    def SetSends(self, sends, process=None):
        if process is None:
            nProcesses = self.GetNProcesses()
            assert(len(sends) == nProcesses)

            self._sends = [[None for i in range(len(sends[i]))]
                           for i in range(nProcesses)]
            for i, sendArray in enumerate(sends):
                for j, send in enumerate(sendArray):
                    self.SetSend(i, j, send)
        else:
            self._sends[process] = [None for i in range(len(sends))]
            for i, send in enumerate(sends):
                self.SetSend(process, i, send)

        return

    def SetReceives(self, receives, process=None):
        if process is None:
            nProcesses = self.GetNProcesses()
            assert(len(receives) == nProcesses)

            self._receives = [[None for i in range(len(receives[i]))]
                              for i in range(nProcesses)]
            for i, receiveArray in enumerate(receives):
                for j, receive in enumerate(receiveArray):
                    self.SetReceive(i, j, receive)
        else:
            self._receives[process] = [None for i in range(len(receives))]
            for i, receive in enumerate(receives):
                self.SetReceive(process, i, receive)

        return

    def TrailingReceivesOrdered(self):
        """
        Return whether this halo is trailing receive ordered
        """

        # If we do not have a number of owned nodes, we can't be usefully
        # trailing receive ordered
        if not self.HasNOwnedNodes():
            return False

        nOwnedNodes = self.GetNOwnedNodes()

        # All sends must be owned
        for sends in self._sends:
            if len(sends) > 0 and max(sends) >= nOwnedNodes:
                return False
        # All receives must be non-owned
        for receives in self._receives:
            if len(receives) > 0 and min(receives) < nOwnedNodes:
                return False
        # All receives must be unique and consecutive
        allReceives = utils.ExpandList(self._receives)
        allReceives.sort()
        if not allReceives == range(nOwnedNodes, nOwnedNodes +
                                    len(allReceives)):
            return False

        return True


class Halos:
    """A set of halos, storing a fixed set of node and element halos (one per
    level), with None for missing halos."""

    def __init__(self, process, nProcesses, nodeHalos=[], elementHalos=[],
                 nLevels=2):
        self._process = process
        self._nProcesses = nProcesses
        self._nLevels = nLevels

        self.SetNodeHalos(nodeHalos)
        self.SetElementHalos(elementHalos)

        return

    def HaloCompatible(self, halo):
        return (halo.GetProcess() == self.GetProcess() and
                halo.GetNProcesses() == self.GetNProcesses())

    def GetProcess(self):
        return self._process

    def GetNProcesses(self):
        return self._nProcesses

    def GetNLevels(self):
        return self._nLevels

    def HasNodeHalo(self, level):
        return not self._nodeHalos[level - 1] is None

    def HasElementHalo(self, level):
        return not self._elementHalos[level - 1] is None

    def NodeHaloCount(self):
        halos = 0
        for i in range(1, self.GetNLevels() + 1):
            if self.HasNodeHalo(i):
                halos += 1

        return halos

    def ElementHaloCount(self):
        halos = 0
        for i in range(1, self.GetNLevels() + 1):
            if self.HasElementHalo(i):
                halos += 1

        return halos

    def HaloCount(self):
        return self.NodeHaloCount() + self.ElementHaloCount()

    def NodeHaloLevels(self):
        levels = []
        for i in range(1, self.GetNLevels() + 1):
            if self.HasNodeHalo(i):
                levels.append(i)

        return levels

    def ElementHaloLevels(self):
        levels = []
        for i in range(1, self.GetNLevels() + 1):
            if self.HasElementHalo(i):
                levels.append(i)

        return levels

    def GetNodeHalo(self, level):
        return self._nodeHalos[level - 1]

    def GetNodeHalos(self):
        return self._nodeHalos

    def SetNodeHalo(self, level, halo):
        assert(self.HaloCompatible(halo))

        self._nodeHalos[level - 1] = halo

        return

    def SetNodeHalos(self, halos):
        assert(len(halos) <= self.GetNLevels())

        self._nodeHalos = [None, None]
        for level, halo in enumerate(halos):
            self.SetNodeHalo(level + 1, halo)

        return

    def GetElementHalo(self, level):
        return self._elementHalos[level - 1]

    def GetElementHalos(self):
        return self._elementHalos

    def SetElementHalo(self, level, halo):
        assert(self.HaloCompatible(halo))

        self._elementHalos[level - 1] = halo

        return

    def SetElementHalos(self, halos):
        assert(len(halos) <= self.GetNLevels())

        self._elementHalos = [None, None]
        for level, halo in enumerate(halos):
            self.SetElementHalo(level + 1, halo)

        return

    def LevelHaloDict(self):
        """
        Return a level-halo dictionary, similar to FLComms halo storage
        """

        halos = {}
        for level in self.NodeHaloLevels():
            halos[level] = self.GetNodeHalo(level)
        for level in self.ElementHaloLevels():
            halos[-level] = self.GetElementHalo(level)

        return halos


def ReadHalos(filename):
    """
    Read a Fluidity .halo file
    """

    xmlFile = xml.dom.minidom.parse(filename)

    halosRootEle = xmlFile.getElementsByTagName("halos")
    assert(len(halosRootEle) == 1)
    halosRootEle = halosRootEle[0]
    haloProcess = int(halosRootEle.attributes["process"].nodeValue)
    nprocs = int(halosRootEle.attributes["nprocs"].nodeValue)

    halosEle = halosRootEle.getElementsByTagName("halo")
    halos = Halos(process=haloProcess, nProcesses=nprocs)
    for haloEle in halosEle:
        try:
            level = int(haloEle.attributes["level"].nodeValue)
        except KeyError:
            # Backwards compatibility
            level = int(haloEle.attributes["tag"].nodeValue)

        n_private_nodes = int(haloEle.attributes["n_private_nodes"].nodeValue)

        halo = Halo(process=haloProcess,
                    nProcesses=nprocs, nOwnedNodes=n_private_nodes)

        processes = []
        haloDataEles = haloEle.getElementsByTagName("halo_data")
        for haloDataEle in haloDataEles:
            process = int(haloDataEle.attributes["process"].nodeValue)
            assert(process >= 0 and process < nprocs)
            assert(not process in processes)
            processes.append(process)

            sendEle = haloDataEle.getElementsByTagName("send")
            assert(len(sendEle) == 1)
            sendEle = sendEle[0]
            sendEleChildren = sendEle.childNodes
            sends = None
            for child in sendEleChildren:
                if child.nodeType == child.TEXT_NODE:
                    sends = map(int, child.nodeValue.split())
                    break
            if not sends is None:
                if level > 0:
                    halo.SetSends(
                        utils.OffsetList(sends, -1), process=process)
                else:
                    halo.SetSends(sends, process=process)

            receiveEle = haloDataEle.getElementsByTagName("receive")
            assert(len(receiveEle) == 1)
            receiveEle = receiveEle[0]
            receiveEleChildren = receiveEle.childNodes
            receives = None
            for child in receiveEleChildren:
                if child.nodeType == child.TEXT_NODE:
                    receives = map(int, child.nodeValue.split())
                    break
            if not receives is None:
                if level > 0:
                    halo.SetReceives(
                        utils.OffsetList(receives, -1), process=process)
                else:
                    halo.SetReceives(receives, process=process)

        if level > 0:
            assert(not halos.HasNodeHalo(level))
            halos.SetNodeHalo(level, halo)
        else:
            assert(not halos.HasElementHalo(-level))
            halos.SetElementHalo(-level, halo)

    return halos


def WriteHalos(halos, filename):
    """
    Write a Fluidity .halo file
    """

    xmlfile = xml.dom.minidom.Document()

    halosRootEle = xmlfile.createElement("halos")
    halosRootEle.setAttribute("process", str(halos.GetProcess()))
    halosRootEle.setAttribute("nprocs", str(halos.GetNProcesses()))
    xmlfile.appendChild(halosRootEle)

    halos = halos.LevelHaloDict()
    for level in halos.keys():
        halo = halos[level]

        haloEle = xmlfile.createElement("halo")
        haloEle.setAttribute("level", str(level))
        haloEle.setAttribute("n_private_nodes", str(halo.GetNOwnedNodes()))
        halosRootEle.appendChild(haloEle)

        for i, process in enumerate(range(halo.GetNProcesses())):
            haloDataEle = xmlfile.createElement("halo_data")
            haloDataEle.setAttribute("process", str(i))
            haloEle.appendChild(haloDataEle)

            sendEle = xmlfile.createElement("send")
            haloDataEle.appendChild(sendEle)

            if level > 0:
                sendText = xmlfile.createTextNode(
                    utils.FormLine(utils.OffsetList(utils.ExpandList(halo.GetSends(process=i)), 1), delimiter=" ", newline=False))
            else:
                sendText = xmlfile.createTextNode(
                    utils.FormLine(utils.ExpandList(halo.GetSends(process=i)), delimiter=" ", newline=False))
            sendEle.appendChild(sendText)

            receiveEle = xmlfile.createElement("receive")
            haloDataEle.appendChild(receiveEle)

            if level > 0:
                receiveText = xmlfile.createTextNode(
                    utils.FormLine(utils.OffsetList(utils.ExpandList(halo.GetReceives(process=i)), 1), delimiter=" ", newline=False))
            else:
                receiveText = xmlfile.createTextNode(
                    utils.FormLine(utils.ExpandList(halo.GetReceives(process=i)), delimiter=" ", newline=False))
            receiveEle.appendChild(receiveText)

    handle = open(filename, "w")
    if XmlExtSupport():
        xml.dom.ext.PrettyPrint(xmlfile, handle)
    else:
        xmlfile.writexml(handle)
    handle.flush()
    handle.close()

    return


class mesh_halosUnittests(unittest.TestCase):

    def testXmlSupport(self):
        import xml.dom.minidom  # noqa: testing
        self.assertTrue(XmlSupport())
        return

    def testHaloTrailingReceivesOrdered(self):
        self.assertTrue(Halo(process=0, nProcesses=1, nOwnedNodes=2,
                        sends=[[0, 1]], receives=[[2, 3]]).TrailingReceivesOrdered())
        self.assertTrue(Halo(process=0, nProcesses=1, nOwnedNodes=2,
                        sends=[[1, 0]], receives=[[3, 2]]).TrailingReceivesOrdered())
        self.assertFalse(Halo(process=0, nProcesses=1, sends=[[0, 1]],
                         receives=[[2, 3]]).TrailingReceivesOrdered())
        self.assertFalse(Halo(process=0, nProcesses=1, nOwnedNodes=2,
                         sends=[[1, 2]], receives=[[2, 3]]).TrailingReceivesOrdered())
        self.assertFalse(Halo(process=0, nProcesses=1, nOwnedNodes=2,
                         sends=[[0, 1]], receives=[[1, 2]]).TrailingReceivesOrdered())
        self.assertFalse(Halo(process=0, nProcesses=1, nOwnedNodes=2,
                         sends=[[0, 1]], receives=[[3, 4]]).TrailingReceivesOrdered())
        self.assertFalse(Halo(process=0, nProcesses=1, nOwnedNodes=2,
                         sends=[[0, 1]], receives=[[2, 2]]).TrailingReceivesOrdered())

        return

    def testHalosIO(self):
        halo1 = Halo(process=0, nProcesses=1,
                     nOwnedNodes=3, sends=[[0, 1]], receives=[[3, 4]])
        halo2 = Halo(process=0, nProcesses=1, nOwnedNodes=3,
                     sends=[[0, 2, 1]], receives=[[3, 5, 4]])
        halos = Halos(process=0, nProcesses=1, nodeHalos=[halo1, halo2])

        tempDir = tempfile.mkdtemp()
        filename = os.path.join(tempDir, "halos")

        WriteHalos(halos, filename)
        halos = ReadHalos(filename)
        self.assertEquals(halos.NodeHaloLevels(), [1, 2])
        self.assertEquals(halos.ElementHaloLevels(), [])
        halo1, halo2 = halos.GetNodeHalos()
        self.assertEquals(halo1.GetProcess(), 0)
        self.assertEquals(halo2.GetProcess(), 0)
        self.assertEquals(halo1.GetNProcesses(), 1)
        self.assertEquals(halo2.GetNProcesses(), 1)
        self.assertEquals(halo1.GetSends(process=0), [0, 1])
        self.assertEquals(halo2.GetSends(process=0), [0, 2, 1])
        self.assertEquals(halo1.GetReceives(process=0), [3, 4])
        self.assertEquals(halo2.GetReceives(process=0), [3, 5, 4])

        filehandling.Rmdir(tempDir, force=True)

        return


class mesh_halosDataUnittests(unittest.TestCase):

    def testReadHalos(self):
        filename = os.path.join(
            os.path.dirname(__file__), os.path.pardir, "test-data", "CoarseCorner_0.halo")
        halos = ReadHalos(filename)
        self.assertEquals(halos.NodeHaloCount(), 2)
        self.assertEquals(halos.GetNodeHalo(1).GetNProcesses(), 4)
        self.assertEquals(halos.GetNodeHalo(2).GetNProcesses(), 4)

        return
