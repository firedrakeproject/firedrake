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
File handling routines
"""

import glob
import os
import shutil
import stat
import tempfile
import unittest

import fluidity.diagnostics.utils as utils


def FileExtension(filename):
    """
    Return the file extension of the supplied filename (including the ".")
    """

    split = os.path.basename(filename).split(".")

    if len(split) == 1:
        return ""
    else:
        return "." + split[-1]


def StripFileExtension(filename):
    """
    Strip the file extension from the supplied filename
    """

    split = os.path.basename(filename).split(".")

    if len(split) == 1:
        return filename
    else:
        return os.path.join(os.path.dirname(filename),
                            utils.FormLine(split[:-1], delimiter=".",
                                           newline=False))


def Touch(filename):
    """
    Update timestamp for given file (or create it if it doesn't exist)
    """

    if FileExists(filename):
        os.utime(filename, None)
    else:
        fileHandle = open(filename, "w")
        fileHandle.flush()
        fileHandle.close()

    return


def Mkdir(path, parents=False):
    """Create a directory at the specified path. Make all parents if parents is
    True."""

    if parents:
        parentPath = ""
        for dir in path.split(os.path.sep):
            parentPath = os.path.join(parentPath, dir)
            Mkdir(parentPath, parents=False)
    else:
        if not FileExists(path) or not Isdir(path):
            os.mkdir(path)

    return


def Cp(donor, target):
    """
    Copy a file
    """

    shutil.copy(donor, target)

    return


def Move(donor, target):
    """
    Move a file
    """

    shutil.move(donor, target)

    return


def FileExists(filename):
    """
    Return whether the supplied file exists
    """

    try:
        os.stat(filename)
        return True
    except OSError:
        return False


def IsExecutable(filename):
    """
    Return whether the supplied file exists and is executable
    """

    if not FileExists(filename):
        return False
    else:
        return os.access(filename, os.X_OK)


def Isdir(path):
    """
    Return whether the supplied path represents a directory
    """

    return stat.S_ISDIR(os.stat(path)[0])


def Rm(path):
    """
    Delete the file at the specified path
    """

    os.remove(path)

    return


def Rmdir(path, force=False):
    """
    Delete the directory at the specified path. Include all of its contents
    if force is True.
    """

    if force:
        for file in glob.glob(os.path.join(path, "*")):
            if Isdir(file):
                Rmdir(file)
            else:
                Rm(file)
    os.rmdir(path)

    return


def FindAndReplace(filename, find, replace):
    """
    Find and replace text in the supplied file
    """

    # Could do this out-of-core

    inputHandle = open(filename, "r")
    inputString = inputHandle.read()
    inputHandle.close()

    inputString = inputString.replace(find, replace)

    outputHandle = open(filename, "w")
    outputHandle.write(inputString)
    outputHandle.close()

    return


class filehandlingUnittests(unittest.TestCase):

    def testFileExtension(self):
        self.assertEquals(FileExtension("a.b"), ".b")
        self.assertEquals(FileExtension("/one/two.three.four"), ".four")
        self.assertEquals(FileExtension("a"), "")
        self.assertEquals(FileExtension("../one.two"), ".two")

        return

    def testStripFileExtension(self):
        self.assertEquals(StripFileExtension("a.b"), "a")
        self.assertEquals(
            StripFileExtension("/one/two.three.four"), "/one/two.three")
        self.assertEquals(StripFileExtension("a"), "a")
        self.assertEquals(StripFileExtension("../one.two"), "../one")

        return

    def testTouch(self):
        tempDir = tempfile.mkdtemp()
        tempFile = os.path.join(tempDir, "test")
        self.assertFalse(FileExists(tempFile))
        Touch(tempFile)
        self.assertTrue(FileExists(tempFile))
        Rmdir(tempDir, force=True)

        return

    def testIsExecutable(self):
        self.assertTrue(FileExists(__file__))
        self.assertFalse(IsExecutable(__file__))

        tempDir = tempfile.mkdtemp()
        tempFile = os.path.join(tempDir, "test")
        Touch(tempFile)
        self.assertTrue(FileExists(tempFile))
        self.assertFalse(IsExecutable(tempFile))
        os.chmod(tempFile, stat.S_IXUSR)
        self.assertTrue(FileExists(tempFile))
        self.assertTrue(IsExecutable(tempFile))

        Rmdir(tempDir, force=True)

        return

    def testFileExists(self):
        self.assertTrue(FileExists(__file__))

        return

    def testIsdir(self):
        tempDir = tempfile.mkdtemp()
        self.assertTrue(Isdir(tempDir))
        Rmdir(tempDir, force=True)

        fileOsHandle, tempFile = tempfile.mkstemp()
        self.assertFalse(Isdir(tempFile))
        os.close(fileOsHandle)
        Rm(tempFile)

        return

    def testRmdir(self):
        tempDir = tempfile.mkdtemp()
        fileHandle = open(os.path.join(tempDir, "tempfile"), "w")
        fileHandle.close()
        os.mkdir(os.path.join(tempDir, "tempdir"))
        Rmdir(tempDir, force=True)
        self.assertFalse(FileExists(tempDir))

        return
