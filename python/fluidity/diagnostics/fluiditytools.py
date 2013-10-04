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
Fluidity related tools
"""

import glob
import os
import subprocess
import tempfile
import unittest

import fluidity.diagnostics.debug as debug

try:
    import numpy
except ImportError:
    debug.deprint("Warning: Failed to import numpy module")

try:
    from fluidity_tools import *
except:
    debug.deprint("Warning: Failed to import fluidity_tools module")

import fluidity.diagnostics.calc as calc
import fluidity.diagnostics.filehandling as filehandling
import fluidity.diagnostics.utils as utils


def FluidityBinary(binaries=["dfluidity-debug", "dfluidity", "fluidity-debug",
                             "fluidity"]):
    """
    Return the command used to call Fluidity
    """

    binary = None

    for possibleBinary in binaries:
        process = subprocess.Popen(["which", possibleBinary],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        process.wait()
        if process.returncode == 0:
            binary = possibleBinary
            debug.dprint("Fluidity binary: " + str(
                process.stdout.readlines()[0]), newline=False)
            break

    if binary is None:
        raise Exception("Failed to find Fluidity binary")

    return binary


class Stat:

    """
    Class for handling .stat files. Similiar to the dictionary returned by
    stat_parser, but with some annoying features fixed.
    """

    def __init__(self, filename=None, delimiter="%", includeMc=False,
                 subsample=1):
        self.SetDelimiter(delimiter)
        self._s = {}
        if not filename is None:
            self.Read(filename, includeMc=includeMc, subsample=subsample)

        return

    def __getitem__(self, key):
        """
        Index into the .stat with the given key (or path)
        """

        def SItem(s, key, delimiter):
            if key in s:
                return s[key]
            else:
                keySplit = key.split(delimiter)
                for i in range(len(keySplit)):
                    key = utils.FormLine(
                        keySplit[:i], delimiter=delimiter, newline=False)
                    if key in s:
                        if isinstance(s[key], dict):
                            try:
                                # Tolerate a failure when recursing, as the key
                                # may have been eroneously split
                                return SItem(s[key], utils.FormLine(keySplit[i:], delimiter=delimiter, newline=False), delimiter)
                            except Exception:
                                pass
                        else:
                            return s[key]

                raise Exception("Key not found")

        item = SItem(self._s, key, self._delimiter)
        if isinstance(item, dict):
            subS = Stat(delimiter=self._delimiter)
            subS._s = item
            return subS
        else:
            return item

    def __setitem__(self, key, value):
        keySplit = self.SplitPath(key)
        assert(len(keySplit) > 0)
        s = self
        for key in keySplit[:-1]:
            if s.haskey(key):
                s = s[key]
                assert(isinstance(s, Stat))
            else:
                s._s[key] = {}
                s = s[key]
        s._s[keySplit[-1]] = value

        return

    def __str__(self):
        paths = self.Paths()
        string = "Stat file:\n"
        for i, path in enumerate(paths):
            string += path
            if i < len(paths):
                string += "\n"

        return string

    def _PathSplit(self, path):
        """
        Return a list of keys into the stat dictionary for the supplied path
        """

        def SPathSplit(s, delimiter, path):
            pathSplit = path.split(delimiter)
            index = 0
            newPath = pathSplit[index]
            while not newPath in s.keys():
                index += 1
                newPath += delimiter + pathSplit[index]

            paths = []
            paths.append(newPath)
            if isinstance(s[newPath], dict):
                paths += SPathSplit(
                    s[newPath], delimiter, path[len(newPath) + len(delimiter):])

            return paths

        return SPathSplit(self._s, self._delimiter, path)

    def haskey(self, key):
        return self.HasPath(key)

    def keys(self):
        return self.Paths()

    def GetDelimiter(self):
        return self._delimiter

    def SetDelimiter(self, delimiter):
        self._delimiter = delimiter

        return

    def Paths(self):
        """
        Return all valid paths
        """

        def SPaths(s, delimiter, base=""):
            if len(base) > 0:
                base += delimiter

            paths = []
            for key in s.keys():
                if isinstance(s[key], dict):
                    paths += SPaths(s[key], delimiter, base=base + key)
                else:
                    paths.append(base + key)

            return paths

        return SPaths(self._s, self._delimiter)

    def PathLists(self):
        """
        Return all valid paths as a series of key lists
        """

        def SPathLists(s, delimiter, base=[]):
            paths = []
            for key in s.keys():
                if isinstance(s[key], dict):
                    paths += SPathLists(s[key], delimiter, base=base + [key])
                else:
                    paths.append(base + [key])

            return paths

        return SPathLists(self._s, self._delimiter)

    def HasPath(self, path):
        """
        Return whether the supplied path is valid for this Stat
        """

        try:
            self[path]
            return True
        except Exception:
            return False

    def FormPath(self, *args):
        path = ""
        for i, arg in enumerate(args):
            path += arg
            if i < len(args) - 1:
                path += self._delimiter

        return path

    def FormPathFromList(self, pathList):
        path = ""
        for i, entry in enumerate(pathList):
            path += entry
            if i < len(pathList) - 1:
                path += self._delimiter

        return path

    def SplitPath(self, path):
        return path.split(self._delimiter)

    def Read(self, filename, includeMc=False, subsample=1):
        """
        Read a .stat file
        """

        def ParseRawS(s, delimiter):
            newS = {}
            for key1 in s.keys():
                assert(not key1 in ["val", "value"])
                if isinstance(s[key1], dict):
                    if len(s[key1].keys()) == 1 and s[key1].keys()[0] in ["val", "value"]:
                        newS[str(key1)] = s[key1][s[key1].keys()[0]]
                    else:
                        subS = ParseRawS(s[key1], delimiter)
                        newS[str(key1)] = {}
                        for key2 in subS.keys():
                            newS[str(key1)][str(key2)] = subS[key2]
                else:
                    rank = len(s[key1].shape)
                    if rank > 1:
                        assert(rank == 2)
                        if includeMc:
                            # Add in this vector

                            # stat_parser gives this in an inconvenient matrix
                            # order. Take the transpose here to make life easier.
                            newS[str(key1)] = s[key1].transpose()

                        # Add in the vector field components
                        for i in range(len(s[key1])):
                            newS[str(key1) + delimiter + str(i + 1)] = s[
                                key1][i]
                    else:
                        try:
                            # Add in this scalar
                            newS[str(key1)] = s[key1]
                        except TypeError:
                            debug.deprint(
                                "Type error for data " + str(s[key1]), 0)
                            raise Exception("ParseRawS failure")
                        except ValueError:
                            debug.deprint(
                                "Value error for data " + str(s[key1]), 0)
                            raise Exception("ParseRawS failure")

            return newS

        debug.dprint("Reading .stat file: " + filename)
        if filehandling.FileExists(filename + ".dat"):
            debug.dprint("Format: binary")
        else:
            debug.dprint("Format: plain text")
        if subsample == 1:
            # Handle this case separately, as it's convenient to be backwards
            # compatible
            statParser = stat_parser(filename)
        else:
            statParser = stat_parser(filename, subsample=subsample)

        self._s = ParseRawS(statParser, self._delimiter)

        if "ElapsedTime" in self.keys():
            t = self["ElapsedTime"]
            if t.shape[0] > 0:
                debug.dprint("Time range: " + str((t[0], t[-1])))
            else:
                debug.dprint("Time range: No data")

        return


def JoinStat(*args):
    """Joins a series of stat files together. Useful for combining checkpoint
    .stat files. Selects data in later stat files over earlier stat files.
    Assumes data in stat files are sorted by ElapsedTime."""

    nStat = len(args)
    assert(nStat > 0)
    times = [stat["ElapsedTime"] for stat in args]

    startT = [t[0] for t in times]
    permutation = utils.KeyedSort(startT, range(nStat))
    stats = [args[index] for index in permutation]
    startT = [startT[index] for index in permutation]
    times = [times[index] for index in permutation]

    endIndices = numpy.array([len(time) for time in times], dtype=int)
    for i, t in enumerate(times[:-1]):
        for j, time in enumerate(t):
            if calc.AlmostEquals(startT[i + 1], time, tolerance=1.0e-6):
                endIndices[i] = max(j - 1, 0)
                break
            elif startT[i + 1] < time:
                endIndices[i] = j
                break
    debug.dprint("Time ranges:")
    if len(times) > 0:
        for i in range(nStat):
            debug.dprint((startT[i], times[i][endIndices[i] - 1]))
    else:
        debug.dprint("No data")

    dataIndices = numpy.empty(len(args) + 1, dtype=int)
    dataIndices[0] = 0
    for i, index in enumerate(endIndices):
        dataIndices[i + 1] = dataIndices[i] + index

    stat = stats[0]
    data = {}
    for key in stat.keys():
        arr = stat[key]
        shape = list(arr.shape)
        shape[0] = dataIndices[-1]
        data[key] = numpy.empty(shape, dtype=arr.dtype)
        data[key][:dataIndices[1]] = arr[:endIndices[0]]
        data[key][dataIndices[1]:] = calc.Nan()
    delimiter = stat.GetDelimiter()

    for i in range(1, nStat):
        stat = stats[i]
        for key in stat.keys():
            arr = stat[key]
            if not key in data:
                shape = list(arr.shape)
                shape[0] = dataIndices[-1]
                data[key] = numpy.empty(shape, dtype=arr.dtype)
                data[key][:dataIndices[i]] = calc.Nan()
                data[key][dataIndices[i + 1]:] = calc.Nan()
            data[key][dataIndices[i]:dataIndices[i + 1]] = arr[:endIndices[i]]

    output = Stat(delimiter=delimiter)
    for key in data.keys():
        output[key] = numpy.array(data[key])

    return output


def DetectorArrays(stat):
    """
    Return a dictionary of detector array lists contained in the supplied stat
    """

    # Detector array data is divided in the stat into one path per array entry.
    # We want to collapse this into a dictionary of one path per array. This
    # involves lots of horrible parsing of stat paths.

    # Find all detector array names and the paths for each entry in the array
    arrays = {}         # The arrays
    notArrayNames = []  # List of candidate array names that are, in fact, not
                        # detector array names
    for path in stat.PathLists():
        if isinstance(stat[stat.FormPathFromList(path)], Stat):
            # This isn't a leaf node
            continue

        # Look for an array marker in the path. This appears as:
        #   [path]%arrayname_index[%component]
        # and for coordinates as:
        #   arrayname_index[%component]

        if len(path) >= 2 and path[1] == "position":
            # This might be a coordinate entry

            firstKey = path[0]
            keySplit = firstKey.split("_")
            # We have an entry in an array if:
            #   1. We have more than one entry in the split
            #   2. The final entry in the split contains one of zero (for
            #      scalars) or two (for vector and tensors) array path delimiters
            #   3. The first stat path split of the final entry is an integer
            #      (the index)
            if len(keySplit) <= 1 \
                or not len(stat.SplitPath(keySplit[-1])) in [1, 2] \
                    or not utils.IsIntString(stat.SplitPath(keySplit[-1])[0]):
                # This definitely can't be an entry in a detector array
                continue

            # OK, we have something that looks a bit like an entry for a
            # detector array

            # Find the array name and the index for this entry
            arrayName = utils.FormLine([utils.FormLine(keySplit[:-1], delimiter="_", newline=False)] + path[
                                       1:], delimiter=stat.GetDelimiter(), newline=False)
            index = int(keySplit[-1])
        else:
            # This might be a field entry

            finalKey = path[-1]
            keySplit = finalKey.split("_")
            # We have an entry in an array if:
            #   1. We have more than one entry in the split
            #   2. The final entry in the split contains one of zero or one
            #      (for field components) array path delimiters
            #   3. The first stat path split of the final entry is an integer
            #      (the index)
            if len(keySplit) <= 1 \
                or not len(stat.SplitPath(keySplit[-1])) in [1, 2] \
                    or not utils.IsIntString(stat.SplitPath(keySplit[-1])[0]):
                # This definitely can't be an entry in a detector array
                continue

            # OK, we have something that looks a bit like an entry for a
            # detector array

            # Find the array name and the index for this entry
            arrayName = utils.FormLine(
                path[:-1] + [utils.FormLine(keySplit[:-1], delimiter="_", newline=False)],
                delimiter=stat.GetDelimiter(), newline=False)
            if len(stat.SplitPath(keySplit[-1])) > 1:
                # This array name references a field component

                # We need to append the component to the array name. This needs
                # to be added to the last but one part of the stat path (the
                # final entry is the name of this detector array as configured
                # in Fluidity).
                splitName = stat.SplitPath(arrayName)
                splitName[-2] = stat.FormPath(
                    splitName[-2], stat.SplitPath(keySplit[-1])[1])
                arrayName = stat.FormPathFromList(splitName)
                index = int(stat.SplitPath(keySplit[-1])[0])
            else:
                # This array name references a field

                index = int(keySplit[-1])

        if arrayName in notArrayNames:
            # We've already discovered that this candidate array name isn't in
            # fact a detector array
            continue

        if index <= 0:
            # This isn't a valid index

            # This candidate array name isn't in fact a detector array
            notArrayNames.append(arrayName)
            if arrayName in arrays:
                arrays.remove(arrayName)
            continue
        if arrayName in arrays and index in arrays[arrayName]:
            # We've seen this index more than once for this array name

            # This candidate apparent array name isn't in fact a detector array
            notArrayNames.append(arrayName)
            arrays.remove(arrayName)
            continue

        if arrayName in arrays:
            arrays[arrayName][index] = stat[stat.FormPathFromList(path)]
        else:
            # This is a new array name
            arrays[arrayName] = {}
            arrays[arrayName][index] = stat[stat.FormPathFromList(path)]

    # Convert the dictionaries of data to lists, and check for consecutive
    # indices
    for name in arrays:
        array = arrays[name]
        indices = array.keys()
        data = [array[index] for index in indices]
        indices, data = utils.KeyedSort(indices, data, returnSortedKeys=True)
        arrays[name] = numpy.array(data)

        for i, index in enumerate(indices):
            if not i + 1 == index:
                # The indices are not consecutive from one. After all the hard
                # work above, we still have an array name that isn't in fact a
                # detector array.
                arrays.remove(name)
                break

    # Fantastic! We have our detectors dictionary!
    debug.dprint("Detector keys:")
    debug.dprint(arrays.keys())

    return arrays


def SplitVtuFilename(filename):
    """
    Split the supplied vtu filename into project, ID and file extension
    """

    first = filehandling.StripFileExtension(filename)
    ext = filehandling.FileExtension(filename)

    split = first.split("_") + [ext]

    idIndex = None
    for i, val in enumerate(split[1:]):
        if utils.IsIntString(val):
            idIndex = i + 1
            break
    assert(not idIndex is None)

    project = utils.FormLine(split[:idIndex], delimiter="_", newline=False)
    id = int(split[idIndex])
    ext = utils.FormLine([""] + split[idIndex + 1:len(split) - 1],
                         delimiter="_", newline=False) + split[-1]

    return project, id, ext


def VtuFilename(project, id, ext):
    """
    Create a vtu filename from a project, ID and file extension
    """

    return project + "_" + str(id) + ext


def VtuFilenames(project, firstId, lastId=None, extension=".vtu"):
    """Return vtu filenames for a Fluidity simulation, in the supplied range of
    IDs."""

    if lastId is None:
        lastId = firstId
    assert(lastId >= firstId)

    filenames = []
    for id in range(firstId, lastId + 1):
        filenames.append(project + "_" + str(id) + extension)

    return filenames


def PVtuFilenames(project, firstId, lastId=None, extension=".pvtu"):
    """Return pvtu filenames for a Fluidity simulation, in the supplied range
    of IDs."""

    return VtuFilenames(project, firstId, lastId=lastId, extension=extension)


def FindVtuFilenames(project, firstId, lastId=None,
                     extensions=[".vtu", ".pvtu"]):
    """
    Find vtu filenames for a Fluidity simulation, in the supplied range of IDs
    """

    if lastId is None:
        lastId = firstId
    assert(lastId >= firstId)

    filenames = []
    for id in range(firstId, lastId + 1):
        filename = project + "_" + str(id)
        for i, ext in enumerate(extensions):
            try:
                os.stat(filename + ext)
                filename += ext
                break
            except OSError:
                pass
            if i == len(extensions) - 1:
                raise Exception("Failed to find input file with ID " + str(id))
        filenames.append(filename)

    return filenames


def FindPvtuVtuFilenames(project, firstId, lastId=None, pvtuExtension=".pvtu",
                         vtuExtension=".vtu"):
    """
    Find vtu filenames for a Fluidity simulation, in the supplied range of IDs,
    retuning the vtus for each given pvtu filename
    """

    pvtuFilenames = FindVtuFilenames(
        project, firstId, lastId=lastId, extensions=[pvtuExtension])

    filenames = []
    for pvtuFilename in pvtuFilenames:
        i = 0
        while True:
            filename = pvtuFilename[
                :-len(pvtuExtension)] + "_" + str(i) + vtuExtension
            try:
                os.stat(filename)
                filenames.append(filename)
                i += 1
            except OSError:
                break
        if i == 0:
            raise Exception("Failed to find vtus for pvtu " + pvtuFilename)

    return filenames


def FindMaxVtuId(project, extensions=[".vtu", ".pvtu"]):
    """
    Find the maximum Fluidity vtu ID for the supplied project name
    """

    filenames = []
    for ext in extensions:
        filenames += glob.glob(project + "_?*" + ext)

    maxId = None
    for filename in filenames:
        id = filename[len(project) + 1:-len(filename.split(".")[-1]) - 1]
        try:
            id = int(id)
        except ValueError:
            continue

        if maxId is None:
            maxId = id
        else:
            maxId = max(maxId, id)

    return maxId


def FindFinalVtu(project, extensions=[".vtu", ".pvtu"]):
    """
    Final the final Fluidity vtu for the supplied project name
    """

    return FindVtuFilenames(project, FindMaxVtuId(project, extensions=extensions),
                            extensions=extensions)[0]


def FindPFilenames(basename, extension):
    filenames = []
    i = 0
    while True:
        filename = basename + "_" + str(i)
        if filehandling.FileExists(filename + extension):
            filenames.append(filename)
        else:
            break
        i += 1

    return filenames


class fluiditytoolsUnittests(unittest.TestCase):

    def testFluidityToolsSupport(self):
        import fluidity_tools  # noqa: testing
        return

    def testSplitVtuFilename(self):
        project, id, ext = SplitVtuFilename("project_0.vtu")
        self.assertEquals(project, "project")
        self.assertEquals(id, 0)
        self.assertEquals(ext, ".vtu")

        project, id, ext = SplitVtuFilename("project_1.vtu")
        self.assertEquals(project, "project")
        self.assertEquals(id, 1)
        self.assertEquals(ext, ".vtu")

        project, id, ext = SplitVtuFilename("project_-1.vtu")
        self.assertEquals(project, "project")
        self.assertEquals(id, -1)
        self.assertEquals(ext, ".vtu")

        project, id, ext = SplitVtuFilename("project_-1_sub.vtu")
        self.assertEquals(project, "project")
        self.assertEquals(id, -1)
        self.assertEquals(ext, "_sub.vtu")

        return

    def testFindVtuFilenames(self):
        tempDir = tempfile.mkdtemp()
        project = os.path.join(tempDir, "project")
        for i in range(2, 4):
            filehandling.Touch(project + "_" + str(i) + ".vtu")
        filenames = FindVtuFilenames(project, firstId=2, lastId=3)
        self.assertEquals(len(filenames), 2)
        filehandling.Touch(project + "_4.pvtu")
        filenames = FindVtuFilenames(project, firstId=2, lastId=4)
        self.assertEquals(len(filenames), 3)
        filehandling.Rmdir(tempDir, force=True)

        return

    def testFindPvtuVtuFilenames(self):
        tempDir = tempfile.mkdtemp()
        project = os.path.join(tempDir, "project")
        for i in range(2, 4):
            for j in range(3):
                filehandling.Touch(
                    project + "_" + str(i) + "_" + str(j) + ".vtu")
            filehandling.Touch(project + "_" + str(i) + ".pvtu")
        filenames = FindPvtuVtuFilenames(project, firstId=2, lastId=3)
        self.assertEquals(len(filenames), 6)
        filehandling.Rmdir(tempDir, force=True)

        return
