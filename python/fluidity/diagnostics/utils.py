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
Some useful utility functions
"""

import copy
import time
import unittest

import fluidity.diagnostics.optimise as optimise


def IsIntString(string):
    """
    Return whether the supplied string contains pure integer data.
    """

    try:
        intVal = int(string)
    except ValueError:
        return False

    strip = string.strip().lstrip("0")

    return len(strip) == 0 or str(intVal) == strip


def CanLen(input):
    """
    Return whether it is valid to call len on the supplied input
    """

    try:
        len(input)
        return True
    except TypeError:
        return False


def Prefix(string, pad, length):
    """
    Prefix the supplied string until it is the desired length, with the given
    padding character
    """

    assert(len(string) <= length)
    assert(len(pad) == 1)

    result = ""
    while len(string) + len(result) < length:
        result += pad
    result += string

    return result


def CurrentDateStamp():
    """
    Return a YYMMDD string
    """

    currentTime = time.localtime()

    return Prefix(str(currentTime[0] % 100), "0", 2) + \
        Prefix(str(currentTime[1]), "0", 2) + \
        Prefix(str(currentTime[2]), "0", 2)


def ExpandList(input):
    """
    Return a list equal to the input with all sub-elements expanded
    """

    parentType = input.__class__
    outputList = []
    for item in input:
        # Trap the case that comparing the item with the input produces an
        # array (as is the case for numpy arrays)
        try:
            equalsInput = item == input
            if CanLen(equalsInput):
                if len(equalsInput) == 1:
                    equalsInput = equalsInput[0]
                else:
                    equalsInput = False
        except ValueError:
            equalsInput = False

        if (not CanLen(equalsInput) or len(equalsInput) == 1) and equalsInput:
            # Trap the case that iterating through the input produces the input
            # (as is the case for single character strings)
            outputList.append(item)
        # If the item is of the same type as the input expand it, unless the
        # item is a string and the input is not
        elif (isinstance(item, parentType) or (CanLen(item)
              and not isinstance(item, str))):
            for subitem in ExpandList(item):
                outputList.append(subitem)
        else:
            outputList.append(item)

    return outputList


def FormLine(inputList, delimiter=" ", newline=True):
    """Form a delimited line out of the supplied list, (by default) terminated
    with a newline."""

    inputList = ExpandList(inputList)

    line = ""
    for i in range(len(inputList)):
        line += str(inputList[i])
        if i < len(inputList) - 1:
            line += delimiter
    if newline:
        line += "\n"

    return line

# Jumping through Python 2.3 hoops for cx1 - in >= 2.4, can just pass a cmp
# argument to list.sort


class Sorter:

    def __init__(self, key, value):
        self._key = key
        self._value = value

        return

    def GetKey(self):
        return self._key

    def GetValue(self):
        return self._value

    def __cmp__(self, val):
        if self._key > val:
            return 1
        elif self._key == val:
            return 0
        else:
            return -1


def KeyedSort(keys, *values, **kargs):
    """Return a re-ordering of values, with the remapping equal to the sort
    mapping of keys. Each key must correspond to a unique value. Optional
    keyword arguments:
      returnSortedKeys  Return the sorted keys as well as the sorted values
    """

    returnSortedKeys = False
    for key in kargs:
        if key == "returnSortedKeys":
            returnSortedKeys = True
        else:
            raise Exception("Unexpected keyword argument \"" + key + "\"")

    sortList = []
    for i, key in enumerate(keys):
        sortList.append(
            Sorter(key, tuple([subValues[i] for subValues in values])))

    sortList.sort()
    if optimise.DebuggingEnabled():
        for i in range(len(sortList) - 1):
            if sortList[i].GetKey() == sortList[i + 1].GetKey():
                assert(sortList[i].GetValue() == sortList[i].GetValue())

    result = []
    if returnSortedKeys:
        result.append([sorter.GetKey() for sorter in sortList])

    for i in range(len(values)):
        result.append([sorter.GetValue()[i] for sorter in sortList])

    if len(result) == 1:
        result = result[0]
    else:
        result = tuple(result)

    return result


def CountUnique(inputList):
    """
    Count the number of unique entries in the supplied list
    """

    lInputList = copy.copy(inputList)
    lInputList.sort()

    count = 0
    if len(lInputList) > 0:
        count += 1
    for i in range(1, len(lInputList)):
        if not lInputList[i] == lInputList[i - 1]:
            count += 1

    return count


def IndexOfMax(inputList):
    """
    Return the index of the max value in the supplied list
    """

    assert(len(inputList) > 0)

    index = 0
    maxVal = inputList[0]
    for i, val in enumerate(inputList[1:]):
        if val > maxVal:
            maxVal = val
            index = i + 1

    return index


def IndexOfMin(inputList):
    """
    Return the index of the min value in the supplied list
    """

    assert(len(inputList) > 0)

    index = 0
    minVal = inputList[0]
    for i, val in enumerate(inputList[1:]):
        if val < minVal:
            minVal = val
            index = i + 1

    return index


def MaskList(inputList, mask):
    """
    Return a list containing elements of the input list where the corresponding
    elements of the input mask are True
    """

    assert(len(inputList) == len(mask))

    outputList = []
    for i, entry in enumerate(inputList):
        if mask[i]:
            outputList.append(entry)

    return outputList


def OffsetList(inputList, offset):
    """
    Return a list equal to the input list with all entries equal to the
    corresponding entries in the input list plus the supplied offset
    """

    outputList = []
    for entry in inputList:
        outputList.append(entry + offset)

    return outputList


def TransposeListList(inputList):
    """
    Transpose the input list of lists (which must not be ragged)
    """

    if optimise.DebuggingEnabled():
        if len(inputList) > 0:
            assert(CanLen(inputList[0]))
            size = len(inputList[0])
            for entry in inputList[1:]:
                assert(CanLen(entry))
                assert(len(inputList[1]) == size)

    if len(inputList) == 0:
        return []

    tList = [[] for i in range(len(inputList[0]))]
    for entry in inputList:
        for i, datum in enumerate(entry):
            tList[i].append(datum)

    return tList


def DictInverse(inputDict):
    """
    Return a dictionary equal to the input dictionary with keys and values
    interchanged
    """

    outputDict = {}
    for key in inputDict:
        outputDict[inputDict[key]] = key

    return outputDict


def StripListDuplicates(list):
    """
    Strip duplicate entries from the supplied list
    """

    listCopy = copy.deepcopy(list)
    listCopy.sort()
    toRemove = []
    for i in range(len(listCopy) - 1):
        if listCopy[i + 1] == listCopy[i]:
            toRemove.append(listCopy[i + 1])

    # Reverse before (and after) stripping, as we choose to keep the earlier
    # duplicates
    list.reverse()
    for entry in toRemove:
        list.remove(entry)
    list.reverse()

    return


def TypeCodeToType(typeCode):
    """
    Convert a type code to the class it represents
    """

    if typeCode in ["b", "d", "f", "s"]:
        return float
    elif typeCode in ["i", "l"]:
        return int
    elif typeCode in ["c"]:
        return str
    else:
        raise Exception("Unrecognised type code: " + typeCode)

    return


class utilsUnittests(unittest.TestCase):

    def testIsIntString(self):
        self.assertFalse(IsIntString(""))
        self.assertTrue(IsIntString("0"))
        self.assertTrue(IsIntString("00"))
        self.assertTrue(IsIntString(" 00  "))
        self.assertTrue(IsIntString("1"))
        self.assertTrue(IsIntString("0010"))
        self.assertTrue(IsIntString(" -1  "))
        self.assertFalse(IsIntString("Hello world"))
        self.assertFalse(IsIntString("1.0"))

        return

    def testCanLen(self):
        self.assertTrue(CanLen((1, 2)))
        self.assertTrue(CanLen([1, 2]))
        self.assertFalse(CanLen(1))

        return

    def testPrefix(self):
        self.assertEquals(Prefix("1", "0", 2), "01")
        self.assertEquals(Prefix("1", "0", 3), "001")
        self.assertRaises(AssertionError, Prefix, "1", "0", 0)
        self.assertRaises(AssertionError, Prefix, "1", "00", 2)

        return

    def testExpandList(self):
        self.assertEquals(ExpandList((1, (2,), [3, 4])), [1, 2, 3, 4])
        self.assertEquals(ExpandList("ab"), ["a", "b"])
        try:
            import numpy
            self.assertEquals(ExpandList(numpy.array([0.0, 1.0])), [0.0, 1.0])
        except ImportError:
            pass
        self.assertEquals(ExpandList(["one", "two"]), ["one", "two"])

        return

    def testFormLine(self):
        self.assertEquals(
            FormLine([1, [2, 3]], delimiter=",", newline=False), "1,2,3")

        return

    def testKeyedSort(self):
        keys = [2, 1, 3]
        values = ["a", "b", "c"]
        keys, values = KeyedSort(keys, values, returnSortedKeys=True)
        self.assertEqual(keys[0], 1)
        self.assertEqual(keys[1], 2)
        self.assertEqual(keys[2], 3)
        self.assertEqual(values[0], "b")
        self.assertEqual(values[1], "a")
        self.assertEqual(values[2], "c")

        keys = [2, 1, 3]
        values1 = ["a", "b", "c"]
        values2 = ["A", "B", "C"]
        keys, values1, values2 = KeyedSort(
            keys, values1, values2, returnSortedKeys=True)
        self.assertEqual(keys[0], 1)
        self.assertEqual(keys[1], 2)
        self.assertEqual(keys[2], 3)
        self.assertEqual(values1[0], "b")
        self.assertEqual(values1[1], "a")
        self.assertEqual(values1[2], "c")
        self.assertEqual(values2[0], "B")
        self.assertEqual(values2[1], "A")
        self.assertEqual(values2[2], "C")

        return

    def testIndexOfMax(self):
        self.assertEquals(IndexOfMax([1, 2, 10, 3, 4]), 2)

        return

    def testIndexOfMin(self):
        self.assertEquals(IndexOfMin([1, 2, -10, 3, 4]), 2)

        return

    def testCountUnique(self):
        self.assertEquals(CountUnique([1, 1, 2]), 2)
        self.assertEquals(
            CountUnique(["b", "cd", "c", "d", "cd", "a", "b"]), 5)

        return

    def testMaskList(self):
        inputList = [1, 2, 3]
        mask = [True, False, True]
        outputList = MaskList(inputList, mask)
        self.assertEqual(len(outputList), 2)
        self.assertEqual(outputList[0], 1)
        self.assertEqual(outputList[1], 3)

        return

    def testOffsetList(self):
        inputList = [1, 2, 3]
        outputList = OffsetList(inputList, -1)
        self.assertEquals(len(outputList), 3)
        self.assertEquals(outputList[0], 0)
        self.assertEquals(outputList[1], 1)
        self.assertEquals(outputList[2], 2)

        return

    def testDictInverse(self):
        dict = {1: "a", 2: "b"}
        dict = DictInverse(dict)
        self.assertEquals(dict["a"], 1)
        self.assertEquals(dict["b"], 2)

        return

    def testStripListDuplicates(self):
        list = [1, 2, 3, 2]
        StripListDuplicates(list)
        self.assertEquals(list, [1, 2, 3])

        return
