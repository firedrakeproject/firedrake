#!/usr/bin/env python
#
# Bibiography:
#
# http://amcg.ese.ic.ac.uk/index.php?title=Local:Fluidity_tools
# Dive Into Python 5.4, Mark Pilgrim, May 2004, http://diveintopython.org/
# Python Documentation 2.5, http://docs.python.org/download.html

"""
stat to csv convertor for Fluidity output stat files.
"""

import getopt
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, "tools"),)
import fluidity_tools


def Help():
    """
    Print program usage information.
    """
    print """Usage: stat2csv [OPTIONS] ... STAT

          Converts a Fluidity .stat file to a .csv file.

          Options:

          -c        Do not output data labels
          -d DELIM  Delimiter to use between output columns (default = \",\")
          -o        Output file
          -s        Output to stdout instead of output file"""
    sys.stdout.flush()

    return


def EPrint(message):
    """
    Send an error message to standard error.
    """

    sys.stderr.write(message + "\n")
    sys.stderr.flush()

    return


def FatalError(message):
    """Send an error message to standard error and terminate with a non-zero
    return value."""

    EPrint(message)

    sys.exit(1)


def Error(message):
    """
    Print an error message, usage information and quit.
    """
    Help()
    FatalError(message)

try:
    opts, args = getopt.getopt(sys.argv[1:], "cd:o:s")
except getopt.GetoptError:
    pass

try:
    inputFile = args[0]
except:
    Help()
    sys.exit(1)

if len(args) > 1:
    Error("Unrecognised option \"" + args[1] + "\" entered")
outputLabels = 1
delimiter = ","
if len(inputFile.split(".")) == 1:
    outputFile = inputFile
else:
    outputFile = inputFile[:-len(inputFile.split(".")[-1]) - 1]
outputFile += ".csv"
useStdout = 0
for opt in opts:
    if opt[0] == "-c":
        outputLabels = 0
    elif opt[0] == "-d":
        delimiter = opt[1]
    elif opt[0] == "-o":
        outputFile = opt[1]
    elif opt[0] == "-s":
        useStdout = 1
    else:
        Error("Unrecognised option \"" + opt[0] + "\" entered")

# Read the input .stat file
s = fluidity_tools.stat_parser(inputFile)


def ExtractData(input, labelPrefix=""):
    """
    Parse the input stat_parser and return a dictionary of scalar field data
    """

    if len(labelPrefix) > 0:
        labelPrefix += "%"

    data = {}
    for key in input:
        if isinstance(input[key], dict):
            # Internal nodes
            newData = ExtractData(input[key], labelPrefix + key)
            for label in newData:
                data[label] = newData[label]
        else:
            # Leaf nodes
            datum = input[key]
            if len(datum.shape) == 1:
                # Rank zero fields (scalars)
                label = labelPrefix + key
                data[label] = datum
            elif len(datum.shape) == 2:
                # Rank one fields (vectors)
                for i in range(datum.shape[0]):
                    label = labelPrefix + key + "%" + str(i + 1)
                    data[label] = datum[i, :]
            else:
                raise Exception(
                    "Unexpected data rank: " + str(len(datum.shape)))

    return data

data = ExtractData(s)
labels = data.keys()

# Open the output file for writing
if useStdout:
    outputHandle = sys.stdout
else:
    outputHandle = open(outputFile, "w")

if outputLabels:
    # Labels
    for i, label in enumerate(labels):
        outputHandle.write(label)
        if i < len(labels) - 1:
            outputHandle.write(delimiter)
        else:
            outputHandle.write("\n")

if len(labels) > 0:
    # Data
    entries = len(data[labels[0]])
    for i in range(entries):
        for j, label in enumerate(labels):
            outputHandle.write(str(data[label][i]))
            if j < len(labels):
                outputHandle.write(delimiter)
        outputHandle.write("\n")

outputHandle.flush()
if not useStdout:
    outputHandle.close()
