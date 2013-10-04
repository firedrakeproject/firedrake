#!/usr/bin/env python

# James Maddison

import getopt
import sys

try:
    import psyco
    psyco.full()
except:
    pass

import vtktools


def EPrint(message):
    """
    Send an error message to standard error
    """

    sys.stderr.write(message + "\n")
    sys.stderr.flush()

    return


def Help():
    """
    Prints program usage information
    """

    print """Usage: vtudiff [OPTIONS] ... INPUT1 INPUT2 OUTPUT [FIRST] [LAST]

Generates vtus with fields equal to the difference between the corresponding
fields in two input vtus (INPUT1 - INPUT2). The fields of INPUT2 are projected
onto the cell points of INPUT1.

If FIRST is supplied, treats INPUT1 and INPUT2 as project names, and generates
a different vtu for the specified range of output files.

Options:

-s  If supplied together with FIRST and LAST, only INPUT1 is treated as a
    project name. Allows a range of vtus to be diffed against a single vtu."""

    return


def Error(message, displayHelp=True):
    """
    Print an error message, usage information and quit
    """

    if displayHelp:
        Help()

    EPrint(message)
    sys.exit(1)

try:
    opts, args = getopt.getopt(sys.argv[1:], "ms")
except:
    Help()
    sys.exit(1)

if len(args) > 5:
    Error("Invalid argument \"" + args[5] + "\" supplied")

diffAgainstSingle = ("-s", "") in opts

try:
    inputFilename1 = args[0]
    inputFilename2 = args[1]
    outputFilename = args[2]
except:
    Help()
    sys.exit(1)

if len(args) > 3:
    try:
        firstId = int(args[3])
        if len(args) > 4:
            try:
                lastId = int(args[4])
            except:
                Error("Invalid last ID entered")
        else:
            lastId = firstId
    except:
        Error("Invalid first ID entered")
else:
    firstId = None

if firstId is None:
    inputFilenames1 = [inputFilename1]
    inputFilenames2 = [inputFilename2]
    outputFilenames = [outputFilename]
else:
    inputFilenames1 = [inputFilename1 + "_" + str(
        i) + ".vtu" for i in range(firstId, lastId + 1)]
    if diffAgainstSingle:
        inputFilenames2 = [inputFilename2 for i in range(firstId, lastId + 1)]
    else:
        inputFilenames2 = [inputFilename2 + "_" + str(
            i) + ".vtu" for i in range(firstId, lastId + 1)]

    outputFilenames = [outputFilename + "_" + str(
        i) + ".vtu" for i in range(firstId, lastId + 1)]

for i in range(len(inputFilenames1)):
    try:
        vtu1 = vtktools.vtu(inputFilenames1[i])
    except:
        Error("Unable to read input vtu \"" + inputFilenames1[i] + "\"", False)
    try:
        vtu2 = vtktools.vtu(inputFilenames2[i])
    except:
        Error("Unable to read input vtu \"" + inputFilenames2[i] + "\"", False)

    diffVtu = vtktools.VtuDiff(vtu1, vtu2, outputFilenames[i])

    try:
        diffVtu.Write()
    except:
        Help()
        Error("Unable to write output file \"" +
              outputFilenames[i] + "\"", False)

    print "Generated vtu diff file \"" + outputFilenames[i] + "\""
