#!/usr/bin/env python

import getopt
import glob
import os
import sys

import diamond.debug as debug
import diamond.schema as schema


def Help():
    debug.dprint("""Usage: update_options [OPTIONS] ... [FILES]

Updates flml, bml, swml and adml files. If FILES is not specified, all .flml,
.bml, .swml and .adml files in tests/*/., tests/*/*/., longtests/*/.,
longtests/*/*/. and examples/*/. will be updated. Options:

-h  Display this help
-v  Verbose mode""", 0)

    return

try:
    opts, args = getopt.getopt(sys.argv[1:], "hv")
except getopt.getoptError:
    Help()
    sys.exit(-1)

if ("-h", "") in opts:
    Help()
    sys.exit(0)

if not ("-v", "") in opts:
    debug.SetDebugLevel(0)

rootDir = os.path.join(os.path.dirname(__file__), os.path.pardir)
testDir = os.path.join(rootDir, "tests")
longtestDir = os.path.join(rootDir, "longtests")
examplesDir = os.path.join(rootDir, "examples")

extdict = {"flml": "fluidity_options.rng",
           "bml": "burgers_equation.rng",
           "swml": "shallow_water_options.rng",
           "adml": "test_advection_diffusion_options.rng"}

# cache parsed schema files
schemadict = {}
for k, v in extdict.items():
    schemadict[k] = schema.Schema(os.path.join(rootDir, "schemas", v))

filenames = args
if len(filenames) == 0:
    filenames = []
    for k, v in extdict.items():
        filenames += glob.glob(os.path.join(testDir, "*", "*." + k))
        filenames += glob.glob(os.path.join(testDir, "*", "*", "*." + k))
        filenames += glob.glob(os.path.join(longtestDir, "*", "*." + k))
        filenames += glob.glob(os.path.join(longtestDir, "*", "*", "*." + k))
        filenames += glob.glob(os.path.join(examplesDir, "*", "*." + k))

invalidFiles = []
updated = 0
for filename in filenames:
    debug.dprint("Processing " + str(filename), 1)

    ext = filename.split(".")[-1]
    sch = schemadict[ext]

    # Read the file and check that either the file is valid, or diamond.schema
    # can make the file valid by adding in the missing elements
    optionsTree = sch.read(filename)
    lost_eles, added_eles, lost_attrs, added_attrs = sch.read_errors()
    if len(lost_eles) + len(lost_attrs) > 0 or not optionsTree.valid:
        debug.deprint(str(filename) + ": Invalid", 0)
        debug.deprint(str(filename) + " errors: " + str(
            (lost_eles, added_eles, lost_attrs, added_attrs)), 1)
        invalidFiles.append(filename)
        continue

    # Write out the updated options file
    optionsTree.write(filename)
    debug.dprint(str(filename) + ": Updated", 0)
    updated += 1

debug.dprint("Summary:", 0)
debug.dprint("Invalid options files:", 0)
for filename in invalidFiles:
    debug.dprint(filename, 0)
debug.dprint("Invalid: " + str(len(invalidFiles)), 0)
debug.dprint("Updated: " + str(updated), 0)
