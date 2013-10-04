#!/usr/bin/env python

import glob
import re
import sys
import sha


def Error(msg):
    sys.stderr.write("Diagnostics error: " + str(msg) + "\n")
    sys.stderr.flush()
    sys.exit(1)

baseName = "Diagnostic_Fields_New"
disabledDiags = ["Diagnostic_Source_Fields.F90",
                 "Diagnostic_Fields_Interfaces.F90"]

inputFilename = baseName + ".F90.in"
outputFilename = baseName + ".F90"

# get sha1 digest of existing generated file.  Can't use 'rw' here
# because it updates the modtime of the file, which we're trying to
# avoid doing.
orig = sha.new()
try:
    f = open(outputFilename, 'r')
    orig.update(f.read())
except IOError:
    pass
else:
    f.close()

# Valid arguments
diagnosticArguments = ["states", "state", "s_field", "v_field", "t_field",
                       "current_time", "dt", "state_index"]
# Additional code used to form above arguments
diagnosticArgumentsCode = """
    real :: current_time, dt
    type(state_type), pointer :: state => null()

    state => states(state_index)

    call get_option("/timestepping/current_time", current_time)
    call get_option("/timestepping/timestep", dt)
"""

# Initialise automagic code
useModulesCode = ""
singleStateScalarDiagnosticsCode = ""
multipleStateScalarDiagnosticsCode = ""
singleStateVectorDiagnosticsCode = ""
multipleStateVectorDiagnosticsCode = ""
singleStateTensorDiagnosticsCode = ""
multipleStateTensorDiagnosticsCode = ""

# Parse fortran files and search for diagnostic algorithms

moduleRe = re.compile(r"^\s*module\s+(\w+)\s*$", re.IGNORECASE | re.MULTILINE)
subroutineRe = re.compile(r"^\s*subroutine\s+(\w+)\(?([\w,\s]*)\)?\s*$",
                          re.IGNORECASE | re.MULTILINE)

diagFiles = glob.glob("*.F90")
for file in [inputFilename, outputFilename] + disabledDiags:
    try:
        diagFiles.remove(file)
    except ValueError:
        pass

for file in diagFiles:
    fileHandle = open(file, "r")
    code = fileHandle.read()
    fileHandle.close()

    modules = moduleRe.findall(code)
    for module in modules:
        useModulesCode += "  use " + module + "\n"

    subroutines = subroutineRe.findall(code)
    for subroutine in subroutines:
        name, args = subroutine[0].lower(), subroutine[1].lower()

        if not name.startswith("calculate_"):
            continue

        alg = name[10:]

        newArgs = []
        for arg in args.split(","):
            arg = arg.strip()
            if len(arg) > 0:
                newArgs.append(arg)
        args = newArgs

        for arg in args:
            if not arg in diagnosticArguments:
                Error("For subroutine \"" + name +
                      "\", invalid argument \"" + arg + "\"")

        if "state" in args:
            if "states" in args:
                Error("For subroutine \"" + name +
                      "\", expected exactly one type of state argument")
            singleState = True
        elif "states" in args:
            singleState = False
        else:
            singleState = True

        if "s_field" in args:
            if "v_field" in args or "t_field" in args:
                Error("For subroutine '%s', expected exactly one type of \
                      diagnostic field argument" % name)
            rank = 0
        elif "v_field" in args:
            if "s_field" in args or "t_field" in args:
                Error("For subroutine '%s', expected exactly one type of \
                      diagnostic field argument" % name)
            rank = 1
        elif "t_field" in args:
            if "s_field" in args or "v_field" in args:
                Error("For subroutine '%s', expected exactly one type of \
                      diagnostic field argument" % name)
            rank = 2
        else:
            Error("For subroutine \"" + name +
                  "\", expected exactly one type of diagnostic field argument")

        algCode = "      case(\"" + alg + "\")\n" + \
            "        call calculate_" + alg
        algCode += "("
        for i, arg in enumerate(args):
            algCode += arg
            if i < len(args) - 1:
                algCode += ", "
            else:
                algCode += ")\n"

        if singleState:
            if rank == 0:
                singleStateScalarDiagnosticsCode += algCode
            elif rank == 1:
                singleStateVectorDiagnosticsCode += algCode
            elif rank == 2:
                singleStateTensorDiagnosticsCode += algCode
            else:
                Error("Unexpected diagnostic field type")
        else:
            if rank == 0:
                multipleStateScalarDiagnosticsCode += algCode
            elif rank == 1:
                multipleStateVectorDiagnosticsCode += algCode
            elif rank == 2:
                multipleStateTensorDiagnosticsCode += algCode
            else:
                Error("Unexpected diagnostic field type")

# Read input
fileHandle = open(inputFilename, "r")
outputCode = fileHandle.read()
fileHandle.close()

# Insert automagic code
outputCode = outputCode.replace("USE_MODULES", useModulesCode)
outputCode = outputCode.replace(
    "DIAGNOSTIC_ARGUMENTS", diagnosticArgumentsCode)
outputCode = outputCode.replace(
    "SINGLE_STATE_SCALAR_DIAGNOSTICS", singleStateScalarDiagnosticsCode)
outputCode = outputCode.replace(
    "MULTIPLE_STATE_SCALAR_DIAGNOSTICS", multipleStateScalarDiagnosticsCode)
outputCode = outputCode.replace(
    "SINGLE_STATE_VECTOR_DIAGNOSTICS", singleStateVectorDiagnosticsCode)
outputCode = outputCode.replace(
    "MULTIPLE_STATE_VECTOR_DIAGNOSTICS", multipleStateVectorDiagnosticsCode)
outputCode = outputCode.replace(
    "SINGLE_STATE_TENSOR_DIAGNOSTICS", singleStateTensorDiagnosticsCode)
outputCode = outputCode.replace(
    "MULTIPLE_STATE_TENSOR_DIAGNOSTICS", multipleStateTensorDiagnosticsCode)

# Write the output
new = sha.new()
new.update(outputCode)

# Only write file if sha1sums differ
if new.digest() != orig.digest():
    try:
        f = open(outputFilename, 'w')
        f.write(outputCode)
    except IOError:
        # Fixme, this should fail better
        pass
    else:
        f.close()
