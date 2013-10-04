#!/usr/bin/env python
import re
import glob
# FIXME: (at some later date)
# Python 2.5 deprecates sha in favour of hashlib, however we support
# Python versions pre 2.5 and the sha module isn't scheduled to be
# removed yet, so just use that for now.
import sha
import StringIO


# File header
header = """
subroutine register_diagnostics

"""
footer = """
end subroutine register_diagnostics
"""

outfile = 'preprocessor/register_diagnostics.F90'

# get sha1 digest of existing generated file.  Can't use 'rw' here
# because it updates the modtime of the file, which we're trying to
# avoid doing.
orig = sha.new()
try:
    f = open(outfile, 'r')
    orig.update(f.read())
except IOError:
    pass
else:
    f.close()

# Now read module files to generate potential new data
output = StringIO.StringIO()
output.write(header)

# List of fortran source files.
fortran_files = glob.glob("*/*.F") + glob.glob("*/*.F90")

module_re = re.compile(r"^\s*module\s+(\w+)\s*$", re.IGNORECASE | re.MULTILINE)

module_list = []

for filename in fortran_files:

    fortran = file(filename, "r").read()

    modules = module_re.findall(fortran)

    for module in modules:

        if re.search(
            r"^\s*subroutine\s+" + module + "_register_diagnostic\s*$",
            fortran,
                re.IGNORECASE | re.MULTILINE):
            module_list.append(module)

for module in module_list:

    output.write("  use " + module +
                 ", only: " + module + "_register_diagnostic\n")

# Ensure that the subroutine is legal in the trivial case.
output.write("""
   continue
   """)

for module in module_list:

    output.write("  call " + module + "_register_diagnostic\n")

output.write(footer)

new = sha.new()
new.update(output.getvalue())

# Only write file if sha1sums differ
if new.digest() != orig.digest():
    try:
        f = open(outfile, 'w')
        f.write(output.getvalue())
    except IOError:
        # Fixme, this should fail better
        pass
    else:
        f.close()
output.close()
