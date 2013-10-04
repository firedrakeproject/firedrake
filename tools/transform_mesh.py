#!/usr/bin/env python

from optparse import OptionParser
import sys
import shutil
import math


#
# Script starts here.
optparser = OptionParser(usage='usage: %prog transformation mesh',
                         add_help_option=True,
                         description="""Applies a coordinate transformation to
                         the given mesh.""")

optparser.set_usage("""usage: %prog <transformation> <mesh>

<transformation> is a python expression giving the coordinate transformation.
<mesh> is the name of the triangle mesh files. You need a mesh.node, mesh.face
and mesh.ele file.

Example:
To rescale the z-dimension by a factor of 1000,
%prog '(x,y,1000*z)' mesh.""")

(options, argv) = optparser.parse_args()

if len(argv) != 2:
    optparser.print_help()
    sys.exit(1)

transformation = argv[0]
mesh_name = argv[1]

# make all definitions of the math module available in
# the transformation expression.
globals = math.__dict__

f = file(mesh_name + '.node', 'r')
newf = file(mesh_name + '.node.tmp', 'w')
header = f.readline()
nodes = int(header.split(' ')[0])
dim = int(header.split(' ')[1])
newf.write(header)

for line in f:
    # remove spaces leading and trailing spaces and the end of line character:
    line = line.lstrip().rstrip()
    if line.startswith('#'):
        continue
    cols = line.split(' ')
    index = int(cols[0])
    globals['x'] = float(cols[1])
    globals['y'] = float(cols[2])
    if dim == 3:
        globals['z'] = float(cols[3])
    xyz = eval(transformation, globals)
    if dim == 2:
        newf.write("%r %r %r\n" % (index, xyz[0], xyz[1]))
    else:
        newf.write("%r %r %r\n" % (index, xyz[0], xyz[1], xyz[2]))

newf.close()
f.close()

shutil.move(mesh_name + '.node', mesh_name + '.node.bak')
shutil.move(mesh_name + '.node.tmp', mesh_name + '.node')
