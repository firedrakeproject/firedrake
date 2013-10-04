#!/usr/bin/env python

from optparse import OptionParser
import sys

face_number = 1


def write_side_boundary(f, face_start, start, stride1, stride2, N1, N2,
                        boundarymarker):
    # adds the faces of one side of the box to the .face file
    face_number = face_start
    for i2 in range(N2 - 1):
        for i1 in range(N1 - 1):
            node1 = start + i1 * stride1 + i2 * stride2 + 1
            node2 = node1 + stride1
            node3 = node2 + stride2
            f.write('%r %r %r %r %r\n' %
                    (face_number, node1, node2, node3, boundarymarker))
            face_number += 1
            node4 = node1 + stride2
            f.write('%r %r %r %r %r\n' %
                    (face_number, node1, node4, node3, boundarymarker))
            face_number += 1

#
# Script starts here.
optparser = OptionParser(
    add_help_option=True,
    description="""Creates mesh that lines up in all directions, so that you
    can make it into a singly, doubly or triply periodic mesh.""")

optparser.set_usage(
    "usage: %prog meshname Lx Ly Lz Nx Ny Nz [Ox Oy Oz]\n\n" +
    "Lx,Ly,Lz are dimensions of the box.\n" +
    "Nx,Ny,Nz are the number of layers in each direction." +
    "Ox,Oy,Oz is the origin (defaults to 0,0,0)."
)

(options, argv) = optparser.parse_args()

if len(argv) < 7:
    optparser.print_help()
    sys.exit(1)

meshname = argv[0]
# dimensions of the box
Lx = float(argv[1])
Ly = float(argv[2])
Lz = float(argv[3])
# number of nodes in each direction
Nx = int(argv[4]) + 1
Ny = int(argv[5]) + 1
Nz = int(argv[6]) + 1
# origin
if len(argv) > 7:
    Ox = float(argv[7])
    Oy = float(argv[8])
    Oz = float(argv[9])
else:
    Ox = 0.0
    Oy = 0.0
    Oz = 0.0

nodes = Nx * Ny * Nz
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dz = Lz / (Nz - 1)

#
# write the .node file
f = file(meshname + '.node', 'w')
f.write('%r 3 1 0\n' % nodes)
for i in range(nodes):
    zlayer = i / (Nx * Ny)
    j = i - zlayer * (Nx * Ny)
    ylayer = j / Nx
    xlayer = j - ylayer * Nx
    x = Ox + xlayer * dx
    y = Oy + ylayer * dy
    z = Oz + zlayer * dz
    column = j + 1
    f.write('%r %r %r %r %r\n' % (i + 1, x, y, z, column))
f.close()

#
# write the .face file
f = file(meshname + '.face', 'w')
faces = 4 * ((Nx - 1) * (Ny - 1) + (Ny - 1) * (Nz - 1) + (Nx - 1) * (Nz - 1))
f.write('%r 1\n' % faces)
# bottom side
face_number = 1
write_side_boundary(f, face_number, 0, 1, Nx, Nx, Ny, 1)
face_number = face_number + 2 * (Nx - 1) * (Ny - 1)
# top side
write_side_boundary(f, face_number, Nx * Ny * (Nz - 1), 1, Nx, Nx, Ny, 2)
face_number = face_number + 2 * (Nx - 1) * (Ny - 1)
# front
write_side_boundary(f, face_number, 0, 1, Nx * Ny, Nx, Nz, 3)
face_number = face_number + 2 * (Nx - 1) * (Nz - 1)
# back
write_side_boundary(f, face_number, Nx * (Ny - 1), 1, Nx * Ny, Nx, Nz, 4)
face_number = face_number + 2 * (Nx - 1) * (Nz - 1)
# left side
write_side_boundary(f, face_number, 0, Nx, Nx * Ny, Ny, Nz, 5)
face_number = face_number + 2 * (Ny - 1) * (Nz - 1)
# right side
write_side_boundary(f, face_number, Nx - 1, Nx, Nx * Ny, Ny, Nz, 6)
f.close()

#
# write the .ele file
f = file(meshname + '.ele', 'w')
elements = 6 * (Nx - 1) * (Ny - 1) * (Nz - 1)
f.write('%r 4 1\n' % elements)
region_id = 0  # this doesn't change at the moment
ele_number = 1
for ix in range(Nx - 1):
    for iy in range(Ny - 1):
        for iz in range(Nz - 1):
            # the first node in the cube
            start = ix + iy * Nx + iz * (Nx * Ny) + 1
            # work out the 8 nodes in the cube
            node1 = start
            node2 = start + 1
            node3 = start + Nx
            node4 = node3 + 1
            node5 = start + Nx * Ny
            node6 = node5 + 1
            node8 = node5 + Nx
            node7 = node8 + 1
            # trust us, the following is right:
            f.write('%r %r %r %r %r %r\n' %
                    (ele_number, node1, node4, node3, node7, region_id))
            ele_number += 1
            f.write('%r %r %r %r %r %r\n' %
                    (ele_number, node1, node7, node8, node5, region_id))
            ele_number += 1
            f.write('%r %r %r %r %r %r\n' %
                    (ele_number, node8, node3, node7, node1, region_id))
            ele_number += 1
            f.write('%r %r %r %r %r %r\n' %
                    (ele_number, node1, node2, node4, node7, region_id))
            ele_number += 1
            f.write('%r %r %r %r %r %r\n' %
                    (ele_number, node1, node7, node5, node6, region_id))
            ele_number += 1
            f.write('%r %r %r %r %r %r\n' %
                    (ele_number, node1, node2, node7, node6, region_id))
            ele_number += 1
f.close()
