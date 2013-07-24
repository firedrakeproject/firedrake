# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This demo verifies that the integral of a unit cube is 1.

The cube will be unstructured in the 2D plane and structured vertically.
"""

from pyop2 import op2, utils
from triangle_reader import read_triangle
from ufl import *
from pyop2.computeind import compute_ind_extr

import numpy as np
import time

parser = utils.parser(group=True, description="PyOP2 2D mass equation demo")
parser.add_argument('-m', '--mesh', action='store', type=str, required=True,
                    help='Base name of triangle mesh \
                          (excluding the .ele or .node extension)')
parser.add_argument('-ll', '--layers', action='store', type=str, required=True,
                    help='Number of extruded layers.')
parser.add_argument('-p', '--partsize', action='store', type=str,
                    required=False, default=1024,
                    help='Partition size in the base mesh.')
opt = vars(parser.parse_args())
op2.init(**opt)
mesh_name = opt['mesh']
layers = int(opt['layers'])
partition_size = int(opt['partsize'])

# Generate code for kernel

mass = op2.Kernel("""
void comp_vol(double A[1], double *x[], double *y[], double *z[])
{
  double area = x[0][0]*(x[2][1]-x[4][1]) + x[2][0]*(x[4][1]-x[0][1])
               + x[4][0]*(x[0][1]-x[2][1]);
  if (area < 0)
    area = area * (-1.0);
  A[0]+=0.5*area*0.1 * y[0][0];

  z[0][0]+=0.2*(0.5*area*0.1*y[0][0]);
  z[1][0]+=0.2*(0.5*area*0.1*y[0][0]);
  z[2][0]+=0.2*(0.5*area*0.1*y[0][0]);
  z[3][0]+=0.2*(0.5*area*0.1*y[0][0]);
  z[4][0]+=0.2*(0.5*area*0.1*y[0][0]);
  z[5][0]+=0.2*(0.5*area*0.1*y[0][0]);
}""", "comp_vol")

# Set up simulation data structures
valuetype = np.float64

nodes, coords, elements, elem_node = read_triangle(mesh_name, layers)

# mesh data
mesh2d = np.array([3, 3, 1])
mesh1d = np.array([2, 1])
A = np.array([[0, 1], [0]])

# the array of dof values for each element type
dofs = np.array([[2, 0], [0, 0], [0, 1]])
dofs_coords = np.array([[2, 0], [0, 0], [0, 0]])
dofs_field = np.array([[0, 0], [0, 0], [0, 1]])
dofs_res = np.array([[1, 0], [0, 0], [0, 0]])

# ALL the nodes, edges amd cells of the 2D mesh
nums = np.array([nodes.size, 0, elements.size])

# compute the various numbers of dofs
dofss = dofs.transpose().ravel()

# number of dofs
noDofs = 0  # number of dofs
noDofs = np.dot(mesh2d, dofs)
noDofs = len(A[0]) * noDofs[0] + noDofs[1]

# Number of elements in the map only counts the first reference to the
# dofs related to a mesh element
map_dofs = 0
for d in range(0, 2):
    for i in range(0, len(mesh2d)):
        for j in range(0, mesh2d[i] * len(A[d])):
            if dofs[i][d] != 0:
                map_dofs += 1

map_dofs_coords = 6
map_dofs_field = 1
map_dofs_res = 6

# EXTRUSION DETAILS
wedges = layers - 1

# NEW MAP
# When building this map we need to make sure we leave space for the maps that
# might be missing. This is because when we construct the ind array we need to
# know which maps is associated with each dof. If the element to node is
# missing then we will have the cell to edges in the first position which is bad
# RULE: if all the dofs in the line are ZERO then skip that mapping else add it

mappp = elem_node.values
mappp = mappp.reshape(-1, 3)

lins, cols = mappp.shape
mapp_coords = np.empty(shape=(lins,), dtype=object)

t0ind = time.clock()
# DERIVE THE MAP FOR THE EDGES
edg = np.empty(shape=(nums[0],), dtype=object)
for i in range(0, nums[0]):
    edg[i] = []

k = 0
count = 0
addNodes = True
addEdges = False
addCells = False

for i in range(0, lins):  # for each cell to node mapping
    ns = mappp[i] - 1
    ns.sort()
    pairs = [(x, y) for x in ns for y in ns if x < y]
    res = np.array([], dtype=np.int32)
    if addEdges:
        for x, y in pairs:
            ys = [kk for yy, kk in edg[x] if yy == y]
            if ys == []:
                edg[x].append((y, k))
                res = np.append(res, k)
                k += 1
            else:
                res = np.append(res, ys[0])
    if addCells:
        res = np.append(res, i)  # add the map of the cell
    if addNodes:
        mapp_coords[i] = np.append(mappp[i], res)
    else:
        mapp_coords[i] = res

mapp_field = np.empty(shape=(lins,), dtype=object)
k = 0
count = 0
addNodes = False
addEdges = False
addCells = True

for i in range(0, lins):  # for each cell to node mapping
    ns = mappp[i] - 1
    ns.sort()
    pairs = [(x, y) for x in ns for y in ns if x < y]
    res = np.array([], dtype=np.int32)
    if addEdges:
        for x, y in pairs:
            ys = [kk for yy, kk in edg[x] if yy == y]
            if ys == []:
                edg[x].append((y, k))
                res = np.append(res, k)
                k += 1
            else:
                res = np.append(res, ys[0])
    if addCells:
        res = np.append(res, i)  # add the map of the cell
    if addNodes:
        mapp_field[i] = np.append(mappp[i], res)
    else:
        mapp_field[i] = res

mapp_res = np.empty(shape=(lins,), dtype=object)
k = 0
count = 0
addNodes = True
addEdges = False
addCells = False

for i in range(0, lins):  # for each cell to node mapping
    ns = mappp[i] - 1
    ns.sort()
    pairs = [(x, y) for x in ns for y in ns if x < y]
    res = np.array([], dtype=np.int32)
    if addEdges:
        for x, y in pairs:
            ys = [kk for yy, kk in edg[x] if yy == y]
            if ys == []:
                edg[x].append((y, k))
                res = np.append(res, k)
                k += 1
            else:
                res = np.append(res, ys[0])
    if addCells:
        res = np.append(res, i)  # add the map of the cell
    if addNodes:
        mapp_res[i] = np.append(mappp[i], res)
    else:
        mapp_res[i] = res

nums[1] = k  # number of edges

# construct the initial indeces ONCE
# construct the offset array ONCE
off = np.zeros(map_dofs, dtype=np.int32)
off_coords = np.zeros(map_dofs_coords, dtype=np.int32)
off_field = np.zeros(map_dofs_field, dtype=np.int32)
off_res = np.zeros(map_dofs_res, dtype=np.int32)

# THE OFFSET array
# for 2D and 3D
count = 0
for d in range(0, 2):  # for 2D and then for 3D
    for i in range(0, len(mesh2d)):  # over [3,3,1]
        for j in range(0, mesh2d[i]):
            for k in range(0, len(A[d])):
                if dofs[i][d] != 0:
                    off[count] = dofs[i][d]
                    count += 1

for i in range(0, map_dofs_coords):
    off_coords[i] = off[i]
for i in range(0, map_dofs_field):
    off_field[i] = off[i + map_dofs_coords]
for i in range(0, map_dofs_res):
    off_res[i] = 1

# assemble the dat
# compute total number of dofs in the 3D mesh
no_dofs = np.dot(nums, dofs.transpose()[0]) * layers + wedges * np.dot(
    dofs.transpose()[1], nums)

#
# THE DAT
#
t0dat = time.clock()

coords_size = nums[0] * layers * 2
coords_dat = np.zeros(coords_size)
count = 0
for k in range(0, nums[0]):
    coords_dat[count:count + layers * dofs[0][0]] = np.tile(
        coords.data[k, :], layers)
    count += layers * dofs[0][0]

field_size = nums[2] * wedges * 1
field_dat = np.zeros(field_size)
field_dat[:] = 3.0

res_size = nums[0] * layers * 1
res_dat = np.zeros(res_size)
res_dat[:] = 0.0

tdat = time.clock() - t0dat

# DECLARE OP2 STRUCTURES

coords_dofsSet = op2.Set(nums[0] * layers, "coords_dofsSet")
coords = op2.Dat(coords_dofsSet ** 2, coords_dat, np.float64, "coords")

wedges_dofsSet = op2.Set(nums[2] * wedges, "wedges_dofsSet")
field = op2.Dat(wedges_dofsSet, field_dat, np.float64, "field")

p1_dofsSet = op2.Set(nums[0] * layers, "p1_dofsSet")
res = op2.Dat(p1_dofsSet, res_dat, np.float64, "res")

# THE MAP from the ind
# create the map from element to dofs for each element in the 2D mesh
lsize = nums[2] * map_dofs_coords
ind_coords = compute_ind_extr(nums, map_dofs_coords, lins, layers, mesh2d,
                              dofs_coords, A, wedges, mapp_coords, lsize)
lsize = nums[2] * map_dofs_field
ind_field = compute_ind_extr(nums, map_dofs_field, lins, layers, mesh2d,
                             dofs_field, A, wedges, mapp_field, lsize)
lsize = nums[2] * map_dofs_res
ind_res = compute_ind_extr(nums, map_dofs_res, lins, layers, mesh2d, dofs_res,
                           A, wedges, mapp_res, lsize)

elem_dofs = op2.Map(elements, coords_dofsSet, map_dofs_coords, ind_coords,
                    "elem_dofs", off_coords)

elem_elem = op2.Map(elements, wedges_dofsSet, map_dofs_field, ind_field,
                    "elem_elem", off_field)

elem_p1_dofs = op2.Map(elements, p1_dofsSet, map_dofs_res, ind_res,
                       "elem_p1_dofs", off_res)

# THE RESULT ARRAY
g = op2.Global(1, data=0.0, name='g')

duration1 = time.clock() - t0ind

# ADD LAYERS INFO TO ITERATION SET
# the elements set must also contain the layers
elements.partition_size = partition_size

# CALL PAR LOOP
# Compute volume
tloop = 0
t0loop = time.clock()
t0loop2 = time.time()
for i in range(0, 100):
    op2.par_loop(mass, elements,
                 g(op2.INC),
                 coords(op2.READ, elem_dofs),
                 field(op2.READ, elem_elem),
                 res(op2.INC, elem_p1_dofs))
tloop += time.clock() - t0loop  # t is CPU seconds elapsed (floating point)
tloop2 = time.time() - t0loop2

ttloop = tloop / 10
print nums[0], nums[1], nums[2], layers, duration1, tloop, tloop2, g.data
print res_dat[0:6]
