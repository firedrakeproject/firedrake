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

import pytest
import numpy
import random

from pyop2 import op2
from pyop2.computeind import compute_ind_extr

backends = ['sequential', 'openmp']

def _seed():
    return 0.02041724

# Large enough that there is more than one block and more than one
# thread per element in device backends
nelems = 32
nnodes = nelems + 2
nedges = 2*nelems + 1

nums = numpy.array([nnodes, nedges, nelems])

layers = 11
wedges = layers - 1
partition_size = 300

mesh2d = numpy.array([3,3,1])
mesh1d = numpy.array([2,1])
A = numpy.array([[0,1],[0]])

dofs = numpy.array([[2,0],[0,0],[0,1]])
dofs_coords = numpy.array([[2,0],[0,0],[0,0]])
dofs_field = numpy.array([[0,0],[0,0],[0,1]])

off1 = numpy.array([2,2,2,2,2,2], dtype=numpy.int32)
off2 = numpy.array([1], dtype=numpy.int32)

noDofs = numpy.dot(mesh2d,dofs)
noDofs = len(A[0])*noDofs[0] + noDofs[1]

map_dofs_coords = 6
map_dofs_field = 1

#CRATE THE MAPS
#elems to nodes
elems2nodes = numpy.zeros(mesh2d[0]*nelems, dtype=numpy.int32)
for i in range(nelems):
    elems2nodes[mesh2d[0]*i:mesh2d[0]*(i+1)] = [i,i+1,i+2]
elems2nodes = elems2nodes.reshape(nelems,3)

#elems to edges
elems2edges = numpy.zeros(mesh2d[1]*nelems, numpy.int32)
c = 0
for i in range(nelems):
    elems2edges[mesh2d[1]*i:mesh2d[1]*(i+1)] = [i+c,i+1+c,i+2+c]
    c = 1
elems2edges = elems2edges.reshape(nelems,3)

#elems to elems
elems2elems = numpy.zeros(mesh2d[2]*nelems, numpy.int32)
elems2elems[:] = range(nelems)
elems2elems = elems2elems.reshape(nelems,1)

@pytest.fixture
def iterset():
    return op2.Set(nelems, 1, "iterset")

@pytest.fixture
def indset():
    return op2.Set(nelems, 1, "indset")

@pytest.fixture
def x(indset):
    return op2.Dat(indset, range(nelems), numpy.uint32, "x")

@pytest.fixture
def iterset2indset(iterset, indset):
    u_map = numpy.array(range(nelems), dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(iterset, indset, 1, u_map, "iterset2indset")

@pytest.fixture
def elements():
    return op2.ExtrudedSet(nelems, 1, layers, "elems")

@pytest.fixture
def node_set1():
    return op2.Set(nnodes * layers, 1, "nodes1")

@pytest.fixture
def node_set2():
    return op2.Set(nnodes * layers, 2, "nodes2")

@pytest.fixture
def edge_set1():
    return op2.Set(nedges * layers, 1, "edges1")

@pytest.fixture
def elem_set1():
    return op2.Set(nelems * wedges, 1, "elems1")

@pytest.fixture
def elems_set2():
    return op2.Set(nelems * wedges, 2, "elems2")

@pytest.fixture
def dat_coords(node_set2):
    coords_size = nums[0] * layers * 2
    coords_dat = numpy.zeros(coords_size)
    count = 0
    for k in range(0, nums[0]):
        coords_dat[count:count+layers*dofs[0][0]] = numpy.tile([(k/2), k%2], layers)
        count += layers*dofs[0][0]
    return op2.Dat(node_set2, coords_dat, numpy.float64, "coords")

@pytest.fixture
def dat_field(elem_set1):
    field_size = nums[2] * wedges * 1
    field_dat = numpy.zeros(field_size)
    field_dat[:] = 1.0
    return op2.Dat(elem_set1, field_dat, numpy.float64, "field")

@pytest.fixture
def dat_c(node_set2):
    coords_size = nums[0] * layers * 2
    coords_dat = numpy.zeros(coords_size)
    count = 0
    for k in range(0, nums[0]):
        coords_dat[count:count+layers*dofs[0][0]] = numpy.tile([0, 0], layers)
        count += layers*dofs[0][0]
    return op2.Dat(node_set2, coords_dat, numpy.float64, "c")

@pytest.fixture
def dat_f(elem_set1):
    field_size = nums[2] * wedges * 1
    field_dat = numpy.zeros(field_size)
    field_dat[:] = -1.0
    return op2.Dat(elem_set1, field_dat, numpy.float64, "f")

@pytest.fixture
def coords_map(elements, node_set2):
    lsize = nums[2]*map_dofs_coords
    ind_coords = compute_ind_extr(nums, map_dofs_coords, nelems, layers, mesh2d, dofs_coords, A, wedges, elems2nodes, lsize)
    return op2.ExtrudedMap(elements, node_set2, map_dofs_coords, off1, ind_coords, "elem_dofs")

@pytest.fixture
def field_map(elements, elem_set1):
    lsize = nums[2]*map_dofs_field
    ind_field = compute_ind_extr(nums, map_dofs_field, nelems, layers, mesh2d, dofs_field, A, wedges, elems2elems, lsize)
    return op2.ExtrudedMap(elements, elem_set1, map_dofs_field, off2, ind_field, "elem_elem")

class TestExtrusion:
    """
    Indirect Loop Tests
    """

    def test_extrusion(self, backend, elements, dat_coords, dat_field, coords_map, field_map):
        g = op2.Global(1, data=0.0, name='g')
        mass = op2.Kernel("""
void comp_vol(double A[1], double *x[], double *y[], int j)
{
    double abs = x[0][0]*(x[2][1]-x[4][1])+x[2][0]*(x[4][1]-x[0][1])+x[4][0]*(x[0][1]-x[2][1]);
    if (abs < 0)
      abs = abs * (-1.0);
    A[0]+=0.5*abs*0.1 * y[0][0];
}""","comp_vol");

        op2.par_loop(mass, elements,
             g(op2.INC),
             dat_coords(coords_map, op2.READ),
             dat_field(field_map, op2.READ)
            )

        assert int(g.data[0]) == int((layers - 1) * 0.1 * (nelems/2))

    def test_write_data_field(self, backend, elements, dat_coords, dat_field, coords_map, field_map, dat_f):
        kernel_wo = "void kernel_wo(double* x[], int j) { x[0][0] = double(42); }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), elements, dat_f(field_map, op2.WRITE))

        assert all(map(lambda x: x==42, dat_f.data))

    def test_write_data_coords(self, backend, elements, dat_coords, dat_field, coords_map, field_map, dat_c):
        kernel_wo_c = """void kernel_wo_c(double* x[], int j) {
                                                               x[0][0] = double(42); x[0][1] = double(42);
                                                               x[1][0] = double(42); x[1][1] = double(42);
                                                               x[2][0] = double(42); x[2][1] = double(42);
                                                               x[3][0] = double(42); x[3][1] = double(42);
                                                               x[4][0] = double(42); x[4][1] = double(42);
                                                               x[5][0] = double(42); x[5][1] = double(42);
                                                            }\n"""
        op2.par_loop(op2.Kernel(kernel_wo_c, "kernel_wo_c"), elements, dat_c(coords_map, op2.WRITE))

        assert all(map(lambda x: x[0]==42 and x[1]==42, dat_c.data))

    def test_read_coord_neighbours_write_to_field(self, backend, elements, dat_coords, dat_field,
                    coords_map, field_map, dat_c, dat_f):
        kernel_wtf = """void kernel_wtf(double* x[], double* y[], int j) {
                                                               double sum = 0.0;
                                                               for (int i=0; i<6; i++){
                                                                    sum += x[i][0] + x[i][1];
                                                               }
                                                               y[0][0] = sum;
                                                            }\n"""
        op2.par_loop(op2.Kernel(kernel_wtf, "kernel_wtf"), elements,
                                dat_coords(coords_map, op2.READ),
                                dat_f(field_map, op2.WRITE))
        assert all(map(lambda x: x[0] >= 0, dat_f.data))

    def test_indirect_coords_inc(self, backend, elements, dat_coords, dat_field,
                    coords_map, field_map, dat_c, dat_f):
        kernel_inc = """void kernel_inc(double* x[], double* y[], int j) {
                                                               for (int i=0; i<6; i++){
                                                                 if (y[i][0] == 0){
                                                                    y[i][0] += 1;
                                                                    y[i][1] += 1;
                                                                 }
                                                               }
                                                            }\n"""
        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"), elements,
                                dat_coords(coords_map, op2.READ),
                                dat_c(coords_map, op2.INC))

        assert sum(sum(dat_c.data)) == nums[0] * layers * 2

    #TODO: extend for higher order elements

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
