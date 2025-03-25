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
from pyop2.mpi import COMM_WORLD


def compute_ind_extr(nums,
                     map_dofs,
                     lins,
                     layers,
                     mesh2d,
                     dofs,
                     A,
                     wedges,
                     map,
                     lsize):
    count = 0
    ind = numpy.zeros(lsize, dtype=numpy.int32)
    len1 = len(mesh2d)
    for mm in range(lins):
        offset = 0
        for d in range(2):
            c = 0
            for i in range(len1):
                a4 = dofs[i, d]
                if a4 != 0:
                    len2 = len(A[d])
                    for j in range(0, mesh2d[i]):
                        m = map[mm][c]
                        for k in range(0, len2):
                            ind[count] = m*(layers - d) + A[d][k] + offset
                            count += 1
                        c += 1
                elif dofs[i, 1-d] != 0:
                    c += mesh2d[i]
                offset += a4*nums[i]*(layers - d)
    return ind


# Data type
valuetype = numpy.float64

# Constants
NUM_ELE = 2
NUM_NODES = 4
NUM_DIMS = 2


def _seed():
    return 0.02041724


nelems = 32
nnodes = nelems + 2
nedges = 2 * nelems + 1

nums = numpy.array([nnodes, nedges, nelems])

layers = 11
wedges = layers - 1
partition_size = 300

mesh2d = numpy.array([3, 3, 1])
mesh1d = numpy.array([2, 1])
A = [[0, 1], [0]]

dofs = numpy.array([[2, 0], [0, 0], [0, 1]])
dofs_coords = numpy.array([[2, 0], [0, 0], [0, 0]])
dofs_field = numpy.array([[0, 0], [0, 0], [0, 1]])

off1 = numpy.array([1, 1, 1, 1, 1, 1], dtype=numpy.int32)
off2 = numpy.array([1], dtype=numpy.int32)

noDofs = numpy.dot(mesh2d, dofs)
noDofs = len(A[0]) * noDofs[0] + noDofs[1]

map_dofs_coords = 6
map_dofs_field = 1

# CRATE THE MAPS
# elems to nodes
elems2nodes = numpy.zeros(mesh2d[0] * nelems, dtype=numpy.int32)
for i in range(nelems):
    elems2nodes[mesh2d[0] * i:mesh2d[0] * (i + 1)] = [i, i + 1, i + 2]
elems2nodes = elems2nodes.reshape(nelems, 3)

# elems to edges
elems2edges = numpy.zeros(mesh2d[1] * nelems, numpy.int32)
c = 0
for i in range(nelems):
    elems2edges[mesh2d[1] * i:mesh2d[1] * (i + 1)] = [
        i + c, i + 1 + c, i + 2 + c]
    c = 1
elems2edges = elems2edges.reshape(nelems, 3)

# elems to elems
elems2elems = numpy.zeros(mesh2d[2] * nelems, numpy.int32)
elems2elems[:] = range(nelems)
elems2elems = elems2elems.reshape(nelems, 1)

xtr_elem_node_map = numpy.asarray(
    [0, 1, 11, 12, 33, 34, 22, 23, 33, 34, 11, 12], dtype=numpy.uint32)


@pytest.fixture
def iterset():
    return op2.Set(nelems, "iterset")


@pytest.fixture
def indset():
    return op2.Set(nelems, "indset")


@pytest.fixture
def diterset(iterset):
    return op2.DataSet(iterset, 1, "diterset")


@pytest.fixture
def dindset(indset):
    return op2.DataSet(indset, 1, "dindset")


@pytest.fixture
def x(dindset):
    return op2.Dat(dindset, range(nelems), numpy.uint32, "x")


@pytest.fixture
def iterset2indset(iterset, indset):
    u_map = numpy.array(range(nelems), dtype=numpy.uint32)
    random.shuffle(u_map, _seed)
    return op2.Map(iterset, indset, 1, u_map, "iterset2indset")


@pytest.fixture
def elements():
    s = op2.Set(nelems)
    return op2.ExtrudedSet(s, layers=layers)


@pytest.fixture
def node_set1():
    return op2.Set(nnodes * layers, "nodes1")


@pytest.fixture
def edge_set1():
    return op2.Set(nedges * layers, "edges1")


@pytest.fixture
def elem_set1():
    return op2.Set(nelems * wedges, "elems1")


@pytest.fixture
def dnode_set1(node_set1):
    return op2.DataSet(node_set1, 1, "dnodes1")


@pytest.fixture
def dnode_set2(node_set1):
    return op2.DataSet(node_set1, 2, "dnodes2")


@pytest.fixture
def dedge_set1(edge_set1):
    return op2.DataSet(edge_set1, 1, "dedges1")


@pytest.fixture
def delem_set1(elem_set1):
    return op2.DataSet(elem_set1, 1, "delems1")


@pytest.fixture
def delems_set2(elem_set1):
    return op2.DataSet(elem_set1, 2, "delems2")


@pytest.fixture
def dat_coords(dnode_set2):
    coords_size = nums[0] * layers * 2
    coords_dat = numpy.zeros(coords_size)
    count = 0
    for k in range(0, nums[0]):
        coords_dat[count:count + layers * dofs[0][0]] = numpy.tile(
            [(k // 2), k % 2], layers)
        count += layers * dofs[0][0]
    return op2.Dat(dnode_set2, coords_dat, numpy.float64, "coords")


@pytest.fixture
def dat_field(delem_set1):
    field_size = nums[2] * wedges * 1
    field_dat = numpy.zeros(field_size)
    field_dat[:] = 1.0
    return op2.Dat(delem_set1, field_dat, numpy.float64, "field")


@pytest.fixture
def dat_c(dnode_set2):
    coords_size = nums[0] * layers * 2
    coords_dat = numpy.zeros(coords_size)
    count = 0
    for k in range(0, nums[0]):
        coords_dat[count:count + layers * dofs[0][0]] = numpy.tile([0, 0], layers)
        count += layers * dofs[0][0]
    return op2.Dat(dnode_set2, coords_dat, numpy.float64, "c")


@pytest.fixture
def dat_f(delem_set1):
    field_size = nums[2] * wedges * 1
    field_dat = numpy.zeros(field_size)
    field_dat[:] = -1.0
    return op2.Dat(delem_set1, field_dat, numpy.float64, "f")


@pytest.fixture
def coords_map(elements, node_set1):
    lsize = nums[2] * map_dofs_coords
    ind_coords = compute_ind_extr(
        nums, map_dofs_coords, nelems, layers, mesh2d, dofs_coords, A, wedges, elems2nodes, lsize)
    return op2.Map(elements, node_set1, map_dofs_coords, ind_coords, "elem_dofs", off1)


@pytest.fixture
def field_map(elements, elem_set1):
    lsize = nums[2] * map_dofs_field
    ind_field = compute_ind_extr(
        nums, map_dofs_field, nelems, layers, mesh2d, dofs_field, A, wedges, elems2elems, lsize)
    return op2.Map(elements, elem_set1, map_dofs_field, ind_field, "elem_elem", off2)


@pytest.fixture
def xtr_elements():
    eset = op2.Set(NUM_ELE)
    return op2.ExtrudedSet(eset, layers=layers)


@pytest.fixture
def xtr_nodes():
    return op2.Set(NUM_NODES * layers)


@pytest.fixture
def xtr_dnodes(xtr_nodes):
    return op2.DataSet(xtr_nodes, 1, "xtr_dnodes")


@pytest.fixture
def xtr_elem_node(xtr_elements, xtr_nodes):
    return op2.Map(xtr_elements, xtr_nodes, 6, xtr_elem_node_map, "xtr_elem_node",
                   numpy.array([1, 1, 1, 1, 1, 1], dtype=numpy.int32))


@pytest.fixture
def xtr_mat(xtr_elem_node, xtr_dnodes):
    sparsity = op2.Sparsity((xtr_dnodes, xtr_dnodes), {(0, 0): [(xtr_elem_node, xtr_elem_node, None, None)]}, "xtr_sparsity")
    return op2.Mat(sparsity, valuetype, "xtr_mat")


@pytest.fixture
def xtr_dvnodes(xtr_nodes):
    return op2.DataSet(xtr_nodes, 3, "xtr_dvnodes")


@pytest.fixture
def xtr_b(xtr_dnodes):
    b_vals = numpy.zeros(NUM_NODES * layers, dtype=valuetype)
    return op2.Dat(xtr_dnodes, b_vals, valuetype, "xtr_b")


@pytest.fixture
def xtr_coords(xtr_dvnodes):
    coord_vals = numpy.asarray([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
                               dtype=valuetype)
    return coord_vals


@pytest.fixture
def extrusion_kernel():
    kernel_code = """
static void extrusion(double *xtr, double *x, int* j)
{
    //Only the Z-coord is increased, the others stay the same
    xtr[0] = x[0];
    xtr[1] = x[1];
    xtr[2] = 0.1*j[0];
}"""
    return op2.Kernel(kernel_code, "extrusion")


class TestExtrusion:

    """
    Extruded Mesh Tests
    """

    def test_extrusion(self, elements, dat_coords, dat_field, coords_map, field_map):
        g = op2.Global(1, data=0.0, name='g', comm=COMM_WORLD)
        mass = op2.Kernel("""
static void comp_vol(double A[1], double x[6][2], double y[1])
{
    double abs = x[0][0]*(x[2][1]-x[4][1])+x[2][0]*(x[4][1]-x[0][1])+x[4][0]*(x[0][1]-x[2][1]);
    if (abs < 0)
      abs = abs * (-1.0);
    A[0]+=0.5*abs*0.1 * y[0];
}""", "comp_vol")

        op2.par_loop(mass, elements,
                     g(op2.INC),
                     dat_coords(op2.READ, coords_map),
                     dat_field(op2.READ, field_map))

        assert int(g.data[0]) == int((layers - 1) * 0.1 * (nelems // 2))

    def test_extruded_nbytes(self, dat_field):
        """Nbytes computes the number of bytes occupied by an extruded Dat."""
        assert dat_field.nbytes == nums[2] * wedges * 8

    def test_direct_loop_inc(self, iterset, diterset):
        dat = op2.Dat(diterset)
        xtr_iterset = op2.ExtrudedSet(iterset, layers=10)
        k = 'static void k(double *x) { *x += 1.0; }'
        dat.data[:] = 0
        op2.par_loop(op2.Kernel(k, 'k'),
                     xtr_iterset, dat(op2.INC))
        assert numpy.allclose(dat.data[:], 9.0)

    def test_extruded_layer_arg(self, elements, field_map, dat_f):
        """Tests that the layer argument is being passed when prompted
        to in the parloop."""

        kernel_blah = """
        static void blah(double* x, int layer_arg){
        x[0] = layer_arg;
        }"""

        op2.par_loop(op2.Kernel(kernel_blah, "blah"),
                     elements, dat_f(op2.WRITE, field_map),
                     pass_layer_arg=True)
        end = layers - 1
        start = 0
        ref = numpy.arange(start, end)
        assert [dat_f.data[end*n:end*(n+1)] == ref
                for n in range(int(len(dat_f.data)/end) - 1)]

    def test_write_data_field(self, elements, dat_coords, dat_field, coords_map, field_map, dat_f):
        kernel_wo = "static void wo(double* x) { x[0] = 42.0; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "wo"),
                     elements, dat_f(op2.WRITE, field_map))

        assert all(map(lambda x: x == 42, dat_f.data))

    def test_write_data_coords(self, elements, dat_coords, dat_field, coords_map, field_map, dat_c):
        kernel_wo_c = """
        static void wo_c(double x[6][2]) {
           x[0][0] = 42.0; x[0][1] = 42.0;
           x[1][0] = 42.0; x[1][1] = 42.0;
           x[2][0] = 42.0; x[2][1] = 42.0;
           x[3][0] = 42.0; x[3][1] = 42.0;
           x[4][0] = 42.0; x[4][1] = 42.0;
           x[5][0] = 42.0; x[5][1] = 42.0;
        }"""
        op2.par_loop(op2.Kernel(kernel_wo_c, "wo_c"),
                     elements, dat_c(op2.WRITE, coords_map))

        assert all(map(lambda x: x[0] == 42 and x[1] == 42, dat_c.data))

    def test_read_coord_neighbours_write_to_field(
        self, elements, dat_coords, dat_field,
            coords_map, field_map, dat_c, dat_f):
        kernel_wtf = """
        static void wtf(double* y, double x[6][2]) {
           double sum = 0.0;
           for (int i=0; i<6; i++){
                sum += x[i][0] + x[i][1];
           }
           y[0] = sum;
        }"""
        op2.par_loop(op2.Kernel(kernel_wtf, "wtf"), elements,
                     dat_f(op2.WRITE, field_map),
                     dat_coords(op2.READ, coords_map),)
        assert all(dat_f.data >= 0)

    def test_indirect_coords_inc(self, elements, dat_coords,
                                 dat_field, coords_map, field_map, dat_c,
                                 dat_f):
        kernel_inc = """
        static void inc(double y[6][2], double x[6][2]) {
           for (int i=0; i<6; i++){
             if (y[i][0] == 0){
                y[i][0] += 1;
                y[i][1] += 1;
             }
           }
        }"""
        op2.par_loop(op2.Kernel(kernel_inc, "inc"), elements,
                     dat_c(op2.RW, coords_map),
                     dat_coords(op2.READ, coords_map))

        assert sum(sum(dat_c.data)) == nums[0] * layers * 2


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
