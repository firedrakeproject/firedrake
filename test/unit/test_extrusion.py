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
from numpy.testing import assert_allclose

from pyop2 import op2
from pyop2.computeind import compute_ind_extr

from coffee.base import *

backends = ['sequential', 'openmp']

# Data type
valuetype = numpy.float64

# Constants
NUM_ELE = 2
NUM_NODES = 4
NUM_DIMS = 2


def _seed():
    return 0.02041724

# Large enough that there is more than one block and more than one
# thread per element in device backends
nelems = 32
nnodes = nelems + 2
nedges = 2 * nelems + 1

nums = numpy.array([nnodes, nedges, nelems])

layers = 11
wedges = layers - 1
partition_size = 300

mesh2d = numpy.array([3, 3, 1])
mesh1d = numpy.array([2, 1])
A = numpy.array([[0, 1], [0]])

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
            [(k / 2), k % 2], layers)
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
        coords_dat[count:count + layers *
                   dofs[0][0]] = numpy.tile([0, 0], layers)
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
    nset = op2.Set(NUM_NODES * layers)
    return op2.ExtrudedSet(nset, layers=layers)


@pytest.fixture
def xtr_dnodes(xtr_nodes):
    return op2.DataSet(xtr_nodes, 1, "xtr_dnodes")


@pytest.fixture
def xtr_elem_node(xtr_elements, xtr_nodes):
    return op2.Map(xtr_elements, xtr_nodes, 6, xtr_elem_node_map, "xtr_elem_node",
                   numpy.array([1, 1, 1, 1, 1, 1], dtype=numpy.int32))


@pytest.fixture
def xtr_mat(xtr_elem_node, xtr_dnodes):
    sparsity = op2.Sparsity((xtr_dnodes, xtr_dnodes), (
        xtr_elem_node, xtr_elem_node), "xtr_sparsity")
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
void extrusion_kernel(double *xtr[], double *x[], int* j[])
{
    //Only the Z-coord is increased, the others stay the same
    xtr[0][0] = x[0][0];
    xtr[0][1] = x[0][1];
    xtr[0][2] = 0.1*j[0][0];
}"""
        return op2.Kernel(kernel_code, "extrusion_kernel")


@pytest.fixture
def vol_comp():
        init = FlatBlock("""
double area = x[0][0]*(x[2][1]-x[4][1]) + x[2][0]*(x[4][1]-x[0][1])
           + x[4][0]*(x[0][1]-x[2][1]);
if (area < 0)
area = area * (-1.0);
""")
        assembly = Incr(Symbol("A", ("i0", "i1")),
                        FlatBlock("0.5 * area * (x[1][2] - x[0][2])"))
        assembly = c_for("i0", 6, c_for("i1", 6, assembly))
        kernel_code = FunDecl("void", "vol_comp",
                              [Decl("double", Symbol("A", (6, 6))),
                               Decl("double", c_sym("*x[]"))],
                              Block([init, assembly], open_scope=False))
        return op2.Kernel(kernel_code, "vol_comp")


@pytest.fixture
def vol_comp_rhs():
        init = FlatBlock("""
double area = x[0][0]*(x[2][1]-x[4][1]) + x[2][0]*(x[4][1]-x[0][1])
           + x[4][0]*(x[0][1]-x[2][1]);
if (area < 0)
area = area * (-1.0);
""")
        assembly = Incr(Symbol("A", ("i0",)),
                        FlatBlock("0.5 * area * (x[1][2] - x[0][2]) * y[0][0]"))
        assembly = c_for("i0", 6, assembly)
        kernel_code = FunDecl("void", "vol_comp_rhs",
                              [Decl("double", Symbol("A", (6,))),
                               Decl("double", c_sym("*x[]")),
                               Decl("int", c_sym("*y[]"))],
                              Block([init, assembly], open_scope=False))
        return op2.Kernel(kernel_code, "vol_comp_rhs")


class TestExtrusion:

    """
    Extruded Mesh Tests
    """

    def test_extrusion(self, backend, elements, dat_coords, dat_field, coords_map, field_map):
        g = op2.Global(1, data=0.0, name='g')
        mass = op2.Kernel("""
void comp_vol(double A[1], double *x[], double *y[])
{
    double abs = x[0][0]*(x[2][1]-x[4][1])+x[2][0]*(x[4][1]-x[0][1])+x[4][0]*(x[0][1]-x[2][1]);
    if (abs < 0)
      abs = abs * (-1.0);
    A[0]+=0.5*abs*0.1 * y[0][0];
}""", "comp_vol")

        op2.par_loop(mass, elements,
                     g(op2.INC),
                     dat_coords(op2.READ, coords_map),
                     dat_field(op2.READ, field_map))

        assert int(g.data[0]) == int((layers - 1) * 0.1 * (nelems / 2))

    def test_extruded_nbytes(self, backend, dat_field):
        """Nbytes computes the number of bytes occupied by an extruded Dat."""
        assert dat_field.nbytes == nums[2] * wedges * 8

    def test_direct_loop_inc(self, backend, xtr_nodes):
        dat = op2.Dat(xtr_nodes)
        k = 'void k(double *x) { *x += 1.0; }'
        dat.data[:] = 0
        op2.par_loop(op2.Kernel(k, 'k'),
                     dat.dataset.set, dat(op2.INC))
        assert numpy.allclose(dat.data[:], 1.0)

    def test_write_data_field(self, backend, elements, dat_coords, dat_field, coords_map, field_map, dat_f):
        kernel_wo = "void kernel_wo(double* x[]) { x[0][0] = 42.0; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"),
                     elements, dat_f(op2.WRITE, field_map))

        assert all(map(lambda x: x == 42, dat_f.data))

    def test_write_data_coords(self, backend, elements, dat_coords, dat_field, coords_map, field_map, dat_c):
        kernel_wo_c = """void kernel_wo_c(double* x[]) {
                                                               x[0][0] = 42.0; x[0][1] = 42.0;
                                                               x[1][0] = 42.0; x[1][1] = 42.0;
                                                               x[2][0] = 42.0; x[2][1] = 42.0;
                                                               x[3][0] = 42.0; x[3][1] = 42.0;
                                                               x[4][0] = 42.0; x[4][1] = 42.0;
                                                               x[5][0] = 42.0; x[5][1] = 42.0;
                                                            }\n"""
        op2.par_loop(op2.Kernel(kernel_wo_c, "kernel_wo_c"),
                     elements, dat_c(op2.WRITE, coords_map))

        assert all(map(lambda x: x[0] == 42 and x[1] == 42, dat_c.data))

    def test_read_coord_neighbours_write_to_field(
        self, backend, elements, dat_coords, dat_field,
            coords_map, field_map, dat_c, dat_f):
        kernel_wtf = """void kernel_wtf(double* x[], double* y[]) {
                                                               double sum = 0.0;
                                                               for (int i=0; i<6; i++){
                                                                    sum += x[i][0] + x[i][1];
                                                               }
                                                               y[0][0] = sum;
                                                            }\n"""
        op2.par_loop(op2.Kernel(kernel_wtf, "kernel_wtf"), elements,
                     dat_coords(op2.READ, coords_map),
                     dat_f(op2.WRITE, field_map))
        assert all(dat_f.data >= 0)

    def test_indirect_coords_inc(self, backend, elements, dat_coords,
                                 dat_field, coords_map, field_map, dat_c,
                                 dat_f):
        kernel_inc = """void kernel_inc(double* x[], double* y[]) {
                                                               for (int i=0; i<6; i++){
                                                                 if (y[i][0] == 0){
                                                                    y[i][0] += 1;
                                                                    y[i][1] += 1;
                                                                 }
                                                               }
                                                            }\n"""
        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"), elements,
                     dat_coords(op2.READ, coords_map),
                     dat_c(op2.INC, coords_map))

        assert sum(sum(dat_c.data)) == nums[0] * layers * 2

    def test_extruded_assemble_mat_rhs_solve(
        self, backend, xtr_mat, xtr_coords, xtr_elements,
        xtr_elem_node, extrusion_kernel, xtr_nodes, vol_comp,
            xtr_dnodes, vol_comp_rhs, xtr_b):
        coords_dim = 3
        coords_xtr_dim = 3  # dimension
        # BIG TRICK HERE:
        # We need the +1 in order to include the entire column of vertices.
        # Extrusion is meant to iterate over the 3D cells which are layer - 1 in number.
        # The +1 correction helps in the case of iteration over vertices which need
        # one extra layer.
        iterset = op2.Set(NUM_NODES, "verts1")
        iterset = op2.ExtrudedSet(iterset, layers=(layers + 1))
        vnodes = op2.DataSet(iterset, coords_dim)

        d_nodes_xtr = op2.DataSet(xtr_nodes, coords_xtr_dim)
        d_lnodes_xtr = op2.DataSet(xtr_nodes, 1)

        # Create an op2.Dat with the base mesh coordinates
        coords_vec = numpy.zeros(vnodes.total_size * coords_dim)
        length = len(xtr_coords.flatten())
        coords_vec[0:length] = xtr_coords.flatten()
        coords = op2.Dat(vnodes, coords_vec, numpy.float64, "dat1")

        # Create an op2.Dat with slots for the extruded coordinates
        coords_new = numpy.array(
            [0.] * layers * NUM_NODES * coords_xtr_dim, dtype=numpy.float64)
        coords_xtr = op2.Dat(d_nodes_xtr, coords_new, numpy.float64, "dat_xtr")

        # Creat an op2.Dat to hold the layer number
        layer_vec = numpy.tile(numpy.arange(0, layers), NUM_NODES)
        layer = op2.Dat(d_lnodes_xtr, layer_vec, numpy.int32, "dat_layer")

        # Map a map for the bottom of the mesh.
        vertex_to_coords = [i for i in range(0, NUM_NODES)]
        v2coords_offset = numpy.array([0], numpy.int32)
        map_2d = op2.Map(iterset, iterset, 1, vertex_to_coords, "v2coords", v2coords_offset)

        # Create Map for extruded vertices
        vertex_to_xtr_coords = [layers * i for i in range(0, NUM_NODES)]
        v2xtr_coords_offset = numpy.array([1], numpy.int32)
        map_xtr = op2.Map(
            iterset, xtr_nodes, 1, vertex_to_xtr_coords, "v2xtr_coords", v2xtr_coords_offset)

        # Create Map for layer number
        v2xtr_layer_offset = numpy.array([1], numpy.int32)
        layer_xtr = op2.Map(
            iterset, xtr_nodes, 1, vertex_to_xtr_coords, "v2xtr_layer", v2xtr_layer_offset)

        op2.par_loop(extrusion_kernel, iterset,
                     coords_xtr(op2.INC, map_xtr, flatten=True),
                     coords(op2.READ, map_2d, flatten=True),
                     layer(op2.READ, layer_xtr))

        # Assemble the main matrix.
        op2.par_loop(vol_comp, xtr_elements,
                     xtr_mat(op2.INC, (xtr_elem_node[op2.i[0]], xtr_elem_node[op2.i[1]])),
                     coords_xtr(op2.READ, xtr_elem_node))

        eps = 1.e-5
        xtr_mat.assemble()
        assert_allclose(sum(sum(xtr_mat.values)), 36.0, eps)

        # Assemble the RHS
        xtr_f_vals = numpy.array([1] * NUM_NODES * layers, dtype=numpy.int32)
        xtr_f = op2.Dat(d_lnodes_xtr, xtr_f_vals, numpy.int32, "xtr_f")

        op2.par_loop(vol_comp_rhs, xtr_elements,
                     xtr_b(op2.INC, xtr_elem_node[op2.i[0]], flatten=True),
                     coords_xtr(op2.READ, xtr_elem_node, flatten=True),
                     xtr_f(op2.READ, xtr_elem_node))

        assert_allclose(sum(xtr_b.data), 6.0, eps)

        x_vals = numpy.zeros(NUM_NODES * layers, dtype=valuetype)
        xtr_x = op2.Dat(d_lnodes_xtr, x_vals, valuetype, "xtr_x")

        op2.solve(xtr_mat, xtr_x, xtr_b)

        assert_allclose(sum(xtr_x.data), 7.3333333, eps)

    # TODO: extend for higher order elements

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
