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

from pyop2 import op2


def _seed():
    return 0.02041724


nnodes = 4096
nele = nnodes // 2


@pytest.fixture(scope='module')
def node():
    return op2.Set(nnodes, 'node')


@pytest.fixture(scope='module')
def ele():
    return op2.Set(nele, 'ele')


@pytest.fixture(scope='module')
def dnode(node):
    return op2.DataSet(node, 1, 'dnode')


@pytest.fixture(scope='module')
def dnode2(node):
    return op2.DataSet(node, 2, 'dnode2')


@pytest.fixture(scope='module')
def dele(ele):
    return op2.DataSet(ele, 1, 'dele')


@pytest.fixture(scope='module')
def dele2(ele):
    return op2.DataSet(ele, 2, 'dele2')


@pytest.fixture
def d1(dnode):
    return op2.Dat(dnode, numpy.zeros(nnodes), dtype=numpy.int32)


@pytest.fixture
def d2(dnode2):
    return op2.Dat(dnode2, numpy.zeros(2 * nnodes), dtype=numpy.int32)


@pytest.fixture
def vd1(dele):
    return op2.Dat(dele, numpy.zeros(nele), dtype=numpy.int32)


@pytest.fixture
def vd2(dele2):
    return op2.Dat(dele2, numpy.zeros(2 * nele), dtype=numpy.int32)


@pytest.fixture(scope='module')
def node2ele(node, ele):
    vals = numpy.arange(nnodes) / 2
    return op2.Map(node, ele, 1, vals, 'node2ele')


class TestVectorMap:

    """
    Vector Map Tests
    """

    def test_sum_nodes_to_edges(self):
        """Creates a 1D grid with edge values numbered consecutively.
        Iterates over edges, summing the node values."""

        nedges = nnodes - 1
        nodes = op2.Set(nnodes, "nodes")
        edges = op2.Set(nedges, "edges")

        node_vals = op2.Dat(
            nodes, numpy.array(range(nnodes), dtype=numpy.uint32), numpy.uint32, "node_vals")
        edge_vals = op2.Dat(
            edges, numpy.array([0] * nedges, dtype=numpy.uint32), numpy.uint32, "edge_vals")

        e_map = numpy.array([(i, i + 1)
                            for i in range(nedges)], dtype=numpy.uint32)
        edge2node = op2.Map(edges, nodes, 2, e_map, "edge2node")

        kernel_sum = """
        static void sum(unsigned int* edge, unsigned int *nodes) {
        *edge = nodes[0] + nodes[1];
        }
        """
        op2.par_loop(op2.Kernel(kernel_sum, "sum"), edges,
                     edge_vals(op2.WRITE),
                     node_vals(op2.READ, edge2node))

        expected = numpy.asarray(
            range(1, nedges * 2 + 1, 2))
        assert all(expected == edge_vals.data)

    def test_read_1d_vector_map(self, node, d1, vd1, node2ele):
        vd1.data[:] = numpy.arange(nele)
        k = """
        static void k(int *d, int *vd) {
        *d = vd[0];
        }"""
        op2.par_loop(op2.Kernel(k, 'k'), node,
                     d1(op2.WRITE),
                     vd1(op2.READ, node2ele))
        assert all(d1.data[::2] == vd1.data)
        assert all(d1.data[1::2] == vd1.data)

    def test_write_1d_vector_map(self, node, vd1, node2ele):
        k = """
        static void k(int *vd) {
        vd[0] = 2;
        }
        """

        op2.par_loop(op2.Kernel(k, 'k'), node,
                     vd1(op2.WRITE, node2ele))
        assert all(vd1.data == 2)

    def test_inc_1d_vector_map(self, node, d1, vd1, node2ele):
        vd1.data[:] = 3
        d1.data[:] = numpy.arange(nnodes).reshape(d1.data.shape)

        k = """
        static void k(int *vd, int *d) {
        vd[0] += *d;
        }"""
        op2.par_loop(op2.Kernel(k, 'k'), node,
                     vd1(op2.INC, node2ele),
                     d1(op2.READ))
        expected = numpy.zeros_like(vd1.data)
        expected[:] = 3
        expected += numpy.arange(
            start=0, stop=nnodes, step=2).reshape(expected.shape)
        expected += numpy.arange(
            start=1, stop=nnodes, step=2).reshape(expected.shape)
        assert all(vd1.data == expected)

    def test_read_2d_vector_map(self, node, d2, vd2, node2ele):
        vd2.data[:] = numpy.arange(nele * 2).reshape(nele, 2)
        k = """
        static void k(int d[2], int vd[1][2]) {
        d[0] = vd[0][0];
        d[1] = vd[0][1];
        }"""
        op2.par_loop(op2.Kernel(k, 'k'), node,
                     d2(op2.WRITE),
                     vd2(op2.READ, node2ele))
        assert all(d2.data[::2, 0] == vd2.data[:, 0])
        assert all(d2.data[::2, 1] == vd2.data[:, 1])
        assert all(d2.data[1::2, 0] == vd2.data[:, 0])
        assert all(d2.data[1::2, 1] == vd2.data[:, 1])

    def test_write_2d_vector_map(self, node, vd2, node2ele):
        k = """
        static void k(int vd[1][2]) {
        vd[0][0] = 2;
        vd[0][1] = 3;
        }
        """

        op2.par_loop(op2.Kernel(k, 'k'), node,
                     vd2(op2.WRITE, node2ele))
        assert all(vd2.data[:, 0] == 2)
        assert all(vd2.data[:, 1] == 3)

    def test_inc_2d_vector_map(self, node, d2, vd2, node2ele):
        vd2.data[:, 0] = 3
        vd2.data[:, 1] = 4
        d2.data[:] = numpy.arange(2 * nnodes).reshape(d2.data.shape)

        k = """
        static void k(int vd[1][2], int d[2]) {
        vd[0][0] += d[0];
        vd[0][1] += d[1];
        }"""
        op2.par_loop(op2.Kernel(k, 'k'), node,
                     vd2(op2.INC, node2ele),
                     d2(op2.READ))

        expected = numpy.zeros_like(vd2.data)
        expected[:, 0] = 3
        expected[:, 1] = 4
        expected[:, 0] += numpy.arange(start=0, stop=2 * nnodes, step=4)
        expected[:, 0] += numpy.arange(start=2, stop=2 * nnodes, step=4)
        expected[:, 1] += numpy.arange(start=1, stop=2 * nnodes, step=4)
        expected[:, 1] += numpy.arange(start=3, stop=2 * nnodes, step=4)
        assert all(vd2.data[:, 0] == expected[:, 0])
        assert all(vd2.data[:, 1] == expected[:, 1])


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
