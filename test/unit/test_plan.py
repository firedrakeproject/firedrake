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
from pyop2 import op_lib_core as core

backends = ['sequential', 'openmp', 'opencl', 'cuda']

def _seed():
    return 0.02041724

# Large enough that there is more than one block and more than one
# thread per element in device backends
nelems = 4096

class TestPlan:
    """
    Plan Construction Tests
    """

    @pytest.fixture
    def iterset(cls, request):
        return op2.Set(nelems, "iterset")

    @pytest.fixture
    def indset(cls, request):
        return op2.Set(nelems, "indset")

    @pytest.fixture
    def x(cls, request, indset):
        return op2.Dat(indset, 1, range(nelems), numpy.uint32, "x")

    @pytest.fixture
    def iterset2indset(cls, request, iterset, indset):
        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        return op2.Map(iterset, indset, 1, u_map, "iterset2indset")

    def test_onecolor_wo(self, backend, iterset, x, iterset2indset):
        # copy/adapted from test_indirect_loop
        kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }\n"

        kernel = op2.Kernel(kernel_wo, "kernel_wo")

        pyplan = core.Plan(kernel,
                             iterset,
                             x(iterset2indset[0], op2.WRITE),
                             partition_size=128,
                             matrix_coloring=False)
        cplan = core.op_plan(kernel,
                             iterset,
                             x(iterset2indset[0], op2.WRITE),
                             partition_size=128,
                             matrix_coloring=False)

        assert pyplan.ninds == cplan.ninds
        assert pyplan.nblocks == cplan.nblocks
        assert pyplan.ncolors == cplan.ncolors
        assert pyplan.nshared == cplan.nshared
        assert (pyplan.nelems == cplan.nelems).all()
        # slice is ok cause op2 plan function seems to allocate an
        # arbitrarily longer array here
        assert (pyplan.ncolblk == cplan.ncolblk[:len(pyplan.ncolblk)]).all()
        assert (pyplan.blkmap == cplan.blkmap).all()
        assert (pyplan.nthrcol == cplan.nthrcol).all()
        assert (pyplan.thrcol == cplan.thrcol).all()
        assert (pyplan.offset == cplan.offset).all()
        assert (pyplan.nindirect == cplan.nindirect).all()
        assert ( (pyplan.ind_map == cplan.ind_map) | (pyplan.ind_map==-1) ).all()
        assert (pyplan.ind_offs == cplan.ind_offs).all()
        assert (pyplan.ind_sizes == cplan.ind_sizes).all()
        assert (pyplan.loc_map == cplan.loc_map).all()

    def test_2d_map(self, backend):
        # copy/adapted from test_indirect_loop
        nedges = nelems - 1
        nodes = op2.Set(nelems, "nodes")
        edges = op2.Set(nedges, "edges")
        node_vals = op2.Dat(nodes, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "node_vals")
        edge_vals = op2.Dat(edges, 1, numpy.array([0] * nedges, dtype=numpy.uint32), numpy.uint32, "edge_vals")

        e_map = numpy.array([(i, i+1) for i in range(nedges)], dtype=numpy.uint32)
        edge2node = op2.Map(edges, nodes, 2, e_map, "edge2node")

        kernel_sum = """
        void kernel_sum(unsigned int *nodes1, unsigned int *nodes2, unsigned int *edge)
        { *edge = *nodes1 + *nodes2; }
        """

        kernel = op2.Kernel(kernel_sum, "kernel_sum")

        pyplan = core.Plan(kernel,
                             edges,
                             node_vals(edge2node[0], op2.READ),
                             node_vals(edge2node[1], op2.READ),
                             edge_vals(op2.IdentityMap, op2.WRITE),
                             matrix_coloring=False,
                             partition_size=96)
        cplan = core.op_plan(kernel,
                             edges,
                             node_vals(edge2node[0], op2.READ),
                             node_vals(edge2node[1], op2.READ),
                             edge_vals(op2.IdentityMap, op2.WRITE),
                             matrix_coloring=False,
                             partition_size=96)

        assert pyplan.ninds == cplan.ninds
        assert pyplan.nblocks == cplan.nblocks
        assert pyplan.ncolors == cplan.ncolors
        assert pyplan.nshared == cplan.nshared
        assert (pyplan.nelems == cplan.nelems).all()
        # slice is ok cause op2 plan function seems to allocate an
        # arbitrarily longer array here
        assert (pyplan.ncolblk == cplan.ncolblk[:len(pyplan.ncolblk)]).all()
        assert (pyplan.blkmap == cplan.blkmap).all()
        assert (pyplan.nthrcol == cplan.nthrcol).all()
        assert (pyplan.thrcol == cplan.thrcol).all()
        assert (pyplan.offset == cplan.offset).all()
        assert (pyplan.nindirect == cplan.nindirect).all()
        sninds = numpy.sum(pyplan.nindirect)
        assert ( (pyplan.ind_map == cplan.ind_map) | (pyplan.ind_map==-1) ).all()
        assert (pyplan.ind_offs == cplan.ind_offs).all()
        assert (pyplan.ind_sizes == cplan.ind_sizes).all()
        assert (pyplan.loc_map == cplan.loc_map).all()

    def test_rhs(self, backend):
        kernel = op2.Kernel("", "dummy")
        elements = op2.Set(2, "elements")
        nodes = op2.Set(4, "nodes")
        elem_node = op2.Map(elements, nodes, 3,
                            numpy.asarray([ 0, 1, 3, 2, 3, 1 ],
                                          dtype=numpy.uint32),
                            "elem_node")
        b = op2.Dat(nodes, 1,
                    numpy.asarray([0.0]*4, dtype=numpy.float64),
                    numpy.float64, "b")
        coords = op2.Dat(nodes, 2,
                         numpy.asarray([ (0.0, 0.0),
                                        (2.0, 0.0),
                                        (1.0, 1.0),
                                        (0.0, 1.5) ],
                                       dtype=numpy.float64),
                         numpy.float64,
                         "coords")
        f = op2.Dat(nodes, 1,
                    numpy.asarray([ 1.0, 2.0, 3.0, 4.0 ], dtype=numpy.float64),
                    numpy.float64, "f")
        pyplan = core.Plan(kernel,
                             elements,
                             b(elem_node[0], op2.INC),
                             b(elem_node[1], op2.INC),
                             b(elem_node[2], op2.INC),
                             coords(elem_node[0], op2.READ),
                             coords(elem_node[1], op2.READ),
                             coords(elem_node[2], op2.READ),
                             f(elem_node[0], op2.READ),
                             f(elem_node[1], op2.READ),
                             f(elem_node[2], op2.READ),
                             matrix_coloring=False,
                             partition_size=2)

        cplan = core.op_plan(kernel,
                             elements,
                             b(elem_node[0], op2.INC),
                             b(elem_node[1], op2.INC),
                             b(elem_node[2], op2.INC),
                             coords(elem_node[0], op2.READ),
                             coords(elem_node[1], op2.READ),
                             coords(elem_node[2], op2.READ),
                             f(elem_node[0], op2.READ),
                             f(elem_node[1], op2.READ),
                             f(elem_node[2], op2.READ),
                             matrix_coloring=False,
                             partition_size=2)

        assert pyplan.ninds == cplan.ninds
        assert pyplan.nblocks == cplan.nblocks
        assert pyplan.ncolors == cplan.ncolors
        assert pyplan.nshared == cplan.nshared
        assert (pyplan.nelems == cplan.nelems).all()
        # slice is ok cause op2 plan function seems to allocate an
        # arbitrarily longer array here
        assert (pyplan.ncolblk == cplan.ncolblk[:len(pyplan.ncolblk)]).all()
        assert (pyplan.blkmap == cplan.blkmap).all()
        assert (pyplan.nthrcol == cplan.nthrcol).all()
        assert (pyplan.thrcol == cplan.thrcol).all()
        assert (pyplan.offset == cplan.offset).all()
        assert (pyplan.nindirect == cplan.nindirect).all()
        assert ( (pyplan.ind_map == cplan.ind_map) | (pyplan.ind_map==-1) ).all()
        assert (pyplan.ind_offs == cplan.ind_offs).all()
        assert (pyplan.ind_sizes == cplan.ind_sizes).all()
        assert (pyplan.loc_map == cplan.loc_map).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
