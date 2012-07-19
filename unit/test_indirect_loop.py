import pytest
import numpy
import random

from pyop2 import op2

def setup_module(module):
    # Initialise OP2
    op2.init(backend='sequential')

def teardown_module(module):
    op2.exit()

def _seed():
    return 0.02041724

#max...
nelems = 92681

class TestIndirectLoop:
    """
    Indirect Loop Tests
    """

    def pytest_funcarg__iterset(cls, request):
        return op2.Set(nelems, "iterset")

    def pytest_funcarg__indset(cls, request):
        return op2.Set(nelems, "indset")

    def pytest_funcarg__x(cls, request):
        return op2.Dat(request.getfuncargvalue('indset'), 1, range(nelems), numpy.uint32, "x")

    def pytest_funcarg__iterset2indset(cls, request):
        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map, _seed)
        return op2.Map(request.getfuncargvalue('iterset'), request.getfuncargvalue('indset'), 1, u_map, "iterset2indset")

    def test_onecolor_wo(self, iterset, x, iterset2indset):
        kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset, x(iterset2indset(0), op2.WRITE))
        assert all(map(lambda x: x==42, x.data))

    def test_onecolor_rw(self, iterset, x, iterset2indset):
        kernel_rw = "void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), iterset, x(iterset2indset(0), op2.RW))
        assert sum(x.data) == nelems * (nelems + 1) / 2

    def test_indirect_inc(self, iterset):
        unitset = op2.Set(1, "unitset")

        u = op2.Dat(unitset, 1, numpy.array([0], dtype=numpy.uint32), numpy.uint32, "u")

        u_map = numpy.zeros(nelems, dtype=numpy.uint32)
        iterset2unit = op2.Map(iterset, unitset, 1, u_map, "iterset2unitset")

        kernel_inc = "void kernel_inc(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"), iterset, u(iterset2unit(0), op2.INC))
        assert u.data[0] == nelems

    def test_global_inc(self, iterset, x, iterset2indset):
        g = op2.Global(1, 0, numpy.uint32, "g")

        kernel_global_inc = "void kernel_global_inc(unsigned int *x, unsigned int *inc) { (*x) = (*x) + 1; (*inc) += (*x); }\n"

        op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), iterset,
                     x(iterset2indset(0), op2.RW),
                     g(op2.INC))
        assert sum(x.data) == nelems * (nelems + 1) / 2
        assert g.data[0] == nelems * (nelems + 1) / 2

    def test_2d_dat(self, iterset, indset, iterset2indset):
        x = op2.Dat(indset, 2, numpy.array([range(nelems), range(nelems)], dtype=numpy.uint32), numpy.uint32, "x")

        kernel_wo = "void kernel_wo(unsigned int* x) { x[0] = 42; x[1] = 43; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset, x(iterset2indset(0), op2.WRITE))
        assert all(map(lambda x: all(x==[42,43]), x.data))

    def test_2d_map(self):
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
        op2.par_loop(op2.Kernel(kernel_sum, "kernel_sum"), edges,
                     node_vals(edge2node(0), op2.READ),
                     node_vals(edge2node(1), op2.READ),
                     edge_vals(op2.IdentityMap, op2.WRITE))

        expected = numpy.asarray(range(1, nedges * 2 + 1, 2)).reshape(nedges, 1)
        assert all(expected == edge_vals.data)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
