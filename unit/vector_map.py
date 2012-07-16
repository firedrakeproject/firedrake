import unittest
import numpy
import random

from pyop2 import op2
# Initialise OP2
op2.init(backend='sequential', diags=0)

def _seed():
    return 0.02041724

#max...
nnodes = 92681

class VectorMapTest(unittest.TestCase):
    """

    Indirect Loop Tests

    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sum_nodes_to_edges(self):
        """Creates a 1D grid with edge values numbered consecutively.
        Iterates over edges, summing the node values."""

        nedges = nnodes-1
        nodes = op2.Set(nnodes, "nodes")
        edges = op2.Set(nedges, "edges")

        node_vals = op2.Dat(nodes, 1, numpy.array(range(nnodes), dtype=numpy.uint32), numpy.uint32, "node_vals")
        edge_vals = op2.Dat(edges, 1, numpy.array([0]*nedges, dtype=numpy.uint32), numpy.uint32, "edge_vals")

        e_map = numpy.array([(i, i+1) for i in range(nedges)], dtype=numpy.uint32)
        edge2node = op2.Map(edges, nodes, 2, e_map, "edge2node")

        kernel_sum = """
void kernel_sum(unsigned int* nodes[1], unsigned int *edge)
{ *edge = nodes[0][0] + nodes[1][0]; }
"""

        op2.par_loop(op2.Kernel(kernel_sum, "kernel_sum"), edges, \
                       node_vals(edge2node,       op2.READ),      \
                       edge_vals(op2.IdentityMap, op2.WRITE))

        expected = numpy.asarray(range(1, nedges*2+1, 2)).reshape(nedges, 1)
        self.assertTrue(all(expected == edge_vals.data))

suite = unittest.TestLoader().loadTestsFromTestCase(VectorMapTest)
unittest.TextTestRunner(verbosity=0, failfast=False).run(suite)
# refactor to avoid recreating input data for each test cases
