import unittest
import numpy

from pyop2 import op2
# Initialise OP2
op2.init(backend='sequential')

#max...
nelems = 92681


class DirectLoopTest(unittest.TestCase):
    """

    Direct Loop Tests

    """

    def setUp(self):
        self._elems = op2.Set(nelems, "elems")
        self._input_x = numpy.array(range(nelems), dtype=numpy.uint32)
        self._x = op2.Dat(self._elems,  1, self._input_x, numpy.uint32, "x")
        self._g = op2.Global(1, 0, numpy.uint32, "natural_sum")

    def tearDown(self):
        del self._elems
        del self._input_x
        del self._x
        del self._g

    def test_wo(self):
        kernel_wo = """
void kernel_wo(unsigned int*);
void kernel_wo(unsigned int* x) { *x = 42; }
"""
        l = op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), self._elems, self._x(op2.IdentityMap, op2.WRITE))
        self.assertTrue(all(map(lambda x: x==42, self._x.data)))

    def test_rw(self):
        kernel_rw = """
void kernel_rw(unsigned int*);
void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }
"""
        l = op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), self._elems, self._x(op2.IdentityMap, op2.RW))
        self.assertEqual(sum(self._x.data), nelems * (nelems + 1) / 2);

    def test_global_incl(self):
        kernel_global_inc = """
void kernel_global_inc(unsigned int*, unsigned int*);
void kernel_global_inc(unsigned int* x, unsigned int* inc) { (*x) = (*x) + 1; (*inc) += (*x); }
"""
        l = op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), self._elems, self._x(op2.IdentityMap, op2.RW), self._g(op2.INC))
        self.assertEqual(self._g.data[0], nelems * (nelems + 1) / 2);

suite = unittest.TestLoader().loadTestsFromTestCase(DirectLoopTest)
unittest.TextTestRunner(verbosity=0).run(suite)

# refactor to avoid recreating input data for each test cases
