import unittest
import numpy
import random

from pyop2 import op2
# Initialise OP2
op2.init(backend='opencl')

#max...
nelems = 92681

class IndirectLoopTest(unittest.TestCase):
    """

    Direct Loop Tests

    """

    def setUp(self):
        self._itset_11 = op2.Set(nelems, "iterset")
        self._elems = op2.Set(nelems, "elems")
        self._input_x = numpy.array(range(nelems), dtype=numpy.uint32)
        self._x = op2.Dat(self._elems,  1, self._input_x, numpy.uint32, "x")
        self._g = op2.Global(1, 0, numpy.uint32, "natural_sum")
        self._input_11 = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(self._input_11)
        self._11_elems = op2.Map(self._itset_11, self._elems, 1, self._input_11, "11_elems")

    def tearDown(self):
        del self._itset_11
        del self._elems
        del self._input_x
        del self._input_11
        del self._x

    def test_wo(self):
        kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }\n"
        l = op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), self._itset_11, self._x(self._11_elems(0), op2.WRITE))
        self.assertTrue(all(map(lambda x: x==42, self._x.value)))

    def test_rw(self):
        kernel_rw = "void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }\n"
        l = op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), self._itset_11, self._x(self._11_elems(0), op2.RW))
        self.assertTrue(sum(self._x.value) == nelems * (nelems + 1) / 2);

    def test_global_inc(self):
        kernel_global_inc = "void kernel_global_inc(unsigned int *x, unsigned int *inc) { (*x) = (*x) + 1; (*inc) += (*x); }"
        l = op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"),
                         self._itset_11, self._x(self._11_elems(0), op2.RW),
                         self._g(op2.INC))
        self.assertTrue(self._g.data[0] == nelems * (nelems + 1) / 2)

suite = unittest.TestLoader().loadTestsFromTestCase(IndirectLoopTest)
unittest.TextTestRunner(verbosity=0).run(suite)

# refactor to avoid recreating input data for each test cases
