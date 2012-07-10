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

    Indirect Loop Tests

    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_onecolor_wo(self):
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map)
        iterset2indset = op2.Map(iterset, indset, 1, u_map, "iterset2indset")

        # temporary fix until we have the user kernel instrumentation code
        kernel_wo = "void kernel_wo(__local unsigned int* x) { *x = 42; }\n"
        #kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset, x(iterset2indset(0), op2.WRITE))
        self.assertTrue(all(map(lambda x: x==42, x.data)))

    def test_onecolor_rw(self):
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map)
        iterset2indset = op2.Map(iterset, indset, 1, u_map, "iterset2indset")

        # temporary fix until we have the user kernel instrumentation code
        kernel_rw = "void kernel_rw(__local unsigned int* x) { (*x) = (*x) + 1; }\n"
        #kernel_rw = "void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), iterset, x(iterset2indset(0), op2.RW))
        self.assertTrue(sum(x.data) == nelems * (nelems + 1) / 2);

    def test_onecolor_global_inc(self):
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")
        g = op2.Global(1, 0, numpy.uint32)

        u_map = numpy.array(range(nelems), dtype=numpy.uint32)
        random.shuffle(u_map)
        iterset2indset = op2.Map(iterset, indset, 1, u_map, "iterset2indset")

        # temporary fix until we have the user kernel instrumentation code
        kernel_global_inc = "void kernel_global_inc(__local unsigned int *x, __private unsigned int *inc) { (*x) = (*x) + 1; (*inc) += (*x); }\n"
        #kernel_global_inc = "void kernel_global_inc(unsigned int *x, unsigned int *inc) { (*x) = (*x) + 1; (*inc) += (*x); }\n"

        op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), iterset,
                     x(iterset2indset(0), op2.RW),
                     g(op2.INC))
        self.assertTrue(g.data[0] == nelems * (nelems + 1) / 2)

suite = unittest.TestLoader().loadTestsFromTestCase(IndirectLoopTest)
unittest.TextTestRunner(verbosity=0).run(suite)

# refactor to avoid recreating input data for each test cases
