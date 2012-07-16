import unittest
import numpy

from pyop2 import op2
# Initialise OP2
op2.init(backend='opencl', diags=0)

#max...
nelems = 92681


class DirectLoopTest(unittest.TestCase):
    """

    Direct Loop Tests

    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_wo(self):
        """Test write only argument."""
        iterset = op2.Set(nelems, "elems")
        x = op2.Dat(iterset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        kernel_wo = """
void kernel_wo(unsigned int* x)
{
  *x = 42;
}
"""

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset, x(op2.IdentityMap, op2.WRITE))
        self.assertTrue(all(map(lambda x: x==42, x.data)))

    def test_rw(self):
        """Test read & write argument."""
        iterset = op2.Set(nelems, "elems")
        x = op2.Dat(iterset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        kernel_rw = """
void kernel_rw(unsigned int* x) {
  *x = *x + 1;
}
"""

        op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), iterset, x(op2.IdentityMap, op2.RW))
        self.assertEqual(sum(x.data), nelems * (nelems + 1) / 2);

    def test_global_inc(self):
        """Test global increment argument."""
        iterset = op2.Set(nelems, "elems")
        x = op2.Dat(iterset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")
        g = op2.Global(1, 0, numpy.uint32, "g")

        kernel_global_inc = """
void kernel_global_inc(unsigned int* x, unsigned int* inc)
{
  *x = *x + 1;
  *inc += *x;
}
"""

        op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), iterset, x(op2.IdentityMap, op2.RW), g(op2.INC))
        self.assertEqual(g.data[0], nelems * (nelems + 1) / 2);
        self.assertEqual(sum(x.data), g.data[0])

    def test_ro_wo_global_inc(self):
        """Test multiple arguments."""
        iterset = op2.Set(nelems, "elems")
        x = op2.Dat(iterset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")
        y = op2.Dat(iterset, 1, numpy.array([0] * nelems, dtype=numpy.uint32), numpy.uint32, "y")
        g = op2.Global(1, 0, numpy.uint32, "g")

        kernel_ro_wo_global_inc = """
void kernel_ro_wo_global_inc(unsigned int* x, unsigned int* y, unsigned int* inc)
{
  *y = *x + 1;
  *inc += *y;
}
"""

        op2.par_loop(op2.Kernel(kernel_ro_wo_global_inc, "kernel_ro_wo_global_inc"), iterset, x(op2.IdentityMap, op2.READ), y(op2.IdentityMap, op2.WRITE), g(op2.INC))
        self.assertEqual(g.data[0], nelems * (nelems + 1) / 2);
        self.assertEqual(sum(y.data), g.data[0])
        self.assertEqual(sum(x.data), g.data[0] - nelems)

    def test_multidim(self):
        """Test dimension > 1 arguments."""
        iterset = op2.Set(nelems, "elems")
        x = op2.Dat(iterset, 2, numpy.array(range(1, 2*nelems + 1), dtype=numpy.uint32), numpy.uint32, "x")
        y = op2.Dat(iterset, 1, numpy.array([0] * nelems, dtype=numpy.uint32), numpy.uint32, "y")
        g = op2.Global(1, 0, numpy.uint32, "g")

        kernel_multidim = """
void kernel_multidim(unsigned int* x, unsigned int* y, unsigned int* inc)
{
  *y = (x[0] + x[1]) / 2;
  *inc += *y;
}
"""

        op2.par_loop(op2.Kernel(kernel_multidim, "kernel_multidim"), iterset, x(op2.IdentityMap, op2.READ), y(op2.IdentityMap, op2.WRITE), g(op2.INC))
        self.assertEqual(sum(y.data), g.data[0])

    def test_multidim_global_inc(self):
        iterset = op2.Set(nelems, "elems")
        x = op2.Dat(iterset, 2, numpy.array(range(1, 2*nelems + 1), dtype=numpy.uint32), numpy.uint32, "x")
        y = op2.Dat(iterset, 1, numpy.array([0] * nelems, dtype=numpy.uint32), numpy.uint32, "y")
        z = op2.Dat(iterset, 1, numpy.array([0] * nelems, dtype=numpy.uint32), numpy.uint32, "z")
        g = op2.Global(2, numpy.array([0, 0], dtype=numpy.uint32), numpy.uint32, "g")

        kernel_multidim_global_inc = """
void kernel_multidim_global_inc(unsigned int* x, unsigned int* y, unsigned int* z, unsigned int* inc)
{
  *y = x[0];
  *z = x[1];
  inc[0] += *y;
  inc[1] += *z;
}
"""

        op2.par_loop(op2.Kernel(kernel_multidim_global_inc, "kernel_multidim_global_inc"), iterset, x(op2.IdentityMap, op2.READ), y(op2.IdentityMap, op2.WRITE), z(op2.IdentityMap, op2.WRITE), g(op2.INC))
        self.assertEqual(sum(y.data), g.data[0])
        self.assertEqual(sum(z.data), g.data[1])

suite = unittest.TestLoader().loadTestsFromTestCase(DirectLoopTest)
unittest.TextTestRunner(verbosity=0).run(suite)

# refactor to avoid recreating input data for each test cases
# TODO:
#  - floating point type computations
#  - constants
