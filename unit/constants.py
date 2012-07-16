import unittest
import numpy

from pyop2 import op2

op2.init(backend='sequential')

size = 100

class ConstantTest(unittest.TestCase):
    """
    Tests of OP2 Constants
    """

    def test_unique_names(self):
        with self.assertRaises(op2.Const.NonUniqueNameError):
            const1 = op2.Const(1, 1, name="constant")
            const2 = op2.Const(1, 2, name="constant")
            const1.remove_from_namespace()
            const2.remove_from_namespace()

    def test_namespace_removal(self):
        const1 = op2.Const(1, 1, name="constant")
        const1.remove_from_namespace()
        const2 = op2.Const(1, 2, name="constant")
        const2.remove_from_namespace()

    def test_1d_read(self):
        kernel = """
        void kernel(unsigned int *x) { *x = constant; }
        """
        constant = op2.Const(1, 100, dtype=numpy.uint32, name="constant")
        itset = op2.Set(size)
        dat = op2.Dat(itset, 1, numpy.zeros(size, dtype=numpy.uint32))
        op2.par_loop(op2.Kernel(kernel, "kernel"),
                     itset, dat(op2.IdentityMap, op2.WRITE))

        self.assertTrue(all(dat.data == constant._data))
        constant.remove_from_namespace()

    def test_2d_read(self):
        kernel = """
        void kernel(unsigned int *x) { *x = constant[0] + constant[1]; }
        """
        constant = op2.Const(2, (100, 200), dtype=numpy.uint32, name="constant")
        itset = op2.Set(size)
        dat = op2.Dat(itset, 1, numpy.zeros(size, dtype=numpy.uint32))
        op2.par_loop(op2.Kernel(kernel, "kernel"),
                     itset, dat(op2.IdentityMap, op2.WRITE))
        self.assertTrue(all(dat.data == constant._data.sum()))
        constant.remove_from_namespace()

suite = unittest.TestLoader().loadTestsFromTestCase(ConstantTest)
unittest.TextTestRunner(verbosity=0).run(suite)
