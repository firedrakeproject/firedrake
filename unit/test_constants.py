import pytest
import numpy

from pyop2 import op2

size = 100

def setup_module(module):
    op2.init(backend='sequential')

def teardown_module(module):
    op2.exit()

class TestConstant:
    """
    Tests of OP2 Constants
    """

    def test_1d_read(self):
        kernel = """
        void kernel(unsigned int *x) { *x = constant; }
        """
        constant = op2.Const(1, 100, dtype=numpy.uint32, name="constant")
        itset = op2.Set(size)
        dat = op2.Dat(itset, 1, numpy.zeros(size, dtype=numpy.uint32))
        op2.par_loop(op2.Kernel(kernel, "kernel"),
                     itset, dat(op2.IdentityMap, op2.WRITE))

        constant.remove_from_namespace()
        assert all(dat.data == constant._data)

    def test_2d_read(self):
        kernel = """
        void kernel(unsigned int *x) { *x = constant[0] + constant[1]; }
        """
        constant = op2.Const(2, (100, 200), dtype=numpy.uint32, name="constant")
        itset = op2.Set(size)
        dat = op2.Dat(itset, 1, numpy.zeros(size, dtype=numpy.uint32))
        op2.par_loop(op2.Kernel(kernel, "kernel"),
                     itset, dat(op2.IdentityMap, op2.WRITE))
        constant.remove_from_namespace()
        assert all(dat.data == constant._data.sum())

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
