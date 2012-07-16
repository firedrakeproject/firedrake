import unittest
import numpy
import random
import warnings
import math

from pyop2 import op2

# Initialise OP2
op2.init(backend='opencl', diags=0)

def _seed():
    return 0.02041724

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
        """Test write only indirect dat without concurrent access."""
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        iterset2indset = op2.Map(iterset, indset, 1, _shuffle(numpy.array(range(nelems), dtype=numpy.uint32)), "iterset2indset")

        kernel_wo = "void kernel_wo(unsigned int* x) { *x = 42; }\n"

        op2.par_loop(op2.Kernel(kernel_wo, "kernel_wo"), iterset, x(iterset2indset(0), op2.WRITE))
        self.assertTrue(all(map(lambda x: x==42, x.data)))

    def test_onecolor_rw(self):
        """Test read & write indirect dat without concurrent access."""
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")

        iterset2indset = op2.Map(iterset, indset, 1, _shuffle(numpy.array(range(nelems), dtype=numpy.uint32)), "iterset2indset")

        kernel_rw = "void kernel_rw(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_rw, "kernel_rw"), iterset, x(iterset2indset(0), op2.RW))
        self.assertEqual(sum(x.data), nelems * (nelems + 1) / 2);

    def test_indirect_inc(self):
        """Test indirect reduction with concurrent access."""
        iterset = op2.Set(nelems, "iterset")
        unitset = op2.Set(1, "unitset")

        u = op2.Dat(unitset, 1, numpy.array([0], dtype=numpy.uint32), numpy.uint32, "u")

        u_map = numpy.zeros(nelems, dtype=numpy.uint32)
        iterset2unit = op2.Map(iterset, unitset, 1, u_map, "iterset2unitset")

        kernel_inc = "void kernel_inc(unsigned int* x) { (*x) = (*x) + 1; }\n"

        op2.par_loop(op2.Kernel(kernel_inc, "kernel_inc"), iterset, u(iterset2unit(0), op2.INC))
        self.assertEqual(u.data[0], nelems)

    def test_global_inc(self):
        """Test global reduction."""
        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(nelems, "indset")

        x = op2.Dat(indset, 1, numpy.array(range(nelems), dtype=numpy.uint32), numpy.uint32, "x")
        g = op2.Global(1, 0, numpy.uint32, "g")

        iterset2indset = op2.Map(iterset, indset, 1, _shuffle(numpy.array(range(nelems), dtype=numpy.uint32)), "iterset2indset")

        kernel_global_inc = "void kernel_global_inc(unsigned int *x, unsigned int *inc) { (*x) = (*x) + 1; (*inc) += (*x); }\n"

        op2.par_loop(op2.Kernel(kernel_global_inc, "kernel_global_inc"), iterset,
                     x(iterset2indset(0), op2.RW),
                     g(op2.INC))

        self.assertEqual(sum(x.data), nelems * (nelems + 1) / 2)
        self.assertEqual(g.data[0], nelems * (nelems + 1) / 2)

    def test_colored_blocks(self):
        """Test colored block execution."""
        #FIX: there is no actual guarantee the randomness will give us blocks of
        #     different color. this would require knowing the partition size and
        #     generates the mapping values in tiles...
        smalln = int(math.log(nelems, 2))

        iterset = op2.Set(nelems, "iterset")
        indset = op2.Set(smalln, "indset")

        a = op2.Dat(iterset, 1, numpy.array([42] * nelems, dtype=numpy.int32), numpy.int32, "a")
        p = op2.Dat(indset, 1, numpy.array([1] * smalln, dtype=numpy.int32), numpy.int32, "p")
        n = op2.Dat(indset, 1, numpy.array([-1] * smalln, dtype=numpy.int32), numpy.int32, "n")
        v = op2.Dat(indset, 1, numpy.array([0] * smalln, dtype=numpy.int32), numpy.int32, "v")

        _map = numpy.random.randint(0, smalln, nelems)
        _map = _map.astype(numpy.int32)
        iterset2indset = op2.Map(iterset, indset, 1, _map, "iterset2indset")

        kernel_colored_blocks = """
void
kernel_colored_blocks(
  int* a,
  int* p,
  int* n,
  int* v)
{
  *a = *p + *n;
  *v += 1;
}
"""
        op2.par_loop(op2.Kernel(kernel_colored_blocks, "kernel_colored_blocks"), iterset,
                                a(op2.IdentityMap, op2.WRITE),
                                p(iterset2indset(0), op2.READ),
                                n(iterset2indset(0), op2.READ),
                                v(iterset2indset(0), op2.INC))

        self.assertTrue(all(map(lambda e: e == 0, a.data)))
        self.assertTrue(numpy.array_equal(v.data, numpy.bincount(_map, minlength=smalln).reshape((smalln, 1))))


    def test_mul_ind(self):
        """ Test multiple indirection maps with concurrent access."""
        n = nelems if (nelems % 2) == 0 else (nelems - 1)

        iterset = op2.Set(n / 2, "iterset")
        setA = op2.Set(n, "A")
        setB = op2.Set(n / 2, "B")

        a = op2.Dat(setA, 1, numpy.array(range(1, (n+1)), dtype=numpy.uint32), numpy.uint32, "a")
        b = op2.Dat(setB, 2, _shuffle(numpy.array(range(1, (n+1)), dtype=numpy.uint32)), numpy.uint32, "b")
        x = op2.Dat(iterset, 1, numpy.zeros(n / 2, dtype=numpy.uint32), numpy.uint32, "x")
        y = op2.Dat(iterset, 1, numpy.zeros(n / 2, dtype=numpy.uint32), numpy.uint32, "y")

        g = op2.Global(2, [0, 0], numpy.uint32, "g")

        iterset2A = op2.Map(iterset, setA, 2, _shuffle(numpy.array(range(n), dtype=numpy.uint32)), "iterset2A")
        iterset2B = op2.Map(iterset, setB, 1, _shuffle(numpy.array(range(n / 2), dtype=numpy.uint32)), "iterset2B")

        kernel_mul_ind = """
void kernel_mul_ind(
  unsigned int* x,
  unsigned int* y,
  unsigned int* a1,
  unsigned int* a2,
  unsigned int* b,
  unsigned int* g)
{

  unsigned int _a = *a1 + *a2;
  unsigned int _b = b[0] + b[1];

  *x = _a;
  *y = _b;

  g[0] += _a;
  g[1] += _b;

}
"""
        op2.par_loop(op2.Kernel(kernel_mul_ind, "kernel_mul_ind"), iterset,\
                     x(op2.IdentityMap, op2.WRITE), y(op2.IdentityMap, op2.WRITE),\
                     a(iterset2A(0), op2.READ), a(iterset2A(1), op2.READ),\
                     b(iterset2B(0), op2.READ),\
                     g(op2.INC))

        self.assertEqual(sum(x.data), n * (n + 1) / 2)
        self.assertEqual(sum(y.data), n * (n + 1) / 2)
        self.assertEqual(g.data[0], n * (n + 1) / 2)
        self.assertEqual(g.data[1], n * (n + 1) / 2)

def _shuffle(arr):
    #FIX: this is probably not a good enough shuffling
    for i in range(int(math.log(nelems,2))):
        numpy.random.shuffle(arr)
    return arr

suite = unittest.TestLoader().loadTestsFromTestCase(IndirectLoopTest)
unittest.TextTestRunner(verbosity=0, failfast=False).run(suite)

# refactor to avoid recreating input data for each test cases
