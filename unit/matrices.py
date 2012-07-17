import unittest
import numpy

from pyop2 import op2
# Initialise OP2
op2.init(backend='sequential')

# Data type
valuetype = numpy.float32

# Constants
NUM_ELE   = 2
NUM_NODES = 4
NUM_DIMS  = 2

class MatricesTest(unittest.TestCase):
    """

    Matrix tests

    """

    def setUp(self):
        elem_node_map = numpy.asarray([ 0, 1, 3, 2, 3, 1 ], dtype=numpy.uint32)
        coord_vals = numpy.asarray([ (0.0, 0.0), (2.0, 0.0),
                                     (1.0, 1.0), (0.0, 1.5) ],
                                   dtype=valuetype)
        f_vals = numpy.asarray([ 1.0, 2.0, 3.0, 4.0 ], dtype=valuetype)
        b_vals = numpy.asarray([0.0]*NUM_NODES, dtype=valuetype)
        x_vals = numpy.asarray([0.0]*NUM_NODES, dtype=valuetype)

        self._nodes = op2.Set(NUM_NODES, "nodes")
        self._elements = op2.Set(NUM_ELE, "elements")
        self._elem_node = op2.Map(self._elements, self._nodes, 3, elem_node_map,
                                  "elem_node")

        # Sparsity(rmaps, cmaps, dims, name)
        sparsity = op2.Sparsity(self._elem_node, self._elem_node, 1, "sparsity")
        # Mat(sparsity, dims, type, name)
        self._mat = op2.Mat(sparsity, 1, valuetype, "mat")
        self._coords = op2.Dat(self._nodes, 2, coord_vals, valuetype, "coords")
        self._f = op2.Dat(self._nodes, 1, f_vals, valuetype, "f")
        self._b = op2.Dat(self._nodes, 1, b_vals, valuetype, "b")
        self._x = op2.Dat(self._nodes, 1, x_vals, valuetype, "x")

        kernel_mass = """
void mass(ValueType* localTensor, ValueType* c0[2], int i_r_0, int i_r_1)
{
  const ValueType CG1[3][6] = { {  0.09157621, 0.09157621, 0.81684757,
                                   0.44594849, 0.44594849, 0.10810302 },
                                {  0.09157621, 0.81684757, 0.09157621,
                                   0.44594849, 0.10810302, 0.44594849 },
                                {  0.81684757, 0.09157621, 0.09157621,
                                   0.10810302, 0.44594849, 0.44594849 } };
  const ValueType d_CG1[3][6][2] = { { {  1., 0. },
                                       {  1., 0. },
                                       {  1., 0. },
                                       {  1., 0. },
                                       {  1., 0. },
                                       {  1., 0. } },

                                     { {  0., 1. },
                                       {  0., 1. },
                                       {  0., 1. },
                                       {  0., 1. },
                                       {  0., 1. },
                                       {  0., 1. } },

                                     { { -1.,-1. },
                                       { -1.,-1. },
                                       { -1.,-1. },
                                       { -1.,-1. },
                                       { -1.,-1. },
                                       { -1.,-1. } } };
  const ValueType w[6] = {  0.05497587, 0.05497587, 0.05497587, 0.11169079,
                            0.11169079, 0.11169079 };
  ValueType c_q0[6][2][2];
  for(int i_g = 0; i_g < 6; i_g++)
  {
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q0[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        };
      };
    };
  };
  for(int i_g = 0; i_g < 6; i_g++)
  {
    ValueType ST0 = 0.0;
    ST0 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    localTensor[0] += ST0 * w[i_g];
  };
}
"""

        kernel_rhs = """
void rhs(ValueType** localTensor, ValueType* c0[2], ValueType* c1[1])
{
  const ValueType CG1[3][6] = { {  0.09157621, 0.09157621, 0.81684757,
                                   0.44594849, 0.44594849, 0.10810302 },
                                {  0.09157621, 0.81684757, 0.09157621,
                                   0.44594849, 0.10810302, 0.44594849 },
                                {  0.81684757, 0.09157621, 0.09157621,
                                   0.10810302, 0.44594849, 0.44594849 } };
  const ValueType d_CG1[3][6][2] = { { {  1., 0. },
                                       {  1., 0. },
                                       {  1., 0. },
                                       {  1., 0. },
                                       {  1., 0. },
                                       {  1., 0. } },

                                     { {  0., 1. },
                                       {  0., 1. },
                                       {  0., 1. },
                                       {  0., 1. },
                                       {  0., 1. },
                                       {  0., 1. } },

                                     { { -1.,-1. },
                                       { -1.,-1. },
                                       { -1.,-1. },
                                       { -1.,-1. },
                                       { -1.,-1. },
                                       { -1.,-1. } } };
  const ValueType w[6] = {  0.05497587, 0.05497587, 0.05497587, 0.11169079,
                            0.11169079, 0.11169079 };
  ValueType c_q1[6];
  ValueType c_q0[6][2][2];
  for(int i_g = 0; i_g < 6; i_g++)
  {
    c_q1[i_g] = 0.0;
    for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
    {
      c_q1[i_g] += c1[q_r_0][0] * CG1[q_r_0][i_g];
    };
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q0[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        };
      };
    };
  };
  for(int i_r_0 = 0; i_r_0 < 3; i_r_0++)
  {
    for(int i_g = 0; i_g < 6; i_g++)
    {
      ValueType ST1 = 0.0;
      ST1 += CG1[i_r_0][i_g] * c_q1[i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
#ifdef __CUDACC__
      op_atomic_add(localTensor[i_r_0], ST1 * w[i_g]);
#else
      localTensor[i_r_0][0] += ST1 * w[i_g];
#endif
    };
  };
}
"""

        self._mass = op2.Kernel(kernel_mass, "mass")
        self._rhs = op2.Kernel(kernel_rhs, "rhs")

    def tearDown(self):
        del self._nodes
        del self._elements
        del self._elem_node
        del self._mat
        del self._coords
        del self._f
        del self._b
        del self._x
        del self._mass
        del self._rhs

    def _assemble_mass(self):
        op2.par_loop(self._mass, self._elements(2,2),
                     self._mat((self._elem_node(op2.i(0)), self._elem_node(op2.i(1))), op2.INC),
                     self._coords(self._elem_node, op2.READ))

    @unittest.expectedFailure
    def test_assemble(self):
        self._assemble_mass()

        expected_vals = [(0.25, 0.125, 0.0, 0.125),
                         (0.125, 0.291667, 0.0208333, 0.145833),
                         (0.0, 0.0208333, 0.0416667, 0.0208333),
                         (0.125, 0.145833, 0.0208333, 0.291667) ]
        expected_matrix = numpy.asarray(expected_vals, dtype=valuetype)
        # Check that the matrix values equal these values, somehow.
        assertTrue(False)

    @unittest.expectedFailure
    def test_rhs(self):
        # Assemble the RHS here, so if solve fails we know whether the RHS
        # assembly went wrong or something to do with the solve.
        assertTrue(False)

    @unittest.expectedFailure
    def test_solve(self):
        # Assemble matrix and RHS and solve.
        assertTrue(False)

suite = unittest.TestLoader().loadTestsFromTestCase(MatricesTest)
unittest.TextTestRunner(verbosity=0).run(suite)
