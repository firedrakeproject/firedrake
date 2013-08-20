# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

import pytest
import numpy as np
from numpy.testing import assert_allclose

from pyop2 import op2
from pyop2.exceptions import MapValueError

# Data type
valuetype = np.float64

# Constants
NUM_ELE = 2
NUM_NODES = 4
NUM_DIMS = 2

elem_node_map = np.asarray([0, 1, 3, 2, 3, 1], dtype=np.uint32)


@pytest.fixture(scope='module')
def nodes():
    return op2.Set(NUM_NODES, "nodes")


@pytest.fixture(scope='module')
def elements():
    return op2.Set(NUM_ELE, "elements")


@pytest.fixture(scope='module')
def dnodes(nodes):
    return op2.DataSet(nodes, 1, "dnodes")


@pytest.fixture(scope='module')
def dvnodes(nodes):
    return op2.DataSet(nodes, 2, "dvnodes")


@pytest.fixture(scope='module')
def delements(elements):
    return op2.DataSet(elements, 1, "delements")


@pytest.fixture(scope='module')
def elem_node(elements, nodes):
    return op2.Map(elements, nodes, 3, elem_node_map, "elem_node")


@pytest.fixture(scope='module')
def mat(elem_node, dnodes):
    sparsity = op2.Sparsity((dnodes, dnodes), (elem_node, elem_node), "sparsity")
    return op2.Mat(sparsity, valuetype, "mat")


@pytest.fixture(scope='module')
def vecmat(elem_node, dvnodes):
    sparsity = op2.Sparsity((dvnodes, dvnodes), (elem_node, elem_node), "sparsity")
    return op2.Mat(sparsity, valuetype, "vecmat")


@pytest.fixture
def coords(dvnodes):
    coord_vals = np.asarray([(0.0, 0.0), (2.0, 0.0),
                             (1.0, 1.0), (0.0, 1.5)],
                            dtype=valuetype)
    return op2.Dat(dvnodes, coord_vals, valuetype, "coords")


@pytest.fixture(scope='module')
def g(request):
    return op2.Global(1, 1.0, np.float64, "g")


@pytest.fixture
def f(dnodes):
    f_vals = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=valuetype)
    return op2.Dat(dnodes, f_vals, valuetype, "f")


@pytest.fixture
def f_vec(dvnodes):
    f_vals = np.asarray([(1.0, 2.0)] * 4, dtype=valuetype)
    return op2.Dat(dvnodes, f_vals, valuetype, "f")


@pytest.fixture(scope='module')
def b(dnodes):
    b_vals = np.zeros(NUM_NODES, dtype=valuetype)
    return op2.Dat(dnodes, b_vals, valuetype, "b")


@pytest.fixture(scope='module')
def b_vec(dvnodes):
    b_vals = np.zeros(NUM_NODES * 2, dtype=valuetype)
    return op2.Dat(dvnodes, b_vals, valuetype, "b")


@pytest.fixture
def x(dnodes):
    x_vals = np.zeros(NUM_NODES, dtype=valuetype)
    return op2.Dat(dnodes, x_vals, valuetype, "x")


@pytest.fixture
def x_vec(dvnodes):
    x_vals = np.zeros(NUM_NODES * 2, dtype=valuetype)
    return op2.Dat(dvnodes, x_vals, valuetype, "x")


@pytest.fixture
def mass():
    kernel_code = """
void mass(double localTensor[1][1], double* c0[2], int i_r_0, int i_r_1)
{
  double CG1[3][6] = { {  0.09157621, 0.09157621, 0.81684757,
                                   0.44594849, 0.44594849, 0.10810302 },
                                {  0.09157621, 0.81684757, 0.09157621,
                                   0.44594849, 0.10810302, 0.44594849 },
                                {  0.81684757, 0.09157621, 0.09157621,
                                   0.10810302, 0.44594849, 0.44594849 } };
  double d_CG1[3][6][2] = { { {  1., 0. },
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
  double w[6] = {  0.05497587, 0.05497587, 0.05497587, 0.11169079,
                            0.11169079, 0.11169079 };
  double c_q0[6][2][2];
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
    double ST0 = 0.0;
    ST0 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    localTensor[0][0] += ST0 * w[i_g];
  };
}"""
    return op2.Kernel(kernel_code, "mass")


@pytest.fixture
def rhs():
    kernel_code = """
void rhs(double** localTensor, double* c0[2], double* c1[1])
{
  double CG1[3][6] = { {  0.09157621, 0.09157621, 0.81684757,
                                   0.44594849, 0.44594849, 0.10810302 },
                                {  0.09157621, 0.81684757, 0.09157621,
                                   0.44594849, 0.10810302, 0.44594849 },
                                {  0.81684757, 0.09157621, 0.09157621,
                                   0.10810302, 0.44594849, 0.44594849 } };
  double d_CG1[3][6][2] = { { {  1., 0. },
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
  double w[6] = {  0.05497587, 0.05497587, 0.05497587, 0.11169079,
                            0.11169079, 0.11169079 };
  double c_q1[6];
  double c_q0[6][2][2];
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
      double ST1 = 0.0;
      ST1 += CG1[i_r_0][i_g] * c_q1[i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
      localTensor[i_r_0][0] += ST1 * w[i_g];
    };
  };
}"""
    return op2.Kernel(kernel_code, "rhs")


@pytest.fixture
def mass_ffc():
    kernel_code = """
void mass_ffc(double A[1][1], double *x[2], int j, int k)
{
    double J_00 = x[1][0] - x[0][0];
    double J_01 = x[2][0] - x[0][0];
    double J_10 = x[1][1] - x[0][1];
    double J_11 = x[2][1] - x[0][1];

    double detJ = J_00*J_11 - J_01*J_10;
    double det = fabs(detJ);

    double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    double FE0[3][3] = \
    {{0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.166666666666667, 0.666666666666667, 0.166666666666667}};

    for (unsigned int ip = 0; ip < 3; ip++)
    {
      A[0][0] += FE0[ip][j]*FE0[ip][k]*W3[ip]*det;
    }
}
"""
    return op2.Kernel(kernel_code, "mass_ffc")


@pytest.fixture
def rhs_ffc():
    kernel_code = """
void rhs_ffc(double **A, double *x[2], double **w0)
{
    double J_00 = x[1][0] - x[0][0];
    double J_01 = x[2][0] - x[0][0];
    double J_10 = x[1][1] - x[0][1];
    double J_11 = x[2][1] - x[0][1];

    double detJ = J_00*J_11 - J_01*J_10;

    double det = fabs(detJ);

    double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    double FE0[3][3] = \
    {{0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.166666666666667, 0.666666666666667, 0.166666666666667}};

    for (unsigned int ip = 0; ip < 3; ip++)
    {
      double F0 = 0.0;

      for (unsigned int r = 0; r < 3; r++)
      {
        F0 += FE0[ip][r]*w0[r][0];
      }

      for (unsigned int j = 0; j < 3; j++)
      {
        A[j][0] += FE0[ip][j]*F0*W3[ip]*det;
      }
    }
}
"""
    return op2.Kernel(kernel_code, "rhs_ffc")


@pytest.fixture
def rhs_ffc_itspace():
    kernel_code = """
void rhs_ffc_itspace(double A[1], double *x[2], double **w0, int j)
{
    double J_00 = x[1][0] - x[0][0];
    double J_01 = x[2][0] - x[0][0];
    double J_10 = x[1][1] - x[0][1];
    double J_11 = x[2][1] - x[0][1];

    double detJ = J_00*J_11 - J_01*J_10;
    double det = fabs(detJ);

    double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    double FE0[3][3] = \
    {{0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.166666666666667, 0.666666666666667, 0.166666666666667}};

    for (unsigned int ip = 0; ip < 3; ip++)
    {
      double F0 = 0.0;

      for (unsigned int r = 0; r < 3; r++)
      {
        F0 += FE0[ip][r]*w0[r][0];
      }

      A[0] += FE0[ip][j]*F0*W3[ip]*det;
    }
}
"""
    return op2.Kernel(kernel_code, "rhs_ffc_itspace")


@pytest.fixture
def mass_vector_ffc():
    kernel_code = """
void mass_vector_ffc(double A[2][2], double *x[2], int j, int k)
{
    const double J_00 = x[1][0] - x[0][0];
    const double J_01 = x[2][0] - x[0][0];
    const double J_10 = x[1][1] - x[0][1];
    const double J_11 = x[2][1] - x[0][1];

    double detJ = J_00*J_11 - J_01*J_10;
    const double det = fabs(detJ);

    const double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    const double FE0_C0[3][6] =
    {{0.666666666666667, 0.166666666666667, 0.166666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.166666666666667, 0.666666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.666666666666667, 0.166666666666667, 0.0, 0.0, 0.0}};
    const double FE0_C1[3][6] =
    {{0.0, 0.0, 0.0, 0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.666666666666667, 0.166666666666667}};

    for (unsigned int ip = 0; ip < 3; ip++)
    {
      for (unsigned int r = 0; r < 2; r++)
      {
        for (unsigned int s = 0; s < 2; s++)
        {
          A[r][s] += (((FE0_C0[ip][r*3+j]))*((FE0_C0[ip][s*3+k])) + ((FE0_C1[ip][r*3+j]))*((FE0_C1[ip][s*3+k])))*W3[ip]*det;
        }
      }
    }
}
"""
    return op2.Kernel(kernel_code, "mass_vector_ffc")


@pytest.fixture
def rhs_ffc_vector():
    kernel_code = """
void rhs_vector_ffc(double **A, double *x[2], double **w0)
{
    const double J_00 = x[1][0] - x[0][0];
    const double J_01 = x[2][0] - x[0][0];
    const double J_10 = x[1][1] - x[0][1];
    const double J_11 = x[2][1] - x[0][1];

    double detJ = J_00*J_11 - J_01*J_10;

    const double det = fabs(detJ);

    const double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    const double FE0_C0[3][6] = \
    {{0.666666666666667, 0.166666666666667, 0.166666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.166666666666667, 0.666666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.666666666666667, 0.166666666666667, 0.0, 0.0, 0.0}};
    const double FE0_C1[3][6] = \
    {{0.0, 0.0, 0.0, 0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.666666666666667, 0.166666666666667}};

    for (unsigned int ip = 0; ip < 3; ip++)
    {
      double F0 = 0.0;
      double F1 = 0.0;
      for (unsigned int r = 0; r < 3; r++)
      {
        for (unsigned int s = 0; s < 2; s++)
        {
          F0 += (FE0_C0[ip][3*s+r])*w0[r][s];
          F1 += (FE0_C1[ip][3*s+r])*w0[r][s];
        }
      }
      for (unsigned int j = 0; j < 3; j++)
      {
        for (unsigned int r = 0; r < 2; r++)
        {
          A[j][r] += (((FE0_C0[ip][r*3+j]))*F0 + ((FE0_C1[ip][r*3+j]))*F1)*W3[ip]*det;
        }
      }
    }
}"""
    return op2.Kernel(kernel_code, "rhs_vector_ffc")


@pytest.fixture
def rhs_ffc_vector_itspace():
    kernel_code = """
void rhs_vector_ffc_itspace(double A[2], double *x[2], double **w0, int j)
{
    const double J_00 = x[1][0] - x[0][0];
    const double J_01 = x[2][0] - x[0][0];
    const double J_10 = x[1][1] - x[0][1];
    const double J_11 = x[2][1] - x[0][1];

    double detJ = J_00*J_11 - J_01*J_10;
    const double det = fabs(detJ);

    const double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    const double FE0_C0[3][6] = \
    {{0.666666666666667, 0.166666666666667, 0.166666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.166666666666667, 0.666666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.666666666666667, 0.166666666666667, 0.0, 0.0, 0.0}};
    const double FE0_C1[3][6] = \
    {{0.0, 0.0, 0.0, 0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.666666666666667, 0.166666666666667}};

    for (unsigned int ip = 0; ip < 3; ip++)
    {
      double F0 = 0.0;
      double F1 = 0.0;
      for (unsigned int r = 0; r < 3; r++)
      {
        for (unsigned int s = 0; s < 2; s++)
        {
          F0 += (FE0_C0[ip][3*s+r])*w0[r][s];
          F1 += (FE0_C1[ip][3*s+r])*w0[r][s];
        }
      }

      for (unsigned int r = 0; r < 2; r++)
      {
        A[r] += (((FE0_C0[ip][r*3+j]))*F0 + ((FE0_C1[ip][r*3+j]))*F1)*W3[ip]*det;
      }
    }
}"""
    return op2.Kernel(kernel_code, "rhs_vector_ffc_itspace")


@pytest.fixture
def zero_dat():
    kernel_code = """
void zero_dat(double *dat)
{
  *dat = 0.0;
}
"""
    return op2.Kernel(kernel_code, "zero_dat")


@pytest.fixture
def zero_vec_dat():
    kernel_code = """
void zero_vec_dat(double *dat)
{
  dat[0] = 0.0; dat[1] = 0.0;
}
"""
    return op2.Kernel(kernel_code, "zero_vec_dat")


@pytest.fixture
def kernel_inc():
    kernel_code = """
void kernel_inc(double entry[1][1], double* g, int i, int j)
{
  entry[0][0] += *g;
}
"""
    return op2.Kernel(kernel_code, "kernel_inc")


@pytest.fixture
def kernel_set():
    kernel_code = """
void kernel_set(double entry[1][1], double* g, int i, int j)
{
  entry[0][0] = *g;
}
"""
    return op2.Kernel(kernel_code, "kernel_set")


@pytest.fixture
def kernel_inc_vec():
    kernel_code = """
void kernel_inc_vec(double entry[2][2], double* g, int i, int j)
{
  entry[0][0] += *g;
  entry[0][1] += *g;
  entry[1][0] += *g;
  entry[1][1] += *g;
}
"""
    return op2.Kernel(kernel_code, "kernel_inc_vec")


@pytest.fixture
def kernel_set_vec():
    kernel_code = """
void kernel_set_vec(double entry[2][2], double* g, int i, int j)
{
  entry[0][0] = *g;
  entry[0][1] = *g;
  entry[1][0] = *g;
  entry[1][1] = *g;
}
"""
    return op2.Kernel(kernel_code, "kernel_set_vec")


@pytest.fixture
def expected_matrix():
    expected_vals = [(0.25, 0.125, 0.0, 0.125),
                     (0.125, 0.291667, 0.0208333, 0.145833),
                     (0.0, 0.0208333, 0.0416667, 0.0208333),
                     (0.125, 0.145833, 0.0208333, 0.291667)]
    return np.asarray(expected_vals, dtype=valuetype)


@pytest.fixture
def expected_vector_matrix():
    expected_vals = [(0.25, 0., 0.125, 0., 0., 0., 0.125, 0.),
                     (0., 0.25, 0., 0.125, 0., 0., 0., 0.125),
                     (0.125, 0., 0.29166667, 0.,
                      0.02083333, 0., 0.14583333, 0.),
                     (0., 0.125, 0., 0.29166667, 0.,
                      0.02083333, 0., 0.14583333),
                     (0., 0., 0.02083333, 0.,
                      0.04166667, 0., 0.02083333, 0.),
                     (0., 0., 0., 0.02083333, 0.,
                      0.04166667, 0., 0.02083333),
                     (0.125, 0., 0.14583333, 0.,
                      0.02083333, 0., 0.29166667, 0.),
                     (0., 0.125, 0., 0.14583333, 0., 0.02083333, 0., 0.29166667)]
    return np.asarray(expected_vals, dtype=valuetype)


@pytest.fixture
def expected_rhs():
    return np.asarray([[0.9999999523522115], [1.3541666031724144],
                       [0.2499999883507239], [1.6458332580869566]],
                      dtype=valuetype)


@pytest.fixture
def expected_vec_rhs():
    return np.asarray([[0.5, 1.0], [0.58333333, 1.16666667],
                       [0.08333333, 0.16666667], [0.58333333, 1.16666667]],
                      dtype=valuetype)


class TestSparsity:

    """
    Sparsity tests
    """

    def test_build_sparsity(self, backend):
        """Building a sparsity from a pair of maps should give the expected
        rowptr and colidx."""
        elements = op2.Set(4)
        nodes = op2.Set(5)
        elem_node = op2.Map(elements, nodes, 3, [0, 4, 3, 0, 1, 4,
                                                 1, 2, 4, 2, 3, 4])
        sparsity = op2.Sparsity((nodes, nodes), (elem_node, elem_node))
        assert all(sparsity._rowptr == [0, 4, 8, 12, 16, 21])
        assert all(sparsity._colidx == [0, 1, 3, 4, 0, 1, 2, 4, 1, 2,
                                        3, 4, 0, 2, 3, 4, 0, 1, 2, 3, 4])

    def test_sparsity_null_maps(self, backend):
        """Building sparsity from a pair of non-initialized maps should fail."""
        s = op2.Set(5)
        with pytest.raises(MapValueError):
            m = op2.Map(s, s, 1)
            op2.Sparsity((s, s), (m, m))


class TestMatrices:

    """
    Matrix tests

    """

    def test_minimal_zero_mat(self, backend, skip_cuda):
        """Assemble a matrix that is all zeros."""
        zero_mat_code = """
void zero_mat(double local_mat[1][1], int i, int j)
{
  local_mat[i][j] = 0.0;
}
"""
        nelems = 128
        set = op2.Set(nelems)
        map = op2.Map(set, set, 1, np.array(range(nelems), np.uint32))
        sparsity = op2.Sparsity((set, set), (map, map))
        mat = op2.Mat(sparsity, np.float64)
        kernel = op2.Kernel(zero_mat_code, "zero_mat")
        op2.par_loop(kernel, set(1, 1), mat(
            (map[op2.i[0]], map[op2.i[1]]), op2.WRITE))

        expected_matrix = np.zeros((nelems, nelems), dtype=np.float64)
        eps = 1.e-12
        assert_allclose(mat.values, expected_matrix, eps)

    def test_assemble_mat(self, backend, mass, mat, coords, elements,
                          elem_node, expected_matrix):
        """Assemble a simple finite-element matrix and check the result."""
        op2.par_loop(mass, elements(3, 3),
                     mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                     coords(elem_node, op2.READ))
        eps = 1.e-5
        assert_allclose(mat.values, expected_matrix, eps)

    def test_assemble_rhs(self, backend, rhs, elements, b, coords, f,
                          elem_node, expected_rhs):
        """Assemble a simple finite-element right-hand side and check result."""
        op2.par_loop(rhs, elements,
                     b(elem_node, op2.INC),
                     coords(elem_node, op2.READ),
                     f(elem_node, op2.READ))

        eps = 1.e-12
        assert_allclose(b.data, expected_rhs, eps)

    def test_solve(self, backend, mat, b, x, f):
        """Solve a linear system where the solution is equal to the right-hand
        side and check the result."""
        op2.solve(mat, x, b)
        eps = 1.e-8
        assert_allclose(x.data, f.data, eps)

    def test_zero_matrix(self, backend, mat):
        """Test that the matrix is zeroed correctly."""
        mat.zero()
        expected_matrix = np.zeros((4, 4), dtype=valuetype)
        eps = 1.e-14
        assert_allclose(mat.values, expected_matrix, eps)

    def test_set_matrix(self, backend, mat, elements, elem_node,
                        kernel_inc, kernel_set, g, skip_cuda):
        """Test accessing a scalar matrix with the WRITE access by adding some
        non-zero values into the matrix, then setting them back to zero with a
        kernel using op2.WRITE"""
        op2.par_loop(kernel_inc, elements(3, 3),
                     mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                     g(op2.READ))
        # Check we have ones in the matrix
        assert mat.array.sum() == 3 * 3 * elements.size
        op2.par_loop(kernel_set, elements(3, 3),
                     mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.WRITE),
                     g(op2.READ))
        # Check we have set all values in the matrix to 1
        assert_allclose(mat.array, np.ones_like(mat.array))
        mat.zero()

    def test_set_matrix_vec(self, backend, vecmat, elements, elem_node,
                            kernel_inc_vec, kernel_set_vec, g, skip_cuda):
        """Test accessing a vector matrix with the WRITE access by adding some
        non-zero values into the matrix, then setting them back to zero with a
        kernel using op2.WRITE"""
        op2.par_loop(kernel_inc_vec, elements(3, 3),
                     vecmat(
                         (elem_node[
                             op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                     g(op2.READ))
        # Check we have ones in the matrix
        assert vecmat.array.sum() == 2 * 2 * 3 * 3 * elements.size
        op2.par_loop(kernel_set_vec, elements(3, 3),
                     vecmat(
                         (elem_node[
                             op2.i[0]], elem_node[op2.i[1]]), op2.WRITE),
                     g(op2.READ))
        # Check we have set all values in the matrix to 1
        assert_allclose(vecmat.array, np.ones_like(vecmat.array))
        vecmat.zero()

    def test_zero_rhs(self, backend, b, zero_dat, nodes):
        """Test that the RHS is zeroed correctly."""
        op2.par_loop(zero_dat, nodes,
                     b(op2.IdentityMap, op2.WRITE))
        assert all(b.data == np.zeros_like(b.data))

    def test_assemble_ffc(self, backend, mass_ffc, mat, coords, elements,
                          elem_node, expected_matrix):
        """Test that the FFC mass assembly assembles the correct values."""
        op2.par_loop(mass_ffc, elements(3, 3),
                     mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                     coords(elem_node, op2.READ))
        eps = 1.e-5
        assert_allclose(mat.values, expected_matrix, eps)

    def test_assemble_vec_mass(self, backend, mass_vector_ffc, vecmat, coords,
                               elements, expected_vector_matrix, elem_node):
        """Test that the FFC vector mass assembly assembles the correct values."""
        op2.par_loop(mass_vector_ffc, elements(3, 3),
                     vecmat(
                         (elem_node[
                             op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                     coords(elem_node, op2.READ))
        eps = 1.e-6
        assert_allclose(vecmat.values, expected_vector_matrix, eps)

    def test_rhs_ffc(self, backend, rhs_ffc, elements, b, coords, f,
                     elem_node, expected_rhs):
        """Test that the FFC rhs assembly assembles the correct values."""
        op2.par_loop(rhs_ffc, elements,
                     b(elem_node, op2.INC),
                     coords(elem_node, op2.READ),
                     f(elem_node, op2.READ))

        eps = 1.e-6
        assert_allclose(b.data, expected_rhs, eps)

    def test_rhs_ffc_itspace(self, backend, rhs_ffc_itspace, elements, b,
                             coords, f, elem_node, expected_rhs,
                             zero_dat, nodes):
        """Test that the FFC right-hand side assembly using iteration spaces
        assembles the correct values."""
        # Zero the RHS first
        op2.par_loop(zero_dat, nodes,
                     b(op2.IdentityMap, op2.WRITE))
        op2.par_loop(rhs_ffc_itspace, elements(3),
                     b(elem_node[op2.i[0]], op2.INC),
                     coords(elem_node, op2.READ),
                     f(elem_node, op2.READ))
        eps = 1.e-6
        assert_allclose(b.data, expected_rhs, eps)

    def test_rhs_vector_ffc(self, backend, rhs_ffc_vector, elements, b_vec,
                            coords, f_vec, elem_node,
                            expected_vec_rhs, nodes):
        """Test that the FFC vector rhs assembly assembles the correct values."""
        op2.par_loop(rhs_ffc_vector, elements,
                     b_vec(elem_node, op2.INC),
                     coords(elem_node, op2.READ),
                     f_vec(elem_node, op2.READ))
        eps = 1.e-6
        assert_allclose(b_vec.data, expected_vec_rhs, eps)

    def test_rhs_vector_ffc_itspace(self, backend, rhs_ffc_vector_itspace,
                                    elements, b_vec, coords, f_vec, elem_node,
                                    expected_vec_rhs, nodes, zero_vec_dat):
        """Test that the FFC vector right-hand side assembly using iteration
        spaces assembles the correct values."""
        # Zero the RHS first
        op2.par_loop(zero_vec_dat, nodes,
                     b_vec(op2.IdentityMap, op2.WRITE))
        op2.par_loop(rhs_ffc_vector_itspace, elements(3),
                     b_vec(elem_node[op2.i[0]], op2.INC),
                     coords(elem_node, op2.READ),
                     f_vec(elem_node, op2.READ))
        eps = 1.e-6
        assert_allclose(b_vec.data, expected_vec_rhs, eps)

    def test_zero_rows(self, backend, mat, expected_matrix):
        """Zeroing a row in the matrix should set the diagonal to the given
        value and all other values to 0."""
        expected_matrix[0] = [12.0, 0.0, 0.0, 0.0]
        mat.zero_rows([0], 12.0)
        eps = 1.e-5
        assert_allclose(mat.values, expected_matrix, eps)

    def test_zero_last_row(self, backend, mat, expected_matrix):
        """Zeroing a row in the matrix should set the diagonal to the given
        value and all other values to 0."""
        which = NUM_NODES - 1
        # because the previous test zeroed the first row
        expected_matrix[0] = [12.0, 0.0, 0.0, 0.0]
        expected_matrix[which] = [0.0, 0.0, 0.0, 4.0]
        mat.zero_rows([which], 4.0)
        eps = 1.e-5
        assert_allclose(mat.values, expected_matrix, eps)

    def test_vector_solve(self, backend, vecmat, b_vec, x_vec, f_vec):
        """Solve a linear system with a vector matrix where the solution is
        equal to the right-hand side and check the result."""
        op2.solve(vecmat, x_vec, b_vec)
        eps = 1.e-12
        assert_allclose(x_vec.data, f_vec.data, eps)

    def test_zero_vector_matrix(self, backend, vecmat):
        """Test that the vector matrix is zeroed correctly."""
        vecmat.zero()
        expected_matrix = np.zeros((8, 8), dtype=valuetype)
        eps = 1.e-14
        assert_allclose(vecmat.values, expected_matrix, eps)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
