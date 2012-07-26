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
import numpy

from pyop2 import op2

# Data type
valuetype = numpy.float64

# Constants
NUM_ELE   = 2
NUM_NODES = 4
NUM_DIMS  = 2

class TestMatrices:
    """
    Matrix tests

    """

    def pytest_funcarg__nodes(cls, request):
        return op2.Set(NUM_NODES, "nodes")

    def pytest_funcarg__elements(cls, request):
        return op2.Set(NUM_ELE, "elements")

    def pytest_funcarg__elem_node(cls, request):
        elements = request.getfuncargvalue('elements')
        nodes = request.getfuncargvalue('nodes')
        elem_node_map = numpy.asarray([ 0, 1, 3, 2, 3, 1 ], dtype=numpy.uint32)
        return op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

    def pytest_funcarg__mat(cls, request):
        elem_node = request.getfuncargvalue('elem_node')
        sparsity = op2.Sparsity(elem_node, elem_node, 1, "sparsity")
        return request.cached_setup(
                setup=lambda: op2.Mat(sparsity, 1, valuetype, "mat"),
                scope='session')

    def pytest_funcarg__coords(cls, request):
        nodes = request.getfuncargvalue('nodes')
        coord_vals = numpy.asarray([ (0.0, 0.0), (2.0, 0.0),
                                     (1.0, 1.0), (0.0, 1.5) ],
                                   dtype=valuetype)
        return op2.Dat(nodes, 2, coord_vals, valuetype, "coords")

    def pytest_funcarg__f(cls, request):
        nodes = request.getfuncargvalue('nodes')
        f_vals = numpy.asarray([ 1.0, 2.0, 3.0, 4.0 ], dtype=valuetype)
        return op2.Dat(nodes, 1, f_vals, valuetype, "f")

    def pytest_funcarg__b(cls, request):
        nodes = request.getfuncargvalue('nodes')
        b_vals = numpy.asarray([0.0]*NUM_NODES, dtype=valuetype)
        return request.cached_setup(
                setup=lambda: op2.Dat(nodes, 1, b_vals, valuetype, "b"),
                scope='session')

    def pytest_funcarg__x(cls, request):
        nodes = request.getfuncargvalue('nodes')
        x_vals = numpy.asarray([0.0]*NUM_NODES, dtype=valuetype)
        return op2.Dat(nodes, 1, x_vals, valuetype, "x")

    def pytest_funcarg__mass(cls, request):
        kernel_code = """
void mass(double* localTensor, double* c0[2], int i_r_0, int i_r_1)
{
  const double CG1[3][6] = { {  0.09157621, 0.09157621, 0.81684757,
                                   0.44594849, 0.44594849, 0.10810302 },
                                {  0.09157621, 0.81684757, 0.09157621,
                                   0.44594849, 0.10810302, 0.44594849 },
                                {  0.81684757, 0.09157621, 0.09157621,
                                   0.10810302, 0.44594849, 0.44594849 } };
  const double d_CG1[3][6][2] = { { {  1., 0. },
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
  const double w[6] = {  0.05497587, 0.05497587, 0.05497587, 0.11169079,
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
    localTensor[0] += ST0 * w[i_g];
  };
}"""
        return op2.Kernel(kernel_code, "mass")

    def pytest_funcarg__rhs(cls, request):

        kernel_code = """
void rhs(double** localTensor, double* c0[2], double* c1[1])
{
  const double CG1[3][6] = { {  0.09157621, 0.09157621, 0.81684757,
                                   0.44594849, 0.44594849, 0.10810302 },
                                {  0.09157621, 0.81684757, 0.09157621,
                                   0.44594849, 0.10810302, 0.44594849 },
                                {  0.81684757, 0.09157621, 0.09157621,
                                   0.10810302, 0.44594849, 0.44594849 } };
  const double d_CG1[3][6][2] = { { {  1., 0. },
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
  const double w[6] = {  0.05497587, 0.05497587, 0.05497587, 0.11169079,
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

    @pytest.mark.xfail
    def test_assemble(self, mass, mat, coords, elements, elem_node):
        op2.par_loop(mass, elements(3,3),
                     mat((elem_node(op2.i(0)), elem_node(op2.i(1))), op2.INC),
                     coords(elem_node, op2.READ))

        expected_vals = [(0.25, 0.125, 0.0, 0.125),
                         (0.125, 0.291667, 0.0208333, 0.145833),
                         (0.0, 0.0208333, 0.0416667, 0.0208333),
                         (0.125, 0.145833, 0.0208333, 0.291667) ]
        expected_matrix = numpy.asarray(expected_vals, dtype=valuetype)
        # Check that the matrix values equal these values, somehow.
        assert False

    def test_rhs(self, rhs, elements, b, coords, f, elem_node):
        op2.par_loop(rhs, elements,
                     b(elem_node, op2.INC),
                     coords(elem_node, op2.READ),
                     f(elem_node, op2.READ))

        expected = numpy.asarray([[0.9999999523522115], [1.3541666031724144],
                                  [0.2499999883507239], [1.6458332580869566]],
                                  dtype=valuetype)
        eps = 1.e-12
        assert all(abs(b.data-expected)<eps)

    def test_solve(self, mat, b, x, f):
        op2.solve(mat, b, x)
        eps = 1.e-12
        assert all(abs(x.data-f.data)<eps)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
