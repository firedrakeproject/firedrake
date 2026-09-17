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
from immutabledict import immutabledict as idict

import pyop3 as op3


ScalarType_c = op3.dtypes.as_cstr(op3.ScalarType)


# Data type
valuetype = op3.ScalarType

# Constants
NUM_ELE = 2
NUM_NODES = 4
NUM_DIMS = 2
layers = 11

elem_node_map = np.asarray([[0, 1, 3], [2, 3, 1]], dtype=np.uint32)


@pytest.fixture(scope='module')
def nodes():
    return op3.Axis(NUM_NODES, "nodes")


@pytest.fixture(scope='module')
def elements():
    return op3.Axis(NUM_ELE, "elements")


@pytest.fixture(scope='module')
def dnodes(nodes):
    return op3.AxisTree.from_iterable([nodes, 1])


@pytest.fixture(scope='module')
def dvnodes(nodes):
    return op3.AxisTree.from_iterable([nodes, 2])


@pytest.fixture(scope='module')
def delements(elements):
    return op3.AxisTree.from_iterable([elements, 1])


@pytest.fixture(scope='module')
def elem_node(elements, nodes):
    elem_node_dat = op3.Dat(op3.AxisTree.from_iterable([elements, 3]), data=elem_node_map)
    return op3.Map(
        {
            idict({"elements": None}): [[op3.TabulatedMapComponent("nodes", None, elem_node_dat)]]
        }
    )


@pytest.fixture
def mat(mass, elements, coords, dnodes, elem_node):
    return op3.Mat.empty(dnodes, dnodes)


@pytest.fixture
def mass_mat(mass, elements, mat, coords, elem_node):
    mat.zero(eager=True)
    op3.loop(
        e := elements.iter(),
        mass(mat[elem_node(e), elem_node(e)], coords[elem_node(e)]),
        eager=True,
    )
    return mat


@pytest.fixture
def coords(dvnodes):
    coord_vals = np.asarray([(0.0, 0.0), (2.0, 0.0),
                             (1.0, 1.0), (0.0, 1.5)],
                            dtype=valuetype)
    return op3.Dat(dvnodes, data=coord_vals)


@pytest.fixture
def g(request):
    return op3.Scalar(1.0)


@pytest.fixture
def f(dnodes):
    f_vals = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=valuetype)
    return op3.Dat(dnodes, data=f_vals)


@pytest.fixture
def f_vec(dvnodes):
    f_vals = np.asarray([(1.0, 2.0)] * 4, dtype=valuetype)
    return op3.Dat(dvnodes, data=f_vals)


@pytest.fixture
def b(dnodes):
    b_vals = np.zeros(NUM_NODES, dtype=valuetype)
    return op3.Dat(dnodes, data=b_vals)


@pytest.fixture
def b_vec(dvnodes):
    b_vals = np.zeros(NUM_NODES * 2, dtype=valuetype)
    return op3.Dat(dvnodes, data=b_vals)


@pytest.fixture
def b_rhs(b, rhs, elements, coords, f, elem_node):
    b.zero(eager=True)
    op3.loop(
        e := elements.iter(),
        rhs(b[elem_node(e)], coords[elem_node(e)], f[elem_node(e)]),
        eager=True,
    )
    return b


@pytest.fixture
def x(dnodes):
    x_vals = np.zeros(NUM_NODES, dtype=valuetype)
    return op3.Dat(dnodes, data=x_vals)


@pytest.fixture
def x_vec(dvnodes):
    x_vals = np.zeros(NUM_NODES * 2, dtype=valuetype)
    return op3.Dat(dvnodes, data=x_vals)


@pytest.fixture
def mass():
    kernel_code = """
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
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0*2+i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        };
      };
    };
  };
  for(int i_g = 0; i_g < 6; i_g++) {
    for (int i_r_0=0; i_r_0<3; ++i_r_0) {
      for (int i_r_1=0; i_r_1<3; ++i_r_1) {
        double ST0 = 0.0;
        ST0 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
        localTensor[i_r_0*3+i_r_1] += ST0 * w[i_g];
      }
    }
  }"""
    return op3.Function.from_c_string(
        "mass",
        kernel_code,
        [("localTensor", op3.ScalarType, op3.INC), ("c0", op3.ScalarType, op3.READ)],
    )


@pytest.fixture
def rhs():
    kernel_code = """
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
      c_q1[i_g] += c1[q_r_0] * CG1[q_r_0][i_g];
    };
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q0[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0*2+i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
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
      localTensor[i_r_0] += ST1 * w[i_g];
    };
  };"""
    return op3.Function.from_c_string(
        "rhs",
        kernel_code,
        [
            ("localTensor", op3.ScalarType, op3.INC),
            ("c0", op3.ScalarType, op3.READ),
            ("c1", op3.ScalarType, op3.READ),
        ],
    )


@pytest.fixture
def mass_ffc():
    kernel_code = """\
  double J_00 = x[2] - x[0];
  double J_01 = x[4] - x[0];
  double J_10 = x[3] - x[1];
  double J_11 = x[5] - x[1];

  double detJ = J_00*J_11 - J_01*J_10;
  double det = fabs(detJ);

  double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
  double FE0[3][3] = \
  {{0.666666666666667, 0.166666666666667, 0.166666666666667},
  {0.166666666666667, 0.166666666666667, 0.666666666666667},
  {0.166666666666667, 0.666666666666667, 0.166666666666667}};

  for (unsigned int ip = 0; ip < 3; ip++)
    for (int j=0; j<3; ++j)
      for (int k=0; k<3; ++k)
        A[j*3+k] += FE0[ip][j]*FE0[ip][k]*W3[ip]*det;"""
    return op3.Function.from_c_string(
        "mass_ffc",
        kernel_code,
        [
            ("A", op3.ScalarType, op3.INC),
            ("x", op3.ScalarType, op3.READ),
        ],
    )


@pytest.fixture
def rhs_ffc():
    kernel_code = """\
    double J_00 = x[2] - x[0];
    double J_01 = x[4] - x[0];
    double J_10 = x[3] - x[1];
    double J_11 = x[5] - x[1];

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
        F0 += FE0[ip][r]*w0[r];
      }

      for (unsigned int j = 0; j < 3; j++)
      {
        A[j] += FE0[ip][j]*F0*W3[ip]*det;
      }
    }"""
    return op3.Function.from_c_string(
        "rhs_ffc",
        kernel_code,
        [
            ("A", op3.ScalarType, op3.INC),
            ("x", op3.ScalarType, op3.READ),
            ("w0", op3.ScalarType, op3.READ),
        ],
    )


@pytest.fixture
def rhs_ffc_itspace():
    kernel_code = """\
  double J_00 = x[2] - x[0];
  double J_01 = x[4] - x[0];
  double J_10 = x[3] - x[1];
  double J_11 = x[5] - x[1];

  double detJ = J_00*J_11 - J_01*J_10;
  double det = fabs(detJ);

  double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
  double FE0[3][3] = \
  {{0.666666666666667, 0.166666666666667, 0.166666666666667},
  {0.166666666666667, 0.166666666666667, 0.666666666666667},
  {0.166666666666667, 0.666666666666667, 0.166666666666667}};

  for (unsigned int ip = 0; ip < 3; ip++) {
    double F0 = 0.0;

    for (unsigned int r = 0; r < 3; r++)
      F0 += FE0[ip][r]*w0[r];
    for (unsigned int j=0; j<3; ++j)
      A[j] += FE0[ip][j]*F0*W3[ip]*det;
  }"""
    return op3.Function.from_c_string(
        "rhs_ffc_itspace",
        kernel_code,
        [
            ("A", op3.ScalarType, op3.INC),
            ("x", op3.ScalarType, op3.READ),
            ("w0", op3.ScalarType, op3.READ),
        ],
    )


@pytest.fixture
def zero_dat():
    return op3.Function.from_c_string(
        "zero_dat", "*dat = 0.0;", [("dat", op3.ScalarType, op3.WRITE)],
    )


@pytest.fixture
def zero_vec_dat():
    return op3.Function.from_c_string(
        "zero_vec_dat", "dat[0] = 0.0; dat[1] = 0.0;", [("dat", op3.ScalarType, op3.WRITE)],
    )


@pytest.fixture
def kernel_inc():
    kernel_code = """\
  for (int i=0; i<3; ++i)
    for (int j=0; j<3; ++j)
      entry[i*3+j] += g[0];"""
    return op3.Function.from_c_string(
        "inc", kernel_code, [("entry", op3.ScalarType, op3.INC), ("g", "double", op3.READ)],
    )


@pytest.fixture
def kernel_set():
    kernel_code = """\
for (int i=0; i<3; ++i)
  for (int j=0; j<3; ++j)
    entry[i*3+j] = g[0];"""
    return op3.Function.from_c_string(
        "set", kernel_code, [("entry", op3.ScalarType, op3.WRITE), ("g", "double", op3.READ)],
    )


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
    return np.asarray([0.9999999523522115, 1.3541666031724144,
                       0.2499999883507239, 1.6458332580869566],
                      dtype=valuetype)


@pytest.fixture
def expected_vec_rhs():
    return np.asarray([[0.5, 1.0], [0.58333333, 1.16666667],
                       [0.08333333, 0.16666667], [0.58333333, 1.16666667]],
                      dtype=valuetype)


@pytest.fixture
def mset():
    return op3.Axis([3, 4], "mset")


@pytest.fixture
def mvset(mset):
    axes = mset.as_tree()
    axes = axes.add_axis({"mset": 0}, op3.Axis(2))
    axes = axes.add_axis({"mset": 1}, op3.Axis(2))
    return axes


@pytest.fixture
def mdat(mset):
    return op3.Dat(mset, data=np.asarray([1, 2, 3, 1, 2, 3, 4]))


@pytest.fixture
def mvdat(mvset):
    return op3.Dat(mvset, data=np.asarray([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4]))


@pytest.fixture
def mmap(mset):
    dat0 = op3.Dat(
        op3.AxisTree.from_iterable([mset.linearize(0), 1]),
        data=np.asarray([[0], [1], [2]], dtype=op3.IntType),
    )
    dat1 = op3.Dat(
        op3.AxisTree.from_iterable([mset.linearize(0), 2]),
        data=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=op3.IntType),
    )
    return op3.Map({
        idict({"mset": 0}): [[
            op3.TabulatedMapComponent("mset", 0, dat0),
            op3.TabulatedMapComponent("mset", 1, dat1),
        ]],
    })


class TestMatrices:

    @pytest.mark.parametrize("intent", [op3.RW, op3.MAX_RW, op3.MAX_WRITE, op3.MIN_RW, op3.MIN_WRITE])
    def test_invalid_intent(self, nodes, mat, intent):
        """Mat args can only have modes READ, WRITE and INC."""
        dummy = op3.Function.from_c_string("dummy", "mat[0] = 0.0;", [("mat", op3.ScalarType, intent)])
        with pytest.raises(op3.exceptions.InvalidIntentException):
            op3.loop(n := nodes.iter(), dummy(mat[n, n]), eager=True)

    # TODO: PetscMatBuffer test
    @pytest.mark.parametrize("dim", [1, 2])
    def test_mat_set_diagonal(self, nodes, dim):
        "Set the diagonal of the entire matrix to 1.0"
        axes = op3.AxisTree.from_iterable([nodes, dim])
        mat = op3.Mat.empty(axes, axes)
        mat.buffer.set_diagonal(1)
        assert np.allclose(mat.values, np.identity(nodes.local_size * dim))

    @pytest.mark.parametrize("dim", [1, 2])
    def test_mat_repeated_set_diagonal(self, nodes, elem_node, dim):
        "Set the diagonal of the entire matrix to 1.0"
        axes = op3.AxisTree.from_iterable([nodes, dim])
        mat = op3.Mat.empty(axes, axes)

        mat.buffer.set_diagonal(1)
        assert np.allclose(mat.values, np.identity(nodes.local_size * dim))

        mat.buffer.set_diagonal(2)
        assert np.allclose(mat.values, np.identity(nodes.local_size * dim)*2)

    def test_minimal_zero_mat(self):
        """Assemble a matrix that is all zeros."""
        axes = op3.Axis(128).as_tree()
        mat = op3.Mat.empty(axes, axes)
        kernel = op3.Function.from_c_string(
            "zero_mat", "local_mat[0] = 0.0;", [("local_mat", op3.ScalarType, op3.WRITE)]
        )
        op3.loop(p := axes.iter(), kernel(mat[p, p]), eager=True)

        expected = np.zeros((128, 128))
        assert np.allclose(mat.values, expected)

    def test_assemble_mat(self, mass, mat, coords, elements,
                          elem_node, expected_matrix):
        """Assemble a simple finite-element matrix and check the result."""
        mat.zero(eager=True)
        op3.loop(
            e := elements.iter(),
            mass(mat[elem_node(e), elem_node(e)], coords[elem_node(e)]),
            eager=True,
        )

        assert np.allclose(mat.values, expected_matrix)

    def test_assemble_rhs(self, rhs, elements, b, coords, f,
                          elem_node, expected_rhs):
        """Assemble a simple finite-element right-hand side and check result."""
        b.zero(eager=True)
        op3.loop(
            e := elements.iter(),
            rhs(b[elem_node(e)], coords[elem_node(e)], f[elem_node(e)]),
            eager=True,
        )
        assert np.allclose(b.data_ro.ravel(), expected_rhs)

    def test_solve(self, mass_mat, b_rhs, x, f):
        """Solve a linear system where the solution is equal to the right-hand
        side and check the result."""
        x = np.linalg.solve(mass_mat.values, b_rhs.data_ro)
        assert np.allclose(x, f.data_ro)

    def test_zero_matrix(self, mat):
        """Test that the matrix is zeroed correctly."""
        mat.zero(eager=True)
        assert np.allclose(mat.values, 0)

    def test_set_matrix(self, mat, elements, elem_node,
                        kernel_inc, kernel_set, g):
        """Test accessing a scalar matrix with the WRITE access by adding some
        non-zero values into the matrix, then setting them back to zero with a
        kernel using op2.WRITE"""
        mat.zero(eager=True)
        op3.loop(
            e := elements.iter(),
            kernel_inc(mat[elem_node(e), elem_node(e)], g),
            eager=True,
        )

        # Check we have ones in the matrix
        assert mat.values.sum() == 3 * 3 * elements.size

        op3.loop(
            e := elements.iter(),
            kernel_set(mat[elem_node(e), elem_node(e)], g),
            eager=True,
        )
        assert mat.values.sum() == (3 * 3 - 2) * elements.size

    def test_zero_rhs(self, b, zero_dat, nodes):
        """Test that the RHS is zeroed correctly."""
        op3.loop(n := nodes.iter(), zero_dat(b), eager=True)
        assert np.allclose(b.data_ro, 0)

    def test_assemble_ffc(self, mass_ffc, mat, coords, elements,
                          elem_node, expected_matrix):
        """Test that the FFC mass assembly assembles the correct values."""
        op3.loop(
            e := elements.iter(),
            mass_ffc(mat[elem_node(e), elem_node(e)], coords[elem_node(e)]),
            eager=True,
        )
        assert np.allclose(mat.values, expected_matrix)

    def test_rhs_ffc(self, rhs_ffc, elements, b, coords, f,
                     elem_node, expected_rhs):
        """Test that the FFC rhs assembly assembles the correct values."""
        op3.loop(
            e := elements.iter(),
            rhs_ffc(b[elem_node(e)], coords[elem_node(e)], f[elem_node(e)]),
            eager=True,
        )
        assert np.allclose(b.data_ro.ravel(), expected_rhs)

    def test_rhs_ffc_itspace(self, rhs_ffc_itspace, elements, b,
                             coords, f, elem_node, expected_rhs,
                             zero_dat, nodes):
        """Test that the FFC right-hand side assembly using iteration spaces
        assembles the correct values."""
        b.zero(eager=True)
        op3.loop(
            e := elements.iter(),
            rhs_ffc_itspace(b[elem_node(e)], coords[elem_node(e)], f[elem_node(e)]),
            eager=True,
        )
        assert np.allclose(b.data_ro.ravel(), expected_rhs)


class TestMixedMatrices:

    # off-diagonal blocks
    od = np.array([[1.0, 2.0, 0.0, 0.0],
                   [0.0, 4.0, 6.0, 0.0],
                   [0.0, 0.0, 9.0, 12.0]])
    # lower left block
    ll = (np.diag([1.0, 8.0, 18.0, 16.0])
          + np.diag([2.0, 6.0, 12.0], -1)
          + np.diag([2.0, 6.0, 12.0], 1))

    @pytest.fixture
    def mat(self, mset, mmap, mdat):
        mat = op3.Mat.empty(mset, mset)

        addone = """\
            for (int i = 0; i < 3; i++)
               for (int j = 0; j < 3; j++)
                  v[i*3 + j] += d[i]*d[j];"""
        addone = op3.Function.from_c_string(
            "addone_mat", addone, [("v", op3.ScalarType, op3.INC), ("d", "double", op3.READ)]
        )

        op3.loop(
            p := mset[0].iter(),
            addone(mat[mmap(p), mmap(p)], mdat[mmap(p)]),
            eager=True,
        )
        return mat

    @pytest.fixture
    def dat(self, mset, mmap, mdat):
        dat = op3.Dat.zeros_like(mdat)
        kernel_code = """\
for (int i=0; i<3; ++i)
  v[i] += d[i];"""
        addone = op3.Function.from_c_string(
            "addone_rhs", kernel_code, [("v", op3.ScalarType, op3.INC), ("d", "double", op3.READ)]
        )
        op3.loop(
            p := mset[0].iter(),
            addone(dat[mmap(p)], mdat[mmap(p)]),
            eager=True,
        )
        return dat

    def test_assemble_mixed_mat(self, mat):
        """Assemble into a matrix declared on a mixed sparsity."""
        assert np.allclose(mat[0, 0].values, np.diag([1.0, 4.0, 9.0]))
        assert np.allclose(mat[0, 1].values, self.od)
        assert np.allclose(mat[1, 0].values, self.od.T)
        assert np.allclose(mat[1, 1].values, self.ll)

    def test_assemble_mixed_rhs(self, dat):
        """Assemble a simple right-hand side over a mixed space and check result."""
        assert np.allclose(dat[0].data_ro, [1, 2, 3])
        assert np.allclose(dat[1].data_ro, [1.0, 4.0, 6.0, 4.0])

    def test_assemble_mixed_rhs_vector(self, mset, mmap, mvdat):
        """Assemble a simple right-hand side over a mixed space and check result."""
        dat = op3.Dat.zeros_like(mvdat)
        kernel_code = """\
for (int i=0; i<3; ++i) {
  v[i*2+0] += d[i*2+0];
  v[i*2+1] += d[i*2+1];
}"""
        addone = op3.Function.from_c_string(
            "addone_rhs_vec", kernel_code, [("v", op3.ScalarType, op3.INC), ("d", "double", op3.READ)]
        )
        op3.loop(
            p := mset[0].iter(),
            addone(dat[mmap(p)], mvdat[mmap(p)]),
            eager=True,
        )
        exp = np.kron(list(zip([1.0, 4.0, 6.0, 4.0])), np.ones(2))
        assert np.allclose(dat[0].data_ro, np.kron(list(zip([1, 2, 3])), np.ones(2)))
        assert np.allclose(dat[1].data_ro, exp)
