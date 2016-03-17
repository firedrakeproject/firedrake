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
from pyop2.exceptions import MapValueError, ModeValueError

from coffee.base import *

backends = ['sequential', 'openmp', 'cuda']

# Data type
valuetype = np.float64

# Constants
NUM_ELE = 2
NUM_NODES = 4
NUM_DIMS = 2
layers = 11

elem_node_map = np.asarray([0, 1, 3, 2, 3, 1], dtype=np.uint32)

xtr_elem_node_map = np.asarray([0, 1, 11, 12, 33, 34, 22, 23, 33, 34, 11, 12], dtype=np.uint32)


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
    init = FlatBlock("""
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
""")
    assembly = Incr(Symbol("localTensor", ("i_r_0", "i_r_1")),
                    FlatBlock("ST0 * w[i_g]"))
    assembly = Block([FlatBlock("double ST0 = 0.0;\nST0 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * \
                                c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);\n"), assembly], open_scope=True)
    assembly = c_for("i_r_0", 3, c_for("i_r_1", 3, assembly))

    kernel_code = FunDecl("void", "mass",
                          [Decl("double", Symbol("localTensor", (3, 3))),
                           Decl("double*", c_sym("c0[2]"))],
                          Block([init, assembly], open_scope=False))

    return op2.Kernel(kernel_code, "mass")


@pytest.fixture
def rhs():
    kernel_code = FlatBlock("""
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
}""")
    return op2.Kernel(kernel_code, "rhs")


@pytest.fixture
def mass_ffc():
    init = FlatBlock("""
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
""")
    assembly = Incr(Symbol("A", ("j", "k")),
                    FlatBlock("FE0[ip][j]*FE0[ip][k]*W3[ip]*det"))
    assembly = c_for("j", 3, c_for("k", 3, assembly))

    kernel_code = FunDecl("void", "mass_ffc",
                          [Decl("double", Symbol("A", (3, 3))),
                           Decl("double*", c_sym("x[2]"))],
                          Block([init, assembly], open_scope=False))

    return op2.Kernel(kernel_code, "mass_ffc")


@pytest.fixture
def rhs_ffc():
    kernel_code = FlatBlock("""
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
""")
    return op2.Kernel(kernel_code, "rhs_ffc")


@pytest.fixture
def rhs_ffc_itspace():
    init = FlatBlock("""
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

""")
    assembly = Incr(Symbol("A", ("j",)), FlatBlock("FE0[ip][j]*F0*W3[ip]*det"))
    assembly = c_for("j", 3, assembly)
    end = FlatBlock("}")

    kernel_code = FunDecl("void", "rhs_ffc_itspace",
                          [Decl("double", Symbol("A", (3,))),
                           Decl("double*", c_sym("x[2]")),
                              Decl("double**", c_sym("w0"))],
                          Block([init, assembly, end], open_scope=False))

    return op2.Kernel(kernel_code, "rhs_ffc_itspace")


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
    code = c_for("i", 3,
                 c_for("j", 3,
                       Incr(Symbol("entry", ("i", "j")), c_sym("*g"))))

    kernel_code = FunDecl("void", "kernel_inc",
                          [Decl("double", Symbol("entry", (3, 3))),
                           Decl("double*", c_sym("g"))],
                          Block([code], open_scope=False))

    return op2.Kernel(kernel_code, "kernel_inc")


@pytest.fixture
def kernel_set():
    code = c_for("i", 3,
                 c_for("j", 3,
                       Assign(Symbol("entry", ("i", "j")), c_sym("*g"))))

    kernel_code = FunDecl("void", "kernel_set",
                          [Decl("double", Symbol("entry", (3, 3))),
                           Decl("double*", c_sym("g"))],
                          Block([code], open_scope=False))

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
    return op2.MixedSet((op2.Set(3), op2.Set(4)))


rdata = lambda s: np.arange(1, s + 1, dtype=np.float64)


@pytest.fixture
def mdat(mset):
    return op2.MixedDat(op2.Dat(s, rdata(s.size)) for s in mset)


@pytest.fixture
def mvdat(mset):
    return op2.MixedDat(op2.Dat(s ** 2, zip(rdata(s.size), rdata(s.size))) for s in mset)


@pytest.fixture
def mmap(mset):
    elem, node = mset
    return op2.MixedMap((op2.Map(elem, elem, 1, [0, 1, 2]),
                         op2.Map(elem, node, 2, [0, 1, 1, 2, 2, 3])))


@pytest.fixture
def msparsity(mset, mmap):
    return op2.Sparsity(mset, mmap)


@pytest.fixture
def non_nest_mixed_sparsity(mset, mmap):
    return op2.Sparsity(mset, mmap, nest=False)


@pytest.fixture
def mvsparsity(mset, mmap):
    return op2.Sparsity(mset ** 2, mmap)


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

    def test_build_mixed_sparsity(self, backend, msparsity):
        """Building a sparsity from a pair of mixed maps should give the
        expected rowptr and colidx for each block."""
        assert all(msparsity._rowptr[0] == [0, 1, 2, 3])
        assert all(msparsity._rowptr[1] == [0, 2, 4, 6])
        assert all(msparsity._rowptr[2] == [0, 1, 3, 5, 6])
        assert all(msparsity._rowptr[3] == [0, 2, 5, 8, 10])
        assert all(msparsity._colidx[0] == [0, 1, 2])
        assert all(msparsity._colidx[1] == [0, 1, 1, 2, 2, 3])
        assert all(msparsity._colidx[2] == [0, 0, 1, 1, 2, 2])
        assert all(msparsity._colidx[3] == [0, 1, 0, 1, 2, 1, 2, 3, 2, 3])

    def test_build_mixed_sparsity_vector(self, backend, mvsparsity):
        """Building a sparsity from a pair of mixed maps and a vector DataSet
        should give the expected rowptr and colidx for each block."""
        assert all(mvsparsity._rowptr[0] == [0, 1, 2, 3])
        assert all(mvsparsity._rowptr[1] == [0, 2, 4, 6])
        assert all(mvsparsity._rowptr[2] == [0, 1, 3, 5, 6])
        assert all(mvsparsity._rowptr[3] == [0, 2, 5, 8, 10])
        assert all(mvsparsity._colidx[0] == [0, 1, 2])
        assert all(mvsparsity._colidx[1] == [0, 1, 1, 2, 2, 3])
        assert all(mvsparsity._colidx[2] == [0, 0, 1, 1, 2, 2])
        assert all(mvsparsity._colidx[3] == [0, 1, 0, 1, 2, 1, 2, 3, 2, 3])

    def test_sparsity_null_maps(self, backend):
        """Building sparsity from a pair of non-initialized maps should fail."""
        s = op2.Set(5)
        with pytest.raises(MapValueError):
            m = op2.Map(s, s, 1)
            op2.Sparsity((s, s), (m, m))

    def test_sparsity_always_has_diagonal_space(self, backend):
        # A sparsity should always have space for diagonal entries
        s = op2.Set(1)
        d = op2.Set(4)
        m = op2.Map(s, d, 1, [2])
        d2 = op2.Set(5)
        m2 = op2.Map(s, d2, 2, [1, 4])
        sparsity = op2.Sparsity((d, d2), (m, m2))

        assert all(sparsity.nnz == [1, 1, 3, 1])


class TestMatrices:

    """
    Matrix tests
    """

    @pytest.mark.parametrize("mode", [op2.READ, op2.RW, op2.MAX, op2.MIN])
    def test_invalid_mode(self, backend, elements, elem_node, mat, mode):
        """Mat args can only have modes WRITE and INC."""
        with pytest.raises(ModeValueError):
            op2.par_loop(op2.Kernel("", "dummy"), elements,
                         mat(mode, (elem_node[op2.i[0]], elem_node[op2.i[1]])))

    @pytest.mark.parametrize('n', [1, 2])
    def test_mat_set_diagonal(self, backend, nodes, elem_node, n, skip_cuda):
        "Set the diagonal of the entire matrix to 1.0"
        mat = op2.Mat(op2.Sparsity(nodes**n, elem_node), valuetype)
        nrows = mat.sparsity.nrows
        mat.set_local_diagonal_entries(range(nrows))
        mat.assemble()
        assert (mat.values == np.identity(nrows * n)).all()

    @pytest.mark.parametrize('n', [1, 2])
    def test_mat_repeated_set_diagonal(self, backend, nodes, elem_node, n, skip_cuda):
        "Set the diagonal of the entire matrix to 1.0"
        mat = op2.Mat(op2.Sparsity(nodes**n, elem_node), valuetype)
        nrows = mat.sparsity.nrows
        mat.set_local_diagonal_entries(range(nrows))
        mat.assemble()
        assert (mat.values == np.identity(nrows * n)).all()
        mat.set_local_diagonal_entries(range(nrows))
        mat.assemble()
        assert (mat.values == np.identity(nrows * n)).all()

    def test_mat_always_has_diagonal_space(self, backend):
        # A sparsity should always have space for diagonal entries
        s = op2.Set(1)
        d = op2.Set(4)
        m = op2.Map(s, d, 1, [2])
        d2 = op2.Set(3)
        m2 = op2.Map(s, d2, 1, [1])
        sparsity = op2.Sparsity((d, d2), (m, m2))

        from petsc4py import PETSc
        # petsc4py default error handler swallows SETERRQ, so just
        # install the abort handler to notice an error.
        PETSc.Sys.pushErrorHandler("abort")
        mat = op2.Mat(sparsity)
        PETSc.Sys.popErrorHandler()

        assert np.allclose(mat.handle.getDiagonal().array, 0.0)

    def test_minimal_zero_mat(self, backend, skip_cuda):
        """Assemble a matrix that is all zeros."""

        code = c_for("i", 1,
                     c_for("j", 1,
                           Assign(Symbol("local_mat", ("i", "j")), c_sym("0.0"))))
        zero_mat_code = FunDecl("void", "zero_mat",
                                [Decl("double", Symbol("local_mat", (1, 1)))],
                                Block([code], open_scope=False))

        nelems = 128
        set = op2.Set(nelems)
        map = op2.Map(set, set, 1, np.array(range(nelems), np.uint32))
        sparsity = op2.Sparsity((set, set), (map, map))
        mat = op2.Mat(sparsity, np.float64)
        kernel = op2.Kernel(zero_mat_code, "zero_mat")
        op2.par_loop(kernel, set,
                     mat(op2.WRITE, (map[op2.i[0]], map[op2.i[1]])))

        mat.assemble()
        expected_matrix = np.zeros((nelems, nelems), dtype=np.float64)
        eps = 1.e-12
        assert_allclose(mat.values, expected_matrix, eps)

    def test_assemble_mat(self, backend, mass, mat, coords, elements,
                          elem_node, expected_matrix):
        """Assemble a simple finite-element matrix and check the result."""
        mat.zero()
        op2.par_loop(mass, elements,
                     mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                     coords(op2.READ, elem_node))
        mat.assemble()
        eps = 1.e-5
        assert_allclose(mat.values, expected_matrix, eps)

    def test_assemble_rhs(self, backend, rhs, elements, b, coords, f,
                          elem_node, expected_rhs):
        """Assemble a simple finite-element right-hand side and check result."""
        b.zero()
        op2.par_loop(rhs, elements,
                     b(op2.INC, elem_node),
                     coords(op2.READ, elem_node),
                     f(op2.READ, elem_node))

        eps = 1.e-12
        assert_allclose(b.data, expected_rhs, eps)

    def test_solve(self, backend, mat, b, x, f):
        """Solve a linear system where the solution is equal to the right-hand
        side and check the result."""
        mat.assemble()
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
        mat.zero()
        op2.par_loop(kernel_inc, elements,
                     mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                     g(op2.READ))
        mat.assemble()
        # Check we have ones in the matrix
        assert mat.values.sum() == 3 * 3 * elements.size
        op2.par_loop(kernel_set, elements,
                     mat(op2.WRITE, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                     g(op2.READ))
        mat.assemble()
        assert mat.values.sum() == (3 * 3 - 2) * elements.size
        mat.zero()

    def test_zero_rhs(self, backend, b, zero_dat, nodes):
        """Test that the RHS is zeroed correctly."""
        op2.par_loop(zero_dat, nodes,
                     b(op2.WRITE))
        assert all(b.data == np.zeros_like(b.data))

    def test_assemble_ffc(self, backend, mass_ffc, mat, coords, elements,
                          elem_node, expected_matrix):
        """Test that the FFC mass assembly assembles the correct values."""
        op2.par_loop(mass_ffc, elements,
                     mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                     coords(op2.READ, elem_node))
        mat.assemble()
        eps = 1.e-5
        assert_allclose(mat.values, expected_matrix, eps)

    def test_rhs_ffc(self, backend, rhs_ffc, elements, b, coords, f,
                     elem_node, expected_rhs):
        """Test that the FFC rhs assembly assembles the correct values."""
        op2.par_loop(rhs_ffc, elements,
                     b(op2.INC, elem_node),
                     coords(op2.READ, elem_node),
                     f(op2.READ, elem_node))

        eps = 1.e-6
        assert_allclose(b.data, expected_rhs, eps)

    def test_rhs_ffc_itspace(self, backend, rhs_ffc_itspace, elements, b,
                             coords, f, elem_node, expected_rhs,
                             zero_dat, nodes):
        """Test that the FFC right-hand side assembly using iteration spaces
        assembles the correct values."""
        # Zero the RHS first
        op2.par_loop(zero_dat, nodes,
                     b(op2.WRITE))
        op2.par_loop(rhs_ffc_itspace, elements,
                     b(op2.INC, elem_node[op2.i[0]]),
                     coords(op2.READ, elem_node),
                     f(op2.READ, elem_node))
        eps = 1.e-6
        assert_allclose(b.data, expected_rhs, eps)

    def test_zero_rows(self, backend, mat, expected_matrix):
        """Zeroing a row in the matrix should set the diagonal to the given
        value and all other values to 0."""
        expected_matrix[0] = [12.0, 0.0, 0.0, 0.0]
        mat.zero_rows([0], 12.0)
        eps = 1.e-5
        assert_allclose(mat.values, expected_matrix, eps)

    def test_zero_rows_subset(self, backend, nodes, mat, expected_matrix):
        """Zeroing rows in the matrix given by a :class:`op2.Subset` should
        set the diagonal to the given value and all other values to 0."""
        expected_matrix[0] = [12.0, 0.0, 0.0, 0.0]
        ss = op2.Subset(nodes, [0])
        mat.zero_rows(ss, 12.0)
        assert_allclose(mat.values, expected_matrix, 1e-5)

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

    def test_mat_nbytes(self, backend, mat):
        """Check that the matrix uses the amount of memory we expect."""
        assert mat.nbytes == 14 * 8


class TestMatrixStateChanges:

    """
    Test that matrix state changes are correctly tracked.  Only used
    on CPU backends (since it matches up with PETSc).
    """

    backends = ['sequential', 'openmp']

    @pytest.fixture(params=[False, True],
                    ids=["Non-nested", "Nested"])
    def mat(self, request, msparsity, non_nest_mixed_sparsity):
        if request.param:
            mat = op2.Mat(msparsity)
        else:
            mat = op2.Mat(non_nest_mixed_sparsity)

        opt = mat.handle.Option.NEW_NONZERO_ALLOCATION_ERR
        opt2 = mat.handle.Option.UNUSED_NONZERO_LOCATION_ERR
        mat.handle.setOption(opt, False)
        mat.handle.setOption(opt2, False)
        for m in mat:
            m.handle.setOption(opt, False)
            m.handle.setOption(opt2, False)
        return mat

    def test_mat_starts_assembled(self, backend, mat):
        assert mat.assembly_state is op2.Mat.ASSEMBLED
        for m in mat:
            assert mat.assembly_state is op2.Mat.ASSEMBLED

    def test_after_set_local_state_is_insert(self, backend, mat):
        mat[0, 0].set_local_diagonal_entries([0])
        mat._force_evaluation()
        assert mat[0, 0].assembly_state is op2.Mat.INSERT_VALUES
        if not mat.sparsity.nested:
            assert mat.assembly_state is op2.Mat.INSERT_VALUES
        if mat.sparsity.nested:
            assert mat[1, 1].assembly_state is op2.Mat.ASSEMBLED

    def test_after_addto_state_is_add(self, backend, mat):
        mat[0, 0].addto_values(0, 0, [1])
        mat._force_evaluation()
        assert mat[0, 0].assembly_state is op2.Mat.ADD_VALUES
        if not mat.sparsity.nested:
            assert mat.assembly_state is op2.Mat.ADD_VALUES
        if mat.sparsity.nested:
            assert mat[1, 1].assembly_state is op2.Mat.ASSEMBLED

    def test_matblock_assemble_runtimeerror(self, backend, mat):
        if mat.sparsity.nested:
            return
        with pytest.raises(RuntimeError):
            mat[0, 0].assemble()

        with pytest.raises(RuntimeError):
            mat[0, 0]._assemble()

    def test_mixing_insert_and_add_works(self, backend, mat):
        mat[0, 0].addto_values(0, 0, [1])
        mat[1, 1].addto_values(1, 1, [3])
        mat[1, 1].set_values(0, 0, [2])
        mat[0, 0].set_values(1, 1, [4])
        mat[1, 1].addto_values(0, 0, [3])
        mat.assemble()

        assert np.allclose(mat[0, 0].values, np.diag([1, 4, 0]))
        assert np.allclose(mat[1, 1].values, np.diag([5, 3, 0, 0]))

        assert np.allclose(mat[0, 1].values, 0)
        assert np.allclose(mat[1, 0].values, 0)

    def test_assembly_flushed_between_insert_and_add(self, backend, mat):
        import types
        flush_counter = [0]

        def make_flush(old_flush):
            def flush(self):
                old_flush()
                flush_counter[0] += 1
            return flush

        oflush = mat._flush_assembly
        mat._flush_assembly = types.MethodType(make_flush(oflush), mat, type(mat))
        if mat.sparsity.nested:
            for m in mat:
                oflush = m._flush_assembly
                m._flush_assembly = types.MethodType(make_flush(oflush), m, type(m))

        mat[0, 0].addto_values(0, 0, [1])
        mat._force_evaluation()
        assert flush_counter[0] == 0
        mat[0, 0].set_values(1, 0, [2])
        mat._force_evaluation()
        assert flush_counter[0] == 1
        mat.assemble()
        mat._force_evaluation()
        assert flush_counter[0] == 1


class TestMixedMatrices:
    """
    Matrix tests for mixed spaces
    """

    # Only working for sequential and OpenMP so far
    backends = ['sequential', 'openmp']

    # off-diagonal blocks
    od = np.array([[1.0, 2.0, 0.0, 0.0],
                   [0.0, 4.0, 6.0, 0.0],
                   [0.0, 0.0, 9.0, 12.0]])
    # lower left block
    ll = (np.diag([1.0, 8.0, 18.0, 16.0]) +
          np.diag([2.0, 6.0, 12.0], -1) +
          np.diag([2.0, 6.0, 12.0], 1))

    @pytest.fixture
    def mat(self, msparsity, mmap, mdat):
        mat = op2.Mat(msparsity)

        code = c_for("i", 3,
                     c_for("j", 3,
                           Incr(Symbol("v", ("i", "j")), FlatBlock("d[i][0] * d[j][0]"))))
        addone = FunDecl("void", "addone_mat",
                         [Decl("double", Symbol("v", (3, 3))),
                          Decl("double", c_sym("**d"))],
                         Block([code], open_scope=False))

        addone = op2.Kernel(addone, "addone_mat")
        op2.par_loop(addone, mmap.iterset,
                     mat(op2.INC, (mmap[op2.i[0]], mmap[op2.i[1]])),
                     mdat(op2.READ, mmap))
        mat.assemble()
        mat._force_evaluation()
        return mat

    @pytest.fixture
    def dat(self, mset, mmap, mdat):
        dat = op2.MixedDat(mset)
        kernel_code = FunDecl("void", "addone_rhs",
                              [Decl("double", Symbol("v", (3,))),
                               Decl("double**", c_sym("d"))],
                              c_for("i", 3, Incr(Symbol("v", ("i")), FlatBlock("d[i][0]"))))
        addone = op2.Kernel(kernel_code, "addone_rhs")
        op2.par_loop(addone, mmap.iterset,
                     dat(op2.INC, mmap[op2.i[0]]),
                     mdat(op2.READ, mmap))
        return dat

    @pytest.mark.xfail(reason="Assembling directly into mixed mats unsupported")
    def test_assemble_mixed_mat(self, backend, mat):
        """Assemble into a matrix declared on a mixed sparsity."""
        eps = 1.e-12
        assert_allclose(mat[0, 0].values, np.diag([1.0, 4.0, 9.0]), eps)
        assert_allclose(mat[0, 1].values, self.od, eps)
        assert_allclose(mat[1, 0].values, self.od.T, eps)
        assert_allclose(mat[1, 1].values, self.ll, eps)

    def test_assemble_mixed_rhs(self, backend, dat):
        """Assemble a simple right-hand side over a mixed space and check result."""
        eps = 1.e-12
        assert_allclose(dat[0].data_ro, rdata(3), eps)
        assert_allclose(dat[1].data_ro, [1.0, 4.0, 6.0, 4.0], eps)

    def test_assemble_mixed_rhs_vector(self, backend, mset, mmap, mvdat):
        """Assemble a simple right-hand side over a mixed space and check result."""
        dat = op2.MixedDat(mset ** 2)
        assembly = Block(
            [Incr(Symbol("v", ("i"), ((2, 0),)), FlatBlock("d[i][0]")),
             Incr(Symbol("v", ("i"), ((2, 1),)), FlatBlock("d[i][1]"))], open_scope=True)
        kernel_code = FunDecl("void", "addone_rhs_vec",
                              [Decl("double", Symbol("v", (6,))),
                               Decl("double**", c_sym("d"))],
                              c_for("i", 3, assembly))
        addone = op2.Kernel(kernel_code, "addone_rhs_vec")
        op2.par_loop(addone, mmap.iterset,
                     dat(op2.INC, mmap[op2.i[0]]),
                     mvdat(op2.READ, mmap))
        eps = 1.e-12
        exp = np.kron(zip([1.0, 4.0, 6.0, 4.0]), np.ones(2))
        assert_allclose(dat[0].data_ro, np.kron(zip(rdata(3)), np.ones(2)), eps)
        assert_allclose(dat[1].data_ro, exp, eps)

    @pytest.mark.xfail(reason="Assembling directly into mixed mats unsupported")
    def test_solve_mixed(self, backend, mat, dat):
        x = op2.MixedDat(dat.dataset)
        op2.solve(mat, x, dat)
        b = mat * x
        eps = 1.e-12
        assert_allclose(dat[0].data_ro, b[0].data_ro, eps)
        assert_allclose(dat[1].data_ro, b[1].data_ro, eps)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
