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

"""
This demo solves the identity equation for a vector variable on a quadrilateral
domain. The initial condition is that all DoFs are [1, 2]^T

This demo requires the fluidity-pyop2 branch of ffc, which can be obtained with:

bzr branch lp:~grm08/ffc/fluidity-pyop2

This may also depend on development trunk versions of other FEniCS programs.
"""

from pyop2 import op2, utils
from ufl import *
import ffc
import numpy as np

op2.init(**utils.parse_args(description="PyOP2 2D mass equation demo (vector field version)"))

# Set up finite element identity problem

E = VectorElement("Lagrange", "triangle", 1)

v = TestFunction(E)
u = TrialFunction(E)
f = Coefficient(E)

a = inner(v,u)*dx
L = inner(v,f)*dx

# Generate code for mass and rhs assembly.

params = ffc.default_parameters()
params['representation'] = 'quadrature'
params['write_file'] = False
mass_code = ffc.compile_form(a, prefix="mass", parameters=params)
#rhs_code  = ffc.compile_form(L, prefix="rhs",  parameters=params)
rhs_code = """
void rhs_cell_integral_0_0(double **A, double *x[2], double **w0)
{
    // Compute Jacobian of affine map from reference cell
    const double J_00 = x[1][0] - x[0][0];
    const double J_01 = x[2][0] - x[0][0];
    const double J_10 = x[1][1] - x[0][1];
    const double J_11 = x[2][1] - x[0][1];

    // Compute determinant of Jacobian
    double detJ = J_00*J_11 - J_01*J_10;

    // Compute inverse of Jacobian

    // Set scale factor
    const double det = fabs(detJ);

    // Cell Volume.

    // Compute circumradius, assuming triangle is embedded in 2D.


    // Facet Area.

    // Array of quadrature weights.
    static const double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    // Quadrature points on the UFC reference element: (0.166666666666667, 0.166666666666667), (0.166666666666667, 0.666666666666667), (0.666666666666667, 0.166666666666667)

    // Value of basis functions at quadrature points.
    static const double FE0_C0[3][6] = \
    {{0.666666666666667, 0.166666666666667, 0.166666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.166666666666667, 0.666666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.666666666666667, 0.166666666666667, 0.0, 0.0, 0.0}};

    static const double FE0_C1[3][6] = \
    {{0.0, 0.0, 0.0, 0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.666666666666667, 0.166666666666667}};


    // Compute element tensor using UFL quadrature representation
    // Optimisations: ('eliminate zeros', False), ('ignore ones', False), ('ignore zero tables', False), ('optimisation', False), ('remove zero terms', False)

    // Loop quadrature points for integral.
    // Number of operations to compute element tensor for following IP loop = 180
    for (unsigned int ip = 0; ip < 3; ip++)
    {

      // Coefficient declarations.
      double F0 = 0.0;
      double F1 = 0.0;

      // Total number of operations to compute function values = 24
      for (unsigned int r = 0; r < 3; r++)
      {
        for (unsigned int s = 0; s < 2; ++s)
        {
          F0 += FE0_C0[ip][r*2 + s]*w0[r][s];
          F1 += FE0_C1[ip][r*2 + s]*w0[r][s];
        }
      }// end loop over 'r'

      // Number of operations for primary indices: 36
      for (unsigned int j = 0; j < 3; j++)
      {
        for (unsigned int s = 0; s < 2; ++s)
        {
          // Number of operations to compute entry: 6
          A[j][s] += (FE0_C0[ip][j*2+s]*F0 + FE0_C1[ip][j*2+s]*F1)*W3[ip]*det;
        }
      }// end loop over 'j'
    }// end loop over 'ip'
}
"""

mass = op2.Kernel(mass_code, "mass_cell_integral_0_0")
rhs  = op2.Kernel(rhs_code,  "rhs_cell_integral_0_0" )

# Set up simulation data structures

NUM_ELE   = 2
NUM_NODES = 4
valuetype = np.float64

nodes = op2.Set(NUM_NODES, "nodes")
elements = op2.Set(NUM_ELE, "elements")

elem_node_map = np.asarray([ 0, 1, 3, 2, 3, 1 ], dtype=np.uint32)
elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

sparsity = op2.Sparsity((elem_node, elem_node), 2, "sparsity")
mat = op2.Mat(sparsity, valuetype, "mat")

coord_vals = np.asarray([ (0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5) ],
                           dtype=valuetype)
coords = op2.Dat(nodes, 2, coord_vals, valuetype, "coords")

f_vals = np.asarray([(1.0, 2.0)]*4, dtype=valuetype)
b_vals = np.asarray([0.0]*2*NUM_NODES, dtype=valuetype)
x_vals = np.asarray([0.0]*2*NUM_NODES, dtype=valuetype)
f = op2.Dat(nodes, 2, f_vals, valuetype, "f")
b = op2.Dat(nodes, 2, b_vals, valuetype, "b")
x = op2.Dat(nodes, 2, x_vals, valuetype, "x")

# Assemble and solve

op2.par_loop(mass, elements(6,6),
             mat((elem_node[op2.i(0)], elem_node[op2.i(1)]), op2.INC),
             coords(elem_node, op2.READ))

op2.par_loop(rhs, elements,
                     b(elem_node, op2.INC),
                     coords(elem_node, op2.READ),
                     f(elem_node, op2.READ))

op2.solve(mat, b, x)

# Print solution

print "Expected solution: %s" % f_vals
print "Computed solution: %s" % x_vals
