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
This demo solves the identity equation on a domain read in from a triangle
file. It requires the fluidity-pyop2 branch of ffc, which can be obtained
with:

bzr branch lp:~grm08/ffc/fluidity-pyop2

This may also depend on development trunk versions of other FEniCS programs.

FEniCS Viper is also required and is used to visualise the solution.
"""

from pyop2 import op2
from triangle_reader import read_triangle
from ufl import *
import ffc, viper
import sys

import numpy as np

if len(sys.argv) is not 2:
    print "Usage: adv_diff <mesh_name>"
    sys.exit(1)
mesh_name = sys.argv[1]

op2.init(backend='sequential')

# Set up finite element problem

dt = 0.0001

T = FiniteElement("Lagrange", "triangle", 1)
V = VectorElement("Lagrange", "triangle", 1)

p=TrialFunction(T)
q=TestFunction(T)
t=Coefficient(T)
u=Coefficient(V)

diffusivity = 0.1

M=p*q*dx

adv_rhs = (q*t+dt*dot(grad(q),u)*t)*dx

d=-dt*diffusivity*dot(grad(q),grad(p))*dx

diff_matrix=M-0.5*d
diff_rhs=action(M+0.5*d,t)

# Generate code for mass and rhs assembly.

params = ffc.default_parameters()
params['representation'] = 'quadrature'
params['write_file'] = False

mass_code        = ffc.compile_form(M,           prefix="mass",        parameters=params)
#adv_rhs_code     = ffc.compile_form(adv_rhs,     prefix="adv_rhs",     parameters=params)
diff_matrix_code = ffc.compile_form(diff_matrix, prefix="diff_matrix", parameters=params)
diff_rhs_code    = ffc.compile_form(diff_rhs,    prefix="diff_rhs",    parameters=params)

adv_rhs_code="""
void adv_rhs_cell_integral_0_0(double **A, double *x[2], double **w0, double **w1)
{
    // Compute Jacobian of affine map from reference cell
    const double J_00 = x[1][0] - x[0][0];
    const double J_01 = x[2][0] - x[0][0];
    const double J_10 = x[1][1] - x[0][1];
    const double J_11 = x[2][1] - x[0][1];

    // Compute determinant of Jacobian
    double detJ = J_00*J_11 - J_01*J_10;

    // Compute inverse of Jacobian
    const double K_00 =  J_11 / detJ;
    const double K_01 = -J_01 / detJ;
    const double K_10 = -J_10 / detJ;
    const double K_11 =  J_00 / detJ;

    // Set scale factor
    const double det = fabs(detJ);

    // Cell Volume.

    // Compute circumradius, assuming triangle is embedded in 2D.


    // Facet Area.

    // Array of quadrature weights.
    static const double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    // Quadrature points on the UFC reference element: (0.166666666666667, 0.166666666666667), (0.166666666666667, 0.666666666666667), (0.666666666666667, 0.166666666666667)

    // Value of basis functions at quadrature points.
    static const double FE0[3][3] = \
    {{0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.166666666666667, 0.666666666666667, 0.166666666666667}};

    static const double FE0_D01[3][3] = \
    {{-1.0, 0.0, 1.0},
    {-1.0, 0.0, 1.0},
    {-1.0, 0.0, 1.0}};

    static const double FE0_D10[3][3] = \
    {{-1.0, 1.0, 0.0},
    {-1.0, 1.0, 0.0},
    {-1.0, 1.0, 0.0}};

    static const double FE1_C0[3][6] = \
    {{0.666666666666667, 0.166666666666667, 0.166666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.166666666666667, 0.666666666666667, 0.0, 0.0, 0.0},
    {0.166666666666667, 0.666666666666667, 0.166666666666667, 0.0, 0.0, 0.0}};

    static const double FE1_C1[3][6] = \
    {{0.0, 0.0, 0.0, 0.666666666666667, 0.166666666666667, 0.166666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.166666666666667, 0.666666666666667},
    {0.0, 0.0, 0.0, 0.166666666666667, 0.666666666666667, 0.166666666666667}};


    // Compute element tensor using UFL quadrature representation
    // Optimisations: ('eliminate zeros', False), ('ignore ones', False), ('ignore zero tables', False), ('optimisation', False), ('remove zero terms', False)

    // Loop quadrature points for integral.
    // Number of operations to compute element tensor for following IP loop = 234
    for (unsigned int ip = 0; ip < 3; ip++)
    {

      // Coefficient declarations.
      double F0 = 0.0;
      double F1 = 0.0;
      double F2 = 0.0;

      // Total number of operations to compute function values = 6
      for (unsigned int r = 0; r < 3; r++)
      {
        F0 += FE0[ip][r]*w0[r][0];
      }// end loop over 'r'

      // Total number of operations to compute function values = 24
      for (unsigned int r = 0; r < 3; r++)
      {
        for (unsigned int s = 0; s < 2; s++)
        {
          F1 += FE1_C0[ip][r*2+s]*w1[r][s];
          F2 += FE1_C1[ip][r*2+s]*w1[r][s];
        }
      }// end loop over 'r'

      // Number of operations for primary indices: 48
      for (unsigned int j = 0; j < 3; j++)
      {
        // Number of operations to compute entry: 16
        A[j][0] += (FE0[ip][j]*F0 + (((((K_01*FE0_D10[ip][j] + K_11*FE0_D01[ip][j]))*F2 + ((K_00*FE0_D10[ip][j] + K_10*FE0_D01[ip][j]))*F1))*0.0001)*F0)*W3[ip]*det;
      }// end loop over 'j'
    }// end loop over 'ip'
}
"""

mass        = op2.Kernel(mass_code,        "mass_cell_integral_0_0")
adv_rhs     = op2.Kernel(adv_rhs_code,     "adv_rhs_cell_integral_0_0" )
diff_matrix = op2.Kernel(diff_matrix_code, "diff_matrix_cell_integral_0_0")
diff_rhs    = op2.Kernel(diff_rhs_code,    "diff_rhs_cell_integral_0_0")

# Set up simulation data structures

valuetype=np.float64

nodes, coords, elements, elem_node = read_triangle(mesh_name)
num_nodes = nodes.size

sparsity = op2.Sparsity(elem_node, elem_node, 1, "sparsity")
mat = op2.Mat(sparsity, 1, valuetype, "mat")

tracer_vals = np.asarray([0.0]*num_nodes, dtype=valuetype)
tracer = op2.Dat(nodes, 1, tracer_vals, valuetype, "tracer")

b_vals = np.asarray([0.0]*num_nodes, dtype=valuetype)
b = op2.Dat(nodes, 1, b_vals, valuetype, "b")

velocity_vals = np.asarray([1.0, 0.0]*num_nodes, dtype=valuetype)
velocity = op2.Dat(nodes, 2, velocity_vals, valuetype, "velocity")

# Set initial condition

i_cond_code="""
void i_cond(double *c, double *t)
{
  double i_t = 0.1; // Initial time
  double A   = 0.1; // Normalisation
  double D   = 0.1; // Diffusivity
  double pi  = 3.141459265358979;
  double x   = c[0]-0.5;
  double y   = c[1]-0.5;
  double r   = sqrt(x*x+y*y);

  if (r<0.25)
    *t = A*(exp((-(r*r))/(4*D*i_t))/(4*pi*D*i_t));
}
"""

i_cond = op2.Kernel(i_cond_code, "i_cond")

op2.par_loop(i_cond, nodes,
             coords(op2.IdentityMap, op2.READ),
             tracer(op2.IdentityMap, op2.WRITE))

zero_dat_code="""
void zero_dat(double *dat)
{
  *dat = 0.0;
}
"""

zero_dat = op2.Kernel(zero_dat_code, "zero_dat")

# Assemble and solve

T = 0.1

vis_coords = np.asarray([ [x, y, 0.0] for x, y in coords.data ],dtype=np.float64)
v = viper.Viper(x=tracer_vals, coordinates=vis_coords, cells=elem_node.values)
v.interactive()

have_advection = True
have_diffusion = True

while T < 0.2:

    # Advection

    if have_advection:
        mat.zero()

        op2.par_loop(mass, elements(3,3),
                     mat((elem_node(op2.i(0)), elem_node(op2.i(1))), op2.INC),
                     coords(elem_node, op2.READ))

        op2.par_loop(zero_dat, nodes,
                     b(op2.IdentityMap, op2.WRITE))

        op2.par_loop(adv_rhs, elements,
                     b(elem_node, op2.INC),
                     coords(elem_node, op2.READ),
                     tracer(elem_node, op2.READ),
                     velocity(elem_node, op2.READ))

        op2.solve(mat, b, tracer)

    # Diffusion

    if have_diffusion:
        mat.zero()

        op2.par_loop(diff_matrix, elements(3,3),
                     mat((elem_node(op2.i(0)), elem_node(op2.i(1))), op2.INC),
                     coords(elem_node, op2.READ))

        op2.par_loop(zero_dat, nodes,
                     b(op2.IdentityMap, op2.WRITE))

        op2.par_loop(diff_rhs, elements,
                     b(elem_node, op2.INC),
                     coords(elem_node, op2.READ),
                     tracer(elem_node, op2.READ))

        op2.solve(mat, b, tracer)

    v.update(tracer_vals)

    T = T + dt

# Interactive visulatisation
v.interactive()
