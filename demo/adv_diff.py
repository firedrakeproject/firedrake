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

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from triangle_reader import read_triangle
from ufl import *
import viper
import sys

import numpy as np

parser = utils.parser(group=True, description="PyOP2 P1 advection-diffusion demo")
parser.add_argument('-m', '--mesh',
                    action='store',
                    type=str,
                    required=True,
                    help='Base name of triangle mesh (excluding the .ele or .node extension)')
parser.add_argument('-v', '--visualize',
                    action='store_true',
                    help='Visualize the result using viper')
opt = vars(parser.parse_args())
op2.init(**opt)
mesh_name = opt['mesh']

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

mass, _, _        = compile_form(M,           "mass")
adv_rhs, _, _     = compile_form(adv_rhs,     "adv_rhs")
diff_matrix, _, _ = compile_form(diff_matrix, "diff_matrix")
diff_rhs, _, _    = compile_form(diff_rhs,    "diff_rhs")

# Set up simulation data structures

valuetype=np.float64

nodes, coords, elements, elem_node = read_triangle(mesh_name)
num_nodes = nodes.size

sparsity = op2.Sparsity((elem_node, elem_node), 1, "sparsity")
mat = op2.Mat(sparsity, valuetype, "mat")

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
  else
    *t = 0.0;
}
"""

i_cond = op2.Kernel(i_cond_code, "i_cond")

op2.par_loop(i_cond, nodes,
             coords(op2.IdentityMap, op2.READ),
             tracer(op2.IdentityMap, op2.WRITE))

# Assemble and solve

def viper_shape(array):
    """Flatten a numpy array into one dimension to make it suitable for
    passing to Viper."""
    return array.reshape((array.shape[0]))

T = 0.1

vis_coords = np.asarray([ [x, y, 0.0] for x, y in coords.data ],dtype=np.float64)
if opt['visualize']:
    v = viper.Viper(x=viper_shape(tracer.data), coordinates=vis_coords, cells=elem_node.values)
    v.interactive()

have_advection = True
have_diffusion = True
solver = op2.Solver()

while T < 0.2:

    # Advection

    if have_advection:
        mat.zero()
        op2.par_loop(mass, elements(3,3),
                     mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                     coords(elem_node, op2.READ))

        b.zero()
        op2.par_loop(adv_rhs, elements(3),
                     b(elem_node[op2.i[0]], op2.INC),
                     coords(elem_node, op2.READ),
                     tracer(elem_node, op2.READ),
                     velocity(elem_node, op2.READ))

        solver.solve(mat, tracer, b)

    # Diffusion

    if have_diffusion:
        mat.zero()
        op2.par_loop(diff_matrix, elements(3,3),
                     mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                     coords(elem_node, op2.READ))

        b.zero()
        op2.par_loop(diff_rhs, elements(3),
                     b(elem_node[op2.i[0]], op2.INC),
                     coords(elem_node, op2.READ),
                     tracer(elem_node, op2.READ))

        solver.solve(mat, tracer, b)

    if opt['visualize']:
        v.update(viper_shape(tracer.data))

    T = T + dt

# Interactive visulatisation
if opt['visualize']:
    v.interactive()
