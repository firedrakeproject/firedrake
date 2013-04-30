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

"""PyOP2 P1 advection-diffusion demo

This demo solves the advection-diffusion equation and is advanced in time using
a theta scheme with theta = 0.5.

The domain read in from a triangle file.

This demo requires the pyop2 branch of ffc, which can be obtained with:

bzr branch lp:~mapdes/ffc/pyop2

This may also depend on development trunk versions of other FEniCS programs.

FEniCS Viper is also required and is used to visualise the solution.
"""

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from triangle_reader import read_triangle
from ufl import *

import numpy as np


def viper_shape(array):
    """Flatten a numpy array into one dimension to make it suitable for
    passing to Viper."""
    return array.reshape((array.shape[0]))

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('-m', '--mesh', required=True,
                    help='Base name of triangle mesh (excluding the .ele or .node extension)')
parser.add_argument('-v', '--visualize', action='store_true',
                    help='Visualize the result using viper')
opt = vars(parser.parse_args())
op2.init(**opt)

# Set up finite element problem

dt = 0.0001

T = FiniteElement("Lagrange", "triangle", 1)
V = VectorElement("Lagrange", "triangle", 1)

p = TrialFunction(T)
q = TestFunction(T)
t = Coefficient(T)
u = Coefficient(V)

diffusivity = 0.1

M = p * q * dx

d = dt * (diffusivity * dot(grad(q), grad(p)) - dot(grad(q), u) * p) * dx

a = M + 0.5 * d
L = action(M - 0.5 * d, t)

# Generate code for mass and rhs assembly.

lhs, = compile_form(a, "lhs")
rhs, = compile_form(L, "rhs")

# Set up simulation data structures

valuetype = np.float64

nodes, vnodes, coords, elements, elem_node, elem_vnode = read_triangle(opt['mesh'])
num_nodes = nodes.size

sparsity = op2.Sparsity((elem_node, elem_node), 1, "sparsity")
mat = op2.Mat(sparsity, valuetype, "mat")

tracer_vals = np.zeros(num_nodes, dtype=valuetype)
tracer = op2.Dat(nodes, tracer_vals, valuetype, "tracer")

b_vals = np.zeros(num_nodes, dtype=valuetype)
b = op2.Dat(nodes, b_vals, valuetype, "b")

velocity_vals = np.asarray([1.0, 0.0] * num_nodes, dtype=valuetype)
velocity = op2.Dat(vnodes, velocity_vals, valuetype, "velocity")

# Set initial condition

i_cond_code = """
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

T = 0.1

if opt['visualize']:
    vis_coords = np.asarray([[x, y, 0.0] for x, y in coords.data_ro], dtype=np.float64)
    import viper
    v = viper.Viper(x=viper_shape(tracer.data_ro), coordinates=vis_coords, cells=elem_node.values)

solver = op2.Solver()

while T < 0.2:

    mat.zero()
    op2.par_loop(lhs, elements(3, 3),
                 mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                 coords(elem_vnode, op2.READ),
                 velocity(elem_vnode, op2.READ))

    b.zero()
    op2.par_loop(rhs, elements(3),
                 b(elem_node[op2.i[0]], op2.INC),
                 coords(elem_vnode, op2.READ),
                 tracer(elem_node, op2.READ),
                 velocity(elem_vnode, op2.READ))

    solver.solve(mat, tracer, b)

    if opt['visualize']:
        v.update(viper_shape(tracer.data_ro))

    T = T + dt
