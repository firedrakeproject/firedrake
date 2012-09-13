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
"""

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from triangle_reader import read_triangle
from ufl import *
import sys

import numpy as np

parser = utils.parser(group=True, description="PyOP2 2D mass equation demo")
parser.add_argument('-m', '--mesh',
                    action='store',
                    type=str,
                    help='Base name of triangle mesh (excluding the .ele or .node extension)')
opt = vars(parser.parse_args())
op2.init(**opt)
mesh_name = opt['mesh']

# Set up finite element identity problem

E = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(E)
u = TrialFunction(E)
f = Coefficient(E)

a = v*u*dx
L = v*f*dx

# Generate code for mass and rhs assembly.

mass_code = compile_form(a, "mass")
rhs_code  = compile_form(L, "rhs")

mass = op2.Kernel(mass_code, "mass_cell_integral_0_0")
rhs  = op2.Kernel(rhs_code,  "rhs_cell_integral_0_0" )

# Set up simulation data structures

valuetype=np.float64

nodes, coords, elements, elem_node = read_triangle(mesh_name)
num_nodes = nodes.size

sparsity = op2.Sparsity((elem_node, elem_node), 1, "sparsity")
mat = op2.Mat(sparsity, valuetype, "mat")

f_vals = np.asarray([ float(i) for i in xrange(num_nodes) ], dtype=valuetype)
b_vals = np.asarray([0.0]*num_nodes, dtype=valuetype)
x_vals = np.asarray([0.0]*num_nodes, dtype=valuetype)
f = op2.Dat(nodes, 1, f_vals, valuetype, "f")
b = op2.Dat(nodes, 1, b_vals, valuetype, "b")
x = op2.Dat(nodes, 1, x_vals, valuetype, "x")

# Assemble and solve

op2.par_loop(mass, elements(3,3),
             mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
             coords(elem_node, op2.READ))

op2.par_loop(rhs, elements(3),
                     b(elem_node[op2.i[0]], op2.INC),
                     coords(elem_node, op2.READ),
                     f(elem_node, op2.READ))

op2.solve(mat, b, x)

# Print solution

print "Expected solution: %s" % f_vals
print "Computed solution: %s" % x_vals
