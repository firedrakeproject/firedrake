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

"""PyOP2 2D mass equation demo

This is a demo of the use of ffc to generate kernels. It solves the identity
equation on a quadrilateral domain. It requires the pyop2 branch of ffc,
which can be obtained with:

bzr branch lp:~mapdes/ffc/pyop2

This may also depend on development trunk versions of other FEniCS programs.
"""

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from ufl import *
import ffc
import numpy as np

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('-s', '--save-output',
                    action='store_true',
                    help='Save the output of the run (used for testing)')
opt = vars(parser.parse_args())
op2.init(**opt)

# Set up finite element identity problem

E = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(E)
u = TrialFunction(E)
f = Coefficient(E)

a = v*u*dx
L = v*f*dx

# Generate code for mass and rhs assembly.

mass, = compile_form(a, "mass")
rhs,  = compile_form(L, "rhs")

# Set up simulation data structures

NUM_ELE   = 2
NUM_NODES = 4
valuetype = np.float64

nodes = op2.Set(NUM_NODES, 1, "nodes")
vnodes = op2.Set(NUM_NODES, 2, "vnodes")
elements = op2.Set(NUM_ELE, 1, "elements")

elem_node_map = np.asarray([ 0, 1, 3, 2, 3, 1 ], dtype=np.uint32)
elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")
elem_vnode = op2.Map(elements, vnodes, 3, elem_node_map, "elem_vnode")

sparsity = op2.Sparsity((elem_node, elem_node), 1, "sparsity")
mat = op2.Mat(sparsity, valuetype, "mat")

coord_vals = np.asarray([ (0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5) ],
                           dtype=valuetype)
coords = op2.Dat(vnodes, coord_vals, valuetype, "coords")

f_vals = np.asarray([ 1.0, 2.0, 3.0, 4.0 ], dtype=valuetype)
b_vals = np.asarray([0.0]*NUM_NODES, dtype=valuetype)
x_vals = np.asarray([0.0]*NUM_NODES, dtype=valuetype)
f = op2.Dat(nodes, f_vals, valuetype, "f")
b = op2.Dat(nodes, b_vals, valuetype, "b")
x = op2.Dat(nodes, x_vals, valuetype, "x")

# Assemble and solve

op2.par_loop(mass, elements(3,3),
             mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
             coords(elem_vnode, op2.READ))

op2.par_loop(rhs, elements(3),
             b(elem_node[op2.i[0]], op2.INC),
             coords(elem_vnode, op2.READ),
             f(elem_node, op2.READ))

solver = op2.Solver()
solver.solve(mat, x, b)

# Print solution

print "Expected solution: %s" % f.data
print "Computed solution: %s" % x.data

# Save output (if necessary)
if opt['save_output']:
    import pickle
    with open("mass2d.out","w") as out:
        pickle.dump((f.data, x.data), out)
