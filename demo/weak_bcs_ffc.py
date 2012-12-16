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

"""PyOP2 laplace equation demo (weak BCs)

This demo uses ffc-generated kernels to solve the Laplace equation on a unit
square with boundary conditions:

  u     = 1 on y = 0
  du/dn = 2 on y = 1

The domain is meshed as follows:

  *-*-*
  |/|/|
  *-*-*
  |/|/|
  *-*-*

This demo requires the pyop2 branch of ffc, which can be obtained with:

bzr branch lp:~mapdes/ffc/pyop2

This may also depend on development trunk versions of other FEniCS programs.
"""

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from ufl import *

import numpy as np

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('-s', '--save-output',
                    action='store_true',
                    help='Save the output of the run (used for testing)')
opt = vars(parser.parse_args())
op2.init(**opt)

# Set up finite element problem

E = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(E)
u = TrialFunction(E)
f = Coefficient(E)
g = Coefficient(E)

a = dot(grad(v,),grad(u))*dx
L = v*f*dx + v*g*ds(2)

# Generate code for Laplacian and rhs assembly.

laplacian, = compile_form(a, "laplacian")
rhs, weak  = compile_form(L, "rhs")

# Set up simulation data structures

NUM_ELE      = 8
NUM_NODES    = 9
NUM_BDRY_ELE = 2
NUM_BDRY_NODE = 3
valuetype = np.float64

nodes = op2.Set(NUM_NODES, "nodes")
elements = op2.Set(NUM_ELE, "elements")
# Elements that Weak BC will be assembled over
top_bdry_elements = op2.Set(NUM_BDRY_ELE, "top_boundary_elements")
# Nodes that Strong BC will be applied over
bdry_nodes = op2.Set(NUM_BDRY_NODE, "boundary_nodes")

elem_node_map = np.asarray([ 0, 1, 4, 4, 3, 0, 1, 2, 5, 5, 4, 1, 3, 4, 7, 7, 6,
                             3, 4, 5, 8, 8, 7, 4], dtype=np.uint32)
elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

top_bdry_elem_node_map = np.asarray([ 7, 6, 3, 8, 7, 4 ], dtype=valuetype)
top_bdry_elem_node = op2.Map(top_bdry_elements, nodes, 3,
                             top_bdry_elem_node_map, "top_bdry_elem_node")

bdry_node_node_map = np.asarray([0, 1, 2 ], dtype=valuetype)
bdry_node_node = op2.Map(bdry_nodes, nodes, 1, bdry_node_node_map, "bdry_node_node")

sparsity = op2.Sparsity((elem_node, elem_node), 1, "sparsity")
mat = op2.Mat(sparsity, valuetype, "mat")

coord_vals = np.asarray([ (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
                          (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
                          (0.0, 1.0), (0.5, 1.0), (1.0, 1.0) ],
                          dtype=valuetype)
coords = op2.Dat(nodes, 2, coord_vals, valuetype, "coords")

f_vals = np.asarray([ 0.0 ]*9, dtype=valuetype)
b_vals = np.asarray([0.0]*NUM_NODES, dtype=valuetype)
x_vals = np.asarray([0.0]*NUM_NODES, dtype=valuetype)
u_vals = np.asarray([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
f = op2.Dat(nodes, 1, f_vals, valuetype, "f")
b = op2.Dat(nodes, 1, b_vals, valuetype, "b")
x = op2.Dat(nodes, 1, x_vals, valuetype, "x")
u = op2.Dat(nodes, 1, u_vals, valuetype, "u")

bdry_vals = np.asarray([1.0, 1.0, 1.0 ], dtype=valuetype)
bdry = op2.Dat(bdry_nodes, 1, bdry_vals, valuetype, "bdry")

# This isn't perfect, defining the boundary gradient on more nodes than are on
# the boundary is couter-intuitive
bdry_grad_vals = np.asarray([2.0]*9, dtype=valuetype)
bdry_grad = op2.Dat(nodes, 1, bdry_grad_vals, valuetype, "gradient")
facet = op2.Global(1, 2, np.uint32, "facet")

# If a form contains multiple integrals with differing coefficients, FFC
# generates kernels that take all the coefficients of the entire form (not
# only the respective integral) as arguments. Arguments that correspond to
# forms that are not used in that integral are simply not referenced.
# We therefore need a dummy argument in place of the coefficient that is not
# used in the par_loop for OP2 to generate the correct kernel call.

# Assemble matrix and rhs

op2.par_loop(laplacian, elements(3,3),
             mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
             coords(elem_node, op2.READ))

op2.par_loop(rhs, elements(3),
             b(elem_node[op2.i[0]], op2.INC),
             coords(elem_node, op2.READ),
             f(elem_node, op2.READ),
             bdry_grad(elem_node, op2.READ)) # argument ignored

# Apply weak BC

op2.par_loop(weak, top_bdry_elements(3),
             b(top_bdry_elem_node[op2.i[0]], op2.INC),
             coords(top_bdry_elem_node, op2.READ),
             f(top_bdry_elem_node, op2.READ), # argument ignored
             bdry_grad(top_bdry_elem_node, op2.READ),
             facet(op2.READ))

# Apply strong BC

mat.zero_rows([ 0, 1, 2 ], 1.0)
strongbc_rhs = op2.Kernel("""
void strongbc_rhs(double *val, double *target) { *target = *val; }
""", "strongbc_rhs")
op2.par_loop(strongbc_rhs, bdry_nodes,
             bdry(op2.IdentityMap, op2.READ),
             b(bdry_node_node[0], op2.WRITE))

solver = op2.Solver(linear_solver='gmres')
solver.solve(mat, x, b)

# Print solution
print "Expected solution: %s" % u.data
print "Computed solution: %s" % x.data

# Save output (if necessary)
if opt['save_output']:
    import pickle
    with open("weak_bcs.out","w") as out:
        pickle.dump((u.data, x.data), out)
