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

"""PyOP2 laplace equation demo

This demo uses ffc-generated kernels to solve the Laplace equation on a unit
square with boundary conditions:

  u = 1 on y = 0
  u = 2 on y = 1

The domain is meshed as follows:

  *-*-*
  |/|/|
  *-*-*
  |/|/|
  *-*-*

This demo requires the MAPDES forks of FFC, FIAT and UFL which are found at:

    https://bitbucket.org/mapdes/ffc
    https://bitbucket.org/mapdes/fiat
    https://bitbucket.org/mapdes/ufl
"""

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from ufl import *

import numpy as np


def main(opt):
    # Set up finite element problem

    E = FiniteElement("Lagrange", "triangle", 1)

    v = TestFunction(E)
    u = TrialFunction(E)
    f = Coefficient(E)

    a = dot(grad(v,), grad(u)) * dx
    L = v * f * dx

    # Generate code for Laplacian and rhs assembly.

    laplacian, = compile_form(a, "laplacian")
    rhs, = compile_form(L, "rhs")

    # Set up simulation data structures

    NUM_ELE = 8
    NUM_NODES = 9
    NUM_BDRY_NODE = 6
    valuetype = np.float64

    nodes = op2.Set(NUM_NODES, "nodes")
    elements = op2.Set(NUM_ELE, "elements")
    bdry_nodes = op2.Set(NUM_BDRY_NODE, "boundary_nodes")

    elem_node_map = np.array([0, 1, 4, 4, 3, 0, 1, 2, 5, 5, 4, 1, 3, 4, 7, 7,
                              6, 3, 4, 5, 8, 8, 7, 4], dtype=np.uint32)
    elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

    bdry_node_node_map = np.array([0, 1, 2, 6, 7, 8], dtype=valuetype)
    bdry_node_node = op2.Map(bdry_nodes, nodes, 1, bdry_node_node_map,
                             "bdry_node_node")

    sparsity = op2.Sparsity((nodes, nodes), (elem_node, elem_node), "sparsity")
    mat = op2.Mat(sparsity, valuetype, "mat")

    coord_vals = np.array([(0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
                           (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
                           (0.0, 1.0), (0.5, 1.0), (1.0, 1.0)],
                          dtype=valuetype)
    coords = op2.Dat(nodes ** 2, coord_vals, valuetype, "coords")

    u_vals = np.array([1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0])
    f = op2.Dat(nodes, np.zeros(NUM_NODES, dtype=valuetype), valuetype, "f")
    b = op2.Dat(nodes, np.zeros(NUM_NODES, dtype=valuetype), valuetype, "b")
    x = op2.Dat(nodes, np.zeros(NUM_NODES, dtype=valuetype), valuetype, "x")
    u = op2.Dat(nodes, u_vals, valuetype, "u")

    bdry_vals = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=valuetype)
    bdry = op2.Dat(bdry_nodes, bdry_vals, valuetype, "bdry")

    # Assemble matrix and rhs

    op2.par_loop(laplacian, elements,
                 mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                 coords(op2.READ, elem_node, flatten=True))

    op2.par_loop(rhs, elements,
                 b(op2.INC, elem_node[op2.i[0]]),
                 coords(op2.READ, elem_node, flatten=True),
                 f(op2.READ, elem_node))

    # Apply strong BCs

    mat.zero_rows([0, 1, 2, 6, 7, 8], 1.0)
    strongbc_rhs = op2.Kernel("""
    void strongbc_rhs(double *val, double *target) { *target = *val; }
    """, "strongbc_rhs")
    op2.par_loop(strongbc_rhs, bdry_nodes,
                 bdry(op2.READ),
                 b(op2.WRITE, bdry_node_node[0]))

    solver = op2.Solver(ksp_type='gmres')
    solver.solve(mat, x, b)

    # Print solution
    if opt['print_output']:
        print "Expected solution: %s" % u.data
        print "Computed solution: %s" % x.data

    # Save output (if necessary)
    if opt['return_output']:
        return u.data, x.data
    if opt['save_output']:
        import pickle
        with open("laplace.out", "w") as out:
            pickle.dump((u.data, x.data), out)

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('--print-output', action='store_true', help='Print output')
parser.add_argument('-r', '--return-output', action='store_true',
                    help='Return output for testing')
parser.add_argument('-s', '--save-output', action='store_true',
                    help='Save output for testing')
parser.add_argument('-p', '--profile', action='store_true',
                    help='Create a cProfile for the run')

if __name__ == '__main__':
    opt = vars(parser.parse_args())
    op2.init(**opt)

    if opt['profile']:
        import cProfile
        cProfile.run('main(opt)', filename='laplace_ffc.cprofile')
    else:
        main(opt)
