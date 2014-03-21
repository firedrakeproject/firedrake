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

"""PyOP2 2D mass equation demo (MPI version)

This is a demo of the use of ffc to generate kernels. It solves the identity
equation on a quadrilateral domain.

This demo requires the MAPDES forks of FFC, FIAT and UFL which are found at:

    https://bitbucket.org/mapdes/ffc
    https://bitbucket.org/mapdes/fiat
    https://bitbucket.org/mapdes/ufl
"""

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from ufl import *
import numpy as np
from petsc4py import PETSc

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('-s', '--save-output',
                    action='store_true',
                    help='Save the output of the run')
parser.add_argument('-t', '--test-output',
                    action='store_true',
                    help='Save output for testing')
opt = vars(parser.parse_args())
op2.init(**opt)

# Set up finite element identity problem

E = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(E)
u = TrialFunction(E)
f = Coefficient(E)

a = v * u * dx
L = v * f * dx

# Generate code for mass and rhs assembly.

mass, = compile_form(a, "mass")
rhs, = compile_form(L, "rhs")

# Set up simulation data structures

NUM_ELE = (0, 1, 2, 2)
NUM_NODES = (0, 2, 4, 4)
valuetype = np.float64

if op2.MPI.comm.size != 2:
    print "MPI mass2d demo only works on two processes"
    op2.MPI.comm.Abort(1)

if op2.MPI.comm.rank == 0:
    node_global_to_universal = np.asarray([0, 1, 2, 3], dtype=PETSc.IntType)
    node_halo = op2.Halo(sends={1: [0, 1]}, receives={1: [2, 3]},
                         gnn2unn=node_global_to_universal)
    element_halo = op2.Halo(sends={1: [0]}, receives={1: [1]})
elif op2.MPI.comm.rank == 1:
    node_global_to_universal = np.asarray([2, 3, 1, 0], dtype=PETSc.IntType)
    node_halo = op2.Halo(sends={0: [0, 1]}, receives={0: [3, 2]},
                         gnn2unn=node_global_to_universal)
    element_halo = op2.Halo(sends={0: [0]}, receives={0: [1]})
else:
    op2.MPI.comm.Abort(1)
nodes = op2.Set(NUM_NODES, "nodes", halo=node_halo)
elements = op2.Set(NUM_ELE, "elements", halo=element_halo)

if op2.MPI.comm.rank == 0:
    elem_node_map = np.asarray([0, 1, 3, 2, 3, 1], dtype=np.uint32)
elif op2.MPI.comm.rank == 1:
    elem_node_map = np.asarray([0, 1, 2, 2, 3, 1], dtype=np.uint32)
else:
    op2.MPI.comm.Abort(1)

elem_node = op2.Map(elements, nodes, 3, elem_node_map, "elem_node")

sparsity = op2.Sparsity((nodes, nodes), (elem_node, elem_node), "sparsity")
mat = op2.Mat(sparsity, valuetype, "mat")

if op2.MPI.comm.rank == 0:
    coord_vals = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.5)],
                            dtype=valuetype)
elif op2.MPI.comm.rank == 1:
    coord_vals = np.asarray([(1, 1), (0, 1.5), (2, 0), (0, 0)],
                            dtype=valuetype)
else:
    op2.MPI.comm.Abort(1)
coords = op2.Dat(nodes ** 2, coord_vals, valuetype, "coords")

if op2.MPI.comm.rank == 0:
    f_vals = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=valuetype)
elif op2.MPI.comm.rank == 1:
    f_vals = np.asarray([3.0, 4.0, 2.0, 1.0], dtype=valuetype)
else:
    op2.MPI.comm.Abort(1)
b_vals = np.asarray([0.0] * NUM_NODES[3], dtype=valuetype)
x_vals = np.asarray([0.0] * NUM_NODES[3], dtype=valuetype)
f = op2.Dat(nodes, f_vals, valuetype, "f")
b = op2.Dat(nodes, b_vals, valuetype, "b")
x = op2.Dat(nodes, x_vals, valuetype, "x")

# Assemble and solve

op2.par_loop(mass, elements,
             mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
             coords(op2.READ, elem_node, flatten=True))

op2.par_loop(rhs, elements,
             b(op2.INC, elem_node[op2.i[0]]),
             coords(op2.READ, elem_node, flatten=True),
             f(op2.READ, elem_node))

solver = op2.Solver()
solver.solve(mat, x, b)


# Compute error in solution
error = (f.data[:f.dataset.size] - x.data[:x.dataset.size])

# Print error solution
print "Rank: %d Expected - computed  solution: %s" % \
    (op2.MPI.comm.rank, error)

# Save output (if necessary)
if opt['save_output']:
    raise RuntimeException('Writing distributed Dats not yet supported')

if opt['test_output']:
    import pickle
    with open("mass2d_mpi_%d.out" % op2.MPI.comm.rank, "w") as out:
        pickle.dump(error, out)
