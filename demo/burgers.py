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

"""Burgers equation demo (unstable forward-Euler integration)

This demo solves the steady-state Burgers equation on a unit interval.
"""

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from ufl import *
import numpy as np
import pylab

parser = utils.parser(group=True,
                      description=__doc__)
parser.add_argument('-p', '--plot',
                    action='store_true',
                    help='Plot the resulting L2 error norm')

opt = vars(parser.parse_args())
op2.init(**opt)

# Simulation parameters
n = 100
nu = 0.0001
timestep = 1.0 / n

# Create simulation data structures

nodes = op2.Set(n, "nodes")
b_nodes = op2.Set(2, "b_nodes")
elements = op2.Set(n - 1, "elements")

elem_node_map = [item for sublist in [(x, x + 1)
                                      for x in xrange(n - 1)] for item in sublist]
dnodes1 = op2.DataSet(nodes, 1)
db_nodes1 = op2.DataSet(nodes, 1)

elem_node = op2.Map(elements, nodes, 2, elem_node_map, "elem_node")

b_node_node_map = [0, n - 1]
b_node_node = op2.Map(b_nodes, nodes, 1, b_node_node_map, "b_node_node")

coord_vals = [i * (1.0 / (n - 1)) for i in xrange(n)]
coords = op2.Dat(dnodes1, coord_vals, np.float64, "coords")

tracer_vals = np.asarray([0.0] * n, dtype=np.float64)
tracer = op2.Dat(dnodes1, tracer_vals, np.float64, "tracer")

tracer_old_vals = np.asarray([0.0] * n, dtype=np.float64)
tracer_old = op2.Dat(dnodes1, tracer_old_vals, np.float64, "tracer_old")

b_vals = np.asarray([0.0] * n, dtype=np.float64)
b = op2.Dat(dnodes1, b_vals, np.float64, "b")

bdry_vals = [0.0, 1.0]
bdry = op2.Dat(db_nodes1, bdry_vals, np.float64, "bdry")

sparsity = op2.Sparsity((dnodes1, dnodes1), (elem_node, elem_node), "sparsity")
mat = op2.Mat(sparsity, np.float64, "mat")

# Set up finite element problem

V = FiniteElement("Lagrange", "interval", 1)
u = Coefficient(V)
u_next = TrialFunction(V)
v = TestFunction(V)

a = (dot(u, grad(u_next)) * v + nu * grad(u_next) * grad(v)) * dx
L = v * u * dx

burgers, = compile_form(a, "burgers")
rhs, = compile_form(L, "rhs")

# Initial condition

i_cond_code = """
void i_cond(double *c, double *t)
{
  double pi = 3.14159265358979;
  *t = *c*2;
}
"""

i_cond = op2.Kernel(i_cond_code, "i_cond")

op2.par_loop(i_cond, nodes,
             coords(op2.IdentityMap, op2.READ),
             tracer(op2.IdentityMap, op2.WRITE))

# Boundary condition

strongbc_rhs = op2.Kernel(
    "void strongbc_rhs(double *v, double *t) { *t = *v; }", "strongbc_rhs")

# Some other useful kernels

assign_dat_code = """
void assign_dat(double *dest, double *src)
{
  *dest = *src;
}"""

assign_dat = op2.Kernel(assign_dat_code, "assign_dat")

l2norm_diff_sq_code = """
void l2norm_diff_sq(double *f, double *g, double *norm)
{
  double diff = abs(*f - *g);
  *norm += diff*diff;
}
"""

l2norm_diff_sq = op2.Kernel(l2norm_diff_sq_code, "l2norm_diff_sq")

# Nonlinear iteration

# Tol = 1.e-8
tolsq = 1.e-16
normsq = op2.Global(1, data=10000.0, name="norm")
solver = op2.Solver()

while normsq.data[0] > tolsq:

    # Assign result from previous timestep

    op2.par_loop(assign_dat, nodes,
                 tracer_old(op2.IdentityMap, op2.WRITE),
                 tracer(op2.IdentityMap, op2.READ))

    # Matrix assembly

    mat.zero()

    op2.par_loop(burgers, elements(2, 2),
                 mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
                 coords(elem_node, op2.READ),
                 tracer(elem_node, op2.READ))

    mat.zero_rows([0, n - 1], 1.0)

    # RHS Assembly

    rhs.zero()

    op2.par_loop(rhs, elements(3),
                 b(elem_node[op2.i[0]], op2.INC),
                 coords(elem_node, op2.READ),
                 tracer(elem_node, op2.READ))

    op2.par_loop(strongbc_rhs, b_nodes,
                 bdry(op2.IdentityMap, op2.READ),
                 b(b_node_node[0], op2.WRITE))

    # Solve

    solver.solve(mat, tracer, b)

    # Calculate L2-norm^2

    normsq = op2.Global(1, data=0.0, name="norm")
    op2.par_loop(l2norm_diff_sq, nodes,
                 tracer(op2.IdentityMap, op2.READ),
                 tracer_old(op2.IdentityMap, op2.READ),
                 normsq(op2.INC))

    print "L2 Norm squared: %s" % normsq.data[0]

if opt['plot']:
    pylab.plot(coords.data, tracer.data)
    pylab.show()
