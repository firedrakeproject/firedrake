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

"""PyOP2 P1 MPI advection-diffusion demo

This demo solves the advection-diffusion equation by splitting the advection
and diffusion terms. The advection term is advanced in time using an Euler
method and the diffusion term is advanced in time using a theta scheme with
theta = 0.5.

The domain read in from a pickle dump.

This demo requires the MAPDES forks of FFC, FIAT and UFL which are found at:

    https://bitbucket.org/mapdes/ffc
    https://bitbucket.org/mapdes/fiat
    https://bitbucket.org/mapdes/ufl
"""

import os
import numpy as np
from cPickle import load
import gzip

from pyop2 import op2, utils
from pyop2.ffc_interface import compile_form
from ufl import *


def main(opt):
    # Set up finite element problem

    dt = 0.0001

    T = FiniteElement("Lagrange", "triangle", 1)
    V = VectorElement("Lagrange", "triangle", 1)

    p = TrialFunction(T)
    q = TestFunction(T)
    t = Coefficient(T)
    u = Coefficient(V)
    a = Coefficient(T)

    diffusivity = 0.1

    M = p * q * dx

    adv_rhs = (q * t + dt * dot(grad(q), u) * t) * dx

    d = -dt * diffusivity * dot(grad(q), grad(p)) * dx

    diff = M - 0.5 * d
    diff_rhs = action(M + 0.5 * d, t)

    # Generate code for mass and rhs assembly.

    adv, = compile_form(M, "adv")
    adv_rhs, = compile_form(adv_rhs, "adv_rhs")
    diff, = compile_form(diff, "diff")
    diff_rhs, = compile_form(diff_rhs, "diff_rhs")

    # Set up simulation data structures

    valuetype = np.float64

    f = gzip.open(opt['mesh'] + '.' + str(op2.MPI.comm.rank) + '.pickle.gz')

    elements, nodes, elem_node, coords = load(f)
    f.close()
    coords = op2.Dat(nodes ** 2, coords.data, np.float64, "dcoords")

    num_nodes = nodes.total_size

    sparsity = op2.Sparsity((nodes, nodes), (elem_node, elem_node), "sparsity")
    if opt['advection']:
        adv_mat = op2.Mat(sparsity, valuetype, "adv_mat")
        op2.par_loop(adv, elements,
                     adv_mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                     coords(op2.READ, elem_node, flatten=True))
    if opt['diffusion']:
        diff_mat = op2.Mat(sparsity, valuetype, "diff_mat")
        op2.par_loop(diff, elements,
                     diff_mat(op2.INC, (elem_node[op2.i[0]], elem_node[op2.i[1]])),
                     coords(op2.READ, elem_node, flatten=True))

    tracer_vals = np.zeros(num_nodes, dtype=valuetype)
    tracer = op2.Dat(nodes, tracer_vals, valuetype, "tracer")

    b_vals = np.zeros(num_nodes, dtype=valuetype)
    b = op2.Dat(nodes, b_vals, valuetype, "b")

    velocity_vals = np.asarray([1.0, 0.0] * num_nodes, dtype=valuetype)
    velocity = op2.Dat(nodes ** 2, velocity_vals, valuetype, "velocity")

    # Set initial condition

    i_cond_code = """void i_cond(double *c, double *t)
{
  double A   = 0.1; // Normalisation
  double D   = 0.1; // Diffusivity
  double pi  = 3.14159265358979;
  double x   = c[0]-(0.45+%(T)f);
  double y   = c[1]-0.5;
  double r2  = x*x+y*y;

  *t = A*(exp(-r2/(4*D*%(T)f))/(4*pi*D*%(T)f));
}
"""

    T = 0.01

    i_cond = op2.Kernel(i_cond_code % {'T': T}, "i_cond")

    op2.par_loop(i_cond, nodes,
                 coords(op2.READ, flatten=True),
                 tracer(op2.WRITE))

    # Assemble and solve

    solver = op2.Solver()

    while T < 0.015:

        # Advection

        if opt['advection']:
            b.zero()
            op2.par_loop(adv_rhs, elements,
                         b(op2.INC, elem_node[op2.i[0]]),
                         coords(op2.READ, elem_node, flatten=True),
                         tracer(op2.READ, elem_node),
                         velocity(op2.READ, elem_node))

            solver.solve(adv_mat, tracer, b)

        # Diffusion

        if opt['diffusion']:
            b.zero()
            op2.par_loop(diff_rhs, elements,
                         b(op2.INC, elem_node[op2.i[0]]),
                         coords(op2.READ, elem_node, flatten=True),
                         tracer(op2.READ, elem_node))

            solver.solve(diff_mat, tracer, b)

        T = T + dt

    if opt['print_output'] or opt['test_output']:
        analytical_vals = np.zeros(num_nodes, dtype=valuetype)
        analytical = op2.Dat(nodes, analytical_vals, valuetype, "analytical")

        i_cond = op2.Kernel(i_cond_code % {'T': T}, "i_cond")

        op2.par_loop(i_cond, nodes,
                     coords(op2.READ, flatten=True),
                     analytical(op2.WRITE))

    # Print error w.r.t. analytical solution
    if opt['print_output']:
        print "Rank: %d Expected - computed  solution: %s" % \
            (op2.MPI.comm.rank, tracer.data - analytical.data)

    if opt['test_output']:
        l2norm = dot(t - a, t - a) * dx
        l2_kernel, = compile_form(l2norm, "error_norm")
        result = op2.Global(1, [0.0])
        op2.par_loop(l2_kernel, elements,
                     result(op2.INC),
                     coords(op2.READ, elem_node, flatten=True),
                     tracer(op2.READ, elem_node),
                     analytical(op2.READ, elem_node)
                     )
        if op2.MPI.comm.rank == 0:
            with open("adv_diff_mpi.%s.out" % os.path.split(opt['mesh'])[-1],
                      "w") as out:
                out.write(str(result.data[0]))
        else:
            # hack to prevent mpi communication dangling
            result.data

if __name__ == '__main__':
    parser = utils.parser(group=True, description=__doc__)
    parser.add_argument('-m', '--mesh', required=True,
                        help='Base name of mesh pickle \
                              (excluding the process number and .pickle extension)')
    parser.add_argument('--no-advection', action='store_false',
                        dest='advection', help='Disable advection')
    parser.add_argument('--no-diffusion', action='store_false',
                        dest='diffusion', help='Disable diffusion')
    parser.add_argument('--print-output', action='store_true', help='Print output')
    parser.add_argument('-t', '--test-output', action='store_true',
                        help='Save output for testing')
    parser.add_argument('-p', '--profile', action='store_true',
                        help='Create a cProfile for the run')

    opt = vars(parser.parse_args())
    op2.init(**opt)

    if op2.MPI.comm.size != 3:
        print "MPI advection-diffusion demo only works on 3 processes"
        op2.MPI.comm.Abort(1)

    if opt['profile']:
        import cProfile
        filename = 'adv_diff.%s.%d.cprofile' % (
            os.path.split(opt['mesh'])[-1], op2.MPI.comm.rank)
        cProfile.run('main(opt)', filename=filename)
    else:
        main(opt)
