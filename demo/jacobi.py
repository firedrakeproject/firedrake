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

# This file contains code from the original OP2 distribution, in the
# 'update' and 'res' variables. The original copyright notice follows:

# Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
# the main source directory for a full list of copyright holders.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Mike Giles may not be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."

"""PyOP2 Jacobi demo

Port of the Jacobi demo from OP2-Common.
"""

from pyop2 import op2, utils
import numpy as np
from math import sqrt

parser = utils.parser(group=True, description=__doc__)
parser.add_argument('-s', '--single',
                    action='store_true',
                    help='single precision floating point mode')
parser.add_argument('-n', '--niter',
                    action='store',
                    default=2,
                    type=int,
                    help='set the number of iteration')

opt = vars(parser.parse_args())
op2.init(**opt)

fp_type = np.float32 if opt['single'] else np.float64

NN = 6
NITER = opt['niter']

nnode = (NN - 1) ** 2
nedge = nnode + 4 * (NN - 1) * (NN - 2)

pp = np.zeros((2 * nedge,), dtype=np.int)

A = np.zeros((nedge,), dtype=fp_type)
r = np.zeros((nnode,), dtype=fp_type)
u = np.zeros((nnode,), dtype=fp_type)
du = np.zeros((nnode,), dtype=fp_type)

e = 0

for i in xrange(1, NN):
    for j in xrange(1, NN):
        n = i - 1 + (j - 1) * (NN - 1)
        pp[2 * e] = n
        pp[2 * e + 1] = n
        A[e] = -1
        e += 1
        for p in xrange(0, 4):
            i2 = i
            j2 = j
            if p == 0:
                i2 += -1
            if p == 1:
                i2 += +1
            if p == 2:
                j2 += -1
            if p == 3:
                j2 += +1

            if i2 == 0 or i2 == NN or j2 == 0 or j2 == NN:
                r[n] += 0.25
            else:
                pp[2 * e] = n
                pp[2 * e + 1] = i2 - 1 + (j2 - 1) * (NN - 1)
                A[e] = 0.25
                e += 1


nodes = op2.Set(nnode, "nodes")
edges = op2.Set(nedge, "edges")

ppedge = op2.Map(edges, nodes, 2, pp, "ppedge")

p_A = op2.Dat(edges, data=A, name="p_A")
p_r = op2.Dat(nodes, data=r, name="p_r")
p_u = op2.Dat(nodes, data=u, name="p_u")
p_du = op2.Dat(nodes, data=du, name="p_du")

alpha = op2.Global(1, data=1.0, name="alpha", dtype=fp_type)

beta = op2.Global(1, data=1.0, name="beta", dtype=fp_type)


res = op2.Kernel("""void res(%(t)s *A, %(t)s *u, %(t)s *du, const %(t)s *beta){
  *du += (*beta)*(*A)*(*u);
}""" % {'t': "double" if fp_type == np.float64 else "float"}, "res")

update = op2.Kernel("""
void update(%(t)s *r, %(t)s *du, %(t)s *u, %(t)s *u_sum, %(t)s *u_max) {
  *u += *du + alpha * (*r);
  *du = %(z)s;
  *u_sum += (*u)*(*u);
  *u_max = *u_max > *u ? *u_max : *u;
}""" % {'t': "double" if fp_type == np.float64 else "float",
        'z': "0.0" if fp_type == np.float64 else "0.0f"}, "update")


for iter in xrange(0, NITER):
    op2.par_loop(res, edges,
                 p_A(op2.READ),
                 p_u(op2.READ, ppedge[1]),
                 p_du(op2.INC, ppedge[0]),
                 beta(op2.READ))
    u_sum = op2.Global(1, data=0.0, name="u_sum", dtype=fp_type)
    u_max = op2.Global(1, data=0.0, name="u_max", dtype=fp_type)

    op2.par_loop(update, nodes,
                 p_r(op2.READ),
                 p_du(op2.RW),
                 p_u(op2.INC),
                 u_sum(op2.INC),
                 u_max(op2.MAX))

    print(" u max/rms = %f %f \n" % (u_max.data[0], sqrt(u_sum.data / nnode)))


print("\nResults after %d iterations\n" % NITER)
for j in range(NN - 1, 0, -1):
    for i in range(1, NN):
        print(" %7.4f" % p_u.data[i - 1 + (j - 1) * (NN - 1)], end='')
    print("")
print("")
