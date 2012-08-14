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

from pyop2 import op2, utils
import numpy as np
import h5py
from math import sqrt
from aero_kernels import dirichlet, dotPV, dotR, init_cg, res_calc, spMV, \
    update, updateP, updateUR

op2.init(**utils.default_parser_parse_args())

# Constants

gam = 1.4
gm1 = op2.Const(1, gam - 1.0, 'gm1', dtype=np.double)
gm1i = op2.Const(1, 1.0/gm1.data, 'gm1i', dtype=np.double)
wtg1 = op2.Const(2, [0.5, 0.5], 'wtg1', dtype=np.double)
xi1 = op2.Const(2, [0.211324865405187, 0.788675134594813], 'xi1', dtype=np.double)
Ng1 = op2.Const(4, [0.788675134594813, 0.211324865405187,
                    0.211324865405187, 0.788675134594813],
                'Ng1', dtype=np.double)
Ng1_xi = op2.Const(4, [-1, -1, 1, 1], 'Ng1_xi', dtype=np.double)
wtg2 = op2.Const(4, [0.25] * 4, 'wtg2', dtype=np.double)
Ng2 = op2.Const(16, [0.622008467928146, 0.166666666666667,
                     0.166666666666667, 0.044658198738520,
                     0.166666666666667, 0.622008467928146,
                     0.044658198738520, 0.166666666666667,
                     0.166666666666667, 0.044658198738520,
                     0.622008467928146, 0.166666666666667,
                     0.044658198738520, 0.166666666666667,
                     0.166666666666667, 0.622008467928146],
                'Ng2', dtype=np.double)
Ng2_xi = op2.Const(32, [-0.788675134594813, 0.788675134594813,
                        -0.211324865405187, 0.211324865405187,
                        -0.788675134594813, 0.788675134594813,
                        -0.211324865405187, 0.211324865405187,
                        -0.211324865405187, 0.211324865405187,
                        -0.788675134594813, 0.788675134594813,
                        -0.211324865405187, 0.211324865405187,
                        -0.788675134594813, 0.788675134594813,
                        -0.788675134594813, -0.211324865405187,
                        0.788675134594813, 0.211324865405187,
                        -0.211324865405187, -0.788675134594813,
                        0.211324865405187, 0.788675134594813,
                        -0.788675134594813, -0.211324865405187,
                        0.788675134594813, 0.211324865405187,
                        -0.211324865405187, -0.788675134594813,
                        0.211324865405187, 0.788675134594813],
                   'Ng2_xi', dtype=np.double)
minf = op2.Const(1, 0.1, 'minf', dtype=np.double)
m2 = op2.Const(1, minf.data**2, 'm2', dtype=np.double)
freq = op2.Const(1, 1, 'freq', dtype=np.double)
kappa = op2.Const(1, 1, 'kappa', dtype=np.double)
nmode = op2.Const(1, 0, 'nmode', dtype=np.double)
mfan = op2.Const(1, 1.0, 'mfan', dtype=np.double)

with h5py.File('FE_grid.h5', 'r') as file:
    # sets
    nodes  = op2.Set.fromhdf5(file, 'nodes')
    bnodes = op2.Set.fromhdf5(file, 'bedges')
    cells  = op2.Set.fromhdf5(file, 'cells')

    # maps
    pbnodes = op2.Map.fromhdf5(bnodes, nodes, file, 'pbedge')
    pcell   = op2.Map.fromhdf5(cells,  nodes, file, 'pcell')

    # dats
    p_xm   = op2.Dat.fromhdf5(nodes, file, 'p_x')
    p_phim = op2.Dat.fromhdf5(nodes, file, 'p_phim')
    p_resm = op2.Dat.fromhdf5(nodes, file, 'p_resm')
    p_K    = op2.Dat.fromhdf5(cells, file, 'p_K')
    p_V    = op2.Dat.fromhdf5(nodes, file, 'p_V')
    p_P    = op2.Dat.fromhdf5(nodes, file, 'p_P')
    p_U    = op2.Dat.fromhdf5(nodes, file, 'p_U')

niter = 20

for i in xrange(1, niter+1):

    op2.par_loop(res_calc, cells,
                 p_xm(pcell, op2.READ),
                 p_phim(pcell, op2.READ),
                 p_K(op2.IdentityMap, op2.WRITE),
                 p_resm(pcell, op2.INC))

    op2.par_loop(dirichlet, bnodes,
                 p_resm(pbnodes(0), op2.WRITE))

    c1 = op2.Global(1, data=0.0, name='c1')
    c2 = op2.Global(1, data=0.0, name='c2')
    c3 = op2.Global(1, data=0.0, name='c3')
    # c1 = R' * R
    op2.par_loop(init_cg, nodes,
                 p_resm(op2.IdentityMap, op2.READ),
                 c1(op2.INC),
                 p_U(op2.IdentityMap, op2.WRITE),
                 p_V(op2.IdentityMap, op2.WRITE),
                 p_P(op2.IdentityMap, op2.WRITE))

    # Set stopping criteria
    res0 = sqrt(c1.data)
    res = res0
    res0 *= 0.1
    it = 0
    maxiter = 200

    while res > res0 and it < maxiter:

        # V = Stiffness * P
        op2.par_loop(spMV, cells,
                     p_V(pcell, op2.INC),
                     p_K(op2.IdentityMap, op2.READ),
                     p_P(pcell, op2.READ))

        op2.par_loop(dirichlet, bnodes,
                     p_V(pbnodes(0), op2.WRITE))

        c2.data = 0.0

        # c2 = P' * V
        op2.par_loop(dotPV, nodes,
                     p_P(op2.IdentityMap, op2.READ),
                     p_V(op2.IdentityMap, op2.READ),
                     c2(op2.INC))

        alpha = op2.Global(1, data=c1.data/c2.data, name='alpha')

        # U = U + alpha * P
        # resm = resm - alpha * V
        op2.par_loop(updateUR, nodes,
                     p_U(op2.IdentityMap, op2.INC),
                     p_resm(op2.IdentityMap, op2.INC),
                     p_P(op2.IdentityMap, op2.READ),
                     p_V(op2.IdentityMap, op2.RW),
                     alpha(op2.READ))

        c3.data = 0.0
        # c3 = resm' * resm
        op2.par_loop(dotR, nodes,
                     p_resm(op2.IdentityMap, op2.READ),
                     c3(op2.INC))

        beta = op2.Global(1, data=c3.data/c1.data, name="beta")
        # P = beta * P + resm
        op2.par_loop(updateP, nodes,
                     p_resm(op2.IdentityMap, op2.READ),
                     p_P(op2.IdentityMap, op2.RW),
                     beta(op2.READ))

        c1.data = c3.data
        res = sqrt(c1.data)
        it += 1

    rms = op2.Global(1, data=0.0, name='rms')

    # phim = phim - Stiffness \ Load
    op2.par_loop(update, nodes,
                 p_phim(op2.IdentityMap, op2.RW),
                 p_resm(op2.IdentityMap, op2.WRITE),
                 p_U(op2.IdentityMap, op2.READ),
                 rms(op2.INC))

    print "rms = %10.5e iter: %d" % (sqrt(rms.data)/sqrt(nodes.size), it)

op2.exit()
