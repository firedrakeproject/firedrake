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

"""PyOP2 aero demo

Port of the aero demo from OP2-Common. Requires an HDF5 mesh file.
"""

import numpy as np
import h5py
from math import sqrt
import os

from pyop2 import op2, utils


def main(opt):
    from aero_kernels import dirichlet, dotPV, dotR, init_cg, res_calc, spMV, \
        update, updateP, updateUR
    try:
        with h5py.File(opt['mesh'], 'r') as f:
            # sets
            nodes = op2.Set.fromhdf5(f, 'nodes')
            bnodes = op2.Set.fromhdf5(f, 'bedges')
            cells = op2.Set.fromhdf5(f, 'cells')

            # maps
            pbnodes = op2.Map.fromhdf5(bnodes, nodes, f, 'pbedge')
            pcell = op2.Map.fromhdf5(cells, nodes, f, 'pcell')
            pvcell = op2.Map.fromhdf5(cells, nodes, f, 'pcell')

            # dats
            p_xm = op2.Dat.fromhdf5(nodes ** 2, f, 'p_x')
            p_phim = op2.Dat.fromhdf5(nodes, f, 'p_phim')
            p_resm = op2.Dat.fromhdf5(nodes, f, 'p_resm')
            p_K = op2.Dat.fromhdf5(cells ** 16, f, 'p_K')
            p_V = op2.Dat.fromhdf5(nodes, f, 'p_V')
            p_P = op2.Dat.fromhdf5(nodes, f, 'p_P')
            p_U = op2.Dat.fromhdf5(nodes, f, 'p_U')
    except IOError:
        import sys
        print "Failed reading mesh: Could not read from %s\n" % opt['mesh']
        sys.exit(1)

    # Constants

    gam = 1.4
    gm1 = op2.Const(1, gam - 1.0, 'gm1', dtype=np.double)
    op2.Const(1, 1.0 / gm1.data, 'gm1i', dtype=np.double)
    op2.Const(2, [0.5, 0.5], 'wtg1', dtype=np.double)
    op2.Const(2, [0.211324865405187, 0.788675134594813], 'xi1',
              dtype=np.double)
    op2.Const(4, [0.788675134594813, 0.211324865405187,
                  0.211324865405187, 0.788675134594813],
              'Ng1', dtype=np.double)
    op2.Const(4, [-1, -1, 1, 1], 'Ng1_xi', dtype=np.double)
    op2.Const(4, [0.25] * 4, 'wtg2', dtype=np.double)
    op2.Const(16, [0.622008467928146, 0.166666666666667,
                   0.166666666666667, 0.044658198738520,
                   0.166666666666667, 0.622008467928146,
                   0.044658198738520, 0.166666666666667,
                   0.166666666666667, 0.044658198738520,
                   0.622008467928146, 0.166666666666667,
                   0.044658198738520, 0.166666666666667,
                   0.166666666666667, 0.622008467928146],
              'Ng2', dtype=np.double)
    op2.Const(32, [-0.788675134594813, 0.788675134594813,
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
    op2.Const(1, minf.data ** 2, 'm2', dtype=np.double)
    op2.Const(1, 1, 'freq', dtype=np.double)
    op2.Const(1, 1, 'kappa', dtype=np.double)
    op2.Const(1, 0, 'nmode', dtype=np.double)
    op2.Const(1, 1.0, 'mfan', dtype=np.double)

    niter = 20

    for i in xrange(1, niter + 1):

        op2.par_loop(res_calc, cells,
                     p_xm(op2.READ, pvcell),
                     p_phim(op2.READ, pcell),
                     p_K(op2.WRITE),
                     p_resm(op2.INC, pcell))

        op2.par_loop(dirichlet, bnodes,
                     p_resm(op2.WRITE, pbnodes[0]))

        c1 = op2.Global(1, data=0.0, name='c1')
        c2 = op2.Global(1, data=0.0, name='c2')
        c3 = op2.Global(1, data=0.0, name='c3')
        # c1 = R' * R
        op2.par_loop(init_cg, nodes,
                     p_resm(op2.READ),
                     c1(op2.INC),
                     p_U(op2.WRITE),
                     p_V(op2.WRITE),
                     p_P(op2.WRITE))

        # Set stopping criteria
        res0 = sqrt(c1.data)
        res = res0
        res0 *= 0.1
        it = 0
        maxiter = 200

        while res > res0 and it < maxiter:

            # V = Stiffness * P
            op2.par_loop(spMV, cells,
                         p_V(op2.INC, pcell),
                         p_K(op2.READ),
                         p_P(op2.READ, pcell))

            op2.par_loop(dirichlet, bnodes,
                         p_V(op2.WRITE, pbnodes[0]))

            c2.data = 0.0

            # c2 = P' * V
            op2.par_loop(dotPV, nodes,
                         p_P(op2.READ),
                         p_V(op2.READ),
                         c2(op2.INC))

            alpha = op2.Global(1, data=c1.data / c2.data, name='alpha')

            # U = U + alpha * P
            # resm = resm - alpha * V
            op2.par_loop(updateUR, nodes,
                         p_U(op2.INC),
                         p_resm(op2.INC),
                         p_P(op2.READ),
                         p_V(op2.RW),
                         alpha(op2.READ))

            c3.data = 0.0
            # c3 = resm' * resm
            op2.par_loop(dotR, nodes,
                         p_resm(op2.READ),
                         c3(op2.INC))

            beta = op2.Global(1, data=c3.data / c1.data, name="beta")
            # P = beta * P + resm
            op2.par_loop(updateP, nodes,
                         p_resm(op2.READ),
                         p_P(op2.RW),
                         beta(op2.READ))

            c1.data = c3.data
            res = sqrt(c1.data)
            it += 1

        rms = op2.Global(1, data=0.0, name='rms')

        # phim = phim - Stiffness \ Load
        op2.par_loop(update, nodes,
                     p_phim(op2.RW),
                     p_resm(op2.WRITE),
                     p_U(op2.READ),
                     rms(op2.INC))

        print "rms = %10.5e iter: %d" % (sqrt(rms.data) / sqrt(nodes.size), it)

if __name__ == '__main__':
    parser = utils.parser(group=True, description=__doc__)
    parser.add_argument('-m', '--mesh', default='meshes/FE_grid.h5',
                        help='HDF5 mesh file to use (default: meshes/FE_grid.h5)')
    parser.add_argument('-p', '--profile', action='store_true',
                        help='Create a cProfile for the run')
    opt = vars(parser.parse_args())
    op2.init(**opt)

    if opt['profile']:
        import cProfile
        filename = 'aero.%s.cprofile' % os.path.split(opt['mesh'])[-1]
        cProfile.run('main(opt)', filename=filename)
    else:
        main(opt)
