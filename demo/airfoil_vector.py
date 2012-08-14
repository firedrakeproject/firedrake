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

from math import atan, sqrt
import numpy as np
from pyop2 import op2, utils
import h5py

op2.init(**utils.parse_args())

from airfoil_vector_kernels import save_soln, adt_calc, res_calc, bres_calc, update

with h5py.File('new_grid.h5', 'r') as file:

    # Declare sets, maps, datasets and global constants

    nodes  = op2.Set.fromhdf5(file, "nodes")
    edges  = op2.Set.fromhdf5(file, "edges")
    bedges = op2.Set.fromhdf5(file, "bedges")
    cells  = op2.Set.fromhdf5(file, "cells")

    pedge   = op2.Map.fromhdf5(edges,  nodes, file, "pedge")
    pecell  = op2.Map.fromhdf5(edges,  cells, file, "pecell")
    pbedge  = op2.Map.fromhdf5(bedges, nodes, file, "pbedge")
    pbecell = op2.Map.fromhdf5(bedges, cells, file, "pbecell")
    pcell   = op2.Map.fromhdf5(cells,  nodes, file, "pcell")

    p_bound = op2.Dat.fromhdf5(bedges, file, "p_bound")
    p_x     = op2.Dat.fromhdf5(nodes,  file, "p_x")
    p_q     = op2.Dat.fromhdf5(cells,  file, "p_q")
    p_qold  = op2.Dat.fromhdf5(cells,  file, "p_qold")
    p_adt   = op2.Dat.fromhdf5(cells,  file, "p_adt")
    p_res   = op2.Dat.fromhdf5(cells,  file, "p_res")

    gam   = op2.Const.fromhdf5(file, "gam")
    gm1   = op2.Const.fromhdf5(file, "gm1")
    cfl   = op2.Const.fromhdf5(file, "cfl")
    eps   = op2.Const.fromhdf5(file, "eps")
    mach  = op2.Const.fromhdf5(file, "mach")
    alpha = op2.Const.fromhdf5(file, "alpha")
    qinf  = op2.Const.fromhdf5(file, "qinf")

# Main time-marching loop

niter = 1000

for i in range(1, niter+1):

    # Save old flow solution
    op2.par_loop(save_soln, cells,
                 p_q   (op2.IdentityMap, op2.READ),
                 p_qold(op2.IdentityMap, op2.WRITE))

    # Predictor/corrector update loop
    for k in range(2):

        # Calculate area/timestep
        op2.par_loop(adt_calc, cells,
                     p_x  (pcell,         op2.READ),
                     p_q  (op2.IdentityMap,  op2.READ),
                     p_adt(op2.IdentityMap,  op2.WRITE))

        # Calculate flux residual
        op2.par_loop(res_calc, edges,
                     p_x  (pedge,  op2.READ),
                     p_q  (pecell, op2.READ),
                     p_adt(pecell, op2.READ),
                     p_res(pecell, op2.INC))

        op2.par_loop(bres_calc, bedges,
                     p_x    (pbedge,       op2.READ),
                     p_q    (pbecell(0),      op2.READ),
                     p_adt  (pbecell(0),      op2.READ),
                     p_res  (pbecell(0),      op2.INC),
                     p_bound(op2.IdentityMap, op2.READ))

        # Update flow field
        rms = op2.Global(1, 0.0, np.double, "rms")
        op2.par_loop(update, cells,
                     p_qold(op2.IdentityMap, op2.READ),
                     p_q   (op2.IdentityMap, op2.WRITE),
                     p_res (op2.IdentityMap, op2.RW),
                     p_adt (op2.IdentityMap, op2.READ),
                     rms(op2.INC))
    # Print iteration history
    rms = sqrt(rms.data/cells.size)
    if i%100 == 0:
        print " %d  %10.5e " % (i, rms)
