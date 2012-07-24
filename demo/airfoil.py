# This file is part of PyOP2.
#
# PyOP2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PyOP2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PyOP2.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.

from math import atan, sqrt
import numpy as np

from pyop2 import op2
# Initialise OP2

import h5py

op2.init(backend='sequential')

from airfoil_kernels import save_soln, adt_calc, res_calc, bres_calc, update

file = h5py.File('new_grid.h5', 'r')

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

file.close()

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
                     p_x  (pcell(0),         op2.READ),
                     p_x  (pcell(1),         op2.READ),
                     p_x  (pcell(2),         op2.READ),
                     p_x  (pcell(3),         op2.READ),
                     p_q  (op2.IdentityMap,  op2.READ),
                     p_adt(op2.IdentityMap,  op2.WRITE))

        # Calculate flux residual
        op2.par_loop(res_calc, edges,
                     p_x  (pedge(0),  op2.READ),
                     p_x  (pedge(1),  op2.READ),
                     p_q  (pecell(0), op2.READ),
                     p_q  (pecell(1), op2.READ),
                     p_adt(pecell(0), op2.READ),
                     p_adt(pecell(1), op2.READ),
                     p_res(pecell(0), op2.INC),
                     p_res(pecell(1), op2.INC))

        op2.par_loop(bres_calc, bedges,
                     p_x    (pbedge(0),       op2.READ),
                     p_x    (pbedge(1),       op2.READ),
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
