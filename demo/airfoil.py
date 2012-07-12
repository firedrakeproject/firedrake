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

# Size of sets
ncell  = file['cells'].value[0].astype('int')
nnode  = file['nodes'].value[0].astype('int')
nedge  = file['edges'].value[0].astype('int')
nbedge = file['bedges'].value[0].astype('int')

# Map values
cell   = file['pcell'].value
edge   = file['pedge'].value
ecell  = file['pecell'].value
bedge  = file['pbedge'].value
becell = file['pbecell'].value

# Data values
bound = file['p_bound'].value
x     = file['p_x'].value
q     = file['p_q'].value
qold  = file['p_qold'].value
res   = file['p_res'].value
adt   = file['p_adt'].value

### End of grid stuff


# Declare sets, maps, datasets and global constants

nodes  = op2.Set(nnode, "nodes")
edges  = op2.Set(nedge, "edges")
bedges = op2.Set(nbedge, "bedges")
cells  = op2.Set(ncell, "cells")

pedge   = op2.Map(edges,  nodes, 2, edge,   "pedge")
pecell  = op2.Map(edges,  cells, 2, ecell,  "pecell")
pbedge  = op2.Map(bedges, nodes, 2, bedge,  "pbedge")
pbecell = op2.Map(bedges, cells, 1, becell, "pbecell")
pcell   = op2.Map(cells,  nodes, 4, cell,   "pcell")

p_bound = op2.Dat(bedges, 1, bound, name="p_bound")
p_x     = op2.Dat(nodes,  2, x,     name="p_x")
p_q     = op2.Dat(cells,  4, q,     name="p_q")
p_qold  = op2.Dat(cells,  4, qold,  name="p_qold")
p_adt   = op2.Dat(cells,  1, adt,   name="p_adt")
p_res   = op2.Dat(cells,  4, res,   name="p_res")

gam   = op2.Const(1, file['gam'].value,   name="gam")
gm1   = op2.Const(1, file['gm1'].value,   name="gm1")
cfl   = op2.Const(1, file['cfl'].value,   name="cfl")
eps   = op2.Const(1, file['eps'].value,   name="eps")
mach  = op2.Const(1, file['mach'].value,  name="mach")
alpha = op2.Const(1, file['alpha'].value, name="alpha")
qinf  = op2.Const(4, file['qinf'].value,  name="qinf")

file.close()

# Main time-marching loop

niter = 1000

for i in range(niter):

    # Save old flow solution
    op2.par_loop(save_soln, cells,
                 p_q   (op2.IdentityMap, op2.READ),
                 p_qold(op2.IdentityMap, op2.WRITE))

    # Predictor/corrector update loop
    for k in range(2):

        # Calculate area/timestep
        op2.par_loop(adt_calc, cells,
                     p_x  (pcell,            op2.READ),
                     p_q  (op2.IdentityMap,  op2.READ),
                     p_adt(op2.IdentityMap,  op2.WRITE))

        # Calculate flux residual
        op2.par_loop(res_calc, edges,
                     p_x  (pedge,  op2.READ),
                     p_q  (pecell, op2.READ),
                     p_adt(pecell, op2.READ),
                     p_res(pecell, op2.INC))

        op2.par_loop(bres_calc, bedges,
                     p_x    (pbedge,          op2.READ),
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
        print "Iteration", i, "RMS:", rms
