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

from airfoil_kernels import save_soln, adt_calc, res_calc, bres_calc, update

### These need to be set by some sort of grid-reading later

# Size of sets
ncell  = 800
nnode  = 1000
nedge  = 500
nbedge = 40

# Map values
cell   = np.array([1]*4*ncell)
edge   = np.array([1]*2*nedge)
ecell  = np.array([1]*2*nedge)
bedge  = np.array([1]*2*nbedge)
becell = np.array([1]*  nbedge)
bound  = np.array([1]*  nbedge)

# Data values
x     = np.array([1.0]*2*nnode)
q     = np.array([1.0]*4*ncell)
qold  = np.array([1.0]*4*ncell)
res   = np.array([1.0]*4*ncell)
adt   = np.array([1.0]*  ncell)

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

p_bound = op2.Dat(bedges, 1, np.long, bound, "p_bound")
p_x     = op2.Dat(nodes,  2, np.double, x,     "p_x")
p_q     = op2.Dat(cells,  4, np.double, q,     "p_q")
p_qold  = op2.Dat(cells,  4, np.double, qold,  "p_qold")
p_adt   = op2.Dat(cells,  1, np.double, adt,   "p_adt")
p_res   = op2.Dat(cells,  4, np.double, res,   "p_res")

gam  = op2.Const(1, 1.4, "gam")
gm1  = op2.Const(1, 0.4, "gm1")
cfl  = op2.Const(1, 0.9, "cfl")
eps  = op2.Const(1, 0.05, "eps")
mach = op2.Const(1, 0.4, "mach")

alpha = op2.Const(1, 3.0*atan(1.0)/45.0, "alpha")

# Constants
p = 1.0
r = 1.0
u = sqrt(1.4/p/r)*0.4
e = p/(r*0.4) + 0.5*u*u

qinf = op2.Const(4, [r, r*u, 0.0, r*e], "qinf")

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
        rms = op2.Global(1, 0, "rms")
        op2.par_loop(update, cells,
                     p_qold(op2.IdentityMap, op2.READ),
                     p_q   (op2.IdentityMap, op2.WRITE),
                     p_res (op2.IdentityMap, op2.RW),
                     p_adt (op2.IdentityMap, op2.READ),
                     rms(op2.INC))

    # Print iteration history
    rms = sqrt(rms.value/cells.size)
    if i%100 == 0:
        print "Iteration", i, "RMS:", rms
