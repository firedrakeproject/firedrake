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

from op2 import *
from math import atan, sqrt

### These need to be set by some sort of grid-reading later
# Size of sets
nnode = 1000
nedge = 500
nbedge = 40
ncell = 800

# Map values
edge = None
ecell = None
bedge = None
becell = None
cell = None

# Data values
bound = None
x = None
q = None
qold = None
adt = None
res = None

### End of grid stuff

# Declare sets, maps, datasets and global constants

nodes  = Set(nnode, "nodes")
edges  = Set(nedge, "edges")
bedges = Set(nbedge, "bedges")
cells  = Set(ncell, "cells")

pedge   = Map(edges,  nodes, 2, edge,   "pedge")
pecell  = Map(edges,  cells, 2, ecell,  "pecell")
pbedge  = Map(bedges, nodes, 2, bedge,  "pbedge")
pbecell = Map(bedges, cells, 1, becell, "pbecell")
pcell   = Map(cells,  nodes, 4, cell,   "pcell")

p_bound = Dat(bedges, 1, "int",    bound, "p_bound")
p_x     = Dat(nodes,  2, "double", x,     "p_x")
p_q     = Dat(cells,  4, "double", q,     "p_q")
p_qold  = Dat(cells,  4, "double", qold,  "p_qold")
p_adt   = Dat(cells,  1, "double", adt,   "p_adt")
p_res   = Dat(cells,  4, "double", res,   "p_res")

gam = Const(1, "double", 1.4, "gam")
gm1 = Const(1, "double", 0.4, "gm1")
cfl = Const(1, "double", 0.9, "cfl")
eps = Const(1, "double", 0.05, "eps")
mach = Const(1, "double", 0.4, "mach")

alpha = Const(1, "double", 3.0*atan(1.0)/45.0, "alpha")

# Values derived from original airfoil - could be tidied up when we've figured
# out the API
p = 1.0
r = 1.0
u = sqrt(1.4/p/r)*0.4
e = p/(r*0.4) + 0.5*u*u

qinf = Const(4, "double", [r, r*u, 0.0, r*e], "qinf")

# Kernels - need populating with code later
save_soln = Kernel("save_soln", None)
adt_calc  = Kernel("adt_calc",  None)
res_calc  = Kernel("res_calc",  None)
bres_calc = Kernel("bres_calc", None)
update    = Kernel("update",    None)


# Main time-marching loop

niter = 1000

for i in range(niter):

    # Save old flow solution
    par_loop(save_soln, cells,
             ArgDat(p_q,    None, None, read),
             ArgDat(p_qold, None, None, write))

    # Predictor/corrector update loop
    for k in range(2):

        # Calculate area/timestep
        par_loop(adt_calc, cells,
                 ArgDat(p_x,   idx_all, pedge, read),
                 ArgDat(p_q,   None,    None,  read),
                 ArgDat(p_adt, None,    None,  write))

        # Calculate flux residual
        par_loop(res_calc, edges,
                 ArgDat(p_x,   idx_all, pedge,  read),
                 ArgDat(p_q,   idx_all, pecell, read),
                 ArgDat(p_adt, idx_all, pecell, read),
                 ArgDat(p_res, idx_all, pecell, inc))

        par_loop(bres_calc, bedges,
                 ArgDat(p_x,     idx_all, pbedge,  read),
                 ArgDat(p_q,     0,       pbecell, read),
                 ArgDat(p_adt,   0,       pbecell, read),
                 ArgDat(p_res,   0,       pbecell, inc),
                 ArgDat(p_bound, None,    None,    read))

        # Update flow field
        rms = Global("rms", val=0)
        par_loop(update, cells,
                 ArgDat(p_qold, None, None, read),
                 ArgDat(p_q,    None, None, write),
                 ArgDat(p_res,  None, None, rw),
                 ArgDat(p_adt,  None, None, read),
                 ArgGbl(rms, inc))

    # Print iteration history
    rms = sqrt(rms.val()/cells.size())
    if i%100 == 0:
        print "Iteration", i, "RMS:", rms

