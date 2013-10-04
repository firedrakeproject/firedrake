#! /usr/bin/env python

# ************************************************************************
# * Copyright 2010 Paulo A. Herrera. All rights reserved.                           *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ************************************************************************

# **************************************************************
# * Example of how to use the low level VtkFile class.         *
# **************************************************************

from evtk.vtk import VtkFile, VtkRectilinearGrid
import numpy as np

nx, ny, nz = 6, 6, 2
lx, ly, lz = 1.0, 1.0, 1.0
dx, dy, dz = lx / nx, ly / ny, lz / nz
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)
x = np.arange(0, lx + 0.1 * dx, dx, dtype='float64')
y = np.arange(0, ly + 0.1 * dy, dy, dtype='float64')
z = np.arange(0, lz + 0.1 * dz, dz, dtype='float64')
start, end = (0, 0, 0), (nx, ny, nz)

w = VtkFile("./evtk_test", VtkRectilinearGrid)
w.openGrid(start=start, end=end)
w.openPiece(start=start, end=end)

# Point data
temp = np.random.rand(npoints)
vx = vy = vz = np.zeros([nx + 1, ny + 1, nz + 1], dtype="float64", order='F')
w.openData("Point", scalars="Temperature", vectors="Velocity")
w.addData("Temperature", temp)
w.addData("Velocity", (vx, vy, vz))
w.closeData("Point")

# Cell data
pressure = np.zeros([nx, ny, nz], dtype="float64", order='F')
w.openData("Cell", scalars="Pressure")
w.addData("Pressure", pressure)
w.closeData("Cell")

# Coordinates of cell vertices
w.openElement("Coordinates")
w.addData("x_coordinates", x)
w.addData("y_coordinates", y)
w.addData("z_coordinates", z)
w.closeElement("Coordinates")

w.closePiece()
w.closeGrid()

w.appendData(data=temp)
w.appendData(data=(vx, vy, vz))
w.appendData(data=pressure)
w.appendData(x).appendData(y).appendData(z)
w.save()
