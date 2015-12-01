import numpy as np
# from math import cos, sin, pi
from netCDF4 import Dataset

rootgrp = Dataset("/home/mh1714/tos_O1_ECHAM5_20c3m_r3_1860-2009.nc")

lon = rootgrp.variables['lon']
lon_bnds = rootgrp.variables['lon_bnds']
assert np.allclose(lon, sorted(lon))
assert np.allclose(lon, np.mean(lon_bnds, axis=1))
assert np.allclose(lon_bnds[1:, 0], lon_bnds[:-1, 1])
assert np.allclose(360, lon_bnds[-1, 1] - lon_bnds[0, 0])
print 'longitude: OK'

lat = rootgrp.variables['lat']
lat_bnds = rootgrp.variables['lat_bnds']
assert np.allclose(lat, sorted(lat))
assert np.allclose(lat, np.mean(lat_bnds, axis=1))
assert np.allclose(lat_bnds[1:, 0], lat_bnds[:-1, 1])
assert np.allclose(-90, lat_bnds[0, 0])
assert np.allclose(90, lat_bnds[-1, 1])
print 'latitude: OK'

lon_bnds_flat = np.hstack([lon_bnds[:, 0], lon_bnds[-1, 1]])
lat_bnds_flat = np.hstack([lat_bnds[:, 0], lat_bnds[-1, 1]])

lon_grid, lat_grid = np.meshgrid(lon_bnds_flat, lat_bnds_flat)
latlon_grid = np.dstack([lat_grid, lon_grid])

tos = rootgrp.variables['tos']
assert set(np.sum(tos[:].mask, axis=0).flat) == set([0, len(rootgrp.variables['time'])])
select_mask = np.logical_not(tos[0].mask)

lat_idx, lon_idx = np.where(select_mask)

coords = latlon_grid[np.dstack([lat_idx, lat_idx, lat_idx + 1, lat_idx + 1]).flatten(),
                     np.dstack([lon_idx, lon_idx + 1, lon_idx, lon_idx + 1]).flatten()]
cells = np.arange(np.sum(select_mask)*4).reshape(-1, 4)
cells = cells[:, [0, 1, 3, 2]]

from firedrake import *
from firedrake import mesh

plex = mesh._from_cell_list(2, cells, coords)
m = Mesh(plex, dim=2)
print 'lat-lon mesh: OK'

V = FunctionSpace(m, 'DG', 0)
sec = V.topological._global_numbering
reordering = np.array([sec.getOffset(i) for i in xrange(V.node_count)])
f = Function(V)
f.dat.data[:] = tos[0].data[select_mask][reordering]
print 'time: 0'

lat_ = m.coordinates.dat.data[:, 0] * np.pi / 180
lon_ = m.coordinates.dat.data[:, 1] * np.pi / 180

Vc = VectorFunctionSpace(m, 'Q', 1, dim=3)
new_coords = Function(Vc)
new_coords.dat.data[:] = np.dstack([np.cos(lat_)*np.cos(lon_),
                                    np.cos(lat_)*np.sin(lon_),
                                    np.sin(lat_)]).reshape(-1, 3)

m_ = Mesh(new_coords)
f_ = Function(functionspace.WithGeometry(f.function_space(), m_), val=f.topological)
print 'change of coordinates'

time = rootgrp.variables['time']
out = File("f.pvd")
for i in xrange(len(time)):
    # print 'Time:', time[i],
    f_.dat.data[:] = tos[i].data[select_mask][reordering]
    print assemble(f*dx)
    # out << f_
    # print '.'
del out
