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
coords = latlon_grid.reshape(-1, 2)

I, J = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)))
size = len(lon_bnds_flat)
cells = np.dstack([I + J*size, I + (J+1)*size, I+1 + (J+1)*size, I+1 + J*size])
cells = cells.reshape(-1, 4)

from firedrake import *
from firedrake import mesh

plex = mesh._from_cell_list(2, cells, coords)
m = Mesh(plex, dim=2)
print 'lat-lon mesh: OK'

V = FunctionSpace(m, 'DG', 0)
sec = V.topological._global_numbering
reordering = np.array([sec.getOffset(i) for i in xrange(V.node_count)])

tos = rootgrp.variables['tos']
f = Function(V)
f.dat.data[reordering] = tos[0].data.flat

markers = np.empty(tos[0].mask.size, dtype=bool)
markers[reordering] = 1 - tos[0].mask.flatten()
sd = mesh.SubDomainData(m.cell_set, markers, np.unique(markers))

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
    f_.dat.data[reordering] = tos[i].data.flat
    print assemble(f*dx(subdomain_data=sd, subdomain_id=1))
    # out << f_
    # print '.'
del out
