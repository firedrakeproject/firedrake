from __future__ import absolute_import, print_function, division

import numpy as np

from firedrake.mesh import Mesh
import firedrake.dmplex as dmplex
from firedrake.petsc import PETSc


__all__ = ['adapt']




def adapt(mesh,metric):
    
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = mesh.geometric_dimension()
    coordSection = mesh._plex.createSection([1], entity_dofs, perm=mesh.topology._plex_renumbering)
    
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    nbrVer = vEnd - vStart
    
    dmCoords = mesh.topology._plex.getCoordinateDM()
    dmCoords.setDefaultSection(coordSection)    

    with mesh.coordinates.dat.vec_ro as coords:
        mesh.topology._plex.setCoordinatesLocal(coords)
    with metric.dat.vec_ro as vec:
    	dmplex.sort_metric(plex, vec, coordSection)
        newplex = dmplex.petscAdapt(mesh.topology._plex, vec)

    newmesh = Mesh(newplex)

    return newmesh





#class Interpolate(object):

