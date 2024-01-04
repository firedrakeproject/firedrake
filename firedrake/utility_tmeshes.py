import numpy as np

import ufl

from pyop2.mpi import COMM_WORLD, temp_internal_comm
from firedrake.utils import IntType, RealType, ScalarType

from firedrake import (
    VectorFunctionSpace,
    Function,
    Constant,
    par_loop,
    dx,
    WRITE,
    READ,
    interpolate,
    FiniteElement,
    interval,
    tetrahedron,
)
from firedrake.cython import tmeshgen
from firedrake import mesh
from firedrake import function
from firedrake import functionspace
from firedrake.petsc import PETSc


__all__ = [
    "TMeshAnnulus",
]


@PETSc.Log.EventDecorator()
def TMeshAnnulus(
    refinement_level=0,
    distribution_parameters=None,
    reorder=None,
    comm=COMM_WORLD,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
):
    """
    Generate a uniform mesh of an interval.

    :arg ncells: The number of the cells over the interval.
    :arg length_or_left: The length of the interval (if ``right``
         is not provided) or else the left hand boundary point.
    :arg right: (optional) position of the right
         boundary point (in which case ``length_or_left`` should
         be the left boundary point).
    :kwarg distribution_parameters: options controlling mesh
           distribution, see :func:`.Mesh` for details.
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on.
    :kwarg name: Optional name of the mesh.
    :kwarg distribution_name: the name of parallel distribution used
           when checkpointing; if `None`, the name is automatically
           generated.
    :kwarg permutation_name: the name of entity permutation (reordering) used
           when checkpointing; if `None`, the name is automatically
           generated.

    The left hand boundary point has boundary marker 1,
    while the right hand point has marker 2.
    """
    plex = tmeshgen._tmesh_annulus(comm)
    plex.setName(mesh._generate_default_mesh_topology_name(name))
    plex.viewFromOptions("-dm_view")
    #m = mesh.Mesh(
    #    plex,
    #    reorder=reorder,
    #    distribution_parameters=distribution_parameters,
    #    name=name,
    #    distribution_name=distribution_name,
    #    permutation_name=permutation_name,
    #    comm=comm,
    #)
    #return m





