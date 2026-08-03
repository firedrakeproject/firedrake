"""Use from functionspaceimpl.py"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import firedrake.cython.dmcommon
import firedrake.mesh
from firedrake.utils import IntType

import pyop3.debug


def _get_ndofs_extruded(mesh: firedrake.mesh.MeshGeometry, entity_dofs: dict) -> np.ndarray:
    ndofs = np.empty(mesh.num_points, dtype=IntType)

    dm = mesh.topology_dm
    dim_label = dm.getLabel("depth")
    base_dim_label = dm.getLabel("base_dim")

    for base_dim in base_dim_label.getValueIS().indices:
        matching_pts = base_dim_label.getStratumIS(base_dim)
        vertex_pts = np.intersect1d(
            matching_pts.indices,
            mesh._plex_indices_for_dim((base_dim, 0)).indices,
            assume_unique=True,
        )
        edge_pts = np.intersect1d(
            matching_pts.indices,
            mesh._plex_indices_for_dim((base_dim, 1)).indices,
            assume_unique=True,
        )
        ndofs[vertex_pts] = entity_dofs[base_dim, 0]
        ndofs[edge_pts] = entity_dofs[base_dim, 1]

    return ndofs


def _partition_constrained_points(
    mesh: firedrake.mesh.MeshGeometry,
    ndofs: np.ndarray,
    boundary_set,
):
    """Split a section into unconstrained and constrained sets."""
    constrained_pts_is = firedrake.cython.dmcommon.get_boundary_set_points(
        mesh.topology_dm,
        boundary_set,
        isinstance(mesh.topology, firedrake.mesh.ExtrudedMeshTopology),
    )
    constrained_pts = constrained_pts_is.indices
    free_pts = np.setdiff1d(
        np.arange(mesh.num_points, dtype=IntType),
        constrained_pts,
        assume_unique=True,
    )

    # These arrays map plex points to number of DoFs, split by whether the point
    # is constrained or not
    num_free_dofs = np.zeros(mesh.num_points, dtype=IntType)
    num_constrained_dofs = np.zeros_like(num_free_dofs)

    num_constrained_dofs[constrained_pts] = ndofs[constrained_pts]
    num_free_dofs[free_pts] = ndofs[free_pts]

    # Finally reorder the arrays because we have a different numbering
    perm = mesh._new_to_old_point_renumbering.indices
    return num_free_dofs[perm], num_constrained_dofs[perm]
