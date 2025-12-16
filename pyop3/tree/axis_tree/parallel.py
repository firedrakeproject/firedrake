from __future__ import annotations

import functools
import numbers
from collections.abc import Sequence

import numpy as np
from immutabledict import immutabledict
from mpi4py import MPI
from pyop3.tree.axis_tree.tree import AbstractAxisTree

from pyop3 import utils
from pyop3.dtypes import IntType, as_numpy_dtype
from pyop3.sf import StarForest, _check_sf, create_petsc_section_sf


def reduction_op(op, invec, inoutvec, datatype):
    dtype = as_numpy_dtype(datatype)
    invec = np.frombuffer(invec, dtype=dtype)
    inoutvec = np.frombuffer(inoutvec, dtype=dtype)
    inoutvec[:] = op(invec, inoutvec)


_contig_min_op = MPI.Op.Create(
    functools.partial(reduction_op, np.minimum), commute=True
)
_contig_max_op = MPI.Op.Create(
    functools.partial(reduction_op, np.maximum), commute=True
)


def partition_ghost_points(axis, sf):
    npoints = sf.size
    is_owned = np.full(npoints, True, dtype=bool)
    is_owned[sf.ileaf] = False

    component_owned_sizes = [0] * len(axis.components)
    numbering = np.empty(npoints, dtype=IntType)
    owned_ptr = 0
    ghost_ptr = npoints - sf.nleaves
    points = axis.numbering.data_ro if axis.numbering is not None else range(npoints)
    for pt in points:
        if is_owned[pt]:
            component_index = axis._axis_number_to_component_index(pt)
            component_owned_sizes[component_index] += 1

            numbering[owned_ptr] = pt
            owned_ptr += 1
        else:
            numbering[ghost_ptr] = pt
            ghost_ptr += 1

    assert owned_ptr == npoints - sf.nleaves
    assert ghost_ptr == npoints
    return component_owned_sizes, numbering


def collect_star_forests(axis_tree: AbstractAxisTree) -> tuple[StarForest, ...]:
    return _collect_sf_graphs_rec(axis_tree, immutabledict())


# NOTE: This function does not check for nested SFs
def _collect_sf_graphs_rec(axis_tree: AbstractAxisTree, path: ConcretePathT) -> tuple[StarForest, ...]:
    axis = axis_tree.node_map[path]

    sfs = []
    for component in axis.components:
        path_ = path | {axis.label: component.label}

        if component.sf is not None:
            # do not recurse further
            if path_ in axis_tree.node_map:
                section = axis_tree.section(path, component)
                petsc_sf = create_petsc_section_sf(component.sf.sf, section)
                _check_sf(petsc_sf)
            else:
                petsc_sf = component.sf.sf
            sf = StarForest(petsc_sf, component.sf.comm)
            sfs.append(sf)
        elif subaxis := axis_tree.node_map.get(path_):
            if isinstance(size := component.size, numbers.Integral) and size > 1:
                raise NotImplementedError("This will be very inefficient")

            # FIXME: Only need to call the inner bit once and repeatedly add?
            for point in range(component.local_size):
                sfs.extend(
                    _collect_sf_graphs_rec(axis_tree, path_)
                )
    return tuple(sfs)


def concatenate_star_forests(star_forests: Sequence[StarForest]) -> StarForest:
    """Combine multiple star forests keeping leaf entries at the end.

    Example
    -------
    Before:

        rank 0:

            size: 9
            ilocal0: [3, 4, 5, 6, 7, 8]
            iremote0: [[1, 3], [1, 4], [1, 0], [1, 1], [1, 5], [1, 2]]

            size: 4
            ilocal1: [1, 2, 3]
            iremote1: [[1, 0], [1, 2], [1, 1]]

        rank 1:

            size: 9
            ilocal0: [6, 7, 8]
            iremote0: [[0, 0], [0, 1], [0, 2]]

            size: 4
            ilocal1: [3]
            iremote1: [[0, 0]]

    After:

        rank 0:

            size: 13
            ilocal: [ 4,  5,  6,  7,  8,  9, 10, 11, 12]
            iremote: [[1, 3], [1, 4], [1, 0], [1, 1], [1, 5], [1, 2], [1, 6], [1, 8], [1, 7]]

        rank 1:

            size: 13
            ilocal: [9, 10, 11, 12]
            iremote: [[0, 0], [0, 1], [0, 2], [0, 3]]

    """
    total_size = sum(sf.size for sf in star_forests)

    local_leaf_indicess = []
    remote_leaf_indicess = []
    total_num_owned = sum(sf.num_owned for sf in star_forests)
    local_leaf_index_start = total_num_owned
    start = 0
    for sf in star_forests:
        size, local_leaf_indices, remote_leaf_indices = sf.graph
        new_local_leaf_indices = local_leaf_indices - sf.num_owned + local_leaf_index_start

        new_offsets = np.arange(start, start+size, dtype=IntType)
        sf.broadcast(new_offsets, MPI.REPLACE)
        new_remote_leaf_indices = new_offsets[sf.num_owned:]

        # but PETSc expects rank information along with the remote indices
        new_remote_leaf_indices = np.stack([remote_leaf_indices[:, 0], new_remote_leaf_indices], axis=1)

        local_leaf_indicess.append(new_local_leaf_indices)
        remote_leaf_indicess.append(new_remote_leaf_indices)

        start += sf.num_owned
        local_leaf_index_start += sf.num_ghost
    assert start == total_num_owned

    ilocal = np.concatenate(local_leaf_indicess)
    iremote = np.concatenate(remote_leaf_indicess)
    comm = utils.single_comm(star_forests, "comm")
    return StarForest.from_graph(total_size, ilocal, iremote, comm)
