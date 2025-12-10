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

            size = axis_tree.component_size(path, component.label)

            if not isinstance(size, numbers.Integral):
                raise NotImplementedError("Assume that star forests have integer size")

            sf = StarForest(petsc_sf)
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


# perhaps I can defer renumbering the SF to here?
# PETSc provides a similar function that composes an SF with a Section, can I use that?
# def grow_dof_sf(axes, axis, path, indices):
#     from pyop3.axtree.layout import step_size
#
#     point_sf = axis.sf
#     # TODO, use convenience methods
#     nroots, ilocal, iremote = point_sf._graph
#
#     component_counts = tuple(c.count for c in axis.components)
#     component_offsets = [0] + list(np.cumsum(component_counts))
#     npoints = component_offsets[-1]
#
#     # renumbering per component, can skip if no renumbering present
#     if axis.numbering is not None:
#         renumbering = [np.empty(c.count, dtype=int) for c in axis.components]
#         counters = [0] * len(axis.components)
#         for new_pt, old_pt in enumerate(axis.numbering.data_ro):
#             for cidx, (min_, max_) in enumerate(
#                 zip(component_offsets, component_offsets[1:])
#             ):
#                 if min_ <= old_pt < max_:
#                     renumbering[cidx][old_pt - min_] = counters[cidx]
#                     counters[cidx] += 1
#                     break
#         assert all(
#             count == c.count for count, c in checked_zip(counters, axis.components)
#         )
#     else:
#         renumbering = [np.arange(c.count, dtype=int) for c in axis.components]
#
#     # effectively build the section
#     new_nroots = 0
#     root_offsets = np.full(npoints, -1, IntType)
#     for pt in point_sf.iroot:
#         # convert to a component-wise numbering
#         selected_component = None
#         component_num = None
#         for cidx, (min_, max_) in enumerate(
#             zip(component_offsets, component_offsets[1:])
#         ):
#             if min_ <= pt < max_:
#                 selected_component = axis.components[cidx]
#                 component_num = renumbering[cidx][pt - component_offsets[cidx]]
#                 break
#         assert selected_component is not None
#         assert component_num is not None
#
#         offset = axes.offset(
#             indices | {axis.label: component_num},
#             path | {axis.label: selected_component.label},
#         )
#         root_offsets[pt] = offset
#         new_nroots += step_size(
#             axes,
#             axis,
#             selected_component,
#             indices | {axis.label: component_num},
#         )
#
#     point_sf.broadcast(root_offsets, MPI.REPLACE)
#
#     # for sanity reasons remove the original root values from the buffer
#     root_offsets[point_sf.iroot] = -1
#
#     local_leaf_offsets = np.empty(point_sf.nleaves, dtype=IntType)
#     leaf_ndofs = local_leaf_offsets.copy()
#     for myindex, pt in enumerate(ilocal):
#         # convert to a component-wise numbering
#         selected_component = None
#         component_num = None
#         for cidx, (min_, max_) in enumerate(
#             zip(component_offsets, component_offsets[1:])
#         ):
#             if min_ <= pt < max_:
#                 selected_component = axis.components[cidx]
#                 component_num = renumbering[cidx][pt - component_offsets[cidx]]
#                 break
#         assert selected_component is not None
#         assert component_num is not None
#
#         # this is wrong?
#         offset = axes.offset(
#             indices | {axis.label: component_num},
#             path | {axis.label: selected_component.label},
#         )
#         local_leaf_offsets[myindex] = offset
#         leaf_ndofs[myindex] = step_size(axes, axis, selected_component)
#
#     # construct a new SF with these offsets
#     ndofs = sum(leaf_ndofs)
#     local_leaf_dof_offsets = np.empty(ndofs, dtype=IntType)
#     remote_leaf_dof_offsets = np.empty((ndofs, 2), dtype=IntType)
#     counter = 0
#     for leaf, pos in enumerate(point_sf.ilocal):
#         for d in range(leaf_ndofs[leaf]):
#             local_leaf_dof_offsets[counter] = local_leaf_offsets[leaf] + d
#
#             rank = point_sf.iremote[leaf][0]
#             remote_leaf_dof_offsets[counter] = [rank, root_offsets[pos] + d]
#             counter += 1
#
#     return (new_nroots, local_leaf_dof_offsets, remote_leaf_dof_offsets)
