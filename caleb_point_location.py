from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from pyop2.datatypes import IntType, ScalarType
import numpy as np

from random import randint


def build_processes_spatial_index(coords, comm):
    # `axis` refers to direction of the operation wrt the NumPy array. Here, a
    # value of 0 means that the output is a list returning the minimum/maximum
    # values of each column (1 would give the same of each row).
    # min/max_c has the format: [x, y, z]
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)

    # Create a numpy array combining both the min and max bounding boxes.
    # New format: [min_x, min_y, min_z, max_x, max_y, max_z]
    local = np.concatenate([min_c, max_c])

    # Create an empty global array that has the size of of the local array
    # multiplied by the world size (number of processes). All local arrays
    # are the same size as they contain only the min/max values for each
    # dimension.
    global_ = np.empty(len(local) * comm.size, dtype=local.dtype)

    # Must build spatial index for processes
    # spatialindex.from_regions(coords_min, coords_max)
    #     - Unpicking from along min axis and max axis (min_c, max_c, local)
    #     - ...
    # Perform `MPI_Allgather` to combine all local arrays into global array.
    comm.Allgather(local, global_)

    # Create lists containing the minimum and maximum bounds of each process,
    # where the index in each list is the rank of the process.
    min_bounds = []
    max_bounds = []
    arr_size = int(len(local) / 2)
    for i in range(0, comm.size):
        j = arr_size * 2 * i
        min_bounds.append(global_[j:j + arr_size])
        max_bounds.append(global_[j + arr_size:j + arr_size * 2])

    # Build spatial indexes from bounds.
    return spatialindex.from_regions(np.array(min_bounds),
                                     np.array(max_bounds))


def get_candidate_processes(rank, processes_index, point):
    candidates = spatialindex.bounding_boxes(processes_index, point)
    i = -1
    for j, candidate in enumerate(candidates):
        if (candidate == comm.rank):
            i = j
            break
    if (i != -1):
        candidates = np.delete(candidates, i)
    return candidates


def locate_cells(mesh, points):
    results = []
    for point in points:
        results.append(mesh.locate_cell(point))
    return results


def perform_sparse_communication_round(comm, recv_buffer_dict, recv_processes,
                                       send_processes):
    # Create lists to hold send and receive communication request objects.
    recv_reqs = []
    send_reqs = []

    # Create all output buffers for each point to be received and prepare to
    # receive.
    for rank, num_points in recv_processes.items():
        # Create all receive requests.
        for i in range(0, num_points):
            req = comm.Irecv(recv_buffer_dict[rank][i], source=rank, tag=i)
            recv_reqs.append(req)

    # Now send all of the point requests we wish to send for this process.

    for rank, points in send_processes.items():
        for i, point_data in enumerate(points):
            # Send the request for this point.
            req = comm.Isend(point_data, dest=rank, tag=i)
            # Wait on request?
            send_reqs.append(req)

    MPI.Request.Waitall(recv_reqs + send_reqs)


def locate_mesh_elements(mesh, processes_index, input_points):
    # Determine whether or not each of the points is held locally. Otherwise
    # find candidate processes that may contain that point.

    comm = mesh.comm

    located_elements = {}
    local_candidates = {}

    local_results = locate_cells(mesh, input_points)

    for i, point in enumerate(input_points):
        result = local_results[i]
        if result is not None:
            # Point found on some local element.
            located_elements[point.tobytes()] = (comm.rank, result)
            PETSc.Sys.syncPrint("[%d] Cell found locally for point %s. Cell: "
                                "%s" % (comm.rank, point, result), comm=comm)
        else:
            # Point not found locally.
            # Get candidates from processes spatial index.
            point_candidates = get_candidate_processes(comm.rank,
                                                       processes_index, point)
            for candidate in point_candidates:
                if candidate in local_candidates:
                    local_candidates[candidate].append(point)
                else:
                    local_candidates[candidate] = [point]

            PETSc.Sys.syncPrint("[%d] Cell not found locally for point %s. "
                                "Candidates: %s" %
                                (comm.rank, point, point_candidates),
                                comm=comm)

    PETSc.Sys.syncFlush(comm=comm)

    PETSc.Sys.syncPrint("[%d] Located elements: %s" %
                        (comm.rank, located_elements), comm=comm)
    PETSc.Sys.syncFlush(comm=comm)
    PETSc.Sys.syncPrint("[%d] Local candidates: %s" %
                        (comm.rank, local_candidates), comm=comm)
    PETSc.Sys.syncFlush(comm=comm)

    # ---

    # Get number of points to receive from each rank through sparse
    # communication round.

    # Create input arrays from candidates dictionary.
    to_ranks = []
    to_data = []
    for rank, points in local_candidates.items():
        to_ranks.append(rank)
        to_data.append(len(points))

    # `build_two_sided()` provides an interface for PetscCommBuildTwoSided,
    # which facilitates a sparse communication round between the processes to
    # identify which processes will be sending points and how many points they
    # wish to send.
    # The output array `from_ranks` holds the ranks of the processes that will
    # be sending points, and the corresponding element in the `from_data` array
    # specifies the number of points that will be sent.
    from_ranks, from_data = dmplex.build_two_sided(
        comm, 1, MPI.INT, np.array(to_ranks, dtype=IntType),
        np.array(to_data, dtype=IntType))

    # Create dictionary to hold number of points to receive from each process.
    recv_points_counts = {}
    # Create dictionary to hold all receive buffers for point requests from
    # each process.
    recv_points_buffers = {}

    # Populate dictionaries.
    for i in range(0, len(from_ranks)):
        recv_points_counts[from_ranks[i]] = from_data[i]
        recv_points_buffers[from_ranks[i]] = np.array(
            [np.empty(mesh.cell_dimension(), dtype=ScalarType)
             for _ in range(from_data[i])])

    # ---

    # Receive all point requests

    perform_sparse_communication_round(comm, recv_points_buffers,
                                       recv_points_counts, local_candidates)

    PETSc.Sys.syncPrint("[%d] Point queries requested: %s"
                        % (comm.rank, str(recv_points_buffers)))
    PETSc.Sys.syncFlush(comm=comm)

    # ---

    # Evaluate all point requests and prepare responses

    # Create dictionary to store results.
    point_responses = {}

    # Evaluate results.
    for rank, num_points in recv_points_counts.items():
        point_responses[rank] = []
        for i in range(0, num_points):
            req_cell = mesh.locate_cell(recv_points_buffers[rank][i])
            if (req_cell is not None):
                point_responses[rank].append(np.array([req_cell],
                                                      dtype=IntType))
            else:
                point_responses[rank].append(np.array([-1], dtype=IntType))

    PETSc.Sys.syncPrint("[%d] Point responses: %s"
                        % (comm.rank, str(point_responses)))
    PETSc.Sys.syncFlush(comm=comm)

    # ---

    # Receive all responses

    # Create dictionary to hold all output buffers indexed by rank.
    recv_results = {}
    # New dictionary with number of points for local candidates.
    local_candidates_lengths = {}
    # Initialise these.
    for rank, points in local_candidates.items():
        num_points = len(points)
        local_candidates_lengths[rank] = num_points
        # Create receive buffer(s).
        recv_results[rank] = np.array([np.array([-1], dtype=IntType)
                                       for _ in range(num_points)])

    perform_sparse_communication_round(comm, recv_results,
                                       local_candidates_lengths,
                                       point_responses)

    PETSc.Sys.syncPrint("[%d] Point location request results: %s"
                        % (comm.rank, str(recv_results)))
    PETSc.Sys.syncFlush(comm=comm)

    # ---

    # Process results

    for rank, results in recv_results.items():
        for i, result in enumerate(results):
            PETSc.Sys.syncPrint("[%d] Result %d from rank %d for point %s: %s"
                                % (comm.rank, i, rank,
                                   local_candidates[rank][i], str(result)))
            if (not np.array_equal([-1], result)):
                if local_candidates[rank][i].tobytes() in located_elements:
                    PETSc.Sys.syncPrint("[%d] COLLISION" % (comm.rank))
                located_elements[local_candidates[rank][i].tobytes()] = (
                    rank, result[0])
    PETSc.Sys.syncFlush(comm=comm)

    # ---

    # Format and return results.

    PETSc.Sys.syncPrint("[%d] located_elements: %s"
                        % (comm.rank, str(located_elements)))
    PETSc.Sys.syncFlush(comm=comm)

    result_list = []
    for point in input_points:
        if (point.tobytes() in located_elements):
            result_list.append(located_elements[point.tobytes()])
        else:
            result_list.append((None, None))

    PETSc.Sys.syncPrint("[%d] Final results: %s"
                        % (comm.rank, str(result_list)))
    PETSc.Sys.syncFlush(comm=comm)

    return np.array(result_list)


if (__name__ == "__main__"):
    # Create an example mesh.
    # UnitCubeMesh is defined in `firedrake/utility_meshes.py`.
    mesh = UnitCubeMesh(4, 4, 4)

    # `data_ro` is a numpy array containing mesh coordinate data values. This
    # comes from PyOP2, which provides a `data_ro` property in `pyop2/base.py`.
    coords = mesh.coordinates.dat.data_ro

    # Defined in `HierarchyBase` class in `mesh.py`. `mesh.comm` returns the
    # `comm` object from the coarsest mesh in `HierarchyBase`'s mesh hierarchy.
    # The `comm` object is an MPI communicator provided by `pyop2.mpi`. This
    # communicator object defaults to `MPI_COMM_WORLD`, as no communicator is
    # specified.
    comm = mesh.comm

    # Build spatial index of processes for point location.
    processes_index = build_processes_spatial_index(coords, comm)

    # Locate a list of points.
    # Generate a list of two points for each process for testing.
    # Points for first 3 processes are fixed.
    _points = []
    if (comm.rank == 0):
        _points = [[0.6, 0.1, 0.6], [0.3, 0.2, 1.0]]
    elif (comm.rank == 1):
        _points = [[0.5, 0.6, 1.0], [0.6, 0.2, 0.6]]
    elif (comm.rank == 2):
        _points = [[0.5, 0.7, 0.7], [0.3, 0.5, 0.4]]
    else:
        for _ in range(0, 2):
            _x = randint(0, 10) / 10
            _y = randint(0, 10) / 10
            _z = randint(0, 10) / 10
            _points.append([_x, _y, _z])

    PETSc.Sys.syncPrint("[%d] Locating points: %s" %
                        (comm.rank, _points), comm=comm)
    PETSc.Sys.syncFlush(comm=comm)

    points = np.array(_points)
    target_process = locate_mesh_elements(mesh, processes_index, points)
