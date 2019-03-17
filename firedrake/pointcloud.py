from firedrake.mesh import spatialindex
from firedrake.dmplex import build_two_sided
from firedrake.utils import cached_property
from firedrake.petsc import PETSc
from firedrake import logging
from pyop2.mpi import MPI
from pyop2.datatypes import IntType, ScalarType
import numpy as np


__all__ = ["PointCloud"]


def syncPrint(*args, **kwargs):
    """Perform a PETSc syncPrint operation with given arguments if the logging level is
    set to at least debug.
    """
    if logging.logger.isEnabledFor(logging.DEBUG):
        PETSc.Sys.syncPrint(*args, **kwargs)


def syncFlush(*args, **kwargs):
    """Perform a PETSc syncFlush operation with given arguments if the logging level is
    set to at least debug.
    """
    if logging.logger.isEnabledFor(logging.DEBUG):
        PETSc.Sys.syncFlush(*args, **kwargs)


class PointCloud(object):
    """Store points for repeated location in a mesh. Facilitates lookup and evaluation
    at these point locations.
    """
    def __init__(self, mesh, points, *args, **kwargs):
        """Initialise the PointCloud.

        :arg mesh: A mesh object.
        :arg points: An N x mesh.geometric_dimension() array of point locations.
        """
        self.mesh = mesh
        self.points = np.asarray(points)
        _, dim = points.shape
        if dim != mesh.geometric_dimension():
            raise ValueError("Points must be %d-dimensional, (got %d)" %
                             (mesh.geometric_dimension(), dim))

        # Build spatial index of processes for point location.
        self.processes_index = self._build_processes_spatial_index()

    @cached_property
    def locations(self):
        """Determine the process rank and element that holds each input point.

        The location algorithm works as follows:
        1. Query local spatial index for list of input points (`self.points`) to determine
           if each is held locally.
        2. For each point not found locally, query spatial index of processes to
           identify those that may contain it. Location requests will be sent to each
           process for the points they may contain.
        3. Perform communication round so that each process knows how many points it will
           receive from the other processes.
        4. Perform sparse communication round to receive points.
        5. Lookup these points in local spatial index and obtain result.
        6. Perform sparse communication round to return results.
        7. Process responses to obtain final results array. If multiple processes report
           containing the same point, choose the process with the lower rank.

        :returns: An array of (rank, cell number) pairs; (-1, -1) if not found.
        """
        rank_cell_pairs = self._locate_mesh_elements()
        return np.array(rank_cell_pairs, dtype=IntType)

    def _build_processes_spatial_index(self):
        """Build a spatial index of processes using the bounding boxes of each process.
        This will be used to determine which processes may hold a given point.

        :returns: A libspatialindex spatial index structure.
        """
        min_c = self.mesh.coordinates.dat.data_ro.min(axis=0)
        max_c = self.mesh.coordinates.dat.data_ro.max(axis=0)

        # Format: [min_x, min_y, min_z, max_x, max_y, max_z]
        local = np.concatenate([min_c, max_c])

        global_ = np.empty(len(local) * self.mesh.comm.size, dtype=local.dtype)
        self.mesh.comm.Allgather(local, global_)

        # Create lists containing the minimum and maximum bounds of each process, where
        # the index in each list is the rank of the process.
        min_bounds, max_bounds = global_.reshape(self.mesh.comm.size, 2,
                                                 len(local) // 2).swapaxes(0, 1)

        # Arrays must be contiguous.
        min_bounds = np.ascontiguousarray(min_bounds)
        max_bounds = np.ascontiguousarray(max_bounds)

        # Build spatial indexes from bounds.
        return spatialindex.from_regions(min_bounds, max_bounds)

    def _get_candidate_processes(self, point):
        """Determine candidate processes for a given point.

        :arg point: A point on the mesh.

        :returns: A numpy array of candidate processes.
        """
        candidates = spatialindex.bounding_boxes(self.processes_index, point)
        return candidates[candidates != self.mesh.comm.rank]

    def _perform_sparse_communication_round(self, recv_buffers, send_data):
        """Perform a sparse communication round in the point location process.

        :arg recv_buffers: A dictionary where the keys are process ranks and the
             corresponding items are numpy arrays of buffers in which to receive data
             from that rank.
        :arg send_data: A dictionary where the keys are process ranks and the
             corresponding items are lists of buffers from which to send data to that
             rank.
        """
        # Create lists to hold send and receive communication request objects.
        recv_reqs = []
        send_reqs = []

        for rank, buffers in recv_buffers.items():
            req = self.mesh.comm.Irecv(buffers, source=rank)
            recv_reqs.append(req)

        for rank, points in send_data.items():
            req = self.mesh.comm.Isend(points, dest=rank)
            send_reqs.append(req)

        MPI.Request.Waitall(recv_reqs + send_reqs)

    def _locate_mesh_elements(self):
        """Determine the location of each input point using the algorithm described in
        `self.locations`.

        :returns: A numpy array of (rank, cell number) pairs; (-1, -1) if not found.
        """

        # Create an array of (rank, element) tuples for storing located
        # elements.
        located_elements = np.full((len(self.points), 2), -1, dtype=IntType)

        # Check if points are located normally.
        local_results = self.mesh.locate_cells(self.points)

        # Update points that have been found locally.
        located_elements[:, 1] = local_results
        located_elements[local_results != -1, 0] = self.mesh.comm.rank

        # Store the points and that have not been found locally, and the indices of those
        # points in `self.points`.
        not_found_indices, = np.where(local_results == -1)
        points_not_found = self.points[not_found_indices]

        # Create dictionaries for storing processes that may contain these points.
        local_candidates = {}
        local_candidate_indices = {}

        for point, idx in zip(points_not_found, not_found_indices):
            # Point not found locally -- get candidates from processes spatial index.
            point_candidates = self._get_candidate_processes(point)
            for candidate in point_candidates:
                local_candidates.setdefault(candidate, []).append(point)
                local_candidate_indices.setdefault(candidate, []).append(idx)

            syncPrint("[%d] Cell not found locally for point %s. Candidates: %s"
                      % (self.mesh.comm.rank, point, point_candidates),
                      comm=self.mesh.comm)

        syncFlush(comm=self.mesh.comm)

        syncPrint("[%d] Located elements: %s" % (self.mesh.comm.rank,
                                                 located_elements.tolist()),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)
        syncPrint("[%d] Local candidates: %s" % (self.mesh.comm.rank, local_candidates),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)

        # Get number of points to receive from each rank through sparse communication
        # round.

        # Create input arrays from candidates dictionary.
        to_ranks = np.zeros(len(local_candidates), dtype=IntType)
        to_data = np.zeros(len(local_candidates), dtype=IntType)
        for i, (rank, points) in enumerate(local_candidates.items()):
            to_ranks[i] = rank
            to_data[i] = len(points)

        # `build_two_sided()` provides an interface for PetscCommBuildTwoSided, which
        # facilitates a sparse communication round between the processes to identify which
        # processes will be sending points and how many points they wish to send.
        # The output array `from_ranks` holds the ranks of the processes that will be
        # sending points, and the corresponding element in the `from_data` array specifies
        # the number of points that will be sent.
        from_ranks, from_data = build_two_sided(self.mesh.comm, 1, MPI.INT, to_ranks,
                                                to_data)

        # Create dictionary to hold all receive buffers for point requests from
        # each process.
        recv_points_buffers = {}
        for i in range(0, len(from_ranks)):
            recv_points_buffers[from_ranks[i]] = np.empty(
                (from_data[i], self.mesh.geometric_dimension()), dtype=ScalarType)

        # Receive all point requests

        local_candidates = {r: np.asarray(p) for r, p in local_candidates.items()}
        self._perform_sparse_communication_round(recv_points_buffers, local_candidates)

        syncPrint("[%d] Point queries requested: %s" % (self.mesh.comm.rank,
                                                        str(recv_points_buffers)),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)

        # Evaluate all point requests and prepare responses

        # Create dictionary to store results.
        point_responses = {}

        # Evaluate results.
        for rank, points_buffers in recv_points_buffers.items():
            point_responses[rank] = self.mesh.locate_cells(points_buffers)

        syncPrint("[%d] Point responses: %s" % (self.mesh.comm.rank,
                                                str(point_responses)),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)

        # Receive all responses

        # Create dictionary to hold all output buffers indexed by rank.
        recv_results = {}
        # Initialise these.
        for rank, points in local_candidates.items():
            # Create receive buffer(s).
            recv_results[rank] = np.empty((len(points), 1), dtype=IntType)

        self._perform_sparse_communication_round(recv_results, point_responses)

        syncPrint("[%d] Point location request results: %s" % (self.mesh.comm.rank,
                                                               str(recv_results)),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)

        # Process and return results.

        # Iterate through all points. If they have not been located locally,
        # iterate through each point request reponse to find the element.
        # Sometimes an element can be reported as found by more than one
        # process -- in this case, choose the process with the lower rank.
        for rank, result in recv_results.items():
            indices = local_candidate_indices[rank]
            found, _ = np.where(result != -1)
            for idx in found:
                i = indices[idx]
                loc_rank = located_elements[i, 0]
                if loc_rank == -1 or rank < loc_rank:
                    located_elements[i, :] = (rank, result[idx])

        syncPrint("[%d] Located elements: %s" % (self.mesh.comm.rank,
                                                 located_elements.tolist()),
                  comm=self.mesh.comm)
        syncFlush(comm=self.mesh.comm)

        return located_elements
