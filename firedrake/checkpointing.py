import functools
import pickle
from petsc4py.PETSc import ViewerHDF5
import finat.ufl
from pyop2 import op2
from pyop2.mpi import COMM_WORLD, internal_comm, MPI
from firedrake.cython import hdf5interface as h5i
from firedrake.cython import dmcommon
from firedrake.petsc import PETSc, OptionsManager
from firedrake.mesh import MeshTopology, ExtrudedMeshTopology, DEFAULT_MESH_NAME, make_mesh_from_coordinates, DistributedMeshOverlapType
from firedrake.functionspace import FunctionSpace
from firedrake import functionspaceimpl as impl
from firedrake.functionspacedata import get_global_numbering, create_element
from firedrake.function import Function, CoordinatelessFunction
from firedrake import extrusion_utils as eutils
from firedrake.embedding import get_embedding_element_for_checkpointing, get_embedding_method_for_checkpointing
from firedrake.parameters import parameters
import firedrake.utils as utils
import firedrake
import numpy as np
import os
import h5py


__all__ = ["DumbCheckpoint", "HDF5File", "FILE_READ", "FILE_CREATE", "FILE_UPDATE", "CheckpointFile"]


FILE_READ = PETSc.Viewer.Mode.READ
r"""Open a checkpoint file for reading.  Raises an error if file does not exist."""

FILE_CREATE = PETSc.Viewer.Mode.WRITE
r"""Create a checkpoint file.  Truncates the file if it exists."""

FILE_UPDATE = PETSc.Viewer.Mode.APPEND
r"""Open a checkpoint file for updating.  Creates the file if it does not exist, providing both read and write access."""


PREFIX = "firedrake"
r"""The prefix attached to the name of the Firedrake objects when saving them with CheckpointFile."""

PREFIX_EXTRUDED = "_".join([PREFIX, "extruded"])
r"""The prefix attached to the attributes associated with extruded meshes."""

PREFIX_IMMERSED = "_".join([PREFIX, "immersed"])
r"""The prefix attached to the attributes associated with immersed meshes."""

PREFIX_EMBEDDED = "_".join([PREFIX, "embedded"])
r"""The prefix attached to the DG function resulting from projecting the original function to the embedding DG space."""

PREFIX_TIMESTEPPING = "_".join([PREFIX, "timestepping"])
r"""The prefix attached to the attributes associated with timestepping."""

PREFIX_TIMESTEPPING_HISTORY = "_".join([PREFIX_TIMESTEPPING, "history"])
r"""The prefix attached to the attributes associated with timestepping history."""

# This is the distribution_parameters and reorder that one must use when
# distribution and permutation are loaded.
distribution_parameters_noop = {"partition": False,
                                "overlap_type": (DistributedMeshOverlapType.NONE, 0)}
reorder_noop = None


class DumbCheckpoint(object):

    r"""A very dumb checkpoint object.

    This checkpoint object is capable of writing :class:`~.Function`\s
    to disk in parallel (using HDF5) and reloading them on the same
    number of processes and a :func:`~.Mesh` constructed identically.

    :arg basename: the base name of the checkpoint file.
    :arg single_file: Should the checkpoint object use only a single
         on-disk file (irrespective of the number of stored
         timesteps)?  See :meth:`~.DumbCheckpoint.new_file` for more
         details.
    :arg mode: the access mode (one of :data:`~.FILE_READ`,
         :data:`~.FILE_CREATE`, or :data:`~.FILE_UPDATE`)
    :arg comm: (optional) communicator the writes should be collective
         over.

    This object can be used in a context manager (in which case it
    closes the file when the scope is exited).

    .. note::

       This object contains both a PETSc ``Viewer``, used for storing
       and loading :class:`~.Function` data, and an :class:`h5py.File`
       opened on the same file handle.  *DO NOT* call
       :meth:`h5py.File.close` on the latter, this will cause
       breakages.

    .. warning::

       DumbCheckpoint class will soon be deprecated.
       Use :class:`~.CheckpointFile` class instead.

    """
    def __init__(self, basename, single_file=True,
                 mode=FILE_UPDATE, comm=None):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn("DumbCheckpoint class will soon be deprecated; use CheckpointFile class instead.",
                          DeprecationWarning)
        self.comm = comm or COMM_WORLD
        self._comm = internal_comm(self.comm, self)
        self.mode = mode

        self._single = single_file
        self._made_file = False
        self._basename = basename
        self._time = None
        self._tidx = -1
        self._fidx = 0
        self.new_file()

    @PETSc.Log.EventDecorator()
    def set_timestep(self, t, idx=None):
        r"""Set the timestep for output.

        :arg t: The timestep value.
        :arg idx: An optional timestep index to use, otherwise an
             internal index is used, incremented by 1 every time
             :meth:`set_timestep` is called.
        """
        if idx is not None:
            self._tidx = idx
        else:
            self._tidx += 1
        self._time = t
        if self.mode == FILE_READ:
            return
        indices = self.read_attribute("/", "stored_time_indices", [])
        new_indices = np.concatenate((indices, [self._tidx]))
        self.write_attribute("/", "stored_time_indices", new_indices)
        steps = self.read_attribute("/", "stored_time_steps", [])
        new_steps = np.concatenate((steps, [self._time]))
        self.write_attribute("/", "stored_time_steps", new_steps)

    @PETSc.Log.EventDecorator()
    def get_timesteps(self):
        r"""Return all the time steps (and time indices) in the current
        checkpoint file.

        This is useful when reloading from a checkpoint file that
        contains multiple timesteps and one wishes to determine the
        final available timestep in the file."""
        indices = self.read_attribute("/", "stored_time_indices", [])
        steps = self.read_attribute("/", "stored_time_steps", [])
        return steps, indices

    @PETSc.Log.EventDecorator()
    def new_file(self, name=None):
        r"""Open a new on-disk file for writing checkpoint data.

        :arg name: An optional name to use for the file, an extension
             of ``.h5`` is automatically appended.

        If ``name`` is not provided, a filename is generated from the
        ``basename`` used when creating the :class:`~.DumbCheckpoint`
        object.  If ``single_file`` is ``True``, then we write to
        ``BASENAME.h5`` otherwise, each time
        :meth:`~.DumbCheckpoint.new_file` is called, we create a new
        file with an increasing index.  In this case the files created
        are::

            BASENAME_0.h5
            BASENAME_1.h5
            ...
            BASENAME_n.h5

        with the index incremented on each invocation of
        :meth:`~.DumbCheckpoint.new_file` (whenever the custom name is
        not provided).
        """
        self.close()
        if name is None:
            if self._single:
                if self._made_file:
                    raise ValueError("Can't call new_file without name with 'single_file'")
                name = "%s.h5" % (self._basename)
                self._made_file = True
            else:
                name = "%s_%s.h5" % (self._basename, self._fidx)
            self._fidx += 1
        else:
            name = "%s.h5" % name

        import os
        exists = os.path.exists(name)
        if self.mode == FILE_READ and not exists:
            raise IOError("File '%s' does not exist, cannot be opened for reading" % name)
        mode = self.mode
        if mode == FILE_UPDATE and not exists:
            mode = FILE_CREATE
        self._vwr = PETSc.ViewerHDF5().create(name, mode=mode,
                                              comm=self._comm)
        if self.mode == FILE_READ:
            nprocs = self.read_attribute("/", "nprocs")
            if nprocs != self.comm.size:
                raise ValueError("Process mismatch: written on %d, have %d" %
                                 (nprocs, self.comm.size))
        else:
            self.write_attribute("/", "nprocs", self.comm.size)

    @property
    def vwr(self):
        r"""The PETSc Viewer used to store and load function data."""
        if hasattr(self, '_vwr'):
            return self._vwr
        self.new_file()
        return self._vwr

    @property
    def h5file(self):
        r"""An h5py File object pointing at the open file handle."""
        if hasattr(self, '_h5file'):
            return self._h5file
        self._h5file = h5i.get_h5py_file(self.vwr)
        return self._h5file

    @PETSc.Log.EventDecorator()
    def close(self):
        r"""Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, "_vwr"):
            self._vwr.destroy()
            del self._vwr
        if hasattr(self, "_h5file"):
            self._h5file.flush()
            del self._h5file

    def _get_data_group(self):
        r"""Return the group name for function data.

        If a timestep is set, this incorporates the current timestep
        index.  See :meth:`.set_timestep`."""
        if self._time is not None:
            return "/fields/%d" % self._tidx
        return "/fields"

    def _write_timestep_attr(self, group):
        r"""Write the current timestep value (if it exists) to the
        specified group."""
        if self._time is not None:
            self.h5file.require_group(group)
            self.write_attribute(group, "timestep", self._time)

    @PETSc.Log.EventDecorator()
    def store(self, function, name=None):
        r"""Store a function in the checkpoint file.

        :arg function: The function to store.
        :arg name: an (optional) name to store the function under.  If
             not provided, uses ``function.name()``.

        This function is timestep-aware and stores to the appropriate
        place if :meth:`set_timestep` has been called.
        """
        if self.mode is FILE_READ:
            raise IOError("Cannot store to checkpoint opened with mode 'FILE_READ'")
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only store functions")
        name = name or function.name()
        group = self._get_data_group()
        self._write_timestep_attr(group)
        with function.dat.vec_ro as v:
            self.vwr.pushGroup(group)
            oname = v.getName()
            v.setName(name)
            v.view(self.vwr)
            v.setName(oname)
            self.vwr.popGroup()

    @PETSc.Log.EventDecorator()
    def load(self, function, name=None):
        r"""Store a function from the checkpoint file.

        :arg function: The function to load values into.
        :arg name: an (optional) name used to find the function values.  If
             not provided, uses ``function.name()``.

        This function is timestep-aware and reads from the appropriate
        place if :meth:`set_timestep` has been called.
        """
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only load functions")
        name = name or function.name()
        group = self._get_data_group()
        with function.dat.vec_wo as v:
            self.vwr.pushGroup(group)
            # PETSc replaces the array in the Vec, which screws things
            # up for us, so read into temporary Vec.
            tmp = v.duplicate()
            tmp.setName(name)
            tmp.load(self.vwr)
            tmp.copy(v)
            tmp.destroy()
            self.vwr.popGroup()

    def write_attribute(self, obj, name, val):
        r"""Set an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        :arg val: The attribute value.

        Raises :exc:`AttributeError` if writing the attribute fails.
        """
        try:
            self.h5file[obj].attrs[name] = val
        except KeyError:
            raise AttributeError("Object '%s' not found" % obj)

    def read_attribute(self, obj, name, default=None):
        r"""Read an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        :arg default: Optional default value to return.  If not
             provided an :exc:`AttributeError` is raised if the
             attribute does not exist.
        """
        try:
            return self.h5file[obj].attrs[name]
        except KeyError:
            if default is not None:
                return default
            raise AttributeError("Attribute '%s' on '%s' not found" % (name, obj))

    def has_attribute(self, obj, name):
        r"""Check for existance of an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        """
        try:
            return (name in self.h5file[obj].attrs)
        except KeyError:
            return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class HDF5File(object):

    r"""An object to facilitate checkpointing.

    This checkpoint object is capable of writing :class:`~.Function`\s
    to disk in parallel (using HDF5) and reloading them on the same
    number of processes and a :func:`~.Mesh` constructed identically.

    :arg filename: filename (including suffix .h5) of checkpoint file.
    :arg file_mode: the access mode, passed directly to h5py, see
        :class:`h5py.File` for details on the meaning.
    :arg comm: communicator the writes should be collective
         over.

    This object can be used in a context manager (in which case it
    closes the file when the scope is exited).

    .. warning::

       HDF5File class will soon be deprecated.
       Use :class:`~.CheckpointFile` class instead.

    """
    def __init__(self, filename, file_mode, comm=None):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn("HDF5File class will soon be deprecated; use CheckpointFile class instead.",
                          DeprecationWarning)
        self.comm = comm or COMM_WORLD
        self._comm = internal_comm(self.comm, self)

        self._filename = filename
        self._mode = file_mode

        exists = os.path.exists(filename)
        if file_mode == 'r' and not exists:
            raise IOError("File '%s' does not exist, cannot be opened for reading" % filename)

        # Create the directory if necessary
        dirname = os.path.dirname(filename)
        try:
            os.makedirs(dirname)
        except OSError:
            pass

        # Try to use MPI
        try:
            self._h5file = h5py.File(filename, file_mode, driver="mpio", comm=self._comm)
        except NameError:  # the error you get if h5py isn't compiled against parallel HDF5
            raise RuntimeError("h5py *must* be installed with MPI support")

        if file_mode == 'r':
            nprocs = self.attributes('/')['nprocs']
            if nprocs != self.comm.size:
                raise ValueError("Process mismatch: written on %d, have %d" %
                                 (nprocs, self.comm.size))
        else:
            self.attributes('/')['nprocs'] = self.comm.size

    def _set_timestamp(self, t):
        r"""Set the timestamp for storing.

        :arg t: The timestamp value.
        """
        if self._mode == 'r':
            return
        attrs = self.attributes("/")
        timestamps = attrs.get("stored_timestamps", [])
        attrs["stored_timestamps"] = np.concatenate((timestamps, [t]))

    def get_timestamps(self):
        r"""Get the timestamps this HDF5File knows about."""

        attrs = self.attributes("/")
        timestamps = attrs.get("stored_timestamps", [])
        return timestamps

    def close(self):
        r"""Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, '_h5file'):
            self._h5file.flush()
            # Need to explicitly close the h5py File so that all
            # objects referencing it are cleaned up, otherwise we
            # close the file, but there are still open objects and we
            # get a refcounting error in HDF5.
            self._h5file.close()
            del self._h5file

    def flush(self):
        r"""Flush any pending writes."""
        self._h5file.flush()

    @PETSc.Log.EventDecorator()
    def write(self, function, path, timestamp=None):
        r"""Store a function in the checkpoint file.

        :arg function: The function to store.
        :arg path: the path to store the function under.
        :arg timestamp: timestamp associated with function, or None for
                        stationary data
        """
        if self._mode == 'r':
            raise IOError("Cannot store to checkpoint opened with mode 'FILE_READ'")
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only store functions")

        if timestamp is not None:
            suffix = "/%.15e" % timestamp
            path = path + suffix

        with function.dat.vec_ro as v:
            dset = self._h5file.create_dataset(path, shape=(v.getSize(),), dtype=function.dat.dtype)

            # Another MPI/non-MPI difference
            try:
                with dset.collective:
                    dset[slice(*v.getOwnershipRange())] = v.array_r
            except AttributeError:
                dset[slice(*v.getOwnershipRange())] = v.array_r

        if timestamp is not None:
            attr = self.attributes(path)
            attr["timestamp"] = timestamp
            self._set_timestamp(timestamp)

    @PETSc.Log.EventDecorator()
    def read(self, function, path, timestamp=None):
        r"""Store a function from the checkpoint file.

        :arg function: The function to load values into.
        :arg path: the path under which the function is stored.
        """
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only load functions")
        if timestamp is not None:
            suffix = "/%.15e" % timestamp
            path = path + suffix

        with function.dat.vec_wo as v:
            dset = self._h5file[path]
            v.array[:] = dset[slice(*v.getOwnershipRange())]

    def attributes(self, obj):
        r""":arg obj: The path to the group."""
        return self._h5file[obj].attrs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class CheckpointFile(object):

    r"""Checkpointing meshes and :class:`~.Function` s in an HDF5 file.

    :arg filename: the name of the HDF5 checkpoint file (.h5 or .hdf5).
    :arg mode: the file access mode (:obj:`~.FILE_READ`, :obj:`~.FILE_CREATE`, :obj:`~.FILE_UPDATE`) or ('r', 'w', 'a').
    :arg comm: the communicator.

    This object allows for a scalable and flexible checkpointing of states.
    One can save and load meshes and :class:`~.Function` s entirely in parallel
    without needing to gather them to or scatter them from a single process.
    One can also use different number of processes for saving and for loading.

    """

    latest_version = '3.0.0'

    def __init__(self, filename, mode, comm=COMM_WORLD):
        self.viewer = ViewerHDF5()
        self.filename = filename
        self.comm = comm
        self._comm = internal_comm(comm, self)
        r"""The neme of the checkpoint file."""
        self.viewer.create(filename, mode=mode, comm=self._comm)
        self.commkey = self._comm.py2f()
        assert self.commkey != MPI.COMM_NULL.py2f()
        self._function_spaces = {}
        self._function_load_utils = {}
        if mode in [PETSc.Viewer.FileMode.WRITE, PETSc.Viewer.FileMode.W, "w"]:
            version = CheckpointFile.latest_version
            self.set_attr_byte_string("/", "dmplex_storage_version", version)
        elif mode in [PETSc.Viewer.FileMode.APPEND, PETSc.Viewer.FileMode.A, "a"]:
            if self.has_attr("/", "dmplex_storage_version"):
                version = self.get_attr_byte_string("/", "dmplex_storage_version")
            else:
                version = CheckpointFile.latest_version
                self.set_attr_byte_string("/", "dmplex_storage_version", version)
        elif mode in [PETSc.Viewer.FileMode.READ, PETSc.Viewer.FileMode.R, "r"]:
            if not self.has_attr("/", "dmplex_storage_version"):
                raise RuntimeError(f"Only files generated with CheckpointFile are supported: got an invalid file ({filename})")
            version = CheckpointFile.latest_version
        else:
            raise NotImplementedError(f"Unsupportd file mode: {mode} not in {'w', 'a', 'r'}")
        self.opts = OptionsManager({"dm_plex_view_hdf5_storage_version": version}, "")
        r"""DMPlex HDF5 version options."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @PETSc.Log.EventDecorator("SaveMesh")
    def save_mesh(self, mesh, distribution_name=None, permutation_name=None):
        r"""Save a mesh.

        :arg mesh: the mesh to save.
        :kwarg distribution_name: the name under which distribution is saved; if `None`, auto-generated name will be used.
        :kwarg permutation_name: the name under which permutation is saved; if `None`, auto-generated name will be used.
        """
        mesh.init()
        # Handle extruded mesh
        tmesh = mesh.topology
        if mesh.extruded:
            # -- Save mesh topology --
            base_tmesh = mesh._base_mesh.topology
            self._save_mesh_topology(base_tmesh)
            if tmesh.name not in self.require_group(self._path_to_topologies()):
                # The tmesh (an ExtrudedMeshTopology) is treated as if it was a first class topology object. It
                # shares the plex data with the base_tmesh, but those data are stored under base_tmesh's path,
                # so we here create a symbolic link:
                # topologies/{tmesh.name}/topology <- topologies/{base_tmesh.name}/topology
                # This is merely to make this group (topologies/{tmesh.name}) behave exactly like standard topology
                # groups (topologies/{some_non_extruded_topology_name}), and not necessary at the moment.
                path = self._path_to_topology(tmesh.name)
                self.require_group(path)
                self.h5pyfile[os.path.join(path, "topology")] = self.h5pyfile[os.path.join(self._path_to_topology(base_tmesh.name), "topology")]
                path = self._path_to_topology_extruded(tmesh.name)
                self.require_group(path)
                self.set_attr(path, PREFIX_EXTRUDED + "_base_mesh", base_tmesh.name)
                self.set_attr(path, PREFIX_EXTRUDED + "_periodic", tmesh.extruded_periodic)
                self.set_attr(path, PREFIX_EXTRUDED + "_variable_layers", tmesh.variable_layers)
                if tmesh.variable_layers:
                    # Save tmesh.layers, which contains (start layer, stop layer)-tuple for each cell
                    # Conceptually, we project these integer pairs onto DG0 vector space of dim=2.
                    cell = base_tmesh.ufl_cell()
                    element = finat.ufl.VectorElement("DP" if cell.is_simplex() else "DQ", cell, 0, dim=2)
                    layers_tV = impl.FunctionSpace(base_tmesh, element)
                    self._save_function_space_topology(layers_tV)
                    # Note that _cell_numbering coincides with DG0 section, so we can use tmesh.layers directly.
                    layers_iset = PETSc.IS().createGeneral(tmesh.layers[:tmesh.cell_set.size, :], comm=tmesh._comm)
                    layers_iset.setName("_".join([PREFIX_EXTRUDED, "layers_iset"]))
                    self.viewer.pushGroup(path)
                    layers_iset.view(self.viewer)
                    self.viewer.popGroup()
                else:
                    self.set_attr(path, PREFIX_EXTRUDED + "_layers", tmesh.layers)
            # -- Save mesh --
            path = self._path_to_meshes(tmesh.name)
            if mesh.name not in self.require_group(path):
                path = self._path_to_mesh(tmesh.name, mesh.name)
                self.require_group(path)
                self._save_ufl_element(path, PREFIX + "_coordinate_element", mesh._coordinates.function_space().ufl_element())
                self.set_attr(path, PREFIX + "_coordinates", mesh._coordinates.name())
                self._save_function_topology(mesh._coordinates)
                if hasattr(mesh, PREFIX + "_radial_coordinates"):
                    # Cannot do: self.save_function(mesh.radial_coordinates)
                    # This will cause infinite recursion.
                    self.set_attr(path, PREFIX + "_radial_coordinate_function", mesh.radial_coordinates.name())
                    radial_coordinates = mesh.radial_coordinates.topological
                    self._save_ufl_element(path, PREFIX + "_radial_coordinate_element", radial_coordinates.function_space().ufl_element())
                    self.set_attr(path, PREFIX + "_radial_coordinates", radial_coordinates.name())
                    self._save_function_topology(radial_coordinates)
                self._update_mesh_name_topology_name_map({mesh.name: tmesh.name})
                # The followings are conceptually redundant, but needed.
                path = os.path.join(self._path_to_mesh(tmesh.name, mesh.name), PREFIX_EXTRUDED)
                self.require_group(path)
                self.save_mesh(mesh._base_mesh)
                self.set_attr(path, PREFIX_EXTRUDED + "_base_mesh", mesh._base_mesh.name)
        else:
            # -- Save mesh topology --
            self._save_mesh_topology(tmesh)
            # -- Save mesh --
            path = self._path_to_meshes(tmesh.name)
            if mesh.name not in self.require_group(path):
                path = self._path_to_mesh(tmesh.name, mesh.name)
                self.require_group(path)
                # Save Firedrake coodinates.
                self._save_ufl_element(path, PREFIX + "_coordinate_element", mesh._coordinates.function_space().ufl_element())
                self.set_attr(path, PREFIX + "_coordinates", mesh._coordinates.name())
                self._save_function_topology(mesh._coordinates)
                # Save DMPlex coordinates for a complete representation of the plex.
                # Practically, plex coordinates will be needed when we checkpoint MeshHierarchy in the future.
                with self.opts.inserted_options():
                    tmesh.topology_dm.coordinatesView(viewer=self.viewer)
                self._update_mesh_name_topology_name_map({mesh.name: tmesh.name})
                # Save cell_orientations for immersed meshes.
                if hasattr(mesh, "_cell_orientations"):
                    path = self._path_to_mesh_immersed(tmesh.name, mesh.name)
                    self.require_group(path)
                    self.set_attr(path, PREFIX_IMMERSED + "_cell_orientations", mesh._cell_orientations.name())
                    cell_orientations_tV = mesh._cell_orientations.function_space()
                    self._save_function_space_topology(cell_orientations_tV)
                    # Compute "canonical" cell_orientations array.
                    # This is the mesh._cell_orientations array (for manifold orientation) that the mesh would
                    # have if all cell orientations of the reference-physical cell mappings were "0" (matching
                    # plex cone ordering).
                    # This can be done by flipping values (0 <-> 1) for cells for which "reflection" have been
                    # introduced relative to orientation 0.
                    canonical_cell_orientations = np.copy(mesh._cell_orientations.dat.data_ro[:tmesh.cell_set.size])
                    o_r_map = np.array(list(cell_orientations_tV.finat_element.cell.cell_orientation_reflection_map().values()), dtype=np.int32)
                    reflected = o_r_map[tmesh.entity_orientations[:tmesh.cell_set.size, -1]]
                    reflected_indices = (reflected == 1)
                    canonical_cell_orientations[reflected_indices] = 1 - canonical_cell_orientations[reflected_indices]
                    cell_orientations_iset = PETSc.IS().createGeneral(canonical_cell_orientations, comm=tmesh._comm)
                    cell_orientations_iset.setName("_".join([PREFIX_IMMERSED, "cell_orientations_iset"]))
                    self.viewer.pushGroup(path)
                    cell_orientations_iset.view(self.viewer)
                    self.viewer.popGroup()

    @PETSc.Log.EventDecorator("SaveMeshTopology")
    def _save_mesh_topology(self, tmesh):
        # -- Save DMPlex --
        tmesh.init()
        topology_dm = tmesh.topology_dm
        tmesh_name = topology_dm.getName()
        distribution_name = tmesh._distribution_name
        perm_is = tmesh._dm_renumbering
        permutation_name = tmesh._permutation_name
        if tmesh_name in self.require_group(self._path_to_topologies()):
            version_str = self.opts.parameters['dm_plex_view_hdf5_storage_version']
            version_major, version_minor, version_patch = tuple(int(ver) for ver in version_str.split('.'))
            # Check if the global number of DMPlex points and
            # the global sum of DMPlex cone sizes are consistent.
            cell_dim = topology_dm.getDimension()
            if version_major < 3:
                path = os.path.join(self._path_to_topology(tmesh_name), "topology", "cells")
            else:
                path = os.path.join(self._path_to_topology(tmesh_name), "topology")
            cell_dim1 = self.get_attr(path, "cell_dim")
            if cell_dim1 != cell_dim:
                raise ValueError(f"Mesh ({tmesh_name}) already exists in {self.filename}, but the topological dimension is inconsistent: {cell_dim1} ({self.filename}) != {cell_dim} ({tmesh_name})")
            order_array_size, ornt_array_size = dmcommon.compute_point_cone_global_sizes(topology_dm)
            if version_major < 3:
                path = os.path.join(self._path_to_topology(tmesh_name), "topology")
                order_array_size1 = self.h5pyfile[path]["order"].size
                ornt_array_size1 = self.h5pyfile[path]["orientation"].size
            else:
                order_array_size1 = 0
                ornt_array_size1 = 0
                for d in range(cell_dim + 1):
                    path = os.path.join(self._path_to_topology(tmesh_name), "topology", "strata", str(d))
                    order_array_size1 += self.h5pyfile[path]["cone_sizes"].size
                    ornt_array_size1 += self.h5pyfile[path]["cones"].size
            if order_array_size1 != order_array_size:
                raise ValueError(f"Mesh ({tmesh_name}) already exists in {self.filename}, but the global number of DMPlex points is inconsistent: {order_array_size1} ({self.filename}) != {order_array_size} ({tmesh_name})")
            if ornt_array_size1 != ornt_array_size:
                raise ValueError(f"Mesh ({tmesh_name}) already exists in {self.filename}, but the global sum of all DMPlex cone sizes is inconsistent: {ornt_array_size1} ({self.filename}) != {ornt_array_size} ({tmesh_name})")
            # We assume that each (conceptually the same) mesh topology (plex)
            # is uniquely named (this is users' responsibility).
            # With the current setup, "distributions" folder will always contain
            # one and only one distribution. This can be amended once we make
            # the global point numbers independent of the distributions.
            path = self._path_to_distributions(tmesh_name)
            if path not in self.h5pyfile:
                # backward campat.: old files do not have "distributions" folder.
                # Just save mesh + dist + perm.
                do_save = True
            else:
                # There should be one and only one distribution.
                _distribution_name, = self.h5pyfile[path].keys()
                if _distribution_name != distribution_name:
                    raise RuntimeError(f"Can only save one distribution: found {_distribution_name} in file, but trying to save {distribution_name}")
                else:
                    path = self._path_to_distribution(tmesh_name, distribution_name)
                    _comm_size = self.h5pyfile[path]["chart_sizes"].size
                    if _comm_size != topology_dm.comm.size:
                        raise RuntimeError(f"Trying to save distribution of the same name ({distribution_name}), but it was previously saved with {_comm_size} processes, and now you are using {topology_dm.comm.size} processes")
                path = self._path_to_permutations(tmesh_name, distribution_name)
                # There should be one and only one permutation.
                _permutation_name, = self.h5pyfile[path].keys()
                if _permutation_name != permutation_name:
                    raise RuntimeError(f"Can only save one permutation: found {_permutation_name} in file, but trying to save {permutation_name}")
                # Regard tmesh + distribution + permutation has already been saved.
                do_save = False
        else:
            do_save = True
        if do_save:
            self.viewer.pushFormat(format=ViewerHDF5.Format.HDF5_PETSC)
            with self.opts.inserted_options():
                topology_dm.distributionSetName(distribution_name)
                topology_dm.topologyView(viewer=self.viewer)
                topology_dm.distributionSetName(None)
                topology_dm.labelsView(viewer=self.viewer)
            self.viewer.popFormat()
            path = self._path_to_permutation(tmesh_name, distribution_name, permutation_name)
            self.require_group(path)
            self.viewer.pushGroup(path)
            perm_is.setName("permutation")
            perm_is.view(self.viewer)
            perm_is.setName(None)
            self.viewer.popGroup()

    @PETSc.Log.EventDecorator("GetTimesteps")
    def get_timestepping_history(self, mesh, name):
        """
        Retrieve the timestepping history and indices for a specified function within a mesh.

        This method is primarily used in checkpointing scenarios during timestepping simulations.
        It returns the indices associated with each function stored in the timestepping mode,
        along with any additional timestepping-related information (like time or timestep values) if available.
        If the specified function has not been stored in timestepping mode, it returns an empty dictionary.

        Parameters
        ----------
        mesh : firedrake.mesh.MeshGeometry
            The mesh containing the function to be queried.
        name : str
            The name of the function whose timestepping history is to be retrieved.

        Returns
        -------
        dict
            - Returns an empty dictionary if the function `name` has not been stored in timestepping mode.
            - If the function `name` is stored in timestepping mode, returns a dictionary with the following contents:
                - 'indices': A list of all stored indices for the function.
                - Additional key-value pairs representing timestepping information, if available.

        Raises
        ------
        RuntimeError
            If the function `name` is not found within the given `mesh` in the current file.

        See Also
        --------
        CheckpointFile.save_function : Describes how timestepping information should be provided.

        Notes
        -----
        The function internally checks whether the specified function is mixed or exists in the file.
        It then retrieves the appropriate data paths and extracts the timestepping information
        as specified in the checkpoint file.
        """

        # check if the function is mixed, or even exists in file
        if name in self._get_mixed_function_name_mixed_function_space_name_map(mesh.name):
            V_name = self._get_mixed_function_name_mixed_function_space_name_map(mesh.name)[name]
            base_path = self._path_to_mixed_function(mesh.name, V_name, name)
            path = os.path.join(base_path, str(0))  # path to subfunction 0
            fsub_name = self.get_attr(path, PREFIX + "_function")
            return self.get_timestepping_history(mesh, fsub_name)
        elif name in self._get_function_name_function_space_name_map(self._get_mesh_name_topology_name_map()[mesh.name], mesh.name):
            tmesh_name = self._get_mesh_name_topology_name_map()[mesh.name]
            V_name = self._get_function_name_function_space_name_map(tmesh_name, mesh.name)[name]
            V = self._load_function_space(mesh, V_name)
            tV = V.topological
            path = self._path_to_function(tmesh_name, mesh.name, V_name, name)
            if PREFIX_EMBEDDED in self.h5pyfile[path]:
                path = self._path_to_function_embedded(tmesh_name, mesh.name, V_name, name)
                _name = self.get_attr(path, PREFIX_EMBEDDED + "_function")
                return self.get_timestepping_history(mesh, _name)
            else:
                tf_name = self.get_attr(path, PREFIX + "_vec")
                dm_name = self._get_dm_name_for_checkpointing(tV.mesh(), tV.ufl_element())
                timestepping_info = {}
                tpath = self._path_to_vec_timestepping(tV.mesh().name, dm_name, tf_name)
                path = self._path_to_function_timestepping(tmesh_name, mesh.name, V_name, name)
                if tpath in self.h5pyfile:
                    assert path in self.h5pyfile
                    timestepping_info["index"] = self.get_attr(tpath, PREFIX_TIMESTEPPING_HISTORY + "_index")
                    for key in self.h5pyfile[path].attrs.keys():
                        if key.startswith(PREFIX_TIMESTEPPING_HISTORY):
                            key_ = key.replace(PREFIX_TIMESTEPPING_HISTORY + "_", "", 1)
                            timestepping_info[key_] = self.get_attr(path, key)
                return timestepping_info
        else:
            raise RuntimeError(
                f"""Function ({name}) not found in {self.filename}""")

    @PETSc.Log.EventDecorator("SaveFunctionSpace")
    def _save_function_space(self, V):
        mesh = V.mesh()
        if isinstance(V.topological, impl.MixedFunctionSpace):
            V_name = self._generate_function_space_name(V)
            base_path = self._path_to_mixed_function_space(mesh.name, V_name)
            self.require_group(base_path)
            self.set_attr(base_path, PREFIX + "_num_sub_spaces", V.num_sub_spaces())
            for i, Vsub in enumerate(V):
                path = os.path.join(base_path, str(i))
                self.require_group(path)
                Vsub_name = self._generate_function_space_name(Vsub)
                self.set_attr(path, PREFIX + "_function_space", Vsub_name)
                self._save_function_space(Vsub)
        else:
            # -- Save mesh --
            self.save_mesh(mesh)
            # -- Save function space topology --
            tV = V.topological
            self._save_function_space_topology(tV)
            # -- Save function space --
            tmesh = tV.mesh()
            element = tV.ufl_element()
            V_name = self._generate_function_space_name(V)
            path = self._path_to_function_spaces(tmesh.name, mesh.name)
            if V_name not in self.require_group(path):
                # Save UFL element
                path = self._path_to_function_space(tmesh.name, mesh.name, V_name)
                self.require_group(path)
                self._save_ufl_element(path, PREFIX + "_ufl_element", element)
                # Test if the loaded UFL element matches the original element
                loaded_element = self._load_ufl_element(path, PREFIX + "_ufl_element")
                if loaded_element != element:
                    raise RuntimeError(f"Loaded UFL element ({loaded_element}) does not match the original element ({element})")

    @PETSc.Log.EventDecorator("SaveFunctionSpaceTopology")
    def _save_function_space_topology(self, tV):
        # -- Save mesh topology --
        tmesh = tV.mesh()
        self._save_mesh_topology(tmesh)
        # -- Save function space topology --
        element = tV.ufl_element()
        dm_name = self._get_dm_name_for_checkpointing(tmesh, element)
        path = self._path_to_dms(tmesh.name)
        if dm_name not in self.require_group(path):
            if element.family() == "Real":
                assert not isinstance(element, (finat.ufl.VectorElement, finat.ufl.TensorElement))
            else:
                dm = self._get_dm_for_checkpointing(tV)
                topology_dm = tmesh.topology_dm
                # If tmesh is an ExtrudedMeshTopology, it inherits plex from the base_tmesh ( = tmesh._base_mesh).
                # In that case we need to save (section) dm under tmesh.name instead of under base_tmesh.name, so
                # we need to switch names of the topology_dm.
                # We could in theory save the topology_dm under tmesh.name as well as under base_tmesh.name or
                # create a symbolic link as /topologies/tmesh.name/topology <- /topologies/base_tmesh.name/topology
                # to have a full structure under tmesh.name, but at least for now we don't need to.
                base_tmesh_name = topology_dm.getName()
                with self.opts.inserted_options():
                    topology_dm.setName(tmesh.name)
                    topology_dm.sectionView(self.viewer, dm)
                    topology_dm.setName(base_tmesh_name)

    @PETSc.Log.EventDecorator("SaveFunction")
    def save_function(self, f, idx=None, name=None, timestepping_info={}):
        r"""Save a :class:`~.Function`.

        :arg f: the :class:`~.Function` to save.
        :kwarg idx: optional timestepping index. A function can
            either be saved in timestepping mode or in normal
            mode (non-timestepping); for each function of interest,
            this method must always be called with the idx parameter
            set or never be called with the idx parameter set.
        :kwarg name: optional alternative name to save the function under.
        :kwarg timestepping_info: optional (requires idx) additional information
            such as time, timestepping that can be stored along a function for
            each index.
        """
        V = f.function_space()
        mesh = V.mesh()
        if name:
            g = Function(V, val=f.dat, name=name)
            return self.save_function(g, idx=idx, timestepping_info=timestepping_info)
        # -- Save function space --
        self._save_function_space(V)
        # -- Save function --
        V_name = self._generate_function_space_name(V)
        if isinstance(V.topological, impl.MixedFunctionSpace):
            base_path = self._path_to_mixed_function(mesh.name, V_name, f.name())
            self.require_group(base_path)
            for i, fsub in enumerate(f.subfunctions):
                path = os.path.join(base_path, str(i))
                self.require_group(path)
                self.set_attr(path, PREFIX + "_function", fsub.name())
                self.save_function(fsub, idx=idx, timestepping_info=timestepping_info)
            self._update_mixed_function_name_mixed_function_space_name_map(mesh.name, {f.name(): V_name})
        else:
            tf = f.topological
            tV = tf.function_space()
            tmesh = tV.mesh()
            self._update_function_name_function_space_name_map(tmesh.name, mesh.name, {f.name(): V_name})
            # Embed if necessary
            element = V.ufl_element()
            _element = get_embedding_element_for_checkpointing(element)
            if _element != element:
                path = self._path_to_function_embedded(tmesh.name, mesh.name, V_name, f.name())
                self.require_group(path)
                method = get_embedding_method_for_checkpointing(element)
                _V = FunctionSpace(mesh, _element)
                _name = "_".join([PREFIX_EMBEDDED, f.name()])
                _f = Function(_V, name=_name)
                self._project_function_for_checkpointing(_f, f, method)
                self.save_function(_f, idx=idx, timestepping_info=timestepping_info)
                self.set_attr(path, PREFIX_EMBEDDED + "_function", _name)
            else:
                # -- Save function topology --
                path = self._path_to_function(tmesh.name, mesh.name, V_name, f.name())
                self.require_group(path)
                self.set_attr(path, PREFIX + "_vec", tf.name())
                self._save_function_topology(tf, idx=idx)
                # store timstepping_info only if in timestepping mode
                if idx is not None:
                    path = self._path_to_function_timestepping(tmesh.name, mesh.name, V_name, f.name())
                    new = path not in self.h5pyfile
                    self.require_group(path)
                    # We make sure the provided timestepping_info is consistent all along timestepping.
                    if not new:
                        existing_keys = {key.replace(PREFIX_TIMESTEPPING_HISTORY + "_", "", 1)
                                         for key in self.h5pyfile[path].attrs.keys()
                                         if key.startswith(PREFIX_TIMESTEPPING_HISTORY)}
                        if timestepping_info.keys() != existing_keys:
                            raise RuntimeError(
                                r"Provided keys in timestepping_info must remain consistent")
                    # store items in timestepping_info accordingly
                    for ts_info_key, ts_info_value in timestepping_info.items():
                        if not isinstance(ts_info_value, float):
                            raise NotImplementedError(f"timestepping_info must have float values: got {type(ts_info_value)}")
                        old_items = [] if new else self.get_attr(path, PREFIX_TIMESTEPPING_HISTORY + f"_{ts_info_key}")
                        items = np.concatenate((old_items, [ts_info_value]))
                        self.set_attr(path, PREFIX_TIMESTEPPING_HISTORY + f"_{ts_info_key}", items)

    @PETSc.Log.EventDecorator("SaveFunctionTopology")
    def _save_function_topology(self, tf, idx=None):
        # -- Save function space topology --
        tV = tf.function_space()
        self._save_function_space_topology(tV)
        # -- Save function topology --
        if idx is not None:
            self.viewer.pushTimestepping()
            self.viewer.setTimestep(idx)
        tmesh = tV.mesh()
        element = tV.ufl_element()
        if element.family() == "Real":
            assert not isinstance(element, (finat.ufl.VectorElement, finat.ufl.TensorElement))
            dm_name = self._get_dm_name_for_checkpointing(tmesh, element)
            path = self._path_to_vec(tmesh.name, dm_name, tf.name())
            self.require_group(path)
            self.set_attr(path, "_".join([PREFIX, "value" if idx is None else "value_" + str(idx)]), tf.dat.data.item())
        else:
            topology_dm = tmesh.topology_dm
            dm = self._get_dm_for_checkpointing(tV)
            dm_name = dm.name
            path = self._path_to_vec(tmesh.name, dm_name, tf.name())
            if path in self.h5pyfile:
                try:
                    timestepping = self.get_attr(os.path.join(path, tf.name()), "timestepping")
                except KeyError:
                    timestepping = False
                if timestepping:
                    assert idx is not None, "In timestepping mode: idx parameter must be set"
                else:
                    assert idx is None, "In non-timestepping mode: idx parameter msut not be set"
            with tf.dat.vec_ro as vec:
                vec.setName(tf.name())
                base_tmesh_name = topology_dm.getName()
                with self.opts.inserted_options():
                    topology_dm.setName(tmesh.name)
                    topology_dm.globalVectorView(self.viewer, dm, vec)
                    topology_dm.setName(base_tmesh_name)
        if idx is not None:
            self.viewer.popTimestepping()
            path = self._path_to_vec_timestepping(tmesh.name, dm_name, tf.name())
            new = path not in self.h5pyfile
            self.require_group(path)
            old_indices = [] if new else self.get_attr(path, PREFIX_TIMESTEPPING_HISTORY + "_index")
            indices = np.concatenate((old_indices, [idx]))
            self.set_attr(path, PREFIX_TIMESTEPPING_HISTORY + "_index", indices)

    @PETSc.Log.EventDecorator("LoadMesh")
    def load_mesh(self, name=DEFAULT_MESH_NAME, reorder=None, distribution_parameters=None, topology=None):
        r"""Load a mesh.

        Parameters
        ----------
        name : str
            the name of the mesh to load (default to `firedrake.mesh.DEFAULT_MESH_NAME`).
        reorder : bool
            whether to reorder the mesh; see `firedrake.Mesh`.
        distribution_parameters : dict
            the `distribution_parameters` used for distributing the mesh; see `firedrake.Mesh`.
        topology : firedrake.mesh.MeshTopology
            the underlying mesh topology if already known.

        Returns
        -------
        firedrake.mesh.MeshGeometry
            the loaded mesh.

        """
        tmesh_name = self._get_mesh_name_topology_name_map()[name]
        path = self._path_to_topology_extruded(tmesh_name)
        if path in self.h5pyfile:
            # -- Load mesh topology --
            base_tmesh_name = self.get_attr(path, PREFIX_EXTRUDED + "_base_mesh")
            base_tmesh = self._load_mesh_topology(base_tmesh_name, reorder, distribution_parameters)
            base_tmesh.init()
            periodic = self.get_attr(path, PREFIX_EXTRUDED + "_periodic") if self.has_attr(path, PREFIX_EXTRUDED + "_periodic") else False
            variable_layers = self.get_attr(path, PREFIX_EXTRUDED + "_variable_layers")
            if variable_layers:
                cell = base_tmesh.ufl_cell()
                element = finat.ufl.VectorElement("DP" if cell.is_simplex() else "DQ", cell, 0, dim=2)
                _ = self._load_function_space_topology(base_tmesh, element)
                base_tmesh_key = self._generate_mesh_key_from_names(base_tmesh.name,
                                                                    base_tmesh._distribution_name,
                                                                    base_tmesh._permutation_name)
                sd_key = self._get_shared_data_key_for_checkpointing(base_tmesh, element)
                _, _, lsf = self._function_load_utils[base_tmesh_key + sd_key]
                nroots, _, _ = lsf.getGraph()
                layers_a = np.empty(nroots, dtype=utils.IntType)
                layers_a_iset = PETSc.IS().createGeneral(layers_a, comm=self._comm)
                layers_a_iset.setName("_".join([PREFIX_EXTRUDED, "layers_iset"]))
                self.viewer.pushGroup(path)
                layers_a_iset.load(self.viewer)
                self.viewer.popGroup()
                layers_a = layers_a_iset.getIndices()
                layers = np.empty((base_tmesh.cell_set.total_size, 2), dtype=utils.IntType)
                unit = MPI._typedict[np.dtype(utils.IntType).char]
                lsf.bcastBegin(unit, layers_a, layers, MPI.REPLACE)
                lsf.bcastEnd(unit, layers_a, layers, MPI.REPLACE)
            else:
                layers = self.get_attr(path, PREFIX_EXTRUDED + "_layers")
            tmesh = ExtrudedMeshTopology(base_tmesh, layers, periodic=periodic, name=tmesh_name)
            # -- Load mesh --
            path = self._path_to_mesh(tmesh_name, name)
            coord_element = self._load_ufl_element(path, PREFIX + "_coordinate_element")
            coord_name = self.get_attr(path, PREFIX + "_coordinates")
            coordinates = self._load_function_topology(tmesh, coord_element, coord_name)
            mesh = make_mesh_from_coordinates(coordinates, name)
            if self.has_attr(path, PREFIX + "_radial_coordinates"):
                radial_coord_element = self._load_ufl_element(path, PREFIX + "_radial_coordinate_element")
                radial_coord_name = self.get_attr(path, PREFIX + "_radial_coordinates")
                radial_coordinates = self._load_function_topology(tmesh, radial_coord_element, radial_coord_name)
                tV_radial_coord = impl.FunctionSpace(tmesh, radial_coord_element)
                V_radial_coord = impl.WithGeometry.create(tV_radial_coord, mesh)
                radial_coord_function_name = self.get_attr(path, PREFIX + "_radial_coordinate_function")
                mesh.radial_coordinates = Function(V_radial_coord, val=radial_coordinates, name=radial_coord_function_name)
            # The followings are conceptually redundant, but needed.
            path = os.path.join(self._path_to_mesh(tmesh_name, name), PREFIX_EXTRUDED)
            base_mesh_name = self.get_attr(path, PREFIX_EXTRUDED + "_base_mesh")
            mesh._base_mesh = self.load_mesh(base_mesh_name, reorder=reorder, distribution_parameters=distribution_parameters, topology=base_tmesh)
        else:
            utils._init()
            # -- Load mesh topology --
            if topology is None:
                tmesh = self._load_mesh_topology(tmesh_name, reorder, distribution_parameters)
            else:
                if topology.name != tmesh_name:
                    raise RuntimeError(f"Got wrong mesh topology (f{topology.name}): expecting f{tmesh_name}")
                tmesh = topology
            # -- Load coordinates --
            # tmesh.topology_dm has already been redistributed.
            path = self._path_to_mesh(tmesh_name, name)
            # Load firedrake coordinates directly.
            # When implementing checkpointing for MeshHierarchy in the future,
            # we will need to postpone calling tmesh.init().
            tmesh.init()
            coord_element = self._load_ufl_element(path, PREFIX + "_coordinate_element")
            coord_name = self.get_attr(path, PREFIX + "_coordinates")
            coordinates = self._load_function_topology(tmesh, coord_element, coord_name)
            mesh = make_mesh_from_coordinates(coordinates, name)
            # Load plex coordinates for a complete representation of plex.
            tmesh.topology_dm.coordinatesLoad(self.viewer, tmesh.sfXC)
            # Load cell_orientations for immersed meshes.
            path = self._path_to_mesh_immersed(tmesh.name, name)
            if path in self.h5pyfile:
                cell = tmesh.ufl_cell()
                element = finat.ufl.FiniteElement("DP" if cell.is_simplex() else "DQ", cell, 0)
                cell_orientations_tV = self._load_function_space_topology(tmesh, element)
                tmesh_key = self._generate_mesh_key_from_names(tmesh.name,
                                                               tmesh._distribution_name,
                                                               tmesh._permutation_name)
                sd_key = self._get_shared_data_key_for_checkpointing(tmesh, element)
                _, _, lsf = self._function_load_utils[tmesh_key + sd_key]
                nroots, _, _ = lsf.getGraph()
                cell_orientations_a = np.empty(nroots, dtype=utils.IntType)
                cell_orientations_a_iset = PETSc.IS().createGeneral(cell_orientations_a, comm=self._comm)
                cell_orientations_a_iset.setName("_".join([PREFIX_IMMERSED, "cell_orientations_iset"]))
                self.viewer.pushGroup(path)
                cell_orientations_a_iset.load(self.viewer)
                self.viewer.popGroup()
                cell_orientations_a = cell_orientations_a_iset.getIndices()
                cell_orientations = np.empty((tmesh.cell_set.total_size, ), dtype=utils.IntType)
                unit = MPI._typedict[np.dtype(utils.IntType).char]
                lsf.bcastBegin(unit, cell_orientations_a, cell_orientations, MPI.REPLACE)
                lsf.bcastEnd(unit, cell_orientations_a, cell_orientations, MPI.REPLACE)
                # Convert the loaded ("canonical") cell_orientations array for use on the loaded mesh
                # by flipping the values (0 <-> 1) for cells that are mapped "reflected" relative to orientation 0.
                o_r_map = np.array(list(cell_orientations_tV.finat_element.cell.cell_orientation_reflection_map().values()), dtype=np.int32)
                reflected = o_r_map[tmesh.entity_orientations[:, -1]]
                reflected_indices = (reflected == 1)
                cell_orientations[reflected_indices] = 1 - cell_orientations[reflected_indices]
                cell_orientations_name = self.get_attr(path, PREFIX_IMMERSED + "_cell_orientations")
                mesh._cell_orientations = CoordinatelessFunction(cell_orientations_tV, val=cell_orientations, name=cell_orientations_name, dtype=np.int32)
        return mesh

    @PETSc.Log.EventDecorator("LoadMeshTopology")
    def _load_mesh_topology(self, tmesh_name, reorder, distribution_parameters):
        """Load the :class:`~.MeshTopology`.

        :arg tmesh_name: The name of the :class:`~.MeshTopology` to load.
        :arg reorder: whether to reorder the mesh (bool); see :func:`~.Mesh`.
        :arg distribution_parameters: the `distribution_parameters` used for
            distributing the mesh; see :func:`~.Mesh`.
        :returns: The loaded :class:`~.MeshTopology`.
        """
        # -- Load DMPlex --
        path = self._path_to_distributions(tmesh_name)
        load_distribution_permutation = False
        if path in self.h5pyfile:
            _distribution_name, = self.h5pyfile[path].keys()
            path = self._path_to_distribution(tmesh_name, _distribution_name)
            _comm_size = self.get_attr(path, "comm_size")
            if _comm_size == self._comm.size and \
               distribution_parameters is None and reorder is None:
                load_distribution_permutation = True
        if load_distribution_permutation:
            path = self._path_to_distributions(tmesh_name)
            distribution_name, = self.h5pyfile[path].keys()
            path = self._path_to_permutations(tmesh_name, distribution_name)
            permutation_name, = self.h5pyfile[path].keys()
            distribution_parameters = distribution_parameters_noop
            reorder = reorder_noop
        else:
            # Here mimic what Mesh function does.
            if reorder is None:
                reorder = parameters["reorder_meshes"]
            if distribution_parameters is None:
                distribution_parameters = {}
            distribution_name = None
            permutation_name = None
            perm_is = None
        plex = PETSc.DMPlex()
        plex.create(comm=self._comm)
        plex.setName(tmesh_name)
        # Check format
        path = os.path.join(self._path_to_topology(tmesh_name), "topology")
        format = ViewerHDF5.Format.HDF5_PETSC
        self.viewer.pushFormat(format=format)
        plex.distributionSetName(distribution_name)
        sfXB = plex.topologyLoad(self.viewer)
        plex.distributionSetName(None)
        self.viewer.popFormat()
        if load_distribution_permutation:
            chart_size = np.empty(1, dtype=utils.IntType)
            chart_sizes_iset = PETSc.IS().createGeneral(chart_size, comm=self._comm)
            chart_sizes_iset.setName("chart_sizes")
            path = self._path_to_distribution(tmesh_name, distribution_name)
            self.viewer.pushGroup(path)
            chart_sizes_iset.load(self.viewer)
            self.viewer.popGroup()
            chart_size = chart_sizes_iset.getIndices().item()
            perm = np.empty(chart_size, dtype=utils.IntType)
            perm_is = PETSc.IS().createGeneral(perm, comm=self._comm)
            path = self._path_to_permutation(tmesh_name, distribution_name, permutation_name)
            self.viewer.pushGroup(path)
            perm_is.setName("permutation")
            perm_is.load(self.viewer)
            perm_is.setName(None)
            self.viewer.popGroup()
        else:
            perm_is = None
        # -- Construct Mesh (Topology) --
        # Use public API so pass user comm (self.comm)
        tmesh = MeshTopology(plex, name=plex.getName(), reorder=reorder,
                             distribution_parameters=distribution_parameters, sfXB=sfXB, perm_is=perm_is,
                             distribution_name=distribution_name, permutation_name=permutation_name,
                             comm=self.comm)
        self.viewer.pushFormat(format=format)
        # tmesh.topology_dm has already been redistributed.
        sfXCtemp = tmesh.sfXB.compose(tmesh.sfBC) if tmesh.sfBC is not None else tmesh.sfXB
        plex.labelsLoad(self.viewer, sfXCtemp)
        self.viewer.popFormat()
        # These labels are distribution dependent.
        # We should be able to save/load labels selectively.
        plex.removeLabel("pyop2_core")
        plex.removeLabel("pyop2_owned")
        plex.removeLabel("pyop2_ghost")
        return tmesh

    @PETSc.Log.EventDecorator("LoadFunctionSpace")
    def _load_function_space(self, mesh, name):
        mesh.init()
        mesh_key = self._generate_mesh_key_from_names(mesh.name,
                                                      mesh.topology._distribution_name,
                                                      mesh.topology._permutation_name)
        V_key = mesh_key + (name, )
        if V_key in self._function_spaces:
            return self._function_spaces[V_key]
        tmesh = mesh.topology
        if self._is_mixed_function_space(mesh.name, name):
            base_path = self._path_to_mixed_function_space(mesh.name, name)
            n = self.get_attr(base_path, PREFIX + "_num_sub_spaces")
            Vsub_list = []
            for i in range(n):
                path = os.path.join(base_path, str(i))
                Vsub_name = self.get_attr(path, PREFIX + "_function_space")
                Vsub = self._load_function_space(mesh, Vsub_name)
                Vsub_list.append(Vsub)
            V = functools.reduce(lambda a, b: a * b, Vsub_list)
        elif self._is_function_space(tmesh.name, mesh.name, name):
            # Load function space data
            path = self._path_to_function_space(tmesh.name, mesh.name, name)
            element = self._load_ufl_element(path, PREFIX + "_ufl_element")
            tV = self._load_function_space_topology(tmesh, element)
            # Construct function space
            V = impl.WithGeometry.create(tV, mesh)
        else:
            raise RuntimeError(f"""
                FunctionSpace ({name}) not found in either of the following path in {self.filename}:

                {self._path_to_mixed_function_space(mesh.name, name)}
                {self._path_to_function_space(tmesh.name, mesh.name, name)}
            """)
        self._function_spaces[V_key] = V
        return V

    @PETSc.Log.EventDecorator("LoadFunctionSpaceTopology")
    def _load_function_space_topology(self, tmesh, element):
        tmesh.init()
        if element.family() == "Real":
            return impl.RealFunctionSpace(tmesh, element, "unused_name")
        tmesh_key = self._generate_mesh_key_from_names(tmesh.name,
                                                       tmesh._distribution_name,
                                                       tmesh._permutation_name)
        sd_key = self._get_shared_data_key_for_checkpointing(tmesh, element)
        if tmesh_key + sd_key not in self._function_load_utils:
            topology_dm = tmesh.topology_dm
            dm = PETSc.DMShell().create(comm=tmesh._comm)
            dm.setName(self._get_dm_name_for_checkpointing(tmesh, element))
            dm.setPointSF(topology_dm.getPointSF())
            section = PETSc.Section().create(comm=tmesh._comm)
            section.setPermutation(tmesh._dm_renumbering)
            dm.setSection(section)
            base_tmesh = tmesh._base_mesh if isinstance(tmesh, ExtrudedMeshTopology) else tmesh
            sfXC = base_tmesh.sfXC
            topology_dm.setName(tmesh.name)
            gsf, lsf = topology_dm.sectionLoad(self.viewer, dm, sfXC)
            topology_dm.setName(base_tmesh.name)
            nodes_per_entity, real_tensorproduct, block_size = sd_key
            # Don't cache if the section has been expanded by block_size
            if block_size == 1:
                cached_section = get_global_numbering(tmesh, (nodes_per_entity, real_tensorproduct), global_numbering=dm.getSection())
                if dm.getSection() is not cached_section:
                    # The same section has already been cached.
                    dm.setSection(cached_section)
            self._function_load_utils[tmesh_key + sd_key] = (dm, gsf, lsf)
        return impl.FunctionSpace(tmesh, element)

    @PETSc.Log.EventDecorator("LoadFunction")
    def load_function(self, mesh, name, idx=None):
        r"""Load a :class:`~.Function` defined on `mesh`.

        :arg mesh: the mesh on which the function is defined.
        :arg name: the name of the :class:`~.Function` to load.
        :kwarg idx: optional timestepping index. A function can
            be loaded with idx only when it was saved with idx.
        :returns: the loaded :class:`~.Function`.
        """
        tmesh = mesh.topology
        if name in self._get_mixed_function_name_mixed_function_space_name_map(mesh.name):
            V_name = self._get_mixed_function_name_mixed_function_space_name_map(mesh.name)[name]
            V = self._load_function_space(mesh, V_name)
            base_path = self._path_to_mixed_function(mesh.name, V_name, name)
            fsub_list = []
            for i, Vsub in enumerate(V):
                path = os.path.join(base_path, str(i))
                fsub_name = self.get_attr(path, PREFIX + "_function")
                fsub = self.load_function(mesh, fsub_name, idx=idx)
                fsub_list.append(fsub)
            dat = op2.MixedDat(fsub.dat for fsub in fsub_list)
            return Function(V, val=dat, name=name)
        elif name in self._get_function_name_function_space_name_map(self._get_mesh_name_topology_name_map()[mesh.name], mesh.name):
            # Load function space
            tmesh_name = self._get_mesh_name_topology_name_map()[mesh.name]
            V_name = self._get_function_name_function_space_name_map(tmesh_name, mesh.name)[name]
            V = self._load_function_space(mesh, V_name)
            # Load vec
            tV = V.topological
            # -- Embed if necessary
            path = self._path_to_function(tmesh_name, mesh.name, V_name, name)
            if PREFIX_EMBEDDED in self.h5pyfile[path]:
                path = self._path_to_function_embedded(tmesh_name, mesh.name, V_name, name)
                _name = self.get_attr(path, PREFIX_EMBEDDED + "_function")
                _f = self.load_function(mesh, _name, idx=idx)
                element = V.ufl_element()
                _element = get_embedding_element_for_checkpointing(element)
                method = get_embedding_method_for_checkpointing(element)
                assert _element == _f.function_space().ufl_element()
                f = Function(V, name=name)
                self._project_function_for_checkpointing(f, _f, method)
                return f
            else:
                tf_name = self.get_attr(path, PREFIX + "_vec")
                tf = self._load_function_topology(tV.mesh(), tV.ufl_element(), tf_name, idx=idx)
                return Function(V, val=tf, name=name)
        else:
            raise RuntimeError(f"""
                Function ({name}) not found under either of the following path in {self.filename}:

                {self._path_to_mixed_mesh(mesh.name)}
                {self._path_to_mesh(tmesh.name, mesh.name)}
            """)

    @PETSc.Log.EventDecorator("LoadFunctionTopology")
    def _load_function_topology(self, tmesh, element, tf_name, idx=None):
        tV = self._load_function_space_topology(tmesh, element)
        topology_dm = tmesh.topology_dm
        dm_name = self._get_dm_name_for_checkpointing(tmesh, element)
        tf = CoordinatelessFunction(tV, name=tf_name)
        path = self._path_to_vec(tmesh.name, dm_name, tf_name)
        if idx is not None:
            self.viewer.pushTimestepping()
            self.viewer.setTimestep(idx)
        if element.family() == "Real":
            assert not isinstance(element, (finat.ufl.VectorElement, finat.ufl.TensorElement))
            value = self.get_attr(path, "_".join([PREFIX, "value" if idx is None else "value_" + str(idx)]))
            tf.dat.data[...] = value
        else:
            if path in self.h5pyfile:
                timestepping = self.has_attr(os.path.join(path, tf.name()), "timestepping")
                if timestepping:
                    assert idx is not None, "In timestepping mode: idx parameter must be set"
                else:
                    assert idx is None, "In non-timestepping mode: idx parameter msut not be set"
            else:
                raise RuntimeError(f"Function {path} not found in {self.filename}")
            with tf.dat.vec_wo as vec:
                vec.setName(tf_name)
                sd_key = self._get_shared_data_key_for_checkpointing(tmesh, element)
                tmesh_key = self._generate_mesh_key_from_names(tmesh.name,
                                                               tmesh._distribution_name,
                                                               tmesh._permutation_name)
                dm, sf, _ = self._function_load_utils[tmesh_key + sd_key]
                base_tmesh_name = topology_dm.getName()
                topology_dm.setName(tmesh.name)
                topology_dm.globalVectorLoad(self.viewer, dm, sf, vec)
                topology_dm.setName(base_tmesh_name)
        if idx is not None:
            self.viewer.popTimestepping()
        return tf

    def _generate_mesh_key(self, mesh_name, distribution_name, permutation_name, reorder, distribution_parameters):
        dist_key = frozenset(distribution_parameters.items())
        return (self.filename, self.commkey, mesh_name, distribution_name, permutation_name, reorder, dist_key)

    def _generate_mesh_key_from_names(self, mesh_name, distribution_name, permutation_name):
        return (self.filename, self.commkey, mesh_name, distribution_name, permutation_name)

    def _generate_function_space_name(self, V):
        """Return a unique function space name."""
        V_names = [PREFIX + "_function_space"]
        for Vsub in V:
            elem = Vsub.ufl_element()
            if isinstance(elem, finat.ufl.RestrictedElement):
                # RestrictedElement.shortstr() contains '<>|{}'.
                elem_name = "RestrictedElement(%s,%s)" % (elem.sub_element().shortstr(), elem.restriction_domain())
            elif isinstance(elem, finat.ufl.EnrichedElement):
                # EnrichedElement.shortstr() contains '<>+'.
                elem_name = "EnrichedElement(%s)" % ",".join(e.shortstr() for e in elem._elements)
            else:
                elem_name = elem.shortstr()
                elem_name = elem_name.replace('?', 'None')
                # MixedElement, VectorElement, TensorElement
                # use '<' and '>' in shortstr(), but changing
                # these to '(' and ')' causes no confusion.
                elem_name = elem_name.replace('<', '(').replace('>', ')')
            V_names.append("_".join([Vsub.mesh().name, elem_name]))
        return "_".join(V_names)

    def _generate_dm_name(self, nodes_per_entity, real_tensorproduct, block_size):
        return "_".join([PREFIX, "dm"]
                        + [str(n) for n in nodes_per_entity]
                        + [str(real_tensorproduct)]
                        + [str(block_size)])

    def _get_shared_data_key_for_checkpointing(self, mesh, ufl_element):
        finat_element = create_element(ufl_element)
        real_tensorproduct = eutils.is_real_tensor_product_element(finat_element)
        entity_dofs = finat_element.entity_dofs()
        nodes_per_entity = tuple(mesh.make_dofs_per_plex_entity(entity_dofs))
        if isinstance(ufl_element, finat.ufl.TensorElement):
            shape = ufl_element.reference_value_shape
            block_size = np.prod(shape)
        elif isinstance(ufl_element, finat.ufl.VectorElement):
            shape = ufl_element.value_shape[:1]
            block_size = np.prod(shape)
        else:
            block_size = 1
        return (nodes_per_entity, real_tensorproduct, block_size)

    def _get_dm_for_checkpointing(self, tV):
        sd_key = self._get_shared_data_key_for_checkpointing(tV.mesh(), tV.ufl_element())
        if isinstance(tV.ufl_element(), (finat.ufl.VectorElement, finat.ufl.TensorElement)):
            nodes_per_entity, real_tensorproduct, block_size = sd_key
            global_numbering, _ = tV.mesh().create_section(nodes_per_entity, real_tensorproduct, block_size=block_size)
            topology_dm = tV.mesh().topology_dm
            dm = PETSc.DMShell().create(tV.mesh()._comm)
            dm.setPointSF(topology_dm.getPointSF())
            dm.setSection(global_numbering)
        else:
            dm = tV.dm
        dm.setName(self._generate_dm_name(*sd_key))
        return dm

    def _get_dm_name_for_checkpointing(self, tmesh, ufl_element):
        if ufl_element.family() == "Real":
            block_size = 1
            return "_".join([PREFIX, "dm", "real", str(block_size)])
        sd_key = self._get_shared_data_key_for_checkpointing(tmesh, ufl_element)
        return self._generate_dm_name(*sd_key)

    def _path_to_topologies(self):
        return "topologies"

    def _path_to_topology(self, tmesh_name):
        return os.path.join(self._path_to_topologies(), tmesh_name)

    def _path_to_topology_extruded(self, tmesh_name):
        return os.path.join(self._path_to_topology(tmesh_name), PREFIX_EXTRUDED)

    def _path_to_dms(self, tmesh_name):
        return os.path.join(self._path_to_topology(tmesh_name), "dms")

    def _path_to_dm(self, tmesh_name, dm_name):
        return os.path.join(self._path_to_dms(tmesh_name), dm_name)

    def _path_to_vecs(self, tmesh_name, dm_name):
        return os.path.join(self._path_to_dm(tmesh_name, dm_name), "vecs")

    def _path_to_vec(self, tmesh_name, dm_name, tf_name):
        return os.path.join(self._path_to_vecs(tmesh_name, dm_name), tf_name)

    def _path_to_vec_timestepping(self, tmesh_name, dm_name, tf_name):
        return os.path.join(self._path_to_vec(tmesh_name, dm_name, tf_name), PREFIX_TIMESTEPPING)

    def _path_to_meshes(self, tmesh_name):
        return os.path.join(self._path_to_topology(tmesh_name), PREFIX + "_meshes")

    def _path_to_mesh(self, tmesh_name, mesh_name):
        return os.path.join(self._path_to_meshes(tmesh_name), mesh_name)

    def _path_to_mesh_immersed(self, tmesh_name, mesh_name):
        return os.path.join(self._path_to_mesh(tmesh_name, mesh_name), PREFIX_IMMERSED)

    def _path_to_function_spaces(self, tmesh_name, mesh_name):
        return os.path.join(self._path_to_mesh(tmesh_name, mesh_name), PREFIX + "_function_spaces")

    def _path_to_function_space(self, tmesh_name, mesh_name, V_name):
        return os.path.join(self._path_to_function_spaces(tmesh_name, mesh_name), V_name)

    def _path_to_functions(self, tmesh_name, mesh_name, V_name):
        return os.path.join(self._path_to_function_space(tmesh_name, mesh_name, V_name), PREFIX + "_functions")

    def _path_to_function(self, tmesh_name, mesh_name, V_name, function_name):
        return os.path.join(self._path_to_functions(tmesh_name, mesh_name, V_name), function_name)

    def _path_to_function_timestepping(self, tmesh_name, mesh_name, V_name, function_name):
        return os.path.join(self._path_to_function(tmesh_name, mesh_name, V_name, function_name), PREFIX_TIMESTEPPING)

    def _path_to_function_embedded(self, tmesh_name, mesh_name, V_name, function_name):
        return os.path.join(self._path_to_function(tmesh_name, mesh_name, V_name, function_name), PREFIX_EMBEDDED)

    def _path_to_mixed_meshes(self):
        return os.path.join(self._path_to_topologies(), PREFIX + "_mixed_meshes")

    def _path_to_mixed_mesh(self, mesh_name):
        return os.path.join(self._path_to_mixed_meshes(), mesh_name)

    def _path_to_mixed_function_spaces(self, mesh_name):
        return os.path.join(self._path_to_mixed_mesh(mesh_name), PREFIX + "_mixed_function_spaces")

    def _path_to_mixed_function_space(self, mesh_name, V_name):
        return os.path.join(self._path_to_mixed_function_spaces(mesh_name), V_name)

    def _path_to_mixed_functions(self, mesh_name, V_name):
        return os.path.join(self._path_to_mixed_function_space(mesh_name, V_name), PREFIX + "_functions")

    def _path_to_mixed_function(self, mesh_name, V_name, function_name):
        return os.path.join(self._path_to_mixed_functions(mesh_name, V_name), function_name)

    def _path_to_distributions(self, tmesh_name):
        return os.path.join(self._path_to_topology(tmesh_name), "distributions")

    def _path_to_distribution(self, tmesh_name, distribution_name):
        return os.path.join(self._path_to_distributions(tmesh_name), distribution_name)

    def _path_to_permutations(self, tmesh_name, distribution_name):
        return os.path.join(self._path_to_distribution(tmesh_name, distribution_name), "permutations")

    def _path_to_permutation(self, tmesh_name, distribution_name, permutation_name):
        return os.path.join(self._path_to_permutations(tmesh_name, distribution_name), permutation_name)

    def _pickle(self, obj):
        return np.void(pickle.dumps(obj))

    def _unpickle(self, obj):
        return pickle.loads(obj.tobytes())

    def _write_pickled_dict(self, path, name, the_dict):
        """Pickle a dict and write it as attribute.

        :arg path: the path at which the attribute is to be written.
        :arg name: the name of the attribute.
        :arg the_dict: the dict to pickle and write.
        """
        self.require_group(path)
        self.set_attr(path, name, self._pickle(the_dict))

    def _read_pickled_dict(self, path, name):
        """Read attribute and unpickle it to get a dict.

        :arg path: the path at which the attribute is found.
        :arg name: the name of the attribute.
        :returns: the unpickled dict
        """
        if path in self.h5pyfile and self.has_attr(path, name):
            return self._unpickle(self.get_attr(path, name))
        else:
            return {}

    def _update_pickled_dict(self, name, new_item, *args):
        the_dict = getattr(self, "_get_" + name)(*args)
        the_dict.update(new_item)
        getattr(self, "_set_" + name)(*args, the_dict)

    def _save_ufl_element(self, path, name, elem):
        self.set_attr(path, name + "_repr", repr(elem))

    def _load_ufl_element(self, path, name):
        if self.has_attr(path, name + "_repr"):
            globals = {}
            locals = {}
            exec("from ufl import *", globals, locals)
            exec("from finat.ufl import *", globals, locals)
            return eval(self.get_attr(path, name + "_repr"), globals, locals)
        else:
            return self._unpickle(self.get_attr(path, name))  # backward compat.

    def _set_mesh_name_topology_name_map(self, new_item):
        path = self._path_to_topologies()
        self._write_pickled_dict(path, PREFIX + "_mesh_name_topology_name_map", new_item)

    def _get_mesh_name_topology_name_map(self):
        path = self._path_to_topologies()
        return self._read_pickled_dict(path, PREFIX + "_mesh_name_topology_name_map")

    def _update_mesh_name_topology_name_map(self, new_item):
        self._update_pickled_dict("mesh_name_topology_name_map", new_item)

    def _set_function_name_function_space_name_map(self, tmesh_name, mesh_name, new_item):
        path = self._path_to_mesh(tmesh_name, mesh_name)
        self._write_pickled_dict(path, PREFIX + "_function_name_function_space_name_map", new_item)

    def _get_function_name_function_space_name_map(self, tmesh_name, mesh_name):
        path = self._path_to_mesh(tmesh_name, mesh_name)
        return self._read_pickled_dict(path, PREFIX + "_function_name_function_space_name_map")

    def _update_function_name_function_space_name_map(self, tmesh_name, mesh_name, new_item):
        self._update_pickled_dict("function_name_function_space_name_map", new_item, tmesh_name, mesh_name)

    def _set_mixed_function_name_mixed_function_space_name_map(self, mesh_name, new_item):
        path = self._path_to_mixed_mesh(mesh_name)
        self._write_pickled_dict(path, PREFIX + "_mixed_function_name_mixed_function_space_name_map", new_item)

    def _get_mixed_function_name_mixed_function_space_name_map(self, mesh_name):
        path = self._path_to_mixed_mesh(mesh_name)
        return self._read_pickled_dict(path, PREFIX + "_mixed_function_name_mixed_function_space_name_map")

    def _update_mixed_function_name_mixed_function_space_name_map(self, mesh_name, new_item):
        self._update_pickled_dict("mixed_function_name_mixed_function_space_name_map", new_item, mesh_name)

    def _is_function_space(self, tmesh_name, mesh_name, V_name):
        path = self._path_to_function_spaces(tmesh_name, mesh_name)
        if path in self.h5pyfile:
            if V_name in self.h5pyfile[path]:
                return True
        return False

    def _is_mixed_function_space(self, mesh_name, V_name):
        base_path = self._path_to_mixed_mesh(mesh_name)
        if base_path in self.h5pyfile:
            path = self._path_to_mixed_function_spaces(mesh_name)
            if V_name in self.h5pyfile[path]:
                return True
        return False

    def _project_function_for_checkpointing(self, f, _f, method):
        if method == "project":
            getattr(f, method)(_f, solver_parameters={"ksp_rtol": 1.e-16})
        elif method == "interpolate":
            getattr(f, method)(_f)
        else:
            raise ValueError(f"Unknown method for projecting: {method}")

    @property
    def h5pyfile(self):
        r"""An h5py File object pointing at the open file handle."""
        if hasattr(self, '_h5pyfile'):
            return self._h5pyfile
        self._h5pyfile = h5i.get_h5py_file(self.viewer)
        return self._h5pyfile

    def set_attr(self, path, key, val):
        r"""Set an HDF5 attribute at specified path.

        :arg path: The path at which the attribute is set.
        :arg key: The attribute key.
        :arg val: The attribute value.
        """
        self.h5pyfile[path].attrs[key] = val

    def get_attr(self, path, key):
        r"""Get an HDF5 attribute at specified path.

        :arg path: The path at which the attribute is found.
        :arg key: The attribute key.
        :returns: The attribute value.
        """
        return self.h5pyfile[path].attrs[key]

    def has_attr(self, path, key):
        r"""Check if an HDF5 attribute exists at specified path.

        :arg path: The path at which the attribute is sought.
        :arg key: The attribute key.
        :returns: `True` if the attribute is found.
        """
        return key in self.h5pyfile[path].attrs

    def set_attr_byte_string(self, path, key, string):
        """Store string as byte string.

        Parameters
        ----------
        path : str
            Path.
        key : str
            Name of the attribute.
        string : str
            String to be stored as a byte string.

        """
        type_id = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        type_id.set_size(len(string) + 1)
        type_id.set_strpad(h5py.h5t.STR_NULLTERM)
        space = h5py.h5s.create(h5py.h5s.SCALAR)
        grp = self.require_group(path)
        aid = h5py.h5a.create(grp.id, key.encode('utf-8'), type_id, space)
        aid.write(np.array(string.encode()))

    def get_attr_byte_string(self, path, key):
        """Return string stored as byte string.

        Parameters
        ----------
        path : str
            Path.
        key : str
            Name of the attribute.

        Returns
        -------
        str
            String stored as byte string.

        """
        return self.get_attr(path, key).decode()

    def close(self):
        r"""Close the checkpoint file."""
        if hasattr(self, "_h5pyfile"):
            self._h5pyfile.flush()
            del self._h5pyfile
        self.viewer.destroy()

    def create_group(self, name, track_order=None):
        r"""Mimic :meth:`h5py.Group.create_group`.

        :arg name: The name of the group.
        :kwarg track_order: Whether to track dataset/group/attribute creation order.

        In this method we customise the :class:`h5py.h5p.PropGCID`
        object from which we create the `h5py.h5g.GroupID` object
        to avoid the "object header message is too large" error
        and/or "record is not in B-tree" error when storing many
        (hundreds of) attributes; see
        `this PR <https://github.com/firedrakeproject/firedrake/pull/2432>`_.

        TODO: Lift this to upstream somehow.
        """
        _self = self.h5pyfile
        if track_order is None:
            track_order = h5py.h5.get_config().track_order
        with h5py._hl.base.phil:
            name, lcpl = _self._e(name, lcpl=True)
            if track_order:
                firedrake.warning("track_order = True is known to cause 'record is not in B-tree' error when trying to store many attributes.")
                gcpl = h5py.Group._gcpl_crt_order
            else:
                gcpl = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
                # The following will remove the 64KiB limit for storing
                # attributes, but it makes checkpointing very slow and
                # actually seems not necessary to avoid the errors.
                # >>> gcpl.set_attr_phase_change(0, 0)
                gcpl.set_link_creation_order(h5py.h5p.CRT_ORDER_TRACKED)
                gcpl.set_attr_creation_order(h5py.h5p.CRT_ORDER_TRACKED)
                # The followings are supposed to improve performance, but
                # makes the program fail with "record is not in B-tree" error.
                # >>> gcpl.set_link_creation_order(h5py.h5p.CRT_ORDER_TRACKED | h5py.h5p.CRT_ORDER_INDEXED)
                # >>> gcpl.set_attr_creation_order(h5py.h5p.CRT_ORDER_TRACKED | h5py.h5p.CRT_ORDER_INDEXED)
            gid = h5py.h5g.create(_self.id, name, lcpl=lcpl, gcpl=gcpl)
            return h5py.Group(gid)

    def require_group(self, name):
        r"""Mimic :meth:`h5py.Group.require_group`.

        :arg name: name of the group.

        This method uses :meth:`~.CheckpointFile.create_group`
        instead of :meth:`h5py.Group.create_group` to create
        an :class:`h5py.Group` object from an :class:`h5py.h5g.GroupID`
        constructed with a custom :class:`h5py.h5p.PropGCID` object
        (often named `gcpl`); see :meth:`h5py.Group.create_group`.

        TODO: Lift this to upstream somehow.
        """
        _self = self.h5pyfile
        with h5py._hl.base.phil:
            if name not in _self:
                return self.create_group(name)
            grp = _self[name]
            if not isinstance(grp, h5py.Group):
                raise TypeError("Incompatible object (%s) already exists" % grp.__class__.__name__)
            return grp
