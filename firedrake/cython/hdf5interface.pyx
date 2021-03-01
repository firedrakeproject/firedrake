# cython: language_level=3
import h5py
cimport petsc4py.PETSc as PETSc


cdef extern from "hdf5.h":
    ctypedef int hid_t


cdef extern from "petscviewerhdf5.h":
    int PetscViewerHDF5GetFileId(PETSc.PetscViewer,hid_t*)


def get_h5py_file(PETSc.Viewer vwr not None):
    """Attempt to convert PETSc viewer file handle to h5py File.

    :arg vwr: The PETSc Viewer (must have type HDF5).

    .. warning::

       For this to work, h5py and PETSc must both have been compiled
       against *the same* HDF5 library (otherwise the file handles are
       not interchangeable).  This is the likeliest reason for failure
       when attempting the conversion."""
    cdef hid_t fid = 0
    cdef int ierr = 0

    if vwr.type != vwr.Type.HDF5:
        raise TypeError("Viewer is not an HDF5 viewer")
    ierr = PetscViewerHDF5GetFileId(vwr.vwr, &fid)
    if ierr != 0:
        raise RuntimeError("Unable to get file handle")

    try:
        objid = h5py.h5i.wrap_identifier(fid)
    except ValueError:
        raise RuntimeError("Unable to convert handle to h5py object. Likely h5py not linked to same HDF5 as PETSc")

    if type(objid) is not h5py.h5f.FileID:
        raise TypeError("Provided handle doesn't reference a file")
    # We got a borrowed reference to the file id from PETSc, need to
    # inc-ref it so that the file isn't closed behind our backs.
    h5py.h5i.inc_ref(objid)
    return h5py.File(objid)
