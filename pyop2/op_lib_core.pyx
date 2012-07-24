"""
Wrap OP2 library for PyOP2

The C level OP2 runtime needs to be aware of the data structures that
the python layer is managing.  So that things like plan construction
and halo swapping actually have some data to deal with.  Equally, the
python level objects need to keep a hold of their C layer counterparts
for interoperability.  All this interfacing is dealt with here.

Naming conventions:

Wrappers around C functions use the same names as in the OP2-Common
library.  Hence, the python classes corresponding to C structs are not
opSet, opDat and so forth, but rather op_set and op_dat.

How it works:

A python object that has a C counterpart has a slot named
_lib_handle.  This is either None, meaning the C initialiser has not
yet been called, or else a handle to the Cython class wrapping the C
data structure.

To get this interfacing library, do something like:

    import op_lib_core as core

To build the C data structure on the python side, the class should do
the following when necessary (probably in __init__):

    if self._lib_handle is None:
        self._lib_handle = core.op_set(self)

The above example is obviously for an op2.Set object.

C layer function calls that require a set as an argument a wrapped
such that you don't need to worry about passing the handle, instead,
just pass the python object.  That is, you do:

   core.op_function(set)

not

   core.op_function(set._lib_handle)

Most C level objects are completely opaque to the python layer.  The
exception is the op_plan structure, whose data must be marshalled to
the relevant device on the python side.  The slots of the op_plan
struct are exposed as properties to python.  Thus, to get the ind_map
array from a plan you do:

   plan = core.op_plan(kernel, set, *args)

   ind_map = plan.ind_map

Scalars are returned as scalars, arrays are wrapped in a numpy array
of the appropriate size.

WARNING, the arrays returned by these properties have their data
buffer pointing to the C layer's data.  As such, they should be
considered read-only.  If you modify them on the python side, the plan
will likely be wrong.


TODO:
Cleanup of C level datastructures is currently not handled.
"""

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport _op_lib_core as core

np.import_array()

cdef data_to_numpy_array_with_template(void * ptr, arr):
    """Return an array with the same properties as ARR with data from PTR."""
    cdef np.npy_intp dim = np.size(arr)
    cdef np.dtype t = arr.dtype
    shape = np.shape(arr)
    return np.PyArray_SimpleNewFromData(1, &dim, t.type_num, ptr).reshape(shape)

cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp size, int t):
    """Return an array of SIZE elements (each of type T) with data from PTR."""
    return np.PyArray_SimpleNewFromData(1, &size, t, ptr)

def op_init(args, diags):
    """Initialise OP2

ARGS should be a list of strings to pass as "command-line" arguments
DIAGS should be an integer specifying the diagnostic level.  The
larger it is, the more chatty OP2 will be."""
    cdef char **argv
    cdef int diag_level = diags
    if args is None:
        core.op_init_core(0, NULL, diag_level)
        return
    args = [bytes(x) for x in args]
    argv = <char **>malloc(sizeof(char *) * len(args))
    if argv is NULL:
        raise MemoryError()
    try:
        for i, a in enumerate(args):
            argv[i] = a
        core.op_init_core(len(args), argv, diag_level)
    finally:
        # We can free argv here, because op_init_core doesn't keep a
        # handle to the arguments.
        free(argv)

def op_exit():
    """Clean up C level data"""
    core.op_rt_exit()
    core.op_exit_core()

cdef class op_set:
    cdef core.op_set _handle
    def __cinit__(self, set):
        """Instantiate a C-level op_set from SET"""
        cdef int size = set._size
        cdef char * name = set._name
        self._handle = core.op_decl_set_core(size, name)

cdef class op_dat:
    cdef core.op_dat _handle
    def __cinit__(self, dat):
        """Instantiate a C-level op_dat from DAT"""
        cdef op_set set = dat._dataset._lib_handle
        cdef int dim = dat._dim[0]
        cdef int size = dat._data.dtype.itemsize
        cdef np.ndarray data = dat._data
        cdef char * name = dat._name
        cdef char * type
        tmp = dat._data.dtype.name + ":soa" if dat.soa else ""
        type = tmp
        self._handle = core.op_decl_dat_core(set._handle, dim, type,
                                             size, <char *>data.data, name)

cdef class op_map:
    cdef core.op_map _handle
    def __cinit__(self, map):
        """Instantiate a C-level op_map from MAP"""
        cdef op_set frm = map._iterset._lib_handle
        cdef op_set to = map._dataset._lib_handle
        cdef int dim = map._dim
        cdef np.ndarray values = map._values
        cdef char * name = map._name
        if values.size == 0:
            self._handle = core.op_decl_map_core(frm._handle, to._handle,
                                                 dim, NULL, name)
        else:
            self._handle = core.op_decl_map_core(frm._handle, to._handle, dim,
                                                 <int *>values.data, name)

cdef class op_arg:
    cdef core.op_arg _handle
    def __cinit__(self, arg, dat=False, gbl=False):
        """Instantiate a C-level op_arg from ARG

If DAT is True, this arg is actually an op_dat.
If GBL is True, this arg is actually an op_gbl.

The reason we have to pass these extra arguments in is because we
can't import sequential into this file, and hence cannot do
isinstance(arg, Dat)."""
        cdef int idx
        cdef op_map map
        cdef core.op_map _map
        cdef int dim
        cdef int size
        cdef char * type
        cdef core.op_access acc
        cdef np.ndarray data
        cdef op_dat _dat
        if not (dat or gbl):
            raise RuntimeError("Must tell me what type of arg this is")

        if dat and gbl:
            raise RuntimeError("An argument cannot be both a Dat and Global!")

        # Map Python-layer access descriptors down to C enum
        acc = {'READ'  : core.OP_READ,
               'WRITE' : core.OP_WRITE,
               'RW'    : core.OP_RW,
               'INC'   : core.OP_INC,
               'MIN'   : core.OP_MIN,
               'MAX'   : core.OP_MAX}[arg.access._mode]

        if dat:
            _dat = arg.data._lib_handle
            if arg.is_indirect():
                idx = arg.idx
                map = arg.map._lib_handle
                _map = map._handle
            else:
                idx = -1
                _map = <core.op_map>NULL
            dim = arg.data._dim[0]
            type = arg.data.dtype.name
            self._handle = core.op_arg_dat_core(_dat._handle, idx, _map,
                                                dim, type, acc)
        elif gbl:
            dim = arg.data._dim[0]
            size = arg.data._data.size/dim
            type = arg.data.dtype.name
            data = arg.data._data
            self._handle = core.op_arg_gbl_core(<char *>data.data, dim,
                                                type, size, acc)

cdef class op_plan:
    cdef core.op_plan *_handle
    cdef int set_size
    cdef int nind_ele
    def __cinit__(self, kernel, iset, *args):
        """Instantiate a C-level op_plan for a parallel loop.

Arguments to this constructor should be the arguments of the parallel
loop, i.e. the KERNEL, the ISET (iteration set) and any
further ARGS."""
        cdef op_set _set = iset._lib_handle
        cdef char * name = kernel._name
        cdef int part_size = 0
        cdef int nargs = len(args)
        cdef op_arg _arg
        cdef core.op_arg *_args
        cdef int ninds
        cdef int *inds
        cdef int i
        cdef int ind = 0

        self.set_size = _set._handle.size
        # Size of the plan is incremented by the exec_size if any
        # argument is indirect and not read-only.  exec_size is only
        # ever non-zero in an MPI setting.
        if any(arg.is_indirect_and_not_read() for arg in args):
            self.set_size += _set._handle.exec_size

        # Count number of indirect arguments.  This will need changing
        # once we deal with vector maps.
        self.nind_ele = sum(arg.is_indirect() for arg in args)

        # Build list of args to pass to C-level op_plan function.
        _args = <core.op_arg *>malloc(nargs * sizeof(core.op_arg))
        if _args is NULL:
            raise MemoryError()
        inds = <int *>malloc(nargs * sizeof(int))
        if inds is NULL:
            raise MemoryError()
        try:
            # _args[i] is the ith argument
            # inds[i] is:
            #   -1 if the ith argument is direct
            #   n >= 0 if the ith argument is indirect
            #    where n counts the number of unique indirect dats.
            #    thus, if there are two arguments, both indirect but
            #    both referencing the same dat/map pair (with
            #    different indices) then ninds = {0,0}
            ninds = 0
            # Keep track of which indirect args we've already seen to
            # get value of inds correct.
            d = {}
            for i in range(nargs):
                inds[i] = -1    # Assume direct
                arg = args[i]
                arg.build_core_arg()
                _arg = arg._lib_handle
                _args[i] = _arg._handle
                # Fix up inds[i] in indirect case
                if arg.is_indirect():
                    if d.has_key(arg):
                        inds[i] = d[arg]
                    else:
                        inds[i] = ind
                        d[arg] = ind
                        ind += 1
                        ninds += 1
            self._handle = core.op_plan_core(name, _set._handle,
                                             part_size, nargs, _args,
                                             ninds, inds)
        finally:
            # We can free these because op_plan_core doesn't keep a
            # handle to them.
            free(_args)
            free(inds)

    property ninds:
        """Return the number of unique indirect arguments"""
        def __get__(self):
            return self._handle.ninds

    property nargs:
        """Return the total number of arguments"""
        def __get__(self):
            return self._handle.nargs

    property part_size:
        """Return the partition size.

Normally this will be zero, indicating that the plan should guess the
best partition size."""
        def __get__(self):
            return self._handle.part_size

    property nthrcol:
        """The number of thread colours in each block.

There are nblocks blocks so nthrcol[i] gives the number of colours in
the ith block."""
        def __get__(self):
            cdef int size = self.nblocks
            return data_to_numpy_array_with_spec(self._handle.nthrcol, size, np.NPY_INT32)

    property thrcol:
        """Thread colours of each element.

The ith entry in this array is the colour of ith element of the
iteration set the plan is defined on."""
        def __get__(self):
            cdef int size = self.set_size
            return data_to_numpy_array_with_spec(self._handle.thrcol, size, np.NPY_INT32)

    property offset:
        """The offset into renumbered mappings for each block.

This tells us where in loc_map (q.v.) this block's renumbered mapping
starts."""
        def __get__(self):
            cdef int size = self.nblocks
            return data_to_numpy_array_with_spec(self._handle.offset, size, np.NPY_INT32)

    property ind_map:
        """Renumbered mappings for each indirect dataset.

The ith indirect dataset's mapping starts at:

    ind_map[(i-1) * set_size]

But we need to fix this up for the block we're currently processing,
so see also ind_offs.
"""
        def __get__(self):
            cdef int size = self.set_size * self.nind_ele
            return data_to_numpy_array_with_spec(self._handle.ind_map, size, np.NPY_INT32)

    property ind_offs:
        """Offsets for each block into ind_map (q.v.).

The ith /unique/ indirect dataset's offset is at:

    ind_offs[(i-1) + blockId * N]

where N is the number of unique indirect datasets."""
        def __get__(self):
            cdef int size = self.nblocks * self.ninds
            return data_to_numpy_array_with_spec(self._handle.ind_offs, size, np.NPY_INT32)

    property ind_sizes:
        """The size of each indirect dataset per block.

The ith /unique/ indirect direct has

    ind_sizes[(i-1) + blockID * N]

elements to be staged in, where N is the number of unique indirect
datasets."""
        def __get__(self):
            cdef int size = self.nblocks * self.ninds
            return data_to_numpy_array_with_spec(self._handle.ind_sizes, size, np.NPY_INT32)

    property nindirect:
        """Total size of each unique indirect dataset"""
        def __get__(self):
            cdef int size = self.ninds
            return data_to_numpy_array_with_spec(self._handle.nindirect, size, np.NPY_INT32)

    property loc_map:
        """Local indirect dataset indices, see also offset

Once the ith unique indirect dataset has been copied into shared
memory (via ind_map), this mapping array tells us where in shared
memory the nth iteration element is:

    arg_i_s + loc_map[(i-1) * set_size + n + offset[blockId]] * dim(arg_i)
"""
        def __get__(self):
            cdef int size = self.set_size * self.nind_ele
            return data_to_numpy_array_with_spec(self._handle.loc_map, size, np.NPY_INT16)

    property nblocks:
        """The number of blocks"""
        def __get__(self):
            return self._handle.nblocks

    property nelems:
        """The number of elements in each block"""
        def __get__(self):
            cdef int size = self.nblocks
            return data_to_numpy_array_with_spec(self._handle.nelems, size, np.NPY_INT32)

    property ncolors_core:
        """Number of core (non-halo colours)

MPI only."""
        def __get__(self):
            return self._handle.ncolors_core

    property ncolors_owned:
        """Number of colours for blocks with only owned elements

MPI only."""
        def __get__(self):
            return self._handle.ncolors_owned

    property ncolors:
        """Number of block colours"""
        def __get__(self):
            return self._handle.ncolors

    property ncolblk:
        """Number of blocks for each colour

This array is allocated to be set_size long, but this is the worst
case scenario (every element interacts with every other).  The number
of "real" elements is ncolors."""
        def __get__(self):
            cdef int size = self.set_size
            return data_to_numpy_array_with_spec(self._handle.ncolblk, size, np.NPY_INT32)

    property blkmap:
        """Mapping from device's block ID to plan's block ID.

There are nblocks entries here, you should index into this with the
device's "block" address plus an offset which is

    sum(ncolblk[i] for i in range(0, current_colour))"""
        def __get__(self):
            cdef int size = self.nblocks
            return data_to_numpy_array_with_spec(self._handle.blkmap, size, np.NPY_INT32)

    property nsharedCol:
        """The amount of shared memory required for each colour"""
        def __get__(self):
            cdef int size = self.ncolors
            return data_to_numpy_array_with_spec(self._handle.nsharedCol, size, np.NPY_INT32)

    property nshared:
        """The total number of bytes of shared memory the plan uses"""
        def __get__(self):
            return self._handle.nshared

    property transfer:
        """Data transfer per kernel call"""
        def __get__(self):
            return self._handle.transfer

    property transfer2:
        """Bytes of cache line per kernel call"""
        def __get__(self):
            return self._handle.transfer2

    property count:
        """Number of times this plan has been used"""
        def __get__(self):
            return self._handle.count
