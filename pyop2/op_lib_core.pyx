"""
Wrap OP2 library for PyOP2

The basic idea is that we need to make the OP2 runtime aware of
python-managed datatypes (sets, maps, dats, and so forth).

All the information we pass to the C library is available in python,
we therefore do not have to expose the details of the C structs.  We
just need a way of initialising a C data structure corresponding to
the python one.  We do this through Cython's "cdef class" feature.
The initialisation takes a python data structure, calls out to the OP2
C library's declaration routine (getting back a pointer to the C
data).  On the python side, we store a reference to the C struct we're
holding.

For example, to declare a set and make the C side aware of it we do
this:

   from pyop2 import op2
   import op_lib_core

   py_set = op2.Set(size, 'name')

   c_set = op_lib_core.op_set(py_set)


py_set._lib_handle now holds a pointer to the c_set, and c_set._handle
is the C pointer we need to pass to routines in the OP2 C library.
"""

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
cimport _op_lib_core as core

np.import_array()

cdef data_to_numpy_array_with_template(void * ptr, arr):
    cdef np.npy_intp dim = np.size(arr)
    cdef np.dtype t = arr.dtype
    shape = np.shape(arr)
    return np.PyArray_SimpleNewFromData(1, &dim, t.type_num, ptr).reshape(shape)

cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp size, int t):
    return np.PyArray_SimpleNewFromData(1, &size, t, ptr)

cdef class op_set:
    cdef core.op_set _handle
    def __cinit__(self, set):
        cdef int size = set._size
        cdef char * name = set._name
        self._handle = core.op_decl_set_core(size, name)

cdef class op_dat:
    cdef core.op_dat _handle
    def __cinit__(self, dat):
        cdef op_set set = dat._dataset._lib_handle
        cdef int dim = dat._dim[0]
        cdef int size = dat._dataset._size
        cdef char * type = dat._data.dtype.name
        cdef np.ndarray data = dat._data
        cdef char * name = dat._name
        self._handle = core.op_decl_dat_core(set._handle, dim, type,
                                             size, <char *>data.data, name)

cdef class op_map:
    cdef core.op_map _handle
    def __cinit__(self, map):
        cdef op_set frm = map._iterset._lib_handle
        cdef op_set to = map._dataset._lib_handle
        cdef int dim = map._dim
        cdef np.ndarray[int, ndim=1, mode="c"] values = map._values.reshape(np.size(map._values))
        cdef char * name = map._name
        if len(map._values) == 0:
            self._handle = core.op_decl_map_core(frm._handle, to._handle,
                                                 dim, NULL, name)
        else:
            self._handle = core.op_decl_map_core(frm._handle, to._handle, dim,
                                                 &values[0], name)

_access_map = {'READ' : core.OP_READ,
               'WRITE' : core.OP_WRITE,
               'RW' : core.OP_RW,
               'INC' : core.OP_INC,
               'MIN' : core.OP_MIN,
               'MAX' : core.OP_MAX}

cdef class op_arg:
    cdef core.op_arg _handle
    def __cinit__(self, arg, dat=False, gbl=False):
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

        acc = _access_map[arg.access._mode]

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
            size = arg.data._size
            type = arg.data.dtype.name
            data = arg.data._data
            self._handle = core.op_arg_gbl_core(<char *>data.data, dim,
                                                type, size, acc)

cdef class op_plan:
    cdef core.op_plan *_handle
    cdef int set_size
    cdef int nind_ele
    def __cinit__(self, kernel, iset, *args):
        cdef op_set _set = iset._lib_handle
        cdef char * name = kernel._name
        cdef int part_size = 0
        cdef int nargs = len(args)
        cdef op_arg _arg
        cdef core.op_arg *_args = <core.op_arg *>malloc(nargs * sizeof(core.op_arg))
        cdef int ninds
        cdef int *inds = <int *>malloc(nargs * sizeof(int))
        cdef int i
        cdef int ind = 0

        self.set_size = _set._handle.size
        if any(arg.is_indirect_and_not_read() for arg in args):
            self.set_size += _set._handle.exec_size

        nind_ele = 0
        for arg in args:
            if arg.is_indirect():
                nind_ele += 1
        self.nind_ele = nind_ele
        ninds = 0

        unique_args = set(args)
        d = {}
        for i in range(nargs):
            arg = args[i]
            arg.build_core_arg()
            _arg = arg._lib_handle
            _args[i] = _arg._handle
            if arg.is_indirect():
                if d.has_key(arg):
                    inds[i] = d[arg]
                else:
                    inds[i] = ind
                    d[arg] = ind
                    ind += 1
                    ninds += 1
            else:
                inds[i] = -1
        self._handle = core.op_plan_core(name, _set._handle, part_size,
                                         nargs, _args, ninds, inds)

        free(_args)

    property ninds:
        def __get__(self):
            return self._handle.ninds

    property nargs:
        def __get__(self):
            return self._handle.nargs

    property part_size:
        def __get__(self):
            return self._handle.part_size

    property nthrcol:
        def __get__(self):
            cdef int size = self.nblocks
            return data_to_numpy_array_with_spec(self._handle.nthrcol, size, np.NPY_INT32)

    property thrcol:
        def __get__(self):
            cdef int size = self.set_size
            return data_to_numpy_array_with_spec(self._handle.thrcol, size, np.NPY_INT32)

    property offset:
        def __get__(self):
            cdef int size = self.nblocks
            return data_to_numpy_array_with_spec(self._handle.offset, size, np.NPY_INT32)

    property ind_map:
        def __get__(self):
            cdef int size = self.set_size * self.nind_ele
            return data_to_numpy_array_with_spec(self._handle.ind_map, size, np.NPY_INT32)

    property ind_offs:
        def __get__(self):
            cdef int size = self.nblocks * self.ninds
            return data_to_numpy_array_with_spec(self._handle.ind_offs, size, np.NPY_INT32)

    property ind_sizes:
        def __get__(self):
            cdef int size = self.nblocks * self.ninds
            return data_to_numpy_array_with_spec(self._handle.ind_sizes, size, np.NPY_INT32)

    property nindirect:
        def __get__(self):
            cdef int size = self.ninds
            return data_to_numpy_array_with_spec(self._handle.nindirect, size, np.NPY_INT32)

    property loc_map:
        def __get__(self):
            cdef int size = self.set_size * self.nind_ele
            return data_to_numpy_array_with_spec(self._handle.loc_map, size, np.NPY_INT16)

    property nblocks:
        def __get__(self):
            return self._handle.nblocks

    property nelems:
        def __get__(self):
            cdef int size = self.nblocks
            return data_to_numpy_array_with_spec(self._handle.nelems, size, np.NPY_INT32)

    property ncolors_core:
        def __get__(self):
            return self._handle.ncolors_core

    property ncolors_owned:
        def __get__(self):
            return self._handle.ncolors_owned

    property ncolors:
        def __get__(self):
            return self._handle.ncolors

    property ncolblk:
        def __get__(self):
            cdef int size = self.set_size
            return data_to_numpy_array_with_spec(self._handle.ncolblk, size, np.NPY_INT32)

    property blkmap:
        def __get__(self):
            cdef int size = self.nblocks
            return data_to_numpy_array_with_spec(self._handle.blkmap, size, np.NPY_INT32)

    property nsharedCol:
        def __get__(self):
            cdef int size = self.ncolors
            return data_to_numpy_array_with_spec(self._handle.nsharedCol, size, np.NPY_INT32)

    property nshared:
        def __get__(self):
            return self._handle.nshared

    property transfer:
        def __get__(self):
            return self._handle.transfer

    property transfer2:
        def __get__(self):
            return self._handle.transfer2

    property count:
        def __get__(self):
            return self._handle.count
