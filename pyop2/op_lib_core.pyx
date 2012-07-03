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
from pyop2 import op2
from libc.stdlib cimport malloc, free
cimport _op_lib_core as core
cimport numpy as np

np.import_array()

cdef data_to_numpy_array_with_template(void * ptr, arr):
    cdef np.npy_intp dim = np.size(arr)
    cdef np.dtype t = arr.dtype
    shape = np.shape(arr)

    return np.PyArray_SimpleNewFromData(1, &dim, t.type_num, ptr).reshape(shape)

cdef data_to_numpy_array_with_spec(void * ptr, np.npy_intp size, np.dtype t):
    return np.PyArray_SimpleNewFromData(1, &size, t.type_num, ptr)

cdef class op_set:
    cdef core.op_set _handle
    def __cinit__(self, set):
        cdef int size = set._size
        cdef char * name = set._name
        self._handle = core.op_decl_set_core(size, name)
        set._lib_handle = self

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
        dat._lib_handle = self

cdef class op_map:
    cdef core.op_map _handle
    def __cinit__(self, map):
        cdef op_set frm = map._iterset._lib_handle
        cdef op_set to = map._dataset._lib_handle
        cdef int dim = map._dim
        cdef np.ndarray[int, ndim=1, mode="c"] values = map._values
        cdef char * name = map._name
        self._handle = core.op_decl_map_core(frm._handle, to._handle, dim,
                                              &values[0], name)
        map._lib_handle = self

cdef class op_arg:
    cdef core.op_arg _handle
    def __cinit__(self, datacarrier):
        datacarrier._arg_handle = self

cdef class op_plan:
    cdef core.op_plan *_handle
    cdef int set_size
    cdef int nind_ele
    def __cinit__(self, kernel, set, args):
        cdef op_set _set = set._lib_handle
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
        if any(arg._map is not op2.IdentityMap and arg._access is not op2.READ
               for arg in args):
            self.set_size += _set._handle.exec_size

        nind_ele = 0
        for arg in args:
            if arg._map is not op2.IdentityMap:
                nind_ele += 1
        ninds = 0

        unique_args = set(args)
        d = {}
        for i in range(nargs):
            _arg = args[i]._arg_handle
            _args[i] = _arg._handle
            arg = args[i]
            if arg.is_indirect():
                if d.has_key(arg):
                    inds[i] = d[arg]
                else:
                    inds[i] = ind
                    d[arg] = ind
                    ind += 1
            else:
                inds[i] = -1
        self._handle = core.op_plan_core(name, _set._handle, part_size,
                                         nargs, _args, ninds, inds)

    def ind_map(self):
        cdef int size = self.set_size * self.nind_ele
        return data_to_numpy_array_with_spec(self._handle.ind_map, size, int)

    def loc_map(self):
        cdef int size = self.set_size * self.nind_ele
        return data_to_numpy_array_with_spec(self._handle.loc_map, size, np.int16)

    def ind_sizes(self):
        cdef int size = self._handle.nblocks * self._handle.ninds
        return data_to_numpy_array_with_spec(self._handle.ind_sizes, size, int)

    def ind_offs(self):
        cdef int size = self._handle.nblocks * self._handle.ninds
        return data_to_numpy_array_with_spec(self._handle.ind_offs, size, int)

    def nthrcol(self):
        cdef int size = self._handle.nblocks
        return data_to_numpy_array_with_spec(self._handle.nthrcol, size, int)

    def thrcol(self):
        cdef int size = self._handle.set.size
        return data_to_numpy_array_with_spec(self._handle.thrcol, size, int)

    def offset(self):
        cdef int size = self._handle.nblocks
        return data_to_numpy_array_with_spec(self._handle.offset, size, int)

    def nelems(self):
        cdef int size = self._handle.nblocks
        return data_to_numpy_array_with_spec(self._handle.nelems, size, int)

    def blkmap(self):
        cdef int size = self._handle.nblocks
        return data_to_numpy_array_with_spec(self._handle.blkmap, size, int)
