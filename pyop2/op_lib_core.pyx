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
cimport _op_lib_core as core
cimport numpy as np

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
        cdef op_set set = dat._dataset
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
        cdef op_set frm = map._iterset
        cdef op_set to = map._dataset
        cdef int dim = map._dim
        cdef np.ndarray[int, ndim=1, mode="c"] values = map._values
        cdef char * name = map._name
        self._handle = core.op_decl_map_core(frm._handle, to._handle, dim,
                                              &values[0], name)
        map._lib_handle = self
