import numpy as np
cimport numpy as np

# python setup_computeind.py build_ext --inplace
# cython -a computeind.pyx

DTYPE = np.int
ctypedef np.int_t DTYPE_t
ctypedef unsigned int ITYPE_t
cimport cython

@cython.boundscheck(False)
def compute_ind_extr(np.ndarray[DTYPE_t, ndim=1] nums,
                ITYPE_t map_dofs1,
                ITYPE_t lins1,
                DTYPE_t layers1,
                np.ndarray[DTYPE_t, ndim=1] mesh2d,
                np.ndarray[DTYPE_t, ndim=2] dofs not None,
                A not None,
                ITYPE_t wedges1,
                map,
                ITYPE_t lsize):
  cdef unsigned int count = 0
  cdef DTYPE_t m
  cdef unsigned int c,offset
  cdef DTYPE_t layers = layers1
  cdef unsigned int map_dofs = <unsigned int>map_dofs1
  cdef unsigned int wedges = <unsigned int>wedges1
  cdef unsigned int lins = <unsigned int>lins1
  cdef unsigned int mm,d,i,j,k,l
  cdef np.ndarray[DTYPE_t, ndim=1] ind = np.zeros(lsize, dtype=DTYPE)
  cdef DTYPE_t a1,a2,a3
  cdef int a4
  cdef int len1 = len(mesh2d)
  cdef int len2
  for mm in range(0,lins):
    offset = 0
    for d in range(0,2):
      c = 0
      for i in range(0,len1):
        a4 = dofs[i, d]
        if a4 != 0:
          len2 = len(A[d])
          for j in range(0, mesh2d[i]):
            m = map[mm][c]
            for k in range(0, len2):
              ind[count] = m*(layers - d) + <DTYPE_t>A[d][k] + offset
              count+=1
            c+=1
        elif dofs[i, 1-d] != 0:
          c+= <unsigned int>mesh2d[i]
        offset += a4*nums[i]*(layers - d)
  return ind