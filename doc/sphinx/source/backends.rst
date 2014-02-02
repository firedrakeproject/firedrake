.. _backends:

PyOP2 Backends
==============

PyOP2 supports a number of different backends to be able to run parallel
computations on different hardware architectures. The currently supported
backends are

* ``sequential``: runs sequentially on a single CPU core.
* ``openmp``: runs multiple threads on an SMP CPU using OpenMP. The number of
  threads is set with the environment variable ``OMP_NUM_THREADS``.
* ``cuda``: offloads computation to a NVIDA GPU (requires :ref:`CUDA and pycuda
  <cuda-installation>`)
* ``opencl``: offloads computation to an OpenCL device, either a multi-core
  CPU or a GPU (requires :ref:`OpenCL and pyopencl <opencl-installation>`)

The ``sequential`` and ``openmp`` backends also support distributed parallel
computations using MPI. For OpenMP this means a hybrid parallel execution
with ``OMP_NUM_THREADS`` threads per MPI rank. Datastructures must be suitably
partitioned in this case with overlapping regions, so called halos. These are
described in detail in :doc:`mpi`.

Sequential backend
------------------

Any computation in PyOP2 requires generating code at runtime specific to each
individual :func:`~pyop2.par_loop`. The sequential backend generates code via
the `Instant`_ utility from the `FEniCS project`_. Since there is no parallel
computation for the sequential backend, the generated code is a C wrapper
function with a ``for`` loop calling the kernel for the respective
:func:`~pyop2.par_loop`. This wrapper also takes care of staging in and out
the data as requested by the access descriptors requested in the parallel
loop. Both the kernel and the wrapper function are just-in-time compiled in a
single compilation unit such that the kernel call can be inlined and does not
incur any function call overhead.

Recall the :func:`~pyop2.par_loop` calling the ``midpoint`` kernel from
:doc:`kernels`: ::

  op2.par_loop(midpoint, cells,
               midpoints(op2.WRITE),
               coordinates(op2.READ, cell2vertex))

.. highlight:: c
   :linenothreshold: 5

The JIT compiled code for this loop is the kernel followed by the generated
wrapper code: ::

  inline void midpoint(double p[2], double *coords[2]) {
    p[0] = (coords[0][0] + coords[1][0] + coords[2][0]) / 3.0;
    p[1] = (coords[0][1] + coords[1][1] + coords[2][1]) / 3.0;
  }

  void wrap_midpoint__(PyObject *_start, PyObject *_end,
                       PyObject *_arg0_0,
                       PyObject *_arg1_0, PyObject *_arg1_0_map0_0) {
    int start = (int)PyInt_AsLong(_start);
    int end = (int)PyInt_AsLong(_end);
    double *arg0_0 = (double *)(((PyArrayObject *)_arg0_0)->data);
    double *arg1_0 = (double *)(((PyArrayObject *)_arg1_0)->data);
    int *arg1_0_map0_0 = (int *)(((PyArrayObject *)_arg1_0_map0_0)->data);
    double *arg1_0_vec[3];
    for ( int n = start; n < end; n++ ) {
      int i = n;
      arg1_0_vec[0] = arg1_0 + arg1_0_map0_0[i * 3 + 0] * 2;
      arg1_0_vec[1] = arg1_0 + arg1_0_map0_0[i * 3 + 1] * 2;
      arg1_0_vec[2] = arg1_0 + arg1_0_map0_0[i * 3 + 2] * 2;
      midpoint(arg0_0 + i * 2, arg1_0_vec);
    }
  }

Note that the wrapper function is called directly from Python and therefore
all arguments are plain Python objects, which first need to be unwrapped. The
arguments ``_start`` and ``_end`` define the iteration set indices to iterate
over. The remaining arguments are :class:`arrays <numpy.ndarray>`
corresponding to a :class:`~pyop2.Dat` or :class:`~pyop2.Map` passed to the
:func:`~pyop2.par_loop`. Arguments are consecutively numbered to avoid name
clashes.

The first :func:`~pyop2.par_loop` argument ``midpoints`` is direct and
therefore no corresponding :class:`~pyop2.Map` is passed to the wrapper
function and the data pointer is passed straight to the kernel with an
appropriate offset. The second argument ``coordinates`` is indirect and hence
a :class:`~pyop2.Dat`-:class:`~pyop2.Map` pair is passed. Pointers to the data
are gathered via the :class:`~pyop2.Map` of arity 3 and staged in the array
``arg1_0_vec``, which is passed to kernel. The coordinate data can therefore
be accessed in the kernel via double indirection as if it was stored
consecutively in memory. Note that for both arguments, the pointers are to two
consecutive double values, since the :class:`~pyop2.DataSet` is of dimension
two in either case.

OpenMP backend
--------------

The OpenMP uses the same infrastructure for code generation and JIT
compilation as the sequential backend described above. In contrast however,
the ``for`` loop is annotated with OpenMP pragmas to make it execute in
parallel with multiple threads. To avoid race conditions on data access, the
iteration set is coloured and a thread safe execution plan is computed as
described in :doc:`colouring`.

The JIT compiled code for the parallel loop from above changes as follows: ::

  void wrap_midpoint__(PyObject* _boffset,
                       PyObject* _nblocks,
                       PyObject* _blkmap,
                       PyObject* _offset,
                       PyObject* _nelems,
                       PyObject *_arg0_0,
                       PyObject *_arg1_0, PyObject *_arg1_0_map0_0) {
    int boffset = (int)PyInt_AsLong(_boffset);
    int nblocks = (int)PyInt_AsLong(_nblocks);
    int* blkmap = (int *)(((PyArrayObject *)_blkmap)->data);
    int* offset = (int *)(((PyArrayObject *)_offset)->data);
    int* nelems = (int *)(((PyArrayObject *)_nelems)->data);
    double *arg0_0 = (double *)(((PyArrayObject *)_arg0_0)->data);
    double *arg1_0 = (double *)(((PyArrayObject *)_arg1_0)->data);
    int *arg1_0_map0_0 = (int *)(((PyArrayObject *)_arg1_0_map0_0)->data);
    double *arg1_0_vec[32][3];
    #ifdef _OPENMP
    int nthread = omp_get_max_threads();
    #else
    int nthread = 1;
    #endif
    #pragma omp parallel shared(boffset, nblocks, nelems, blkmap)
    {
      int tid = omp_get_thread_num();
      #pragma omp for schedule(static)
      for (int __b = boffset; __b < boffset + nblocks; __b++)
      {
        int bid = blkmap[__b];
        int nelem = nelems[bid];
        int efirst = offset[bid];
        for (int n = efirst; n < efirst+ nelem; n++ )
        {
          int i = n;
          arg1_0_vec[tid][0] = arg1_0 + arg1_0_map0_0[i * 3 + 0] * 2;
          arg1_0_vec[tid][1] = arg1_0 + arg1_0_map0_0[i * 3 + 1] * 2;
          arg1_0_vec[tid][2] = arg1_0 + arg1_0_map0_0[i * 3 + 2] * 2;
          midpoint(arg0_0 + i * 2, arg1_0_vec[tid]);
        }
      }
    }
  }

Computation is split in ``nblocks`` blocks which start at an initial offset
``boffset`` and correspond to colours that can be executed conflict free in
parallel. This loop over colours is therefore wrapped in an OpenMP parallel
region and is annotated with an ``omp for`` pragma. The block id ``bid`` for
each of these blocks is given by the block map ``blkmap`` and is the index
into the arrays ``nelems`` and ``offset`` provided as part of the execution
plan. These are the number of elements that are part of the given block and
its starting index. Note that each thread needs its own staging array
``arg1_0_vec``, which is therefore scoped by the thread id.

.. _Instant: https://bitbucket.org/fenics-project/instant
.. _FEniCS project: http://fenicsproject.org
