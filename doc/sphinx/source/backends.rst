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

.. _Instant: https://bitbucket.org/fenics-project/instant
.. _FEniCS project: http://fenicsproject.org
