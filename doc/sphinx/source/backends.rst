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
