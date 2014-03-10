.. _architecture:

PyOP2 Architecture
==================

As described in :ref:`concepts`, the PyOP2 API allows users to declare the
topology of unstructured meshes in the form of :class:`Sets <pyop2.Set>` and
:class:`Maps <pyop2.Map>` and data in the form of :class:`Dats <pyop2.Dat>`,
:class:`Mats <pyop2.Mat>`, :class:`Globals <pyop2.Global>` and :class:`Consts
<pyop2.Const>`. Any computations on this data happen in :class:`Kernels
<pyop2.Kernel>` described in :ref:`kernels` executed via :func:`parallel loops
<pyop2.par_loop>`. A schematic overview of the PyOP2 architecture is given
below:

.. figure:: images/pyop2_architecture.svg
  :align: center

  Schematic overview of the PyOP2 architecture

For the most part, PyOP2 is a conventional Python library, with performance
critical library functions implemented in Cython_. A user's application code
makes calls to the PyOP2 API, most of which are conventional library calls.
The exception are :func:`~pyop2.par_loop` calls, which encapsulate most of
PyOP2's runtime core functionality performing backend-specific code
generation. Executing a parallel loop comprises the following steps:

1. Compute a parallel execution plan, including the amount of shared memory
   required and a colouring of the iteration set for conflict-free parallel
   execution.  This process is described in :doc:`plan` and does not apply to
   the sequential backend.
2. Generate code for executing the computation specific to the backend and the
   given :func:`~pyop2.par_loop` arguments as detailed in :doc:`backends`
   according to the execution plan computed in the previous step.
3. Pass the generated code to a backend-specific toolchain for just-in-time
   compilation, producing a shared library callable as a Python module which
   is dynamically loaded. This module is cached on disk to save recompilation
   when the same :func:`~pyop2.par_loop` is called again for the same backend.
4. Build the backend-specific list of arguments to be passed to the generated
   code, which may initiate host to device data transfer for the CUDA and
   OpenCL backends.
5. Call into the generated module to perform the actual computation. For
   distributed parallel computations this involves separate calls for the
   regions owned by the current processor and the halo as described in
   :doc:`mpi`.
6. Perform any necessary reductions for :class:`Globals <pyop2.Global>`.
7. Call the backend-specific matrix assembly procedure on any
   :class:`~pyop2.Mat` arguments.

In practice, the computation is defered and executed lazily only when the
result is requested. At this point, the current execution trace is analyzed
and computation is enforced according to the read and write dependencies of
the requested result as described in :doc:`lazy`.

.. _Cython: http://cython.org
