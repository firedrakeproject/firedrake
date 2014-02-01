.. _kernels:

PyOP2 Kernels
=============

Kernels in PyOP2 define the local operations that are to be performed for each
element of the iteration set the kernel is executed over. There must be a one
to one match between the arguments declared in the kernel signature and the
actual arguments passed to the parallel loop executing this kernel. As
described in :doc:`concepts`, data is accessed directly on the iteration set
or via mappings passed in the :func:`~pyop2.par_loop` call.

The kernel only sees data corresponding to the current element of the
iteration set it is invoked for. Any data read by the kernel i.e. accessed as
:data:`~pyop2.READ`, :data:`~pyop2.RW` or :data:`~pyop2.INC` is automatically
gathered via the mapping relationship in the *staging in* phase and the kernel
is passed pointers to the staging memory. Similarly, after the kernel has been
invoked, any modified data i.e. accessed as :data:`~pyop2.WRITE`,
:data:`~pyop2.RW` or :data:`~pyop2.INC` is scattered back out via the
:class:`~pyop2.Map` in the *staging out* phase. It is only safe for a kernel
to manipulate data in the way declared via the access descriptor in the
parallel loop call. Any modifications to an argument accessed read-only would
not be written back since the staging out phase is skipped for this argument.
Similarly, the result of reading an argument declared as write-only is
undefined since the data has not been staged in.
