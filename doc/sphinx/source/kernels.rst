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

.. _kernel-api:

Kernel API
----------

Consider a :func:`~pyop2.par_loop` computing the midpoint of a triangle given
the three vertex coordinates. Note that we make use of a covenience in the
PyOP2 syntax, which allow declaring an anonymous :class:`~pyop2.DataSet` of a
dimension greater one by using the ``**`` operator. We omit the actual data in
the declaration of the :class:`~pyop2.Map` ``cell2vertex`` and
:class:`~pyop2.Dat` ``coordinates``. ::

  vertices = op2.Set(num_vertices)
  cells = op2.Set(num_cells)

  cell2vertex = op2.Map(cells, vertices, 3, [...])

  coordinates = op2.Dat(vertices ** 2, [...], dtype=float)
  midpoints = op2.Dat(cells ** 2, dtype=float)

  op2.par_loop(midpoint, cells,
               midpoints(op2.WRITE),
               coordinates(op2.READ, cell2vertex))

Kernels are implemented in a restricted subset of C99 and are declared by
passing a *C code string* and the *kernel function name*, which must match the
name in the C kernel signature, to the :class:`~pyop2.Kernel` constructor: ::

  midpoint = op2.Kernel("""
  void midpoint(double p[2], double *coords[2]) {
    p[0] = (coords[0][0] + coords[1][0] + coords[2][0]) / 3.0;
    p[1] = (coords[0][1] + coords[1][1] + coords[2][1]) / 3.0;
  }""", "midpoint")

Since kernels cannot return any value, the return type is always ``void``. The
kernel argument ``p`` corresponds to the third :func:`~pyop2.par_loop`
argument ``midpoints`` and ``coords`` to the fourth argument ``coordinates``
respectively. Argument names need not agree, the matching is by position.

Data types of kernel arguments must match the type of data passed to the
parallel loop. The Python types :class:`float` and :class:`numpy.float64`
correspond to a C :class:`double`, :class:`numpy.float32` to a C
:class:`float`, :class:`int` or :class:`numpy.int64` to a C :class:`long` and
:class:`numpy.int32` to a C :class:`int`.

Direct :func:`~pyop2.par_loop` arguments such as ``midpoints`` are passed to
the kernel as a ``double *``, indirect arguments such as ``coordinates`` as a
``double **`` with the first indirection due to the map and the second
indirection due the data dimension. The kernel signature above uses arrays
with explicit sizes to draw attention to the fact that these are known. We
could have interchangibly used a kernel signature with plain pointers:

.. code-block:: c

  void midpoint(double * p, double ** coords)
