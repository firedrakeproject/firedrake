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
  
.. _local-iteration-spaces:

Local iteration spaces
----------------------

PyOP2 supports complex kernels with large local working set sizes, which may
not run very efficiently on architectures with a limited amount of registers
and on-chip resources. In many cases the resource usage is proportional to the
size of the *local iteration space* the kernel operates on.

Consider a finite-element local assembly kernel for a mass matrix from linear
basis functions on triangles. For each element in the iteration set, the
kernel computes a 3x3 local tensor:

.. code-block:: c

  void mass(double A[3][3], double **vertex_coordinates) {
    double J[4];
    J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0];
    J[1] = vertex_coordinates[2][0] - vertex_coordinates[0][0];
    J[2] = vertex_coordinates[4][0] - vertex_coordinates[3][0];
    J[3] = vertex_coordinates[5][0] - vertex_coordinates[3][0];
    double detJ;
    detJ = J[0]*J[3] - J[1]*J[2];
    const double det = fabs(detJ);

    double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    double FE0[3][3] = {{0.666666666666667, 0.166666666666667, 0.166666666666667},
                        {0.166666666666667, 0.166666666666667, 0.666666666666667},
                        {0.166666666666667, 0.666666666666667, 0.166666666666667}};

    for (int ip = 0; ip<3; ip++) {
      for (int j = 0; j<3; j++) {
        for (int k = 0; k<3; k++) {
          A[j][k] += (det*W3[ip]*FE0[ip][k]*FE0[ip][j]);
        }
      }
    }
  }

This kernel is the simplest commonly found in finite-element computations and
only serves to illustrate the concept. To improve the efficiency of executing
complex kernels on manycore platforms, their operation can be distributed
among several threads which each compute a single point in this local
iteration space to increase the level of parallelism and to lower the amount
of resources required per thread. In the case of the ``mass`` kernel from
above we obtain:

.. code-block:: c

  void mass(double A[1][1], double **vertex_coordinates, int j, int k) {
    double J[4];
    J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0];
    J[1] = vertex_coordinates[2][0] - vertex_coordinates[0][0];
    J[2] = vertex_coordinates[4][0] - vertex_coordinates[3][0];
    J[3] = vertex_coordinates[5][0] - vertex_coordinates[3][0];
    double detJ;
    detJ = J[0]*J[3] - J[1]*J[2];
    const double det = fabs(detJ);

    double W3[3] = {0.166666666666667, 0.166666666666667, 0.166666666666667};
    double FE0[3][3] = {{0.666666666666667, 0.166666666666667, 0.166666666666667},
                        {0.166666666666667, 0.166666666666667, 0.666666666666667},
                        {0.166666666666667, 0.666666666666667, 0.166666666666667}};

    for (int ip = 0; ip<3; ip++) {
      A[0][0] += (det*W3[ip]*FE0[ip][k]*FE0[ip][j]);
    }
  }

Note how the doubly nested loop over basis function is hoisted out of the
kernel, which receives its position in the local iteration space to compute as
additional arguments j and k. PyOP2 needs to be told to loop over this local
iteration space by indexing the corresponding maps with an
:class:`~pyop2.base.IterationIndex` :data:`~pyop2.i`. The
:func:`~pyop2.par_loop` over ``elements`` to assemble the matrix ``mat`` with
``coordinates`` as read-only coefficient both indirectly accessed via
``ele2nodes`` is defined as follows: ::

  op2.par_loop(mass, elements,
               mat(op2.INC, (ele2nodes[op2.i[0]], ele2nodes[op2.i[1]])),
               coordinates(op2.READ, ele2nodes))
