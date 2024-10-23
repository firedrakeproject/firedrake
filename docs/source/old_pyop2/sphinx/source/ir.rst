The PyOP2 Intermediate Representation
=====================================

The :class:`parallel loop <pyop2.par_loop>` is the main construct of PyOP2.
It applies a specific :class:`~pyop2.Kernel` to all elements in the iteration
set of the parallel loop. Here, we describe how to use the PyOP2 API to build
a kernel and, also, we provide simple guidelines on how to write efficient
kernels.

Using the Intermediate Representation
-------------------------------------

In the :doc:`previous section <kernels>`, we described the API for
PyOP2 kernels in terms of the C code that gets executed.
Passing in a string of C code is the simplest way of creating a
:class:`~pyop2.Kernel`.  Another possibility is to use PyOP2 Intermediate
Representation (IR) objects to express the :class:`~pyop2.Kernel` semantics.

An Abstract Syntax Tree of the kernel code can be manually built using IR
objects. Since PyOP2 has been primarily thought to be fed by higher layers
of abstractions, rather than by users, no C-to-AST parser is currently provided.
The advantage of providing an AST, instead of C code, is that it enables PyOP2
to inspect and transform the kernel, which is aimed at achieving performance
portability among different architectures and, more generally, better execution
times.

For the purposes of exposition, let us consider a simple
kernel ``init`` which initialises the members of a :class:`~pyop2.Dat`
to zero.

.. code-block:: python

  from op2 import Kernel

  code = """void init(double* edge_weight) {
    for (int i = 0; i < 3; i++)
      edge_weight[i] = 0.0;
  }"""
  kernel = Kernel(code, "init")

Here, we describe how we can use PyOP2 IR objects to build an AST for
the this kernel. For example, the most basic AST one can come up with
is

.. code-block:: python

  from op2 import Kernel
  from ir.ast_base import *

  ast = FlatBlock("""void init(double* edge_weight) {
    for (int i = 0; i < 3; i++)
      edge_weight[i] = 0.0;
  }""")
  kernel = Kernel(ast, "init")

The :class:`~pyop2.ir.ast_base.FlatBlock` object encapsulates a "flat" block
of code, which is not modified by the IR engine. A
:class:`~pyop2.ir.ast_base.FlatBlock` is used to represent (possibly large)
fragments of code for which we are not interested in any kind of
transformation, so it may be particularly useful to speed up code development
when writing, for example, test cases or non-expensive kernels.  On the other
hand, time-demanding kernels should be properly represented using a "real"
AST. For example, an useful AST for ``init`` could be the following

.. code-block:: python

  from op2 import Kernel
  from ir.ast_base import *

  ast_body = [FlatBlock("...some code can go here..."),
              c_for("i", 3, Assign(Symbol("edge_weight", ("i",)), c_sym("0.0")))]
  ast = FunDecl("void", "init",
                [Decl("double*", c_sym("edge_weight"))],
                ast_body)
  kernel = Kernel(ast, "init")

In this example, we first construct the body of the kernel function. We have
an initial :class:`~pyop2.ir.ast_base.FlatBlock` that contains, for instance,
some sort of initialization code. :func:`~pyop2.ir.ast_base.c_for` is a shortcut
for building a :class:`for loop <pyop2.ir.ast_base.For>`.  It takes an
iteration variable (``i``), the extent of the loop and its body.  Multiple
statements in the body can be passed in as a list.
:func:`~pyop2.ir.ast_base.c_sym` is a shortcut for building :class:`symbols
<pyop2.ir.ast_base.Symbol>`. You may want to use
:func:`~pyop2.ir.ast_base.c_sym` when the symbol makes no explicit use of
iteration variables.

We use :class:`~pyop2.ir.ast_base.Symbol` instead of
:func:`~pyop2.ir.ast_base.c_sym`,  when ``edge_weight`` accesses a specific
element using the iteration variable ``i``. This is fundamental to allow the
IR engine to perform many kind of transformations involving the kernel's
iteration space(s). Finally, the signature of the function is constructed
using the :class:`~pyop2.ir.ast_base.FunDecl`.

Other examples on how to build ASTs can be found in the tests folder,
particularly looking into ``test_matrices.py`` and
``test_iteration_space_dats.py``.


Achieving Performance Portability with the IR
---------------------------------------------

One of the key objectives of PyOP2 is obtaining performance portability.
This means that exactly the same program can be executed on a range of
different platforms, and that the PyOP2 engine will strive to get the best
performance out of the chosen platform. PyOP2 allows users to write kernels
by completely abstracting from the underlying machine. This is mainly
achieved in two steps:

* Given the AST of a kernel, PyOP2 applies a first transformation aimed at
  mapping the parallelism inherent to the kernel to that available in the
  backend.
* Then, PyOP2 applies optimizations to the sequential code, depending on the
  underlying backend.

To maximize the outcome of the transformation process, it is important that
kernels are written as simply as possible. That is, premature optimization,
possibly for a specific backend, might harm performance.

A minimal language, the so-called PyOP2 Kernel Domain-Specific Language, is
used to trigger specific transformations. If we had had a parser from C
code to AST, we would have embedded this DSL in C by means of ``pragmas``.
As we directly build an AST, we achieve the same goal by decorating AST nodes
with specific attributes, added at node creation-time. An overview of the
language follows

* ``pragma pyop2 itspace``. This is added to :class:`~pyop2.ir.ast_base.For`
  nodes (i.e. written on top of for loops). It tells PyOP2 that the following
  is a fully-parallel loop, that is all of its iterations can be executed in
  parallel without any sort of synchronization.
* ``pragma pyop2 assembly(itvar1, itvar2)``. This is added to a statement node,
  to denote that we are performing a local assembly operation along to the
  ``itvar1`` and ``itvar2`` dimensions.
* ``pragma pyop2 simd``. This is added on top of the kernel signature. It is
  used to suggest PyOP2 to apply SIMD vectorization along the ParLoop's
  iteration set dimension. This kind of vectorization is also known as
  *inter-kernel vectorization*. This feature is currently not supported
  by PyOP2, and will be added only in a future release.

The ``itspace`` pragma tells PyOP2 how to extract parallelism from the kernel.
Consider again our usual example. To expose a parallel iteration space, one
one must write

.. code-block:: python

  from op2 import Kernel

  code = """void init(double* edge_weight) {
    #pragma pyop2 itspace
    for (int i = 0; i < 3; i++)
      edge_weight[i] = 0.0;
  }"""
  kernel = Kernel(code, "init")

The :func:`~pyop2.ir.ast_base.c_for` shortcut when creating an AST expresses
the same semantics of a for loop decorated with a ``pragma pyop2 itspace``.

Now, imagine we are executing the ``init`` kernel on a CPU architecture.
Typically we want a single core to execute the entire kernel, because it is
very likely that the kernel's iteration space is small and its working set
fits the L1 cache, and no benefit would be gained by splitting the computation
between distinct cores. On the other end, if the backend is a GPU or an
accelerator, a different execution model might give better performance.
There's a huge amount of parallelism available, for example, in a GPU, so
delegating the execution of an individual iteration (or a chunk of iterations)
to a single thread could pay off. If that is the case, the PyOP2 IR engine
re-structures the kernel code to exploit such parallelism.

Optimizing kernels on CPUs
--------------------------

So far, some effort has been spent on optimizations for CPU platforms. Being a
DSL, PyOP2 provides specific support for those (linear algebra) operations that
are common among unstructured-mesh-based numerical methods. For example, PyOP2
is capable of aggressively optimizing local assembly codes for applications
based on the Finite Element Method. We therefore distinguish optimizations in
two categories:

* Generic optimizations, such as data alignment and support for autovectorization.
* Domain-specific optimizations (DSO)

To trigger DSOs, statements must be decorated using the kernel DSL. For example,
if the kernel computes the local assembly of an element in an unstructured mesh,
then a ``pragma pyop2 assembly(itvar1, itvar2)`` should be added on top of the
corresponding statement. When constructing the AST of a kernel, this can be
simply achieved by

.. code-block:: python

  from ir.ast_base import *

  s1 = Symbol("X", ("i",))
  s2 = Symbol("Y", ("j",))
  tensor = Symbol("A", ("i", "j"))
  pragma = "#pragma pyop2 outerproduct(j,k)"
  code = c_for("i", 3, c_for("j", 3, Incr(tensor, Prod(s1, s2), pragma)))

That, conceptually, corresponds to

.. code-block:: c

  #pragma pyop2 itspace
  for (int i = 0; i < 3; i++)
    #pragma pyop2 itspace
    for (int j = 0; j < 3; j++)
      #pragma pyop2 assembly(i, j)
      A[i][j] += X[i]*Y[j]

Visiting the AST, PyOP2 finds a 2-dimensional iteration space and an assembly
statement. Currently, ``#pragma pyop2 itspace`` is ignored when the backend is
a CPU. The ``#pragma pyop2 assembly(i, j)`` can trigger multiple DSOs.
PyOP2 currently lacks an autotuning system that automatically finds out the
best possible kernel implementation; that is, the optimizations that minimize
the kernel run-time. To drive the optimization process, the user (or the
higher layer) can specify which optimizations should be applied. Currently,
PyOP2 can automate:

* Alignment and padding of data structures: for issuing aligned loads and stores.
* Loop trip count adjustment according to padding: useful for autovectorization
  when the trip count is not a multiple of the vector length
* Loop-invariant code motion and autovectorization of invariant code: this is
  particularly useful since trip counts are typically small, and hoisted code
  can still represent a significant proportion of the execution time
* Register tiling for rectangular iteration spaces
* (DSO for pragma assembly): Outer-product vectorization + unroll-and-jam of
  outer loops to improve register re-use or to mitigate register pressure

How to select specific kernel optimizations
-------------------------------------------

When constructing a :class:`~pyop2.Kernel`, it is possible to specify the set
of optimizations we want PyOP2 to apply. The IR engine will analyse the kernel
AST and will try to apply, incrementally, such optimizations. The PyOP2's FFC
interface, which build a :class:`~pyop2.Kernel` object given an AST provided
by FFC, makes already use of the available optimizations. Here, we take the
emblematic case of the FFC interface and describe how to play with the various
optimizations through a series of examples.

.. code-block:: python

  ast = ...
  opts = {'licm': False,
          'tile': None,
          'ap': False,
          'vect': None}
  kernel = Kernel(ast, 'my_kernel', opts)

In this example, we have an AST ``ast`` and we specify optimizations through
the dictionary ``opts``; then, we build the :class:`~pyop2.Kernel`, passing in
the optional argument ``opts``. No optimizations are enabled here. The
possible options are:

* ``licm``: Loop-Invariant Code Motion.
* ``tile``: Register Tiling (of rectangular iteration spaces)
* ``ap``: Data alignment, padding. Trip count adjustment.
* ``vect``: SIMD intra-kernel vectorization.

If we wanted to apply both loop-invariant code motion and data alignment, we
would simply write

.. code-block:: python

  ast = ...
  opts = {'licm': True,
          'ap': True}
  kernel = Kernel(ast, 'my_kernel', opts)

Now, let's assume we know the kernel has a rectangular iteration space. We want
to try register tiling, with a particular tile size. The way to get it is

.. code-block:: python

  ast = ...
  opts = {'tile': (True, 8)}
  kernel = Kernel(ast, 'my_kernel', opts)

In this case, the iteration space is sliced into tiles of size 8x8. If the
iteration space is smaller than the slice, then the transformation is not
applied. By specifying ``-1`` instead of ``8``, we leave PyOP2 free to choose
automatically a certain tile size.

A fundamental optimization for any PyOP2 kernel is SIMD vectorization. This is
because almost always kernels fit the L1 cache and are likely to be compute-
bound. Backend compilers' AutoVectorization (AV) is therefore an opportunity.
By enforcing data alignment and padding, we can increase the chance AV is
successful. To try AV, one should write

.. code-block:: python

  import ir.ast_plan as ap

  ast = ...
  opts = {'ap': True,
          'vect': (ap.AUTOVECT, -1)}
  kernel = Kernel(ast, 'my_kernel', opts)

The ``vect``'s second parameter (-1) is ignored when AV is requested.
If our kernel is computing an assembly-like operation, then we can ask PyOP2
to optimize for register locality and register pressure, by resorting to a
different vectorization technique. Early experiments show that this approach
can be particularly useful when the amount of data movement in the assembly
loops is "significant". Of course, this depends on kernel parameters (e.g.
size of assembly loop, number and size of arrays involved in the assembly) as
well as on architecture parameters (e.g. size of L1 cache, number of available
registers). This strategy takes the name of *Outer-Product Vectorization*
(OP), and can be activated in the following way (again, we suggest to use it
along with data alignment and padding).

.. code-block:: python

  import ir.ast_plan as ap

  ast = ...
  opts = {'ap': True,
          'vect': (ap.V_OP_UAJ, 1)}
  kernel = Kernel(ast, 'my_kernel', opts)

``UAJ`` in ``V_OP_UAJ`` stands for ``Unroll-and-Jam``. It has been proved that
OP shows a much better performance when used in combination with unrolling the
outer assembly loop and incorporating (*jamming*) the unrolled iterations
within the inner loop. The second parameter, therefore, specifies the unroll-
and-jam factor: the higher it is, the larger is the number of iterations
unrolled. A factor 1 means that no unroll-and-jam is performed. The optimal
factor highly depends on the computational characteristics of the kernel.
