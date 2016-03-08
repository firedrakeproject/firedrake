.. default-role:: math

A plausibly efficient implementation of `R`
===========================================

The function space `R` (for "Real" or, possibly, `Rognes
<http://home.simula.no/~meg/>`_) is employed to model concepts such as
global constraints. When employed as an unknown in an equation, it
presents implementation difficulties because it couples with all of
the other degrees of freedom. This results in a dense row in the
resulting matrix. Using the distributed CSR format which Firedrake
employs for other function spaces, both the assembly and action of
this row will require the entire system state to be gathered onto one
MPI process. This is clearly a horribly non-performant option.

Representing matrices involving `R`
-----------------------------------

Instead, we can observe that a dense matrix row (or column) is
isomorphic to a :class:`~firedrake.function.Function` and model these
blocks of the matrix accordingly. A non-trivial system
involving a function over `R` will always be a mixed system. The
resulting matrix will have four blocks, one a conventional sparse
matrix, one a dense column, one a dense row, and one a single
double. The dense row and column blocks can be implemented as matrix
shells. The row block will implement matrix multiplication as a dot
product returning a :class:`~pyop2.base.Global` while the column block
will implement matrix multiplication by pointwise scaling the input
:class:`~pyop2.base.Dat` and will return another
:class:`~pyop2.base.Dat`. This arrangement enables both the row block
and the column block to have the same parallel data distribution as a
:class:`~pyop2.base.Dat`, which removes the key scalability problem.


Assembling matrices involving `R`
---------------------------------

Assembling the column block will be as simple as replacing the trial
function with the constant 1, thereby transforming a 2-form into a
1-form, and assembling.

Similarly, assembling the row block simply requires the replacement of
the test function with the constant 1, and assembling.

The one by one block in the corner can be assembled by replacing both
the test and trial functions of the corresponding form with 1 and
assembling.

Clearly the remaining block does not involve `R` and can be assembled
as usual.
