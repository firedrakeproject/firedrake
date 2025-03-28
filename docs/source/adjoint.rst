.. default-role:: math

Solving adjoint PDEs
====================

Mathematical background
-----------------------

Suppose we have a parametrised finite element problem, given `m \in M` find
`u\in V` such that:

.. math::
    :label: eq:forward

    f(u, m; v) = 0 \qquad \forall v \in V.

Further suppose that we have some scalar quantity of interest `J(u, m) \in
\mathbb{R}`. We call `m` the *control*, `J` the *functional* and `u` the
*state*. We can initially assume that both `V` and `M` are finite element
spaces.

Our objective is to compute `\frac{\mathrm{d}J}{\mathrm{d}m}`. We can assume
that `J` itself is amenable to differentiation with respect to either of its
arguments, but the challenge is that `u` is implicitly a function of `m`
because `u` solves the PDE. In order to capture this dependency, we introduce
the *reduced functional* defined as:

.. math::
    :label:

    \hat{J}(m) = J(u(m), m)

Our differentiation task can now be expressed as:

.. math::
    :label: eq:functional_chain

    \frac{\mathrm{d}\hat{J}}{\mathrm{d}m} = \frac{\partial J}{\partial{u}}\frac{\partial u}{\partial m} + \frac{\partial J}{\partial m}

Assuming, again, that we can differentiate `J` with respect to its arguments,
the challenge here remains the term `\frac{\partial u}{\partial m}`. Here the
key insight is that the dependence of `u` on `m` is given by the fact that `u`
solves the PDE for any value of `m`. In other words, changing `m` doesn't ever
change the value of `f(u, m)`, because we simply solve for the value of `u`
which solves the equation. To be more precise:

.. math::
    :label:

    \frac{\mathrm{d}f}{\mathrm{d} m} = 0

Applying the chain rule yields:

.. math::
    :label:

    \frac{\partial f}{\partial u}\frac{\partial u}{\partial m} + \frac{\partial f}{\partial m} = 0

or:

.. math::
    :label:

    \frac{\partial f}{\partial u}\frac{\partial u}{\partial m} = - \frac{\partial f}{\partial m}

Hence:

.. math::
    :label:

    \frac{\partial u}{\partial m} = - \frac{\partial f}{\partial u}^{-1}\frac{\partial f}{\partial m}

Substituting back into :eq:`eq:functional_chain` yields:

.. math::
    :label: eq:djdm

    \frac{\mathrm{d}\hat{J}}{\mathrm{d}m} = -\frac{\partial J}{\partial u}\frac{\partial f}{\partial u}^{-1}\frac{\partial f}{\partial m} + \frac{\partial J}{\partial m}

Consider now the function signatures of the symbols in :eq:`eq:djdm`. Here we are
only concerned with the arguments, as these determine the sizes of the
resulting assembled tensors:

.. math:: 
    :label:

    \frac{\mathrm{d}\hat{J}}{\mathrm{d}m}: M\rightarrow \mathbb{R}

    \frac{\partial J}{\partial u}: V\rightarrow \mathbb{R}

    \frac{\partial f}{\partial u}: V \times V \rightarrow \mathbb{R}

    \frac{\partial f}{\partial m}: V \times M \rightarrow \mathbb{R}

    \frac{\partial J}{\partial m}: M\rightarrow \mathbb{R}

The consequence of this is that the term `\frac{\partial f}{\partial
u}^{-1}\frac{\partial f}{\partial m}` requires the inversion of one potentially
large matrix onto another, which is an intractable calculation in general.

Instead, we define:

.. math::
    :label: eq:adjoint
    
    \lambda^*(\in V\rightarrow\mathbb{R}) = -\frac{\partial J}{\partial u}\frac{\partial f}{\partial u}^{-1}.

We actually solve the adjoint to this equation. That is find `\lambda \in V`
such that: 

.. math::
    :label:

    \frac{\partial f}{\partial u}^{*}(u, m; \lambda, v) = 
    -\frac{\partial J}{\partial u}(u, m; v) \qquad \forall v \in V.

Note that these terms include `u`, so it is first necessary to solve
:eq:`eq:forward` to obtain this value. The value of `m` is an input to the
whole calculation and is hence known in advance. The adjoint operator
`\frac{\partial f}{\partial u}^{*}` is given by the following identity:

.. math::
    :label:

    \frac{\partial f}{\partial u}^{*}(u, m; \lambda, v)
    = \frac{\partial f}{\partial u}(u, m; v, \lambda).

Note that the form arguments are reversed between the left and right hand
sides. This is the mechanism by which the adjoint (transpose) form is
assembled.

Having obtained `\lambda`, we can obtain the first right hand side term to
:eq:`eq:djdm` by evaluating:

.. math::
    :label: eq:djdm_final

    \frac{\partial f}{\partial m}(u, m; \lambda, \tilde{m})\qquad \forall \tilde{m} \in M.

Since `\lambda` is known at this stage, this is simply the evaluation of a
linear form.

How Firedrake and Pyadjoint automate derivative calculation
-----------------------------------------------------------

Firedrake automates the process in the preceding section using the methodology
first published in :cite:`Farrell2013` using the implementation in 
`Pyadjoint <https://pyadjoint.org>`__ :cite:`Mitusch2019`. 

The essence of this process is:

1. The user's forward solve and objective functional computations are recorded
   (this provides access to the definitions of `f` and `J`)
2. The user defines a reduced functional `\hat{J}` which specifies which
   recorded variable should be used as the functional `J`, and which as the
   control `m`.
3. When the user requests a derivative calculation, :eq:`eq:djdm` is evaluated
   via :eq:`eq:adjoint` and :eq:`eq:djdm_final`. The various forms required are
   derived automatically by applying UFL's :func:`~ufl.derivative`,
   :func:`~ufl.adjoint`, and :func:`~ufl.action` operators.

Taping an example calculation
-----------------------------

The adjoint computation depends on the operations that result in the functional
evaluation being recorded by Pyadjoint, a process known as *taping*, or
*annotation*.

First, the user code must access the adjoint module:

.. code-block:: python3

    from firedrake.adjoint import *
    continue_annotation()

The call to :func:`~pyadjoint.continue_annotation` starts the taping process:
all subsequent relevant operations will be recorded until taping is paused.
This can be accomplished with a call to :func:`~pyadjoint.pause_annotation`, or
temporarily within a :class:`~pyadjoint.stop_annotating` context manager, or
within a function decorated with :func:`~pyadjoint.tape.no_annotations`.

The following code then solves the Burgers equation in one dimension with
homogeneous Dirichlet Boundary conditions and (for simplicity) implicit Euler
in time. Along the way we accumulate the quantity `J`, which is the sum of the
squared `L_2` norm of the solution at every timestep. We will use this as our
functional.

Note that no explicit adjoint code is included. This code is shown to
provide context for the material that follows.

.. literalinclude:: ../../tests/firedrake/adjoint/test_burgers_newton.py
    :start-after: start solver
    :end-before: end solver
    :dedent:
    :language: python3

We've now solved the PDE over time and computed our functional. Observe that
we paused taping at the end of the computation.

For future reference, the value of :obj:`!J` printed at the end is
`5.006`.

Reduced functionals
-------------------

A :class:`~pyadjoint.ReducedFunctional` is the key object encapsulating adjoint
calculations. It ties together a functional value, which can be any result of a
taped calculation, and one or more controls, which are created from almost any
quantity which is an input of the computation of the functional value (for
details of object types that can be used as functional values and controls, see
:ref:`overloaded_types`).

In this case we use :obj:`!J` as the functional value and the initial condition
as the control:

.. literalinclude:: ../../tests/firedrake/adjoint/test_burgers_newton.py
    :start-after: end solver
    :end-before: end reduced functional
    :dedent:
    :language: python3

Each control must be wrapped in :class:`~pyadjoint.Control`.

Reduced functional evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A reduced functional is a callable, differentiable function object whose inputs
are the control and whose output is the functional value. The most basic
operation that can be undertaken on a reduced functional is to evaluate the
functional for a new value of the control(s). This is achieved by calling the
reduced functional passing an object of the same type as the control for each
control. For example, we can evaluate :obj:`!Jhat` for new initial conditions
using:

.. literalinclude:: ../../tests/firedrake/adjoint/test_burgers_newton.py
    :start-after: start functional evaluation
    :end-before: end functional evaluation
    :dedent:
    :language: python3

This time the printed output is `5.415` which is different from the first
evaluation. The documentation for calling reduced functionals is to be found on
the :meth:`~pyadjoint.ReducedFunctional.__call__` special method.

Reduced functional derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The derivative of the reduced functional with respect to the controls can be
evaluated using the :meth:`~pyadjoint.ReducedFunctional.derivative` method. The
derivative so calculated will be linearised about the state resulting from the
last evaluation of the reduced functional (or the state that was originally
taped if the functional has never been re-evaluated). This is as simple as:

.. literalinclude:: ../../tests/firedrake/adjoint/test_burgers_newton.py
    :start-after: start derivative
    :end-before: end derivative
    :dedent:
    :language: python3

The derivative, :meth:`!dJ`, will have the same type as the controls.

.. note::

    Strictly, :meth:`!ReducedFunctional.derivative` returns the gradient, which
    is the Riesz representer of the derivative. A future release of Firedrake
    and Pyadjoint will change this to return the true derivative, which is of
    the type dual to the controls.

The tape
--------

The sequence of recorded operations is stored on an object called the tape. The
currently active tape can be accessed by calling
:func:`~pyadjoint.get_working_tape`. The user usually has limited direct
interaction with the tape, but there is some useful information which can be
extracted. 

Visualising the tape
~~~~~~~~~~~~~~~~~~~~

A PDF visualisation of the tape can be constructed by calling the
:meth:`pyadjoint.Tape.visualise` method and passing a filename ending in
:file:`.pdf`. This requires the installation of two additional Python modules,
:mod:`!networkx` and :mod:`!pygraphviz`. The former can simply be installed with
:program:`pip` but the latter depends on the external :program:`graphviz` package.
Installation instructions for both :mod:`!pygraphviz` and :program:`graphviz`
are to be found on `the pygraphviz website
<https://pygraphviz.github.io/documentation/stable/install.html#recommended>`__.

.. _fig-tape:

.. figure:: images/tape.pdf

    A visualisation of the Burgers equation example above shortened to a single
    timestep. Operations (blocks) recorded on the tape are shown as grey
    rectangles, while taped variables are shown as ovals. 

The numbered blocks in the tape visualisation are as follows:

1.  The initial condition is projected.
2.  The initial condition is copied into `u_{\mathrm{old}}`.
3.  The squared norm of the initial condition is computed.
4.  The timestep PDE is solved.
5.  The squared norm of the new solution is computed.
6.  The result of step 5 is added to step 3 resulting in the functional value.

The oval variables with labels of the form `w_n` are of type Firedrake
:class:`~.function.Function` while the variables labelled with numbers
are annotated scalars of type :class:`~pyadjoint.AdjFloat`.

Progress bars
~~~~~~~~~~~~~


Taylor tests
------------

.. _overloaded_types:

Overloaded types
----------------



.. bibliography:: _static/references.bib
    :filter: False

    Farrell2013
    Mitusch2019