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

Note that these terms include $u$, so it is first necessary to solve
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
first published in :cite:`Farrell2012` using the implementation in 
`Pyadjoint <https://pyadjoint.org>`__ :cite:`Mitusch`. 

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

Controlling the taping process
------------------------------

The adjoint computation depends on the operations that result in the functional
evaluation being recorded by Pyadjoint, a process known as *taping*, or
*annotation*.

First, the user code must access the adjoint module:

.. code-block:: python3

    from firedrake.adjoint import *



