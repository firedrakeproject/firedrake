Dual spaces 
=====================================

Mathematical background
--------------------------

If :math:`V` is a vector space, then define the (anti-)dual space :math:`V^*` to be the space of bounded conjugate linear functionals :math:`V \to K`. Therefore, the dual space is the space containing functions from :math:`V` to :math:`K`, where :math:`K` can either be :math:`\mathbb{R}` or :math:`\mathbb{C}`.

If :math:`\{\phi_i\}` is a basis for a finite vector space :math:`V` then there exists :math:`\{\phi_i^*\}` a basis for :math:`V^*` such that:

.. math::

    \phi_i^*(\phi_j) = \delta_{ij}

The basis :math:`\{\phi_i^*\}` is termed the *dual basis*. Where it is
necessary to make the distinction, we will refer to the space to which a dual
space is dual as the *primal space* and its basis as the *primal basis*. 

Since UFL function spaces are finite-dimensional Hilbert spaces which result
from the discretisation of infinite-dimensional Hilbert spaces, all of the
function spaces with which we are concerned are reflexive, ie :math:`(V^*)^*`
is isomorphic to V under the canonical map. That is, we can identify
:math:`(V^*)^*` and :math:`V`:

.. math::

    (V^*)^* \equiv V

A form defined over an unknown :math:`a` in the primal space :math:`V` is a
known object in the dual space. For example:

.. math::
    h(a) &= \int_\Omega \phi_i\, \mathrm{ d}x\  a_i \\
    &= \int_\Omega  \phi_i\, \mathrm{ d}x\ I_{ij}\ a_j \\
    &= \int_\Omega \phi_i\, \mathrm{ d}x\ \phi_i^*(\phi_j)\ a_j \\
    &= \int_\Omega \phi_i\, \mathrm{ d}x\ \phi_i^*(\phi_j\ a_j ) \\
    &= \int_\Omega \phi_i\, \mathrm{ d}x\ \phi_i^*(a)\\
    &= h_i \phi_i^*(a)

with basis coefficients :math:`h_i = \displaystyle\int_\Omega \phi_i \text{
d}x`.




Dual objects in UFL
--------------------------

For an arbitrary :py:class:`~.ufl.FunctionSpace`, ``V``, the corresponding dual space :math:`V^*` can be obtained by calling the :py:meth:`~.ufl.FunctionSpace.dual` method:

.. code-block:: python3

    from firedrake import *
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    V_star = V.dual()

A :py:class:`~.ufl.Coefficient` defines a *known* function ``c`` in ``V``. A :py:class:`~.Function` is a subclass of :py:class:`~.ufl.Coefficient`.
Consequently, 

.. code-block:: python3

    c = Function(V)
    f_0 = c * dx

is a symbolic expression for the integral of ``c`` over the domain and represents a scalar value. ``f_0`` is a Python object of type :py:class:`~.ufl.Form`, once assembled, it is a scalar object.


Conversely, :py:class:`~.Argument` defines a placeholder symbol ``a`` for an *unknown* function in ``V``. :py:class:`~firedrake.ufl_expr.TestFunction` and :py:class:`~firedrake.ufl_expr.TrialFunction` are syntactic sugar for ``Argument(V, 0)`` and ``Argument(V, 1)`` respectively.

.. code-block:: python3

    a = TrialFunction(V)
    f_1 = a * dx

represents the integration of the unknown function ``a`` over the domain. It's therefore a linear 1-form, or a function in the dual space :math:`V^* = V \rightarrow K`. ``f_1`` is also a Python object of type :py:class:`~.ufl.Form`. When assembled, it is an object of type :py:class:`~.ufl.Cofunction`:

.. code-block:: python3

    cf = assemble(f_1) # type Cofunction

``cf`` is a known object in the dual space, and the dual equivalent of :py:class:`~.ufl.Coefficient`. The more consistent name ``Cocoefficient`` was rejected as confusing and risible. :py:class:`~.ufl.Cofunction` objects can be combined with symbolic :py:class:`~.ufl.Form` objects:

.. code-block::

    v = TestFunction(V) 
    a = v * dx
    b = assemble(a)
    res = a + b
    c = assemble(res)


Furthermore, we will want to express unknown objects in the dual space. For example, in order to represent interpolation from a space :math:`U` to a space :math:`V`,  it is convienent to reframe this as a problem involving the dual space:

.. math::

    V \to U = V \times U^* \to R

Using the reflexivity of the function space :math:`U`. This form therefore has two arguments, one in the primal space :math:`V` and one in the dual space :math:`U^*`. Therefore, we need to represent *arguments* in the dual space - we will call these *coarguments*. The details of interpolation will be discussed in :ref:`its own section <firedrake_interpolation>`.

A :py:class:`~.Coargument` can be constructed by either calling :py:class:`~.ufl.Argument` on a dual space object or calling :py:class:`~.Coargument` on a dual space.

.. code-block::

    v = Argument(V, 1) # type Argument
    u = Argument(V.dual(), 2) # type Coargument
    w = Coargument(V.dual(), 3) # type Coargument


There is a further dual-related type avalilable in UFL. In :py:class:`~.ufl.Cofunction`, we have represented an assembled 1-form. However, commonly we also assemble 2-forms. :py:class:`~.Matrix` allows an analogous use, and assembled 2-forms can be naturally combined with 2-forms that have not yet been assembled:

.. code-block::

    mesh = UnitSquareMesh(10,10)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V) 

    a = (u*v - inner(grad(u),grad(v)) ) * dx
    M = assemble(a) # type Matrix
    res = assemble(M + a)

Operations supported symbolically, such as the adjoint and action, are also supported on the dual space equivalent. 

.. code-block::

    mesh = UnitSquareMesh(10,10)
    V = FunctionSpace(mesh, "Lagrange", 1)
    U = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(U)
    v = TestFunction(V) 

    a = u * v * dx
    a = assemble(a) # type Matrix

    adj = adjoint(a)

    b = Matrix(V, U.dual())
    u = Coefficient(U)
    u_a = Argument(U, 0)
    u_form = u_a * dx

    primal_action = action(a, u)
    dual_action = action(b, u_form)


In summary, this table describes the dual types corresponding to primal finite element spaces, and to known and unknown functions in those spaces:

.. list-table:: 
   :widths: 25 25
   :header-rows: 1

   * - Primal quantity 
     - Dual quantity
   * - :py:class:`~.ufl.FunctionSpace`
     - :py:class:`~.ufl.classes.DualSpace`
   * - :py:class:`~.ufl.Coefficient`
     - :py:class:`~.ufl.Cofunction`
   * - :py:class:`~.Argument`
     - :py:class:`~.Coargument`