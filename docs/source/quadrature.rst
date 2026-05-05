.. only:: html

  .. contents::

Quadrature rules
================

To numerically compute the integrals in the variational formulation,
a quadrature rule is required.
By default, Firedrake obtains a quadrature rule by estimating the polynomial
degree of the integrands within a :py:class:`ufl.Form`. Sometimes
this estimate might be quite large, and a warning like this one will be raised:

.. code-block::

   tsfc:WARNING Estimated quadrature degree 13 more than tenfold greater than any argument/coefficient degree (max 1)

For integrals with very complicated or non-polynomial nonlinearities, the
estimated quadrature degree might be in the hundreds or thousands, rendering
the integration prohibitively expensive, or leading to segfaults.

Specifying the quadrature rule in the variational formulation
-------------------------------------------------------------

To manually override the default, the
quadrature degree can be prescribed on each integral :py:class:`~ufl.measure.Measure`,

.. code-block:: python3

   inner(sin(u)**4, v) * dx(degree=4)

Setting ``degree=4`` means that the quadrature rule will be exact only for integrands
of total polynomial degree up to 4. This, of course, will introduce a greater numerical
error than the default.

Rather than enforcing a fixed quadrature degree, it is also possible to set
a maximum allowable degree. This value will only be used if UFL estimates a larger
degree.
The maximum allowable degree can be set for a particular integral by
adding the ``"max_quadrature_degree"`` entry to the ``metadata`` of the ``Measure``:

.. code-block:: python3

   inner(sin(u)**4, v) * dx(metadata={"max_quadrature_degree": 4})

The ``metadata`` can also be used to set a fixed quadrature degree by adding
a ``"quadrature_degree"`` entry to the dictionary. However, because setting a
fixed degree is quite common the ``degree`` keyword argument shown above is
provided as a convenient shorthand.

For integrals that do not specify a fixed or maximum quadrature degree, a default value
may be keyed as the ``"quadrature_degree"`` or ``"max_quadrature_degree"`` entry
respectively in the ``form_compiler_parameters`` dictionary passed on to :py:func:`~.solve`,
:py:func:`~.project`, :py:class:`~.NonlinearVariationalProblem`, or :py:func:`~.assemble`.

.. code-block:: python3

   F = inner(grad(u), grad(v))*dx(degree=0) + inner(exp(u), v)*dx - inner(1, v)*dx

   solve(F == 0, u, form_compiler_parameters={"quadrature_degree": 4})

   assemble(F, form_compiler_parameters={"max_quadrature_degree": 4})

In the example above:

* In the ``solve`` call, the integrals with unspecified quadrature degree
  will be computed on a quadrature rule that exactly integrates polynomials of
  the degree set in ``form_compiler_parameters``.

* In the ``assemble`` call, the integrals with unspecified quadrature degree will
  be computed with the ``"max_quadrature_degree"`` set in the ``"form_compiler_parameters"``
  if the estimated quadrature degree is greater than 4, otherwise they will be computed
  with the estimated quadrature degree.

Another way to specify the quadrature rule is through the ``scheme`` keyword. This could be
either a :py:class:`~finat.quadrature.QuadratureRule`, or a string. Supported string values
are ``"default"``, ``"canonical"``, and ``"KMV"``. For more details see
:py:func:`~FIAT.quadrature_schemes.create_quadrature`.

Lumped quadrature schemes
-------------------------

Spectral elements, such as Gauss-Legendre-Lobatto and `KMV`_, may be used with
lumped quadrature schemes to produce a diagonal mass matrix.

.. literalinclude:: ../../tests/firedrake/regression/test_quadrature_manual.py
   :language: python3
   :dedent:
   :start-after: [test_lump_scheme 1]
   :end-before: [test_lump_scheme 2]

.. Note::

   To obtain the lumped mass matrix with ``scheme="KMV"``,
   the ``degree`` argument should match the degree of the :py:class:`~finat.ufl.finiteelement.FiniteElement`.

The Quadrature space
--------------------

It is possible to define a finite element :py:class:`~.Function` on a quadrature rule.
The ``"Quadrature"`` and ``"Boundary Quadrature"`` spaces are useful to
interpolate data at quadrature points on cell interiors and cell boundaries,
respectively.

.. literalinclude:: ../../tests/firedrake/regression/test_quadrature_manual.py
   :language: python3
   :dedent:
   :start-after: [test_quadrature_space 1]
   :end-before: [test_quadrature_space 2]

The ``quad_scheme`` keyword argument again may be either
:py:class:`~finat.quadrature.QuadratureRule` or a string.
If a :py:class:`~.Function` in the ``"Quadrature"`` space appears within an
integral, Firedrake will automatically select the quadrature rule that corresponds
to ``dx(degree=quad_degree, scheme=quad_scheme)`` to match the one associated
with the quadrature space.

.. _element_quad_scheme:

Specifying the quadrature for integral-type degrees of freedom
--------------------------------------------------------------

Finite element spaces with :ref:`integral-type degrees of freedom <element_variants>`
support different quadrature rules.
These are selected by passing a string to the ``"quad_scheme"`` keyword argument of
the :py:class:`~finat.ufl.finiteelement.FiniteElement` or
:py:func:`~.FunctionSpace` constructors. For example, to construct a
Crouzeix-Raviart space with degrees of freedom consisting of integrals along the edges
computed from a 2-point average at the endpoints, one can set ``quad_scheme="KMV"``:

.. code-block:: python3

    fe = FiniteElement("Crouzeix-Raviart", triangle, 1, variant="integral", quad_scheme="KMV")

.. Note::

    Finite elements with integral-type degrees of freedom only accept string
    values for ``quad_scheme``, since it does not make sense to specify a
    concrete :py:class:`~finat.quadrature.QuadratureRule` when the degrees of
    freedom are defined on cell entities of different dimensions.


.. _KMV : https://defelement.org/elements/kong-mulder-veldhuizen.html
