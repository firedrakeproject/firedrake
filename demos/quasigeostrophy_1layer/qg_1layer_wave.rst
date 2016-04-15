Quasi-Geostrophic Model in Firedrake
====================================

The Quasi-Geostrophic (QG) model is very important in geophysical fluid
dynamics as it describes some aspects of large-scale flows in the oceans
and atmosphere very well. The interested reader can find derivations in
Pedlosky (1987) and Vallis (2006).

In these notes we present the nonlinear equations for the one-layer QG
model with a free-surface. Then, the weak form will be derived as is
needed for the firedrake library.

Governing Equations
-------------------

The Quasi-Geostrophic (QG) model is very similar to the 2D vorticity
equation. Since the leading order geostrophic velocity is incompressible
in the horizontal, the governing equations can be written as

.. math::

   \begin{aligned}
   \partial_t q + \vec \nabla \cdot \left( \vec u q \right)  + \beta v &= 0, \\
   \vec u & = \vec\nabla^\perp \psi, \\
   \nabla^2 \psi - \frac{1}{L_d^2} \psi &= q. \end{aligned}

where the :math:`\psi` and :math:`q` are the streamfunction and
Potential Vorticity (PV). The Laplacian is 2D since we are only in the
horizontal plane and we defined

.. math:: \vec\nabla^\perp =  \hat e_z \times \vec\nabla.

The first equation above states that the PV is conserved following the
flow. The second equation forces the leading order velocity to be
geostrophic and the third equation is the definition for the QG PV for
this barotropic model. To solve this using Finite Elements it is
necessary to establish the weak form of the model, which is done in the
next subsection.

Weak Form
---------

Evolving the nonlinear equations consists of two steps. First, the
elliptic problem must be solved to compute the streamfunction given the
PV. Second, the PV equation must be integrated forward in time. This is
done using a strong stability preserving Runge Kutta 3 (SSPRK3) method.

Elliptic Equation
~~~~~~~~~~~~~~~~~

First, we focus on the elliptic inversion in the case of a flat bottom.
If we compute the inner product of the equation with the test function
:math:`\phi` we obtain,

.. math::

   \begin{aligned}
   \langle \nabla^2 \psi, \phi \rangle - \frac{1}{L_d^2} \langle \psi, \phi \rangle  &= \langle q, \phi \rangle, \\
   \langle \nabla \psi, \nabla \phi \rangle +  \frac{1}{L_d^2} \langle \psi, \phi \rangle &= -\langle q, \phi \rangle,\end{aligned}

where in the second equation we used the divergence theorem and the
homogeneous Dirichlet boundary conditions on the test function.

Evolution Equation
~~~~~~~~~~~~~~~~~~

The SSPRK3 method used as explained in Gottleib (2005) can be written as

.. math::

   \begin{aligned}
   q^{(1)} &= q^n - \Delta t \left[ \vec \nabla \cdot \left( \vec u^n q^n \right) +  \beta v^n \right] , \\
   q^{(2)} &= \frac34 q^n + \frac14 \left[ q^{(1)} - \Delta t  \vec \nabla \cdot \left( \vec u^{(1)} q^{(1)} \right) 
   - \Delta t \beta v^{(1)}\right], \\
   q^{n+1} &= \frac13 q^n + \frac23 \left[ q^{(2)} - \Delta t \vec \nabla \cdot \left( \vec u^{(2)} q^{(2)} \right) - \Delta t \beta v^{(1)} \right].\end{aligned}

To get the weak form we need to introduce a test function, :math:`p`,
and take the inner product of the first equation with :math:`p`.

.. math::

   \begin{aligned}
   \langle q^{(1)}, p \rangle &= \langle q^n, p \rangle  - \Delta t \langle \vec \nabla \cdot \left( \vec u^n q^n \right), p \rangle 
   - \Delta t \langle \beta  v, q \rangle, \\
   \langle q^{(1)}, p \rangle - \Delta t \langle \vec u^n q^n, \vec\nabla p \rangle  +  \Delta t \langle \beta  v, q \rangle
   &= \langle q^n, p \rangle  - \Delta t \langle \vec u^n q^n, p \rangle_{bdry}\end{aligned}

The first and second terms on the left hand side are referred to as
:math:`a_{mass}` and :math:`a_{int}` in the code. The first term on the
right-hand side is referred to as :math:`a_{mass}` in the code. The
second term on the right-hand side is the extra term due to the DG
framework, which does not exist in the CG version of the problem and it
is referred to as :math:`a_{flux}`. This above problem must be solved
for :math:`q^{(1)}` and then :math:`q^{(2)}` and then these are used to
compute the numerical approximation to the PV at the new time
:math:`q^{n+1)}`.
