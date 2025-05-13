Assimilating point data
=======================

.. rst-class:: emphasis

    This example is based on work done by Reuben Nixon-Hill et al. in :cite:`Nixon-Hill:2024`, and was written up by Leo Collins.


General theory
--------------

We have a model

.. math::

    F(u,m)=0

where :math:`u` is our solution, :math:`m` are the model parameters, and :math:`F` is our model, for example a PDE. 
We have a set of :math:`N` observations :math:`u_{\text{obs}}^i` at locations :math:`X_i`, for :math:`i=1,\ldots,N`.
We want our solution field :math:`u` to match the observations at the locations :math:`X_i`, so we define a "misfit" (or "objective") function

.. math::

    J=J_{\text{model-data misfit}} + J_{\text{regularisation}}.

Here :math:`J_{\text{regularisation}}` is a regularisation term, which is there to ensure that the problem is well-posed. Often this uses some known properties of the model, such as smoothness requirements.
The :math:`J_{\text{model-data misfit}}` term is taken to be the :math:`L^2` norm of the difference between the observations :math:`u_{\text{obs}}^i` and the model solution :math:`u` point evaluated 
at the observation locations: :math:`u(X_i)`, i.e. :math:`\lVert u_{\text{obs}}^i-u(X_{i}) \rVert_{L^2}`.

The aim is to minimise the misfit functional :math:`J`. This is possible because point evaluation as implemented using the vertex-only mesh formalism is differentiable 


Vertex-only mesh formalism
----------------------------

Our mesh consists of the our observation locations :math:`X_i`,

.. math::

    \Omega_{v}=\{X_i\}_{i=1}^{N}

which is embedded in the parent mesh :math:`\Omega`, and our observations :math:`u_{\text{obs}}^i` live in the function space of zero-order discontinuous Lagrange polynomials,

.. math::
  
    u_{\text{obs}}^i \in \operatorname{P0DG}(\Omega_{v}).


Integrating a function over a vertex-only mesh is equivalent to summing the function values at the vertices of the mesh, i.e for a function :math:`f\in\operatorname{P0DG}(\Omega_{v})` we have

.. math::

    \int_{\Omega_{v}} f \, dx = \sum_{i=1}^{N} f(X_{i}).


We can interpolate a function defined on some function space on the parent mesh :math:`u\in\operatorname{FS}(\Omega)` into the function space :math:`\operatorname{P0DG}(\Omega_{v})` by evaluating the function :math:`u` at the vertices :math:`X_i` of the mesh.
This is done by the interpolation operator

.. math::

    \mathcal{I}_{\operatorname{P0DG}(\Omega_{v})}&\rightarrow\operatorname{FS}(\Omega)
    \mathcal{I}_{\operatorname{P0DG}(\Omega_{v})}(u)&\mapsto u_{v}.


Finally we can write down our misfit term as

.. math::

    J_{\text{model-data misfit}} &= \sum_{i=1}^{N} \lVert u_{\text{obs}}^i-u(X_{i}) \rVert_{L^2}^2
    &= \int_{\Omega_{v}} (u_{\text{obs}}^i-\mathcal{I}_{\operatorname{P0DG}(\Omega_{v})}(u))^2 \, dx
    &= \sum_{i=1}^{N} (u_{\text{obs}}^i-u(X_{i))^2


Unknown conductivity
--------------------

We consider the PDE 

.. math::

    -\nabla\cdot k\nabla u=1

defined on the domain :math:`\Omega`. Our solution field is :math:`u:\Omega\rightarrow\mathbb{R}` and :math:`k` is the conductivity. We take the Dirichlet boundary condition

.. math::

    u=0 \text{ on } \partial\Omega.

We assume that the conductivity is of the form

.. math::

    k=k_{0}e^{q}

with :math:`k_{0}=\frac{1}{2}`, where :math:`q` is the log-conductivity field. We want to estimate the log-conductivity field :math:`q` from the (noisy) point observations :math:`u_{\text{obs}}^i`.
We take the true conductivity :math:`q_{\text{true}}` to be in :math:`\operatorname{P2CG}(\Omega)`, and solve the PDE on the same function space for :math:`u_{\text{true}}\in\operatorname{P2CG}(\Omega)`.
The PDE can be written in weak form as

.. math::

    k_{0}e^{q}\int_{\Omega}\nabla u\cdot\nabla v \, dx = \int_{\Omega} fv\,dx

where :math:`v` is a test function in :math:`\operatorname{P2CG}(\Omega)`.

Firedrake implementation
------------------------