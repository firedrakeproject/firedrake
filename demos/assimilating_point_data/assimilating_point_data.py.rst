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
