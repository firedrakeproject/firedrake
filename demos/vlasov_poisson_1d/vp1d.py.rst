1D Vlasov-Poisson Equation
===========================

.. rst-class:: emphasis

   This tutorial was contributed by `Colin Cotter
   <mailto:colin.cotter@imperial.ac.uk> and Werner Bauer`__.

.. math::
   m_t + mu_x + (mu)_x = 0, \quad u - \alpha^2u_{xx} = m,


As usual, to implement this problem, we start by importing the
Firedrake namespace. ::

  from firedrake import *


A python script version of this demo can be found :demo:`here <vp1d.py>`.

.. rubric:: References

.. bibliography:: demo_references.bib
   :filter: docname in docnames
