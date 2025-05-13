1D Vlasov-Poisson Equation
===========================

   This tutorial was contributed by `Colin Cotter
   <mailto:colin.cotter@imperial.ac.uk>`__ and Werner Bauer.

.. math::
   m_t + mu_x + (mu)_x = 0, \quad u - \alpha^2u_{xx} = m,


As usual, to implement this problem, we start by importing the
Firedrake namespace. ::

  from firedrake import *


A python script version of this demo can be found :demo:`here <vp1d.py>`.
