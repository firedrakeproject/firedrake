:orphan:

.. _firedrake_tutorials:

.. only:: html

   .. sidebar:: Current development information.

      Firedrake is continually tested using `GitHub actions <https://docs.github.com/en/actions>`__.

      .. only:: release

         Latest Firedrake **release** branch status:

         |firedrakereleasebuild|_

      .. only:: not release

         Latest Firedrake **master** branch status:

         |firedrakemasterbuild|_

      .. |firedrakereleasebuild| image:: https://github.com/firedrakeproject/firedrake/actions/workflows/push.yml/badge.svg?branch=release
      .. _firedrakereleasebuild: https://github.com/firedrakeproject/firedrake/actions/workflows/push.yml?branch=release
      .. |firedrakemasterbuild| image:: https://github.com/firedrakeproject/firedrake/actions/workflows/push.yml/badge.svg?query=branch%3Amaster
      .. _firedrakemasterbuild: https://github.com/firedrakeproject/firedrake/actions/workflows/push.yml?query=branch%3Amaster

      Firedrake and its components are developed on `GitHub
      <http://github.com>`__.

      * `Firedrake on GitHub <https://github.com/firedrakeproject/firedrake/>`__
      * `FIAT on GitHub <https://github.com/firedrakeproject/fiat>`__

   Getting started
   ===============

   The first step is to download and install Firedrake and its
   dependencies. For full instructions, see :doc:`obtaining Firedrake
   <install>`.

   .. include:: intro_tut.rst

   Jupyter notebooks
   -----------------

   In addition to the documented tutorials, we also have some `Jupyter
   notebooks <https://jupyter.org/>`__ that are a more interactive way of
   getting to know Firedrake. They are described in more detail :doc:`on
   their own page <notebooks>`.

   Youtube Channel
   ---------------
   Firedrake has a `youtube channel <https://www.youtube.com/channel/UCwwT3kL0HHCv_O3VaeX3GUg>`__ where recorded tutorials are occasionally uploaded.

   API documentation
   =================

   The complete list of all the classes and methods in Firedrake is
   available at the :doc:`firedrake` page. The same information is
   :ref:`indexed <genindex>` in alphabetical order. Another very
   effective mechanism is the site :ref:`search engine <search>`.

   .. include:: manual.rst
   .. include:: advanced_tut.rst
