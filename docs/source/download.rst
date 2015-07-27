Obtaining Firedrake
===================

Firedrake depends on PyOP2_, FFC_, FIAT_, and UFL_. It is easiest to obtain
all of these components on Ubuntu Linux and related distributions such as Mint
or Debian. Installation on other Unix-like operating systems is likely to be
possible, although harder. Installation on a Mac is straightforward using the
commands below.

PyOP2
-----

Instructions for obtaining PyOP2_ and its dependencies are at
:doc:`obtaining_pyop2`. Note that PyOP2_ is updated frequently and Firedrake
requires an up-to-date version.

FFC, FIAT and UFL
-----------------

Firedrake currently requires a fork of FFC_, UFL_ and FIAT_.  Note that FFC_
requires a version of Instant_.

FFC_ currently depends on Swig_, which you can install from
package. On Ubuntu and relatives type::

  sudo apt-get install swig

while on Mac OS it's::

  brew install swig

Install FFC_ and all dependencies via pip::

  sudo pip install \
    six \
    sympy \
    git+https://bitbucket.org/mapdes/ffc.git#egg=ffc \
    git+https://bitbucket.org/mapdes/ufl.git#egg=ufl \
    git+https://bitbucket.org/mapdes/fiat.git#egg=fiat \
    git+https://bitbucket.org/fenics-project/instant.git#egg=instant

These dependencies are regularly updated. If you already have the packages
installed and want to upgrade to the latest versions, do the following::

  sudo pip install -U --no-deps ...

To install for your user only, which does not require sudo permissions,
modify the pip invocation for either case above as follows::

  pip install --user ...

Potential installation errors on Mac OS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The installation of FFC_ requires a C++11 compatible compiler and
standard library, some Mac OS systems (for example OS X "Lion")
supply the former, but not the latter.  Should you obtain errors
installing FFC_ of the following form:

.. code-block:: c

   ufc/ufc_wrap.cpp:3841:8: error: no member named 'shared_ptr' in namespace 'std'
   std::shared_ptr< ufc::function > tempshared1 ;

It's possible that you just need to tell the compiler to pick the
correct standard library.  To do so, try running with
``CXXFLAGS='-stdlib=libc++'`` when installing::

  sudo CXXFLAGS='-stdlib=libc++' pip install -U --no-deps ...

Visualisation software
----------------------

Firedrake can output data in VTK format, suitable for viewing in
Paraview_.  On Ubuntu and similar systems, you can obtain Paraview by
installing the ``paraview`` package.  On Mac OS, the easiest approach
is to download a binary from the `paraview website <Paraview_>`_.

Firedrake
---------

In addition to PyOP2, you will need to install Firedrake. There are two
routes, depending on whether you intend to contribute to Firedrake
development.

For performance reasons, there are various levels of caching with
eviction policies.  To support these, you will need to install
cachetools::

   sudo pip install cachetools

or (for your user only)::

   pip install --user cachetools

Firedrake will perform entirely correctly without this package, but
will be less efficient for tight time-stepping loops.

In order to have the form assembly cache operate in the most automatic
fashion possible, you are also advised to install psutil (version 2.0.0
or newer is required)::

  sudo pip install psutil

or (to install for your user only)::

  pip install --user psutil

Pip instructions for users
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you only wish to use Firedrake, and will not be contributing to
development at all, you can install Firedrake using pip::

  sudo pip install git+https://github.com/firedrakeproject/firedrake.git

or (to install for your user only)::

  pip install --user git+https://github.com/firedrakeproject/firedrake.git

You're now ready to go. You might like to start with the tutorial
examples on the :doc:`documentation page <documentation>`.

Git instructions for developers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cython >= 0.22 is required to build Firedrake. Install it using pip ::

 pip install "Cython>=0.22"

Next, obtain the Firedrake source from GitHub_ ::

 git clone https://github.com/firedrakeproject/firedrake.git

You will also need to point Python at the right directories. You might
want to consider setting this permanently in your
``.bashrc`` or similar::

  cd firedrake
  export PYTHONPATH=$PWD:$PYTHONPATH

From the Firedrake directory build the relevant modules::

 make

Cleaning disk caches after upgrade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After upgrading, you may need to clear any disk caches that Firedrake
maintains to ensure that your problem does not pick up any out of date
compiled modules.  This can be carried out by executing the
``firedrake-clean`` script.  If you carried out a sudo install of
Firedrake using pip, ``firedrake-clean`` should be in your ``PATH``
and so you should just be able to execute it.  If you carried out a
user install using pip, you will need to add ``$HOME/.local/bin`` to
your ``PATH`` ::

  export PATH=$HOME/.local/bin:$PATH

If you are using a checkout of Firedrake, ``firedrake-clean`` lives in
the ``scripts`` subdirectory.

Additional dependencies for developers
--------------------------------------

If you plan to develop Firedrake then you will require a few more
packages. 

Documentation dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Building the documention requires Sphinx_
(including the Youtube and Bibtex plugins) and wget_. For example on Ubuntu-like
Linux systems::

  sudo apt-get install python-sphinx wget

and on Mac OS::

  brew install wget
  sudo pip install sphinx 

note that the Sphinx in Homebrew is not the python documentation tool!

The Sphinx Youtube plugin is obtained by cloning the sphinx-contrib
repository::

  hg clone https://bitbucket.org/birkenfeld/sphinx-contrib

Then install the Youtube plugin::

  cd sphinx-contrib/youtube
  sudo python setup.py install

Note that the ``sphinxcontrib.youtube`` Ubuntu package does not work
for our purposes.

Finally install the Bibtex plugin::

  sudo pip install sphinxcontrib-bibtex

.. _PyOP2: http://op2.github.io/PyOP2
.. _FFC: https://bitbucket.org/mapdes/ffc
.. _FIAT: https://bitbucket.org/mapdes/fiat
.. _UFL: https://bitbucket.org/mapdes/ufl
.. _Instant: https://bitbucket.org/fenics-project/instant
.. _GitHub: https://github.com/firedrakeproject/firedrake
.. _Paraview: http://www.paraview.org
.. _Sphinx: http://www.sphinx-doc.org
.. _wget: http://www.gnu.org/software/wget/
.. _Swig: http://www.swig.org/
