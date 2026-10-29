.. only:: html

   .. contents::

Optimising Firedrake Performance
================================

  "Premature optimisation is the root of all evil"

  -- Donald Knuth

Performance of a Firedrake script is rarely optimal from the outset.
Choice of solver options, discretisation and variational form all have
an impact on the amount of time your script takes to run. More general
programming considerations such as not repeating unnecessary work inside
of a loop can also be signficant.

It is always a bad idea to attempt to optimise your code without a solid
understanding of where the bottlenecks are, else you could spend vast
amounts of developer time resulting in little to no improvement in performance.
The best strategy for performance optimisation should therefore always be to start
at the highest level possible with an overview of the entire problem before
drilling down into specific hotspots. To get this high level understanding of
your script we strongly recommend that you first profile your script using a
flame graph (see :ref:`below <generating-flame-graphs>`).

.. _generating-flame-graphs:

Automatic flame graph generation with PETSc
-------------------------------------------

`Flame graphs <https://www.brendangregg.com/flamegraphs.html>`_ are a very
useful entry point when trying to optimise your application since they make
hotspots easy to find. PETSc can generate a flame graph input file using
its logging infrastructure that Firedrake has extended by annotating many of
its own functions with PETSc events. This allows users to easily generate
informative flame graphs giving a lot of insight into the internals of
Firedrake and PETSc.

As an example, here is a flame graph showing the performance of the
:doc:`scalar wave equation with higher-order mass lumping demo
<demos/higher_order_mass_lumping.py>`.
It is interactive and you can zoom in on functions by clicking.

.. raw:: html
    :file: images/flame_graph_demo.svg

One can immediately see that the dominant hotspots for this code are
assembly and writing to output so any optimisation effort should be
spent in those. Some time is also spent in ``firedrake.__init__`` but
this corresponds to the amount of time spent importing Firedrake and
would be amortized for longer-running problems.

Flame graphs can also be generated for codes run in parallel with the
reported times in the graph given by the maximum value across all ranks.

Generating the flame graph
~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate a flame graph from your Firedrake script you need to:

1. Run your code with the extra flag ``-log_view :foo.txt:ascii_flamegraph``.
   For example:

   .. code-block:: console

     $ python myscript.py -log_view :foo.txt:ascii_flamegraph

   This will run your program as usual but output an additional file
   called ``foo.txt`` containing the profiling information.

2. Visualise the results. This can be done in one of two ways:

   * Generate an SVG file using the ``flamegraph.pl`` script from
     `this repository <https://github.com/brendangregg/FlameGraph>`_
     with the command:

     .. code-block:: console

       $ ./flamegraph.pl foo.txt > foo.svg

     You can then view ``foo.svg`` in your browser.

   * Upload the file to `speedscope <https://www.speedscope.app/>`_ and view it there.

Adding your own events
~~~~~~~~~~~~~~~~~~~~~~

It is very easy to add your own events to the flame graph and there
are a few different ways of doing it. The simplest methods are:

* With a context manager:

  .. code-block:: python

      from firedrake.petsc import PETSc

      with PETSc.Log.Event("foo"):
          do_something_expensive()

* With a decorator:

  .. code-block:: python

      from firedrake.petsc import PETSc

      @PETSc.Log.EventDecorator("foo")
      def do_something_expensive():
          ...

  If no arguments are passed to ``PETSc.Log.EventDecorator`` then the
  event name will be the same as the function.

Caveats
~~~~~~~

* The ``flamegraph.pl`` script assumes by default that the values
  in the stack traces are sample counts. This means that if you
  hover over functions in the SVG it will report the count in terms
  of 'samples' rather than the correct unit of microseconds. A simple
  fix to this is to include the command line option ``--countname us``
  when you generate the SVG. For example:

  .. code-block:: console

    $ ./flamegraph.pl --countname us foo.txt > foo.svg

* If you use PETSc stages in your code these will be ignored in the flame graph.

* If you call ``PETSc.Log.begin()`` as part of your script/package
  then profiling will not work as expected. This is because this
  function starts PETSc's default (flat) logging while we need to
  use nested logging instead.

  This issue can be avoided with the simple guard:

  .. code-block:: python

    import petsctools

    # If the -log_view flag is passed you don't need to call
    # PETSc.Log.begin because it is done automatically.
    if "log_view" not in petsctools.get_commandline_options():
        PETSc.Log.begin()

Common performance issues
-------------------------

Calling ``solve`` repeatedly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When solving PDEs, Firedrake uses a PETSc ``SNES`` (nonlinear solver)
under the hood. Every time the user calls :py:func:`~firedrake.solving.solve`
a new ``SNES`` is created and used to solve the problem. This is a
convenient shorthand for scripts that only need to solve a problem
once, but it is fairly expensive to set up a new ``SNES`` and so
repeated calls to :py:func:`~firedrake.solving.solve` will introduce
some overhead.

To get around this problem, users should instead instantiate
a variational problem (e.g. :py:class:`~.NonlinearVariationalProblem`)
and solver (e.g. :py:class:`~.NonlinearVariationalSolver`) outside of
the loop body. An example showing how this is done can be found
in `this demo <https://firedrakeproject.org/demos/DG_advection.py.html>`_.

Other useful tools
------------------

Here we present a handful of performance analysis tools that users may
find useful to run with their codes.

py-spy
~~~~~~

`py-spy <https://github.com/benfred/py-spy>`_ is a great sampling
profiler that outputs directly to SVG flame graphs. It allows users
to see the entire stack trace of the program rather than just the
annotated PETSc events and unlike most Python profilers it can also
profile native code.

A flame graph for your Firedrake script can be generated from py-spy with:

.. code-block:: console

   $ py-spy record -o foo.svg --native -- python myscript.py

Beyond the inherent uncertainty that comes from using a sampling profiler,
one substantial limitation of py-spy is that it does not work when run
in parallel.

pyinstrument
~~~~~~~~~~~~~

`pyinstrument <https://github.com/joerick/pyinstrument>`_ is a great
sample-based profiling tool that you can use to easily identify
hotspots in your code. To use the profiler simply run:

.. code-block:: console

   $ pyinstrument myscript.py

This will print out a timed callstack to the terminal. To instead
generate an interactive graphic you can view in your browser pass
the ``-r html`` flag.

Unfortunately, pyinstrument cannot profile native code. This means
that information about the code's execution inside of PETSc is largely
lost.

memory_profiler
~~~~~~~~~~~~~~~

`memory_profiler <https://github.com/pythonprofilers/memory_profiler>`_
is a useful tool that you can use to monitor the memory usage of your
script. After installing it you can simply run:

.. code-block:: console

   $ mprof run python myscript.py
   $ mprof plot

The former command will run your script and generate a file containing the
profiling information. The latter then displays a plot of the memory usage
against execution time for the whole script.

memory_profiler also works in parallel. You can pass either of the
``--include-children`` or ``--multiprocess`` flags to ``mprof``
depending on whether or not you want to accumulate the memory usage
across ranks or plot them separately. For example:

.. code-block:: console

   $ mprof run --include-children mpiexec -n 4 python myscript.py

Score-P
~~~~~~~

`Score-P <https://www.vi-hps.org/projects/score-p/>`_ is a tool aimed
at HPC users. We found it to provide some useful insight into MPI
considerations such as load balancing and communication overhead.

To use it with Firedrake, users will also need to install Score-P's
`Python bindings <https://github.com/score-p/scorep_binding_python>`_.

TAU
~~~

These instructions describe how to install the `TAU profiling tools <https://www.cs.uoregon.edu/research/tau/home.php>`_
for use with Firedrake on an Ubuntu system (including Ubuntu run using WSL).

#. Start by making a directory for installing TAU, for example ``${HOME}/TAU``. We will store the path to this directory in an environment variable so it can be used later:

   .. code-block:: console

     $ export MY_TAU_DIR=${HOME}/TAU
     $ mkdir $MY_TAU_DIR
     $ cd $MY_TAU_DIR

#. Download and unpack TAU into this directory:

   .. code-block:: console

     $ wget http://tau.uoregon.edu/tau.tgz
     $ tar zxvf tau.tgz

#. Download and unpack PDT (Program Database Toolkit):

   .. code-block:: console

     $ wget http://tau.uoregon.edu/pdt.tgz
     $ tar zxvf pdt.tgz

#. Start the Firedrake venv so that you have the same python as used by Firedrake. Use the path to your firedrake installation in the following command:

   .. code-block:: console

     $ . /path/to/firedrake/venv-firedrake/bin/activate

#. Build PDT:

   .. code-block:: console

     $ cd pdtoolkit-3.25.2
     $ ./configure
     $ make
     $ make install

#. Download the third-party library package. This needs to be untarred in the top level directory of the tau distribution (``${HOME}/TAU/tau-2.35.2`` in this example). If you have a different version of TAU change the version number ``2.35.2`` to the version number that you have:

   .. code-block:: console

     $ cd ..
     $ wget http://tau.uoregon.edu/ext.tgz
     $ cd tau-2.35.2
     $ tar zxvf ../ext.tgz

#. Build and install TAU

   Run the configure command:

   .. code-block:: console

     $ ./configure -bfd=download -dwarf=download -unwind=download -iowrapper -pdt=${MY_TAU_DIR}/pdtoolkit-3.25.2 -prefix=${MY_TAU_DIR}/tau -python -mpi

   At the end of the output from the configure command will be an entry to add to your PATH environment variable e.g.

   .. code-block:: console

     Configuration complete!
     Please add  /path/to/TAU/tau/x86_64/bin  to your path

   Add this directory to your PATH e.g.

   .. code-block:: console

     $ export PATH=${PATH}:/path/to/TAU/tau/x86_64/bin

   where ``/path/to/TAU`` will be replaced by the path to your TAU installation (e.g. ``${HOME}/TAU``). The ``x86_64`` part of the part is platform dependent and may differ on your system.

   You will need to add this entry to your path whenever you use TAU.

   Now make and install TAU:

   .. code-block:: console

     $ make install

#. Run an example:

   Set up a new directory for the example:

   .. code-block:: console

     $ mkdir -p ~/tau_examples/sampling
     $ cd ~/tau_examples/sampling

   Download a script for Firedrake to run, e.g. the Burgers equation example:

   .. code-block:: console

     $ wget https://firedrakeproject.org/demos/burgers.py

   Make sure threading is not used. Firedrake complains if this is not done:

   .. code-block:: console

     $ export OMP_NUM_THREADS=1

   Run the example without TAU to make sure that all the JIT compilation has been done

   .. code-block:: console

     $ mpiexec -n 2 tau_exec python burgers.py

   Run the example with TAU to get a profile:

   .. code-block:: console

     $ mpiexec -n 2 tau_exec python burgers.py

#. Post process and view the output:

   .. code-block:: console

     $ pprof > profile_combined.txt
     $ less profile_combined.txt

#. We will now add the capability for TAU to do tracing (rather than sampling). This will show a timeline of what the application is doing. This requires running configure and make again but this can be doing using the existing installation directory:

   .. code-block:: console

     $ cd $MY_TAU_DIR/tau-2.35.2
     $ ./configure -bfd=download -dwarf=download -unwind=download -iowrapper -pdt=${MY_TAU_DIR}/pdtoolkit-3.25.2 -prefix=${MY_TAU_DIR}/tau -python -mpi -TRACE -perfetto
     $ make install

#. Run a tracing example:

   Set up a new directory for the example:

   .. code-block:: console

     $ mkdir -p ~/tau_examples/tracing
     $ cd ~/tau_examples/tracing

   Download a script for Firedrake to run, e.g. the Burgers equation example:

   .. code-block:: console

     $ wget https://firedrakeproject.org/demos/burgers.py

   Set the trace output format for the Perfetto viewer:

   .. code-block:: console

     $ export TAU_TRACE_FORMAT=perfetto

   Run the tracing experiment:

   .. code-block:: console

     $ mpiexec -n 2 tau_exec -T TRACE python burgers.py

   There should now be a file called tau.perfetto.gz Unzip this file:

   .. code-block:: console

     $ gunzip tau.perfetto.gz

   and upload to ui.perfetto.dev for viewing using the 'Open trace file' option from the menu.

#. View the trace.

   In the Perfetto viewer click on the trace for one of the ranks to expand the view. You can navigate the trace using:

   - W: zoom in
   - S: zoom out
   - A: move left (earlier time)
   - D: move right (later time)
