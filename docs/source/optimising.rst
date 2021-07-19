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

The best strategy for performance optimisation should always be to start
at the highest level possible with an overview of the entire problem before
drilling down into specific hotspots. For this we recommend first profiling
your script using a flame graph (see :ref:`below <generating-flame-graphs>`).

.. _generating-flame-graphs:

Generating flame graphs
-----------------------

`Flame graphs <https://www.brendangregg.com/flamegraphs.html>`_ are a very
useful entry point when trying to optimise your application since they make
hotspots easy to find. PETSc can generate a flame graph input file using
its logging infrastructure that Firedrake has extended by annotating many of
its own functions with PETSc events. This allows users to easily generate
informative flame graphs giving a lot of insight into the internals of
Firedrake and PETSc.

As an example, here is a flame graph showing the performance of the
`scalar wave equation with higher-order mass lumping demo
<https://firedrakeproject.org/demos/higher_order_mass_lumping.py.html>`_.
It is interactive and you can zoom in on functions by clicking.

.. raw:: html
    :file: images/flame_graph_demo.svg

One can immediately see that the dominant hotspots for this code are
assembly and writing to output so any optimisation effort should be
spent in those. Some time is also spent in ``firedrake.__init__`` but
this corresponds to the amount of time spent importing Firedrake and
would be amortized for longer-running problems.

.. note::

   The flame graph output works in parallel too. The time per function in
   that case is given by the maximum value across all ranks.

..
  ## Generating the flame graph

  To generate a flame graph from your Firedrake script you need to:

  1. Run your code with the extra flag `-log_view :foo.txt:ascii_flamegraph`. This will run your program as usual but output an additional file called `foo.txt`.

  2. Visualise the results. This can be done in one of two ways:
    
      - Generate an SVG file using the `flamegraph.pl` script from [this repository](https://github.com/brendangregg/FlameGraph) with the command:

          ```bash
          $ ./flamegraph.pl foo.txt > foo.svg
          ```

          You can then view the output file in your browser.

      - Upload the file to [speedscope](https://www.speedscope.app/) and view it there.

  ## Adding your own events

  It is very easy to add your own events to the flame graph and there are a few different ways of doing it.
  The simplest methods are:

  - With a context manager:
      
      ```python
      from firedrake.petsc import PETSc

      with PETSc.Log.Event("foo"):
          do_something_expensive()
      ```

  - With a decorator:

      ```python
      from firedrake.petsc import PETSc

      @PETSc.Log.EventDecorator("foo")
      def do_something_expensive():
          ...
      ```

      If no arguments are passed to `PETSc.Log.EventDecorator` then the event name will be the same as the function.

  ## Extra information

  - The `flamegraph.pl` script assumes by default that the values in the stack traces are sample counts.
    This means that if you hover over functions in the SVG it will report the count in terms of 'samples' rather than the correct unit of microseconds.
    A simple fix to this is to include the command line option `--countname us` when you generate the SVG.

  - If you use PETSc stages in your code these will be ignored in the flame graph.

  - If you call `PETSc.Log.begin()` as part of your script/package then profiling will not work as expected. 
    This is because it will start PETSc's default (flat) logging while we need to use nested logging instead.

    This issue can be avoided with the simple guard:
    
    ```python
    from firedrake.petsc import OptionsManager

    # If the -log_view flag is passed you don't need to call 
    # PETSc.Log.begin because it is done automatically.
    if "log_view" in OptionsManager.commandline_options:
        PETSc.Log.begin()
    ```

Common issues
-------------

Calling ``solve`` repeatedly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HPC-specific considerations
---------------------------

Filesystem contention
~~~~~~~~~~~~~~~~~~~~~

* tarball the venv

Parallel communication
~~~~~~~~~~~~~~~~~~~~~~

* Score-P
