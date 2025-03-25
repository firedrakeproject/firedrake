Profiling
=========

Profiling PyOP2 programs
------------------------

Profiling a PyOP2 program is as simple as profiling any other Python
code. You can profile the jacobi demo in the PyOP2 ``demo`` folder as
follows: ::

  python -m cProfile -o jacobi.dat jacobi.py

This will run the entire program under cProfile_ and write the profiling
data to ``jacobi.dat``. Omitting ``-o`` will print a summary to stdout,
which is not very helpful in most cases.

Creating a graph
................

There is a much more intuitive way of representing the profiling data
using the excellent gprof2dot_ to generate a graph. Install from `PyPI
<http://pypi.python.org/pypi/gprof2dot/>`__ with ::

  sudo pip install gprof2dot

Use as follows to create a PDF: ::

  gprof2dot -f pstats -n 1 jacobi.dat | dot -Tpdf -o jacobi.pdf

``-f pstats`` tells ``gprof2dot`` that it is dealing with Python
cProfile_ data (and not actual *gprof* data) and ``-n 1`` ignores
everything that makes up less than 1% of the total runtime - most likely
you are not interested in that (the default is 0.5).

Consolidating profiles from different runs
..........................................

To aggregate profiling data from different runs, save the following as
``concat.py``: ::

  """Usage: concat.py PATTERN FILE"""

  import sys
  from glob import glob
  from pstats import Stats

  if len(sys.argv) != 3:
      print __doc__
      sys.exit(1)
  files = glob(sys.argv[1])
  s = Stats(files[0])
  for f in files[1:]: s.add(f)
  s.dump_stats(sys.argv[2])

With profiles from different runs named ``<basename>.*.part``, use it
as ::

  python concat.py '<basename>.*.part' <basename>.dat

and then call ``gprof2dot`` as before.

Using PyOP2's internal timers
-----------------------------

PyOP2 automatically times the execution of certain regions:

* Sparsity building
* Plan construction
* Parallel loop kernel execution
* Halo exchange
* Reductions
* PETSc Krylov solver

To output those timings, call :func:`~pyop2.profiling.summary` in your
PyOP2 program or run with the environment variable
``PYOP2_PRINT_SUMMARY`` set to 1.

To query e.g. the timer for parallel loop execution programatically,
use the :func:`~pyop2.profiling.timing` helper: ::

  from pyop2 import timing
  timing("ParLoop compute")               # get total time
  timing("ParLoop compute", total=False)  # get average time per call

To add additional timers to your own code, you can use the
:func:`~pyop2.profiling.timed_region` and
:func:`~pyop2.profiling.timed_function` helpers: ::

  from pyop2.profiling import timed_region, timed_function

  with timed_region("my code"):
      # my code

  @timed_function("my function")
  def my_func():
      # my func

Line-by-line profiling
----------------------

To get a line-by-line profile of a given function, install Robert Kern's
`line profiler`_ and:

1. Import the :func:`~pyop2.profiling.profile` decorator: ::

     from pyop2.profiling import profile

2. Decorate the function to profile with ``@profile``
3. Run your script with ``kernprof.py -l <script.py>``
4. Generate an annotated source file with ::

     python -m line_profiler <script.py.lprof>

Note that ``kernprof.py`` injects the ``@profile`` decorator into the
Python builtins namespace. PyOP2 provides a passthrough version of this
decorator which does nothing if ``profile`` is not found in
``__builtins__``. This means you can run your script regularly without
having to remove the decorators again.

The :func:`~pyop2.profiling.profile` decorator also works with the
memory profiler (see below). PyOP2 therefore provides the
:func:`~pyop2.profiling.lineprof` decorator which is only enabled when
running with ``kernprof.py``.

A number of PyOP2 internal functions are decorated such that running
your PyOP2 application with ``kernprof.py`` will produce a line-by-line
profile of the parallel loop computation (but not the generated code!).

Memory profiling
----------------

To profile the memory usage of your application, install Fabian
Pedregosa's `memory profiler`_ and:

1. Import the :func:`~pyop2.profiling.profile` decorator: ::

     from pyop2.profiling import profile

2. Decorate the function to profile with ``@profile``.
3. Run your script with ::

      python -m memory_profiler <script.py>

   to get a line-by-line memory profile of your function.
4. Run your script with ::
   
      memprof run --python <script.py>

   to record memory usage of your program over time.
5. Generate a plot of the memory profile with ``memprof plot``.

Note that ``memprof`` and ``python -m memory_profiler`` inject the
``@profile`` decorator into the Python builtins namespace. PyOP2
provides a passthrough version of this decorator which does nothing if
``profile`` is not found in ``__builtins__``. This means you can run
your script regularly without having to remove the decorators again.

The :func:`~pyop2.profiling.profile` decorator also works with the line
profiler (see below). PyOP2 therefore provides the
:func:`~pyop2.profiling.memprof` decorator which is only enabled when
running with ``memprof``.

A number of PyOP2 internal functions are decorated such that running
your PyOP2 application with ``memprof run`` will produce a memory
profile of the parallel loop computation (but not the generated code!).

.. _cProfile: https://docs.python.org/2/library/profile.html#cProfile
.. _gprof2dot: https://code.google.com/p/jrfonseca/wiki/Gprof2Dot
.. _line profiler: https://pythonhosted.org/line_profiler/
.. _memory profiler: https://github.com/fabianp/memory_profiler
