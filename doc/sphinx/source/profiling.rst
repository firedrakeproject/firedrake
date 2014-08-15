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

.. _cProfile: https://docs.python.org/2/library/profile.html#cProfile
.. _gprof2dot: https://code.google.com/p/jrfonseca/wiki/Gprof2Dot
