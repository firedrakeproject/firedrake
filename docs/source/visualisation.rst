.. only:: html

  .. contents::

Visualising the results of simulations
======================================

Having run a simulation, it is likely that we will want to look at the
results.  To do this, Firedrake supports saving data in VTK_
format, suitable for visualisation in Paraview_ (amongst others).

In addition, 1D and 2D function could be plotted and displayed using the python
library of matplotlib (an optional dependency of firedrake)

Creating output files
~~~~~~~~~~~~~~~~~~~~~

Output for visualisation purposes is managed with a :class:`~.File`
object.  To create one, we just need to pass the name of the output
file on disk.  The file Firedrake creates is in PVD_ and therefore the
requested file name must end in ``.pvd``.

.. code-block:: python

   outfile = File("output.pvd")
   # The following raises an error
   badfile = File("output.vtu")

To save functions to the :class:`~.File` we use the
:meth:`~.File.write` method.

.. code-block:: python

   mesh = UnitSquareMesh(1, 1)
   V = FunctionSpace(mesh, "DG", 0)
   f = Function(V)
   f.interpolate(sin(SpatialCoordinate(mesh)[0]))

   outfile = File("output.pvd")
   outfile.write(f)

.. note::

   Output created for visualisation purposes is in most cases lossy.
   If you need to save data for checkpointing purposes, you should
   instead use Firedrake's :doc:`checkingpointing capabilities
   <checkpointing>`.

Saving time-dependent data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, we have a time-dependent simulation and would like to save the
same function at multiple timesteps.  This is straightforward, we must
create the output :class:`~.File` outside the time loop and call
:meth:`~.File.write` inside.

.. code-block:: python

   ...
   outfile = File("timesteps.pvd")

   while t < T:
       ...
       outfile.write(f)
       t += dt


The PVD_ data format supports specifying the timestep value for
time-dependent data.  We do not have to provide it to
:meth:`~.File.write`, by default an integer counter is used that is
incremented by 1 each time :meth:`~.File.write` is called.  It is
possible to override this by passing the keyword argument ``time``.

.. code-block:: python

   ...
   outfile = File("timesteps.pvd")

   while t < T:
       ...
       outfile.write(f, time=t)
       t += dt

Saving multiple functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Often we will want to save, and subsequently visualise, multiple
different fields from a simulation.  For example the velocity and
pressure in a fluids models.  This is possible either by having a
separate output file for each field, or by saving multiple fields to
the same output file.  The latter may be more convenient for
subsequent analysis.  To do this, we just need to pass multiple
:class:`~.Function`\s to :meth:`~.File.write`.

.. code-block:: python

   u = Function(V, name="Velocity")
   p = Function(P, name="Pressure")

   outfile = File("output.pvd")

   outfile.write(u, p, time=0)

   # We can happily do this in a timeloop as well.
   while t < t:
       ...
       outfile.write(u, p, time=t)

.. note::

   Subsequent writes to the same file *must* use the same number of
   functions, and the functions must have the *same* names.  The
   following example results in an error.

   .. code-block:: python

      u = Function(V, name="Velocity")
      p = Function(P, name="Pressure")

      outfile = File("output.pvd")

      outfile.write(u, p, time=0)
      ...
      # This raises an error
      outfile.write(u, time=1)
      # as does this
      outfile.write(p, u, time=1)

Visualising high-order data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The file format Firedrake outputs to currently only supports
visualisation of *linear* scalar-, vector-, or tensor-valued fields
represented with either a Lagrange or discontinuous Lagrange basis.
To visualise fields in anything other than one of these spaces we must
therefore decimate the data to this format first.  One option is to do
so by hand before outputting.  Either by :doc:`interpolating
<interpolation>` or else :func:`projecting <~.project>` the data to a
linear space.  Since this is such a common operation, the
:class:`~.File` object is set up to manage these operations
automatically, we just need to choose whether we want data to be
interpolated or projected.  The default is to use interpolation.  For
example, assume we wish to output a vector-valued function that lives
in an :math:`H(\operatorname{div})` space.  If we want it to be
interpolated in the output file we can use

.. code-block:: python

   V = FunctionSpace(mesh, "RT", 2)
   f = Function(V)
   ...
   outfile = File("output.pvd")
   outfile.write(f)

If instead we want projection, we use

.. code-block:: python

   projected = File("proj_output.pvd", project_output=True)
   projected.write(f)

Plotting with `matplotlib`
~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting 1D and 2D functions could be as easy as calling the built-in plot
function :func:`plot <firedrake.plot.plot>` with the :class:`~.Function` you wish to
plot.

Currently, firedrake supports plotting 1D and 2D functions, this is made
possible with an optional dependency matplotlib package.

To install matplotlib_, please look at the installation instructions of
matplotlib.

For 1D functions with degree less than 4, the plot of the function would be
exact using Bezier curves. For higher order 1D functions, the plot would be the
linear approximation by sampling points of the function. The number of sample
points per element could be specfied to when calling :func:`plot
<firedrake.plot.plot>`.

For multiple 1D functions, for example, in the case of time-dependent functions
at different times. They could be plotted together by passing the list of
function when calling the function :func:`plot <firedrake.plot.plot>`. The returned
figure will contain a slider and an autoplay button so that it could be viewed
in a animated fashion. The plus and minus buttons can change the speed of the
animation.

When used in Jupyter Notebook, plotting multiple 1D functions using additional
keyword argument ``interactive=True`` when calling the function 
:func:`plot <firedrake.plot.plot>` will generate an interactive slider for
selecting the figures. 

For 2D functions, both surface plots and contour plots are supported. By
default, the :func:`plot <firedrake.plot.plot>` will return a surface plot in the
colour map of coolwarm. Contour plotting could be enabled by passing the keyword
argument ``contour=True``.


Selecting the output space when outputting multiple functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

All functions that are output to the same file must be represented in
the same space, the rules for selecting the output space are as
follows.  If *all* the functions, including the mesh's coordinate
field are continuous (i.e. they live in :math:`H^1`), then the output
will be decimated to a piecewise linear Lagrange space.  If any of the
functions are at least partially discontinuous, again including the
coordinate field (this occurs when using periodic meshes), then the
output will be decimated to a piecewise linear discontinuous Lagrange
space.

.. _Paraview: http://www.paraview.org
.. _VTK: http://www.vtk.org
.. _PVD: http://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
.. _matplotlib: http://matplotlib.org
