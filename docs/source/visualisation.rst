.. only:: html

  .. contents::

Visualising the results of simulations
======================================

Having run a simulation, it is likely that we will want to look at the
results.  To do this, Firedrake supports saving data in VTK_ format,
suitable for visualisation in Paraview_ (amongst others).

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

   Output created for visualisation purposes is
   not intended for purposes other than visualisation. If you need
   to save data for checkpointing purposes, you should
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


Visualising high-order data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The file format Firedrake outputs to currently supports the
visualisation of scalar-, vector-, or tensor-valued fields
represented with an `arbitrary order (possibly discontinuous) Lagrange basis`__.
Furthermore, the fields must be in an isoparametric function space, meaning
the :doc:`mesh coordinates <mesh-coordinates>` associated to a field must be represented
with the same basis as the field. To visualise fields in anything
other than these spaces we must transform the data to this
format first. One option is to do so by hand before outputting.
Either by :doc:`interpolating <interpolation>` or else :func:`projecting <project>`
the :doc:`mesh coordinates <mesh-coordinates>` and then the field. Since this is
such a common operation, the :class:`~.File` object is set up to manage these
operations automatically, we just need to choose whether we want data to be
interpolated or projected. The default is to use interpolation.  For example,
assume we wish to output a vector-valued function that lives in an :math:`H(\operatorname{div})`
space. If we want it to be interpolated in the output file we can use

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

.. note::

   This feature requires Paraview version 5.5.0 or better. If you must use an
   older version of Paraview, you must manually interpolate mesh coordinates
   and field coordinates to a piecewise linear function space, represented
   with either a Lagrange (H1) or discontinuous Lagrange (L2) basis. The :class:`~.File`
   is also setup to manage this issue. For instance, we can force the output
   to be discontinuous piecewise linears via

   .. code-block:: python

      projected = File("proj_output.pvd", target_degree=1, target_continuity=H1)
      projected.write(f)


Using Paraview on higher order data
+++++++++++++++++++++++++++++++++++

Paraview's visualisation algorithims are typically exact on piecewise linear data,
but if you write higher order data, Paraview will produce an approximate visualisation.
This approximation can be controlled in at least two ways:

1. Under the display properties of an unstructured grid,
   the Nonlinear Subdivision Level can be increased; this option controls
   the display of unstructured grid data and can be used to present a plausible
   curved geometry. Further, the Nonlinear Subdivision Level can also be
   changed after applying filters such as Extract Surface.
2. The Tessellate_ filter can be applied to unstructured grid data
   and has two parameters: Chord Error, Maximum Number of Subdivisions,
   and Field Error. Tessellation_ is the process of approximating a higher
   order geometry via subdividing cells into smaller linear cells. Chord Error
   is a tessellation error metric, the distance between the midpoint of any
   edge on the tessellated geometry and a corresponding point in the original
   geometry. Field Error is analogous to Chord Error: the error of the field
   on the tessellated data is compared pointwise to the original data at
   the midpoints of the edges of the tessellated geometry and the corresponding
   points on the original geometry. The Maximum Number of Subdivisions is the
   maximum number of times an edge in the original geometry can be subdivided. 

Besides the two tools listed above, Paraview provides many other tools (filters)
that might be applied to the original data or composed with the tools listed above.
Documentation on these interactions is sparse, but tessellation can be used to understand
this issue: the Tessellate_ filter produces another unstructured grid from its inputs so
algorithms can be applied to both the tessellated and input unstructured grid. The tessellated
data can also be saved for future reference.

.. note::

   Field Error is hidden in the current Paraview UI (5.7) so we
   include a visual guide wherein the field error is set via the
   highlighted field directly below Chord Error:

   .. image:: images/paraview-field-error.png

   We also note that the Tessellate_ filter (and other filters) can
   be more clearly controlled via the Paraview Python shell (under
   the View menu). For instance, Field Error can be more clearly
   specified via an argument to the Tessellate_ filter constructor.

   .. code-block:: python

      from paraview.simple import *
      pvd = PVDReader(FileName="Example.pvd")
      tes = Tessellate(pvd, FieldError=0.001)


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

Selecting the output space when outputting multiple functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

All functions, including the mesh coordinates, that are output
to the same file must be represented in the same space, the rules
for selecting the output space are as follows. First, all functions
must be defined via the same cell type otherwise an exception will be
thrown. Second, if all functions are continuous (i.e. they live in
:math:`H^1`), then the output space will be a piecewise continuous space. If any of the
functions are at least partially discontinuous, again including the
coordinate field (this occurs when using periodic meshes), then the
output will use a piecewise discontinuous space. Third, the degree of
the basis will be the maximum degree used over the spaces
of all input functions. For elements where the degree is a tuple
(this occurs when using tensor product elements), the the maximum
will be over the elements of the tuple too, meaning a tensor
product of elements of degree 4 and 2 will be turned into a tensor
product of elements of degree 4 and 4.


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


.. _Paraview: http://www.paraview.org
.. _VTK: http://www.vtk.org
.. _PVD: http://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
.. _matplotlib: http://matplotlib.org
.. _Arbitrary: https://blog.kitware.com/modeling-arbitrary-order-lagrange-finite-elements-in-the-visualization-toolkit/
__ Arbitrary_
.. _Tessellate: https://kitware.github.io/paraview-docs/latest/python/paraview.simple.Tessellate.html
.. _Tessellation: https://ieeexplore.ieee.org/document/1634311/
