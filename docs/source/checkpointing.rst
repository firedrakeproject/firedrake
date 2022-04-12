.. only:: html

   .. contents::

=====================
 Checkpointing state
=====================

In addition to the ability to :doc:`write field data to vtu files
<visualisation>`, suitable for visualisation in Paraview_, Firedrake
has support for checkpointing state to disk.  This enables
pausing, and subsequently resuming, a simulation at a later time.

Checkpointing with CheckpointFile
=================================

:class:`~.CheckpointFile` class facilitates saving/loading meshes and
:class:`~.Function` s to/from an HDF5_ file. 
The implementation is scalable in that :class:`~.Function` s are
saved to and loaded from the file entirely in parallel without needing
to pass through a single process.
It also supports flexible checkpointing, where one can save meshes and
:class:`~.Function` s on :math:`N` processes and later load them on
:math:`P` processes.

Saving
------

In the following example we save in "example.h5" file two :class:`~.Function` s,
along with the mesh on which they are defined.

.. code-block:: python3

    mesh = UnitSquareMesh(10, 10, name="meshA")
    V = FunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
    Z = V * W
    f = Function(V, name="f")
    g = Function(Z, name="g")
    with CheckpointFile("example.h5", 'w') as afile:
        afile.save_mesh(mesh)  # optional
        afile.save_function(f)
        afile.save_function(g)

If the mesh name is not provided by the user when constructing the mesh, the
default mesh name, :data:`~.DEFAULT_MESH_NAME`, is assigned, which is then
used when saving in the file. We, however, strongly encourage users to name
each mesh.

Inspecting saved data
---------------------

Now "example.h5" file has been created and the mesh and :class:`~.Function`
data have been saved.
One can view the contents of the HDF5 file with "h5dump" utility shipped with
the HDF5 installation; "h5dump -n example.h5", for instance, shows:

::

    HDF5 "example.h5" {
    FILE_CONTENTS {
     group      /
     group      /topologies
     group      /topologies/firedrake_mixed_meshes
     group      /topologies/firedrake_mixed_meshes/meshA
     group      /topologies/firedrake_mixed_meshes/meshA/firedrake_mixed_function_spaces
     group      /topologies/firedrake_mixed_meshes/meshA/firedrake_mixed_function_spaces/firedrake_function_space_meshA_CG2(None,None)_meshA_CG1(None,None)
     group      /topologies/firedrake_mixed_meshes/meshA/firedrake_mixed_function_spaces/firedrake_function_space_meshA_CG2(None,None)_meshA_CG1(None,None)/0
     group      /topologies/firedrake_mixed_meshes/meshA/firedrake_mixed_function_spaces/firedrake_function_space_meshA_CG2(None,None)_meshA_CG1(None,None)/1
     group      /topologies/firedrake_mixed_meshes/meshA/firedrake_mixed_function_spaces/firedrake_function_space_meshA_CG2(None,None)_meshA_CG1(None,None)/firedrake_functions
     group      /topologies/firedrake_mixed_meshes/meshA/firedrake_mixed_function_spaces/firedrake_function_space_meshA_CG2(None,None)_meshA_CG1(None,None)/firedrake_functions/g
     group      /topologies/firedrake_mixed_meshes/meshA/firedrake_mixed_function_spaces/firedrake_function_space_meshA_CG2(None,None)_meshA_CG1(None,None)/firedrake_functions/g/0
     group      /topologies/firedrake_mixed_meshes/meshA/firedrake_mixed_function_spaces/firedrake_function_space_meshA_CG2(None,None)_meshA_CG1(None,None)/firedrake_functions/g/1
     group      /topologies/meshA_topology
     group      /topologies/meshA_topology/dms
     group      /topologies/meshA_topology/dms/coordinateDM
     dataset    /topologies/meshA_topology/dms/coordinateDM/order
     group      /topologies/meshA_topology/dms/coordinateDM/section
     dataset    /topologies/meshA_topology/dms/coordinateDM/section/atlasDof
     dataset    /topologies/meshA_topology/dms/coordinateDM/section/atlasOff
     group      /topologies/meshA_topology/dms/coordinateDM/section/field0
     dataset    /topologies/meshA_topology/dms/coordinateDM/section/field0/atlasDof
     dataset    /topologies/meshA_topology/dms/coordinateDM/section/field0/atlasOff
     group      /topologies/meshA_topology/dms/coordinateDM/section/field0/component0
     group      /topologies/meshA_topology/dms/coordinateDM/section/field0/component1
     group      /topologies/meshA_topology/dms/coordinateDM/vecs
     group      /topologies/meshA_topology/dms/coordinateDM/vecs/coordinates
     dataset    /topologies/meshA_topology/dms/coordinateDM/vecs/coordinates/coordinates
     group      /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_1
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_1/order
     group      /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_1/section
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_1/section/atlasDof
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_1/section/atlasOff
     group      /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_1/vecs
     group      /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_1/vecs/g[1]
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_1/vecs/g[1]/g[1]
     group      /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_2
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_2/order
     group      /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_2/section
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_2/section/atlasDof
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_2/section/atlasOff
     group      /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_2/vecs
     group      /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_2/vecs/meshA_coordinates
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_0_0_False_2/vecs/meshA_coordinates/meshA_coordinates
     group      /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/order
     group      /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/section
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/section/atlasDof
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/section/atlasOff
     group      /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/vecs
     group      /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/vecs/f
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/vecs/f/f
     group      /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/vecs/g[0]
     dataset    /topologies/meshA_topology/dms/firedrake_dm_1_1_0_False_1/vecs/g[0]/g[0]
     group      /topologies/meshA_topology/firedrake_meshes
     group      /topologies/meshA_topology/firedrake_meshes/meshA
     group      /topologies/meshA_topology/firedrake_meshes/meshA/firedrake_function_spaces
     group      /topologies/meshA_topology/firedrake_meshes/meshA/firedrake_function_spaces/firedrake_function_space_meshA_CG1(None,None)
     group      /topologies/meshA_topology/firedrake_meshes/meshA/firedrake_function_spaces/firedrake_function_space_meshA_CG1(None,None)/firedrake_functions
     group      /topologies/meshA_topology/firedrake_meshes/meshA/firedrake_function_spaces/firedrake_function_space_meshA_CG1(None,None)/firedrake_functions/g[1]
     group      /topologies/meshA_topology/firedrake_meshes/meshA/firedrake_function_spaces/firedrake_function_space_meshA_CG2(None,None)
     group      /topologies/meshA_topology/firedrake_meshes/meshA/firedrake_function_spaces/firedrake_function_space_meshA_CG2(None,None)/firedrake_functions
     group      /topologies/meshA_topology/firedrake_meshes/meshA/firedrake_function_spaces/firedrake_function_space_meshA_CG2(None,None)/firedrake_functions/f
     group      /topologies/meshA_topology/firedrake_meshes/meshA/firedrake_function_spaces/firedrake_function_space_meshA_CG2(None,None)/firedrake_functions/g[0]
     group      /topologies/meshA_topology/labels
     group      /topologies/meshA_topology/labels/...
     ...
     group      /topologies/meshA_topology/topology
     dataset    /topologies/meshA_topology/topology/cells
     dataset    /topologies/meshA_topology/topology/cones
     dataset    /topologies/meshA_topology/topology/order
     dataset    /topologies/meshA_topology/topology/orientation
     }
    }

Loading
-------

We can load the mesh and :class:`~.Function` s in "example.h5" as in the
following.

.. code-block:: python3

    with CheckpointFile("example.h5", 'r') as afile:
        mesh = afile.load_mesh("meshA")
        f = afile.load_function(mesh, "f")
        g = afile.load_function(mesh, "g") 

Note that one needs to load the mesh before loading the :class:`~.Function` s
that are defined on it. If the default mesh name, :data:`~.DEFAULT_MESH_NAME`,
was used when saving, the mesh name can be ommitted when loading.

Extrusion
---------

Extruded meshes can be saved and loaded seamlessly as the following:

.. code-block:: python3

    mesh = UnitSquareMesh(10, 10, name="meshA")
    extm = ExtrudedMesh(mesh, layers=4)
    V = FunctionSpace(extm, "CG", 2)
    f = Function(V, name="f")
    with CheckpointFile("example_extrusion.h5", 'w') as afile:
        afile.save_mesh(mesh)  # optional
        afile.save_function(f)
    with CheckpointFile("example_extrusion.h5", 'r') as afile:
        extm = afile.load_mesh("meshA_extruded")
        f = afile.load_function(extm, "f")

Note that if the name was not directly provided by the user, the base mesh's
name postfixed by "_extruded" is given to the extruded mesh.

Timestepping
------------

The following demonstrates how a :class:`~.Function` can be saved and loaded
at each timestep in a time-series simulation by setting the `idx` parameter:

.. code-block:: python3

    mesh = UnitSquareMesh(2, 2, name="meshA")
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V, name="f")
    x, y = SpatialCoordinate(mesh)
    with CheckpointFile("example_timestepping.h5", 'w') as afile:
        afile.save_mesh(mesh)  # optional
        for i in range(4):
            f.interpolate(x * i)
            afile.save_function(f, idx=i)
    with CheckpointFile("example_timestepping.h5", 'r') as afile:
        mesh = afile.load_mesh("meshA")
        for i in range(4):
            f = afile.load_function(mesh, "f", idx=i)

Note that each :class:`~.Function` can either be saved in the timestepping mode
with `idx` parameter always set or in the normal mode (non-timestepping mode)
with `idx` parameter always unset, and the same :class:`~.Function` can only be
loaded using the same mode.


Checkpointing with DumbCheckpoint
=================================

.. warning::

   :class:`~.DumbCheckpoint` will be deprecated after 01/01/2023.
   Instead, users are encouraged to use :class:`~.CheckpointFile`,
   which is more robust and scalable.

The support for :class:`~.DumbCheckpoint` is somewhat limited.  One may
only store :class:`~.Function`\s in the checkpoint object.  Moreover,
no remapping of data is performed.  This means that resuming the
checkpoint is only possible on the same number of processes as used to
create the checkpoint file.  Additionally, the *same* ``Mesh``
must be used: that is a ``Mesh`` constructed identically to the
mesh used to generate the saved checkpoint state.


Opening a checkpoint
--------------------

A checkpoint file is created using the :class:`~.DumbCheckpoint`
constructor.  We pass a filename argument, and an access mode.
Available modes are:

:data:`~.FILE_READ`

     Open the checkpoint file for reading.  Raises :exc:`OSError` if
     the file does not already exist.

:data:`~.FILE_CREATE`

     Open the checkpoint file for reading and writing, creating the
     file if it does not exist, and *erasing* any existing contents if
     it does.

:data:`~.FILE_UPDATE`

     Open the checkpoint file for reading and writing, creating it if
     it does not exist, without erasing any existing contents.


For example, to open a checkpoint file for writing solution state,
truncating any existing contents we use:

.. code-block:: python3

   chk = DumbCheckpoint("dump", mode=FILE_CREATE)

note how we only provide the base name of the on-disk file, ``".h5"`` is
appended automatically.

Storing data
------------

Once a checkpoint file is opened, :class:`~.Function` data can be
stored in the checkpoint using :meth:`~.DumbCheckpoint.store`.
A :class:`~.Function` is referenced in the checkpoint file by its
:meth:`~.Function.name`, but this may be overridden by explicitly
passing an optional `name` argument.  For example, to store a
:class:`~.Function` using its default name use:

.. code-block:: python3

   f = Function(V, name="foo")
   chk.store(f)

If instead we want to override the name we use:

.. code-block:: python3

   chk.store(f, name="bar")

.. warning::

   No warning is provided when storing multiple :class:`~.Function`\s
   with the same name, existing values are overwritten.

   Moreover, attempting to store a :class:`~.Function` with a
   different number of degrees of freedom into an existing name will
   cause an error.

Loading data
------------

Once a checkpoint is created, we can use it to load saved state into
:class:`~.Function`\s to resume a simulation.  To load data into a
:class:`~.Function` from a checkpoint, we pass it to
:meth:`~.DumbCheckpoint.load`.  As before, the data is looked up by
its :meth:`~.Function.name`, although once again this may be
overridden by optionally specifying the ``name`` as an argument.

For example, assume we had previously saved a checkpoint containing
two different :class:`~.Function`\s with names ``"A"`` and
``"B"``.  We can load these as follows:

.. code-block:: python3

   chk = DumbCheckpoint("dump.h5", mode=FILE_READ)

   a = Function(V, name="A")

   b = Function(V)

   # Use a.name() to look up value
   chk.load(a)

   # Look up value by explicitly specifying name="B"
   chk.load(b, name="B")

.. note::

   Since Firedrake does not currently support reading data from a
   checkpoint file on a different number of processes from that it was
   written with, whenever a :class:`~.Function` is stored, an
   attribute is set recording the number of processes used.  When
   loading data from the checkpoint, this value is validated against
   the current number of processes and an error is raised if they do
   not match.

Closing a checkpoint
--------------------

The on-disk file inside a checkpoint object is automatically closed
when the checkpoint object is garbage-collected.  However, since this
may not happen at a predictable time, it is possible to manually close
a checkpoint file using :meth:`~.DumbCheckpoint.close`.  To facilitate
this latter usage, checkpoint objects can be used as `context
managers`_ which ensure that the checkpoint file is closed as soon as
the object goes out of scope.  To use this approach, we use the python
``with`` statement:

.. code-block:: python3

   # Normal code here
   with DumbCheckpoint("dump.h5", mode=FILE_UPDATE) as chk:
       # Checkpoint file open for reading and writing
       chk.store(...)
       chk.load(...)

   # Checkpoint file closed, continue with normal code


Writing attributes
------------------

In addition to storing :class:`~.Function` data, it is also possible
to store metadata in :class:`~.DumbCheckpoint` files using HDF5
attributes.  This is carried out using h5py_ to manipulate the file.
The interface allows setting attribute values, reading them, and
checking if a file has a particular attribute:

:meth:`~.DumbCheckpoint.write_attribute`

      Write an attribute, specifying the object path the attribute
      should be set on, the name of the attribute and its value.

:meth:`~.DumbCheckpoint.read_attribute`

      Read an attribute with specified name from at a given object
      path.

:meth:`~.DumbCheckpoint.has_attribute`

      Check if a particular attribute exists.  Does not raise an error
      if the object also does not exist.


Support for multiple timesteps
------------------------------

The checkpoint object supports multiple timesteps in the same on-disk
file.  The primary interface to this is via
:meth:`~.DumbCheckpoint.set_timestep`.  If never called on a
checkpoint file, no timestep support is enabled, and storing a
:class:`~.Function` with the same name as an existing object
overwrites it (data is stored in the HDF5 group ``"/fields"``).  If
one wishes to store multiple timesteps, one should call
:meth:`~.DumbCheckpoint.set_timestep`, providing the timestep value
(and optionally a timestep "index").  Storing a :class:`~.Function`
will now write to the group ``"/fields/IDX"``.  To store the same
function at a different time level, we just call
:meth:`~.DumbCheckpoint.set_timestep` again with a new timestep
value.

Inspecting available time levels
--------------------------------

The stored time levels in the checkpoint object are available as
attributes in the file.  They may be inspected by calling
:meth:`~.DumbCheckpoint.get_timesteps`.  This returns a list of the
timesteps stored in the file, along with the indices they map to.  In
addition, the timestep value is available as an attribute on the
appropriate field group: reading the attribute
``"/fields/IDX/timestep"`` returns the timestep value corresponding to
``IDX``.

Support for multiple on-disk files
----------------------------------

For large simulations, it may not be expedient to store all timesteps
in the same on-disk file.  To this end, the :class:`~.DumbCheckpoint`
object offers the facility to retain the same checkpoint object, but
change the on-disk file used to store the data.  To switch to a new
on-disk file one uses :meth:`~.DumbCheckpoint.new_file`.  There are
two method of choosing the new file name.  If the
:class:`~.DumbCheckpoint` object was created passing
``single_file=False`` then calling :meth:`~.DumbCheckpoint.new_file`
without any additional arguments will use an internal counter to
create file names by appending this counter to the provided base
name.  This selection can be overridden by explicitly passing the
optional ``name`` argument.

As an example, consider the following sequence:

.. code-block:: python3

   with DumbCheckpoint("dump", single_file=False, mode=FILE_CREATE) as chk:
       chk.store(a)
       chk.store(b)
       chk.new_file()
       chk.store(c)
       chk.new_file(name="special")
       chk.store(d)
       chk.new_file()
       chk.store(e)

Will create four on-disk files:

``dump_0.h5``

   Containing ``a`` and ``b``;

``dump_1.h5``

   Containing ``c``;

``special.h5``

   Containing ``d``;

``dump_2.h5``

   Containing ``e``.


Implementation details
======================

The on-disk representation of checkpoints is as HDF5_ files.
Firedrake uses the PETSc_ HDF5 Viewer_ object to write and read state.
As such, writing data is collective across processes.  h5py_ is used
for attribute manipulation.  To this end, h5py_ *must* be linked
against the same version of the HDF5 library that PETSc was built
with.  The ``firedrake-install`` script automates this, however, if
you build PETSc manually, you will need to ensure that h5py_ is linked
correctly following the instructions for custom installation here_.

.. warning::

   Calling :py:meth:`h5py:File.close` on the h5py representation will
   likely result in errors inside PETSc (since it is not aware that
   the file has been closed).  So don't do that!


.. _Paraview: http://www.paraview.org

.. _context managers: https://www.python.org/dev/peps/pep-0343/

.. _HDF5: https://www.hdfgroup.org/HDF5/

.. _PETSc: http://www.mcs.anl.gov/petsc/

.. _Viewer: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Viewer/index.html
.. _h5py: http://www.h5py.org

.. _here: http://docs.h5py.org/en/latest/build.html#custom-installation
