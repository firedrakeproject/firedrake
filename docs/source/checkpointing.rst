.. only:: html

   .. contents::

=====================
 Checkpointing state
=====================

In addition to the ability to :doc:`write field data to vtu files
<visualisation>`, suitable for visualisation in Paraview_, Firedrake
has some support for checkpointing state to disk.  This enables
pausing, and subsequently resuming, a simulation at a later time.

Restrictions on checkpointing
=============================

The current support for checkpointing is somewhat limited.  One may
only store :class:`~.Function`\s in the checkpoint object.  Moreover,
no remapping of data is performed.  This means that resuming the
checkpoint is only possible on the same number of processes as used to
create the checkpoint file.  Additionally, the *same* ``Mesh``
must be used: that is a ``Mesh`` constructed identically to the
mesh used to generate the saved checkpoint state.

.. note::

   In the future, this restriction will be lifted and Firedrake will
   be able to store and resume checkpoints on different number of
   processes, as long as the mesh topology remains unchanged.


Creating and using checkpoint files
===================================

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
