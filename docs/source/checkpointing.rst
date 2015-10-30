.. only:: html

   .. contents::

=====================
 Checkpointing state
=====================

In addition to the ability to write field data to vtu files, suitable
for visualisation in Paraview_, Firedrake has some support for
checkpointing state to disk.  This enables pausing, and subsequently
resuming, a simulation at a later time.

Restrictions on checkpointing
=============================

The current support for checkpointing is somewhat limited.  One may
only store :class:`~.Function`\s in the checkpoint object.  Moreover,
no remapping of data is performed.  This means that resuming the
checkpoint is only possible on the same number of processes as used to
create the checkpoint file.  Additionally, the *same* :class:`~.Mesh`
must be used: that is a :class:`~.Mesh` constructed identically to the
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

.. code-block:: python

   chk = DumbCheckpoint("dump.h5", mode=FILE_WRITE)


Storing data
------------

Once a checkpoint file is opened, :class:`~.Function` data can be
stored in the checkpoint using :meth:`~.DumbCheckpoint.store`.
A :class:`~.Function` is referenced in the checkpoint file by its
:meth:`~.Function.name`, but this may be overridden by explicitly
passing an optional `name` argument.  For example, to store a
:class:`~.Function` using its default name use:

.. code-block:: python

   f = Function(V, name="foo")
   chk.store(f)

If instead we want to override the name we use:

.. code-block:: python

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

.. code-block:: python

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

.. code-block:: python

   # Normal code here
   with DumbCheckpoint("dump.h5", mode=FILE_UPDATE) as chk:
       # Checkpoint file open for reading and writing
       chk.store(...)
       chk.load(...)

   # Checkpoint file closed, continue with normal code


Implementation details
======================

The on-disk representation of checkpoints is as HDF5_ files.
Firedrake uses the PETSc_ HDF5 Viewer_ object to write and read state.
As such, writing data is collective across processes.  Checkpoint
files can be inspected using the Python h5py_ package.  If h5py_ is
installed and was linked against the same version of the HDF5 library
that PETSc was built with, it is possible to obtain a h5py
:py:class:`h5py:File` object corresponding to an open checkpoint file
by using :meth:`~.DumbCheckpoint.as_h5py`.  An error is raised if this
conversion fails.

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
