=================================================
Automated validation of simulation configurations
=================================================

Firedrake provides an extensible mechanism for defining simulation parameters
that supports validation of the provided inputs, as well as automatic creation
of a browser-based :doc:`configuration GUI <configuration>`.
The core object that facilitates this is a :class:`~.Parameters` instance.

The :class:`~firedrake.parameters.Parameters` class allows users to attach
meta-data to the key and the meta-data can be used to validate the parameter
as well as providing helps for configuration.

The parameters instance is a nested dictionary of configuration parameters,
to which one can attach metadata for both validation and presentation of
additional information in the GUI.

We create a :class:`~.Parameters` instance by providing a name for the
parameters and an optional summary (in reStructuredText format).
Any further arguments are treated as for the :class:`~.dict` constructor.

Although this object behaves like a normal python dictionary, its keys instead
contain metadata relevant for validation and presentation. This is achieved
through the :class:`~.TypedKey` class.

Creating a TypedKey
===================

By default, the :class:`~firedrake.parameters.Parameters` class will infer the
basic types of data and create :class:`~firedrake.parameters.TypedKey` for the
parameter accordingly with first inserting into the dict. This is done by
:func:`~firedrake.parameters.KeyType.get_type`.

However, if the user wishes to add a custom-configured TypedKey, the following
syntax could be used at the time of first insertion.

.. code-block:: python

    parameters[TypedKey("foo", type=IntType(), help="A sample key",
    visible_level=1, depends="baz")] = "bar"

Note that if the user wishes to use pre-defined ``Keytypes``, it is necessary
to import the types manually. For example: ``from firedrake.parameters import IntType``

See also the constructor of :class:`~firedrake.parameters.TypedKey`.

Getting the TypedKey
====================

To add meta-data to key, the user should first get a reference to the key. This
can be done by using the method
:func:`~firedrake.parameters.Parameters.get_key`. This returns the
key from the :class:`~firedrake.parameters.Parameters` instance and further
meta-data can be edit for the key.

TypedKeys
=========

Types
-----

Types are stored as a property named ``type``. Types must be subclasses of the 
abstract class :class:`~firedrake.parameters.KeyType`.

There are two methods to implement for this abstract class
:meth:`~firedrake.parameters.KeyType.parse` and
:meth:`~firedrake.parameters.KeyType.validate`.

For most use cases, there are built-in types for integer values ``Inttype``,
float values ``FloatType``, string values ``StrType``, bool values ``BoolType``.
For advanced types, multiple types can be combined using ``OrType``. Lists can
also be formed using ``ListType``.

By default, the type of each value is inferred automatically if not explicitly
specified; however, if the user wish to add more information, it is necessary
to set types manually.

Help
----

Help information is stored as a property named ``help``. If no help has been
set, the help inforamtion will be displayed as ``No help available``.

Dependency
----------

Currently, the dependency supported is only limited to bool values of a key in
the same :class:`~firedrake.parameters.Parameters` instance.

To specify a dependency, simply set the property ``depends`` of the key to be
name of the key the key is dependent on.

For example, if ``param`` is an instance of ``Parameters`` and ``foo`` is the
key for a bool value, ``bar`` is a key dependent on ``foo``.

.. code-block:: python

    param.get_key("bar").depends = "foo"

will set the dependency.

With the dependency set, the parameters for dependent parameters will not be
shown unless the parameter being depended on is set to be true.

Visible Level
-------------

Each key can be set a visible level. This feature can be used to control
the number of parameters shown to user. The visible level of a key is contained
as a property named ``visible_level``. The visible level should be a
non-negative integer, default to be 0.

After the visible levels have been set, the web interface by default will only
show level 0 keys for configuration. The visible level can be changed via
``Show more options`` and ``Show fewer options`` buttons on the web interface.

Wrapper and Unwrapper
---------------------

Wrapper and unwrappers may be useful for pre- or post-processing of the
parameter. They are configurable via
:meth:`~firedrake.parameters.TypedKey.set_wrapper` and
:meth:`~firedrake.parameters.TypedKey.set_unwrapper`

To call a wrapper or unwrapper, simply use
:meth:`~firedrake.parameters.TypedKey.wrap` or
:meth:`~firedrake.parameters.TypedKey.unwrap`

