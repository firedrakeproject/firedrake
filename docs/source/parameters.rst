===================================================
Parameters for Firedrake and Firedrake applications
===================================================

Firedrake provides an extensible mechanism for defining parameters
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
:func:`~firedrake.parameter_types.KeyType.get_type`.

However, if the user wishes to add a custom-configured TypedKey, the following
syntax could be used at the time of first insertion.

.. code-block:: python

    parameters[TypedKey("foo", val_type=IntType(), help="A sample key",
        visible_level=1, depends="baz")] = "bar"

Note that if the user wishes to use pre-defined :class:`Keytypes
<firedrake.parameter_types.KeyType>`,
it is necessary to import the types manually.   i.e.:
``from firedrake.parameter_types import *``

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
abstract class :class:`~firedrake.parameter_types.KeyType`.

There are two methods to implement for this abstract class
:meth:`~firedrake.parameter_types.KeyType.parse` and
:meth:`~firedrake.parameter_types.KeyType.validate`.

For most use cases, there are built-in types for integer values
:class:`~firedrake.parameter_types.IntType`, float values
:class:`~firedrake.parameter_types.FloatType`, string values
:class:`~firedrake.parameter_types.StrType`, bool values
:class:`~firedrake.parameter_types.BoolType`.
For advanced types, multiple types can be combined using
:class:`~firedrake.parameter_types.OrType`. Lists can
also be formed using :class:`~firedrake.parameter_types.ListType`.

By default, the type of each value is inferred automatically if not explicitly
specified; however, if the user wish to add more information, it is necessary
to set types manually.

Help
----

Help information is stored as a property named ``help``. If no help has been
set, the help inforamtion will be displayed as ``No help available``.

Dependency
----------

Currently,  dependency support is limited to bool values of a key in
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

Visibility Level
-------------

The visibility level of each key can be set. This feature can be used to control
the number of parameters shown to user. The visibility level of a key is contained
as a property named ``visibility_level``. The visibility level should be a
non-negative integer, default to be 0.

After the visibility levels have been set, the web interface by default will only
show level 0 keys for configuration. The visibility level can be changed via
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

For Developers: Branding the web interface
==========================================

The HTML page header and footer is user-configurable. By default, it contains
the Firedrake logo and copyright information for Firedrake.

To add a custom HTML page header and/or footer, simply set the attributes
``html_header`` and/or ``html_footer`` of the :class:`~firedrake.parameters.Parameters`
instance to strings containing the HTML code of the page header
or footer respectively.

Note that in order to include static files (stylesheets, images, etc.), you'll
have to place them in a folder and use the following stytax to reference the
static file of name ``FILENAME`` you need.

.. code-block:: html

    {{ url_for('static', filename="FILENAME")  }}

Then, please set the attribute ``static_files_folder`` of the
:class:`~firedrake.parameters.Parameters` instance to be the folder storing the
static files, as a relative path to the module containing the
:class:`~firedrake.parameters.Parameters` instance.

Recommended header and footer design
------------------------------------

It is recommended to enclose the header in a ``div`` of ``row`` class, i.e.

.. code-block:: html
    <div class="row" style="text-align: center;">
        INSERT YOUR HEADER HERE.
    </div>

It is recommended to enclose the footer in a ``footer`` of ``footer`` class,
i.e.

.. code-block:: html
    <footer class="footer">
        INSERT YOUR FOOTER HERE.
    </footer>
