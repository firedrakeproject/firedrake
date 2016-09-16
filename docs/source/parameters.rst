====================================
Configuring meta-data for Parameters
====================================

The :class:`firedrake.parameters.Parameters` class allows users to attach
meta-data to the key and the meta-data could be used to validate the parameter
as well as providing helps for configuration.

Each ``Parameters`` instance could contain a summary, the summary could be
written in RST(reStructuredText) and it will be rendered in the web interface.

The class is subclassed from the python native type ``dict``, but the keys are
subclasses of :class:`firedrake.parameters.TypedKey` so that they could contain
meta-data.

To get the TypedKey
===================

To add meta-data to key, the user should first get a reference to the key. This
could be done by using the method
:func:`firedrake.paramaters.Parameters.get_key`. This returns the
key from the :class:`firedrake.parameters.Parameters` instance and further
meta-data could be edit for the key.


Types
=====

Types are stored as a property named ``type``. Types must be subclasses of the 
abstract class :class:`firedrake.parameters.KeyType`.

There are two methods to implement for this abstract class
``parse(self, val)`` and ``validate(self, val)``.

For most use cases, there are built-in types for integer values ``Inttype``,
float values ``FloatType``, string values ``StrType``, bool values ``BoolType``.
For advanced types, multiple types could be combined using ``OrType`` or form
a list using ``ListType``.

By default, the type of each value is inferred automatically if not explicitly
specified; however, if the user wish to add more information, it is necessary
to set types manually.

Help
====

Help information is stored as a property named ``help``. If no help has been
set, the help inforamtion will be displayed as ``No help available``.

Dependency
==========

Currently, the dependency supported is only limited to bool values of a key in
the same :class:`firedrake.parameters.Parameters` instance. 

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
=============

Each key could be set a visible level. This feature could be used to control
the number of parameters shown to user. The visible level of a key is contained
as a property named ``visible_level``. The visible level should be a
non-negative integer, default to be 0.

After the visible levels have been set, the web interface by default will only
show level 0 keys for configuration. The visible level could be changed via
``Show more options`` and ``Show fewer options`` buttons on the web interface.

Wrapper and Unwrapper
=====================

Wrapper and unwrappers may be useful for pre- or post-processing of the
parameter. They are configurable via ``set_wrapper(callable)`` and 
``set_unwrapper(callable)``

To call a wrapper or unwrapper, simply use ``wrap(value)`` or
``unwrap(value)``

