# Utility module that imports and initialises petsc4py
import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import itertools
from contextlib import contextmanager


__all__ = ("PETSc", "OptionsManager")


def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    from firedrake.logging import warning
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in parameters.items():
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            warning("Ignoring duplicate option: %s (existing value %s, new value %s)",
                    option, new[option], value)
        new[option] = value
    return new


class OptionsManager(object):

    # What appeared on the commandline, we should never clear these.
    # They will override options passed in as a dict if an
    # options_prefix was supplied.
    commandline_options = frozenset(PETSc.Options().getAll())

    options_object = PETSc.Options()

    count = itertools.count()

    """Mixin class that helps with managing setting petsc options.

    :arg parameters: The dictionary of parameters to use.
    :arg options_prefix: The prefix to look up items in the global
        options database (may be ``None``, in which case only entries
        from ``parameters`` will be considered.  If no trailing
        underscore is provided, one is appended.  Hence ``foo_`` and
        ``foo`` are treated equivalently.  As an exception, if the
        prefix is the empty string, no underscore is appended.

    To use this, you must call its constructor to with the parameters
    you want in the options database.

    You then call :meth:`set_from_options`, passing the PETSc object
    you'd like to call ``setFromOptions`` on.  Note that this will
    actually only call ``setFromOptions`` the first time (so really
    this parameters object is a once-per-PETSc-object thing).

    So that the runtime monitors which look in the options database
    actually see options, you need to ensure that the options database
    is populated at the time of a ``SNESSolve`` or ``KSPSolve`` call.
    Do that using the :meth:`inserted_options` context manager.

    .. code-block:: python

       with self.inserted_options():
           self.snes.solve(...)

    This ensures that the options database has the relevant entries
    for the duration of the ``with`` block, before removing them
    afterwards.  This is a much more robust way of dealing with the
    fixed-size options database than trying to clear it out using
    destructors.

    This object can also be used only to manage insertion and deletion
    into the PETSc options database, by using the context manager.
    """
    def __init__(self, parameters, options_prefix):
        super().__init__()
        if parameters is None:
            parameters = {}
        else:
            # Convert nested dicts
            parameters = flatten_parameters(parameters)
        if options_prefix is None:
            self.options_prefix = "firedrake_%d_" % next(self.count)
            self.parameters = parameters
            self.to_delete = set(parameters)
        else:
            if len(options_prefix) and not options_prefix.endswith("_"):
                options_prefix += "_"
            self.options_prefix = options_prefix
            # Remove those options from the dict that were passed on
            # the commandline.
            self.parameters = {k: v for k, v in parameters.items()
                               if options_prefix + k not in self.commandline_options}
            self.to_delete = set(self.parameters)
            # Now update parameters from options, so that they're
            # available to solver setup (for, e.g., matrix-free).
            # Can't ask for the prefixed guy in the options object,
            # since that does not DTRT for flag options.
            for k, v in self.options_object.getAll().items():
                if k.startswith(self.options_prefix):
                    self.parameters[k[len(self.options_prefix):]] = v
        self._setfromoptions = False

    def set_default_parameter(self, key, val):
        """Set a default parameter value.

        :arg key: The parameter name
        :arg val: The parameter value.

        Ensures that the right thing happens cleaning up the options
        database.
        """
        k = self.options_prefix + key
        if k not in self.options_object and key not in self.parameters:
            self.parameters[key] = val
            self.to_delete.add(key)

    def set_from_options(self, petsc_obj):
        """Set up petsc_obj from the options database.

        :arg petsc_obj: The PETSc object to call setFromOptions on.

        Matt says: "Only ever call setFromOptions once".  This
        function ensures we do so.
        """
        if not self._setfromoptions:
            with self.inserted_options():
                petsc_obj.setOptionsPrefix(self.options_prefix)
                # Call setfromoptions inserting appropriate options into
                # the options database.
                petsc_obj.setFromOptions()
                self._setfromoptions = True

    @contextmanager
    def inserted_options(self):
        """Context manager inside which the petsc options database
    contains the parameters from this object."""
        from firedrake.logging import warning
        try:
            for k, v in self.parameters.items():
                key = self.options_prefix + k
                if type(v) is bool:
                    if "monitor" in k or "view" in k:
                        warning("""Firedrake will stop translating True/False options soon.\n"""
                                """This is to allow controlling boolean options with a default value of True.\n"""
                                """To obtain the old behaviour you should translate:\n"""
                                """  {%r: True} => {"option": None}\n"""
                                """and\n"""
                                """  {%r: False} => {}\n""" % (k, k))
                    if v:
                        self.options_object[key] = None
                else:
                    self.options_object[key] = v
            yield
        finally:
            for k in self.to_delete:
                del self.options_object[self.options_prefix + k]
