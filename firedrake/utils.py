# Some generic python utilities not really specific to our work.
from __future__ import absolute_import
from decorator import decorator
from pyop2.utils import cached_property  # noqa: imported from here elsewhere


# after https://micheles.googlecode.com/hg/decorator/documentation.html and
# http://code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
def _memoize(func, obj, *args, **kw):
    try:
        cache = obj.__cache
    except AttributeError:
        cache = obj.__cache = {}
    if kw:
        key = func, args, tuple(kw.iteritems())
    else:
        key = func, args
    if key in cache:
        return cache[key]
    else:
        cache[key] = result = func(obj, *args, **kw)
        return result


def memoize(f):
    return decorator(_memoize, f)


_current_uid = 0


def _new_uid():
    global _current_uid
    _current_uid += 1
    return _current_uid


def _init():
    """Cause :func:`pyop2.init` to be called in case the user has not done it
    for themselves. The result of this is that the user need only call
    :func:`pyop2.init` if she wants to set a non-default option, for example
    to switch the backend or the debug or log level."""
    from pyop2 import op2
    from firedrake.parameters import parameters
    if not op2.initialised():
        op2.init(**parameters["pyop2_options"])


def unique_name(name, nameset):
    """Return name if name is not in nameset, or a deterministic
    uniquified name if name is in nameset. The new name is inserted into
    nameset to prevent further name clashes."""

    if name not in nameset:
        nameset.add(name)
        return name

    idx = 0
    while True:
        newname = "%s_%d" % (name, idx)
        if newname in nameset:
            idx += 1
        else:
            nameset.add(name)
            return newname
