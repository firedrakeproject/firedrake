# Some generic python utilities not really specific to our work.

from decorator import decorator


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


# from http://www.toofishes.net/blog/python-cached-property-decorator/
class cached_property(object):

    '''A read-only @property that is only evaluated once. The value is cached
    on the object itself rather than the function or class; this should prevent
    memory leakage.'''

    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        obj.__dict__[self.__name__] = result = self.fget(obj)
        return result
