"""
Backwards compatibility for some functionality.
"""
import numpy
from distutils.version import StrictVersion


if StrictVersion(numpy.__version__) < StrictVersion("1.10"):
    def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
        """Wrapper around ``numpy.allclose``, which see.

        If ``equal_nan`` is ``True``, consider nan values to be equal
        in comparisons.
        """
        if not equal_nan:
            return numpy.allclose(a, b, rtol=rtol, atol=atol)
        nana = numpy.isnan(a)
        if not nana.any():
            return numpy.allclose(a, b, rtol=rtol, atol=atol)
        nanb = numpy.isnan(b)
        equal_nan = numpy.allclose(nana, nanb)
        return equal_nan and numpy.allclose(a[~nana], b[~nanb],
                                            rtol=rtol, atol=atol)
else:
    allclose = numpy.allclose
