from pyop2 import op2


class Matrix(object):
    """A representation of an assembled bilinear form.

    :arg a: the bilinear form this :class:`Matrix` represents.

    A :class:`pyop2.op2.Mat` will be built from the remaining
    arguments, for valid values, see :class:`pyop2.op2.Mat`.

    .. note::

    This object acts to the right on an assembled :class:`Function`
    and to the left on an assembled co-Function (currently represented
    by a :class:`Function`).

    """

    def __init__(self, a, *args, **kwargs):
        self._a = a
        self._M = op2.Mat(*args, **kwargs)

    @property
    def a(self):
        """The bilinear form this :class:`Matrix` was assembled from"""
        return self._a

    @property
    def M(self):
        """The :class:`pyop2.op2.Mat` representing the assembled form"""
        return self._M
