import abc

import numpy as np

from pyop2 import (
    datatypes as dtypes,
    mpi,
    utils
)
from pyop2.types.access import Access


class DataCarrier(abc.ABC):

    """Abstract base class for OP2 data.

    Actual objects will be :class:`DataCarrier` objects of rank 0
    (:class:`Global`), rank 1 (:class:`Dat`), or rank 2
    (:class:`Mat`)"""

    @utils.cached_property
    def dtype(self):
        """The Python type of the data."""
        return self._data.dtype

    @utils.cached_property
    def ctype(self):
        """The c type of the data."""
        return dtypes.as_cstr(self.dtype)

    @utils.cached_property
    def name(self):
        """User-defined label."""
        return self._name

    @utils.cached_property
    def dim(self):
        """The shape tuple of the values for each element of the object."""
        return self._dim

    @utils.cached_property
    def cdim(self):
        """The scalar number of values for each member of the object. This is
        the product of the dim tuple."""
        return self._cdim

    def increment_dat_version(self):
        pass


class EmptyDataMixin(abc.ABC):
    """A mixin for :class:`Dat` and :class:`Global` objects that takes
    care of allocating data on demand if the user has passed nothing
    in.

    Accessing the :attr:`_data` property allocates a zeroed data array
    if it does not already exist.
    """
    def __init__(self, data, dtype, shape):
        if data is None:
            self._dtype = np.dtype(dtype if dtype is not None else dtypes.ScalarType)
        else:
            self._numpy_data = utils.verify_reshape(data, dtype, shape, allow_none=True)
            self._dtype = self._data.dtype

    @utils.cached_property
    def _data(self):
        """Return the user-provided data buffer, or a zeroed buffer of
        the correct size if none was provided."""
        if not self._is_allocated:
            self._numpy_data = np.zeros(self.shape, dtype=self._dtype)
        return self._numpy_data

    @property
    def _is_allocated(self):
        """Return True if the data buffer has been allocated."""
        return hasattr(self, '_numpy_data')


class VecAccessMixin(abc.ABC):

    def __init__(self, petsc_counter=None):
        if petsc_counter:
            # Use lambda since `_vec` allocates the data buffer
            # -> Dat/Global should not allocate storage until accessed
            self._dat_version = lambda: self._vec.stateGet()
            self.increment_dat_version = lambda: self._vec.stateIncrease()
        else:
            # No associated PETSc Vec if incompatible type:
            # -> Equip Dat/Global with their own counter.
            self._version = 0
            self._dat_version = lambda: self._version

            def _inc():
                self._version += 1
            self.increment_dat_version = _inc

    @property
    def dat_version(self):
        return self._dat_version()

    @abc.abstractmethod
    def vec_context(self, access):
        pass

    @abc.abstractproperty
    def _vec(self):
        pass

    @property
    @mpi.collective
    def vec(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're allowed to modify the data you get back from this view."""
        return self.vec_context(access=Access.RW)

    @property
    @mpi.collective
    def vec_wo(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're allowed to modify the data you get back from this view,
        but you cannot read from it."""
        return self.vec_context(access=Access.WRITE)

    @property
    @mpi.collective
    def vec_ro(self):
        """Context manager for a PETSc Vec appropriate for this Dat.

        You're not allowed to modify the data you get back from this view."""
        return self.vec_context(access=Access.READ)
