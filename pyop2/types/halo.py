import abc


class Halo(abc.ABC):

    """A description of a halo associated with a :class:`pyop2.types.set.Set`.

    The halo object describes which :class:`pyop2.types.set.Set` elements are sent
    where, and which :class:`pyop2.types.set.Set` elements are received from where.
    """

    @abc.abstractproperty
    def comm(self):
        """The MPI communicator for this halo."""
        pass

    @abc.abstractproperty
    def local_to_global_numbering(self):
        """The mapping from process-local to process-global numbers for this halo."""
        pass

    @abc.abstractmethod
    def global_to_local_begin(self, dat, insert_mode):
        """Begin an exchange from global (assembled) to local (ghosted) representation.

        :arg dat: The :class:`pyop2.types.dat.Dat` to exchange.
        :arg insert_mode: The insertion mode.
        """
        pass

    @abc.abstractmethod
    def global_to_local_end(self, dat, insert_mode):
        """Finish an exchange from global (assembled) to local (ghosted) representation.

        :arg dat: The :class:`pyop2.types.dat.Dat` to exchange.
        :arg insert_mode: The insertion mode.
        """
        pass

    @abc.abstractmethod
    def local_to_global_begin(self, dat, insert_mode):
        """Begin an exchange from local (ghosted) to global (assembled) representation.

        :arg dat: The :class:`pyop2.types.dat.Dat` to exchange.
        :arg insert_mode: The insertion mode.
        """
        pass

    @abc.abstractmethod
    def local_to_global_end(self, dat, insert_mode):
        """Finish an exchange from local (ghosted) to global (assembled) representation.

        :arg dat: The :class:`pyop2.types.dat.Dat` to exchange.
        :arg insert_mode: The insertion mode.
        """
        pass
