from abc import ABCMeta, abstractmethod, abstractproperty

from gem.utils import make_proxy_class


class KernelInterface(metaclass=ABCMeta):
    """Abstract interface for accessing the GEM expressions corresponding
    to kernel arguments."""

    @abstractmethod
    def coordinate(self, ufl_domain):
        """A function that maps :class:`ufl.Domain`s to coordinate
        :class:`ufl.Coefficient`s."""

    @abstractmethod
    def coefficient(self, ufl_coefficient, restriction):
        """A function that maps :class:`ufl.Coefficient`s to GEM
        expressions."""

    @abstractmethod
    def constant(self, const):
        """Return the GEM expression corresponding to the constant."""

    @abstractmethod
    def cell_orientation(self, restriction):
        """Cell orientation as a GEM expression."""

    @abstractmethod
    def cell_size(self, restriction):
        """Mesh cell size as a GEM expression.  Shape (nvertex, ) in FIAT vertex ordering."""

    @abstractmethod
    def entity_number(self, restriction):
        """Facet or vertex number as a GEM index."""

    @abstractmethod
    def entity_orientation(self, restriction):
        """Entity orientation as a GEM index."""

    @abstractmethod
    def create_element(self, element, **kwargs):
        """Create a FInAT element (suitable for tabulating with) given
        a UFL element."""

    @abstractproperty
    def unsummed_coefficient_indices(self):
        """A set of indices that coefficient evaluation should not sum over.
        Used for macro-cell integration."""


ProxyKernelInterface = make_proxy_class('ProxyKernelInterface', KernelInterface)
