from __future__ import absolute_import, print_function, division
from six import with_metaclass

from abc import ABCMeta, abstractmethod

from gem.utils import make_proxy_class


class KernelInterface(with_metaclass(ABCMeta)):
    """Abstract interface for accessing the GEM expressions corresponding
    to kernel arguments."""

    @abstractmethod
    def coefficient(self, ufl_coefficient, restriction):
        """A function that maps :class:`ufl.Coefficient`s to GEM
        expressions."""

    @abstractmethod
    def cell_orientation(self, restriction):
        """Cell orientation as a GEM expression."""

    @abstractmethod
    def entity_number(self, restriction):
        """Facet or vertex number as a GEM index."""


ProxyKernelInterface = make_proxy_class('ProxyKernelInterface', KernelInterface)
