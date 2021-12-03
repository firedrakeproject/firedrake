import abc
import enum
import itertools

import finat
import gem
from gem.optimise import remove_componenttensors as prune
import loopy as lp
import numpy as np


class Intent(enum.IntEnum):
    IN = enum.auto()
    OUT = enum.auto()


class KernelArg(abc.ABC):
    """Class encapsulating information about kernel arguments."""

    @abc.abstractproperty
    def name(self):
        ...

    @abc.abstractproperty
    def dtype(self):
        ...

    @abc.abstractproperty
    def intent(self):
        ...

    @abc.abstractproperty
    def loopy_arg(self):
        ...


class RankZeroKernelArg(KernelArg, abc.ABC):
    ...


class RankOneKernelArg(KernelArg, abc.ABC):
    ...


class RankTwoKernelArg(KernelArg, abc.ABC):
    ...


class DualEvalOutputKernelArg(KernelArg):

    name = "A"
    intent = Intent.OUT

    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self._shape)


class CoordinatesKernelArg(RankOneKernelArg):

    name = "coords"
    intent = Intent.IN

    def __init__(self, size, dtype, interior_facet=False):
        self._size = size
        self._dtype = dtype
        self._interior_facet = interior_facet

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=(self._size,))


class ConstantKernelArg(RankZeroKernelArg):

    def __init__(self, name, number, shape, dtype):
        self._name = name
        self.number = number
        self._shape = shape
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def intent(self):
        return Intent.IN

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.shape)


class CoefficientKernelArg(RankOneKernelArg):

    def __init__(self, name, number, size, dtype):
        self._name = name
        self.number = number
        self._size = size
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def intent(self):
        return Intent.IN

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=(self._size,))


class CellOrientationsKernelArg(RankOneKernelArg):

    name = "cell_orientations"
    intent = Intent.IN

    def __init__(self, shape=(1,), dtype=np.int32, interior_facet=False, interior_facet_horiz=False):
        self._shape = shape
        self._dtype = dtype
        self._interior_facet = interior_facet
        assert not interior_facet_horiz

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        shape = np.prod([self.node_shape, *self.shape], dtype=int)
        return lp.GlobalArg(self.name, self.dtype, shape=shape)

    @property
    def shape(self):
        return self._shape

    @property
    def node_shape(self):
        return 2 if self._interior_facet else 1


class CellSizesKernelArg(RankOneKernelArg):

    name = "cell_sizes"
    intent = Intent.IN

    def __init__(self, size, dtype):
        self._size = size
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=(self._size,))


class FacetKernelArg(RankOneKernelArg, abc.ABC):

    name = "facet"
    intent = Intent.IN
    dtype = np.uint32

    node_shape = None  # Must be None because of direct addressing - this is obscure

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.shape)


class ExteriorFacetKernelArg(FacetKernelArg):

    shape = (1,)


class InteriorFacetKernelArg(FacetKernelArg):

    shape = (2,)


class TabulationKernelArg(RankOneKernelArg):

    intent = Intent.IN
    shape = (1,)

    def __init__(self, name, shape, dtype):
        self._name = name
        self._shape = shape
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def node_shape(self):
        return np.prod(self._shape, dtype=int)

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self._name, self._dtype, shape=self._shape)


class OutputKernelArg(KernelArg, abc.ABC):

    name = "A"
    intent = Intent.OUT


class ScalarOutputKernelArg(RankZeroKernelArg, OutputKernelArg):

    def __init__(self, dtype):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.shape, is_output=True)

    @property
    def shape(self):
        return (1,)

    def make_gem_exprs(self, multiindices):
        assert len(multiindices) == 0
        return [gem.Indexed(gem.Variable(self.name, self.shape), (0,))]


class VectorOutputKernelArg(RankOneKernelArg, OutputKernelArg):

    def __init__(self, index_shape, dtype, *, interior_facet=False, diagonal=False):
        self.index_shape, = index_shape
        self._dtype = dtype

        self._interior_facet = interior_facet
        self._diagonal = diagonal

    @property
    def dtype(self):
        return self._dtype

    @property
    def ushape(self):
        return np.array([np.prod(self.index_shape, dtype=int)])

    @property
    def cshape(self):
        return tuple(2*self.ushape) if self._interior_facet else tuple(self.ushape)

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.cshape)

    # TODO Function please
    def make_gem_exprs(self, multiindices):
        u_shape = self.ushape
        c_shape = self.cshape

        if self._diagonal:
            multiindices = multiindices[:1]

        if self._interior_facet:
            slicez = [
                [slice(r*s, (r + 1)*s) for r, s in zip(restrictions, u_shape)]
                for restrictions in [(0,), (1,)]
            ]
        else:
            slicez = [[slice(s) for s in u_shape]]

        var = gem.Variable(self.name, c_shape)
        exprs = [self._make_expression(gem.view(var, *slices), multiindices) for slices in slicez]
        return prune(exprs)

    # TODO More descriptive name
    def _make_expression(self, restricted, multiindices):
        return gem.Indexed(gem.reshape(restricted, self.index_shape),
                           tuple(itertools.chain(*multiindices)))


class MatrixOutputKernelArg(RankTwoKernelArg, OutputKernelArg):

    def __init__(self, shapes, dtype, *, interior_facet=False):
        self._rshape, self._cshape = shapes
        self._dtype = dtype
        self._interior_facet = interior_facet

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self.c_shape)

    @property
    def u_shape(self):
        return np.array([np.prod(s, dtype=int)
                            for s in [self._rshape, self._cshape]])

    @property
    def c_shape(self):
        return tuple(2*self.u_shape) if self._interior_facet else tuple(self.u_shape)

    def make_gem_exprs(self, multiindices):
        u_shape = self.u_shape
        c_shape = self.c_shape

        if self._interior_facet:
            slicez = [
                [slice(r*s, (r + 1)*s) for r, s in zip(restrictions, u_shape)]
                for restrictions in itertools.product((0, 1), repeat=2)
            ]
        else:
            slicez = [[slice(s) for s in u_shape]]

        var = gem.Variable(self.name, c_shape)
        exprs = [self._make_expression(gem.view(var, *slices), multiindices) for slices in slicez]
        return prune(exprs)

    # TODO More descriptive name
    def _make_expression(self, restricted, multiindices):
        return gem.Indexed(gem.reshape(restricted, self._rshape, self._cshape),
                           tuple(itertools.chain(*multiindices)))


class _ElementHandler:

    def __init__(self, elem, interior_facet=False, interior_facet_horiz=False):
        self._elem = elem
        self._interior_facet = interior_facet
        self._interior_facet_horiz = interior_facet_horiz

    @property
    def node_shape(self):
        if self._is_tensor_element:
            shape = self._elem.index_shape[:-len(self.tensor_shape)]
        else:
            shape = self._elem.index_shape

        shape = np.prod(shape, dtype=int)

        if self._interior_facet and not self._interior_facet_horiz:
            return 2 * shape
        else:
            return shape

    @property
    def tensor_shape(self):
        return self._elem._shape if self._is_tensor_element else (1,)

    @property
    def loopy_shape(self):
        if self._is_tensor_element:
            shape = self._elem.index_shape[:-len(self.tensor_shape)]
        else:
            shape = self._elem.index_shape

        shape = np.prod(shape, dtype=int)

        # We have to treat facets carefully as the local kernel needs double size but
        # external map does not.
        if self._interior_facet:
            shape *= 2
        return np.prod([shape, *self.tensor_shape], dtype=int)

    @property
    def is_mixed(self):
        return isinstance(self._elem, finat.EnrichedElement) and self._elem.is_mixed

    def split(self):
        if not self.is_mixed:
            raise ValueError("Cannot split a non-mixed element")

        return tuple([type(self)(subelem.element) for subelem in self._elem.elements])

    @property
    def _is_tensor_element(self):
        return isinstance(self._elem, finat.TensorFiniteElement)
