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

    @abc.abstractproperty
    def shape(self):
        """The shape of the per-node tensor.

        For example, a scalar-valued element will have shape == (1,) whilst a 3-vector
        would have shape (3,).
        """
        ...


class RankOneKernelArg(KernelArg, abc.ABC):

    @abc.abstractproperty
    def shape(self):
        ...

    @abc.abstractproperty
    def node_shape(self):
        ...


class RankTwoKernelArg(KernelArg, abc.ABC):

    @abc.abstractproperty
    def rshape(self):
        ...

    @abc.abstractproperty
    def cshape(self):
        ...

    @abc.abstractproperty
    def rnode_shape(self):
        ...

    @abc.abstractproperty
    def cnode_shape(self):
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

    def __init__(self, elem, isreal, dtype, interior_facet=False, interior_facet_horiz=False):
        self._elem = _ElementHandler(elem, interior_facet, interior_facet_horiz, isreal)
        self._dtype = dtype
        self._interior_facet = interior_facet

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._elem.tensor_shape

    @property
    def node_shape(self):
        return self._elem.node_shape

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self._elem.loopy_shape)


class ConstantKernelArg(RankZeroKernelArg):

    def __init__(self, name, shape, dtype):
        self._name = name
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

    def __init__(self, name, elem, isreal, dtype, *, interior_facet=False, interior_facet_horiz=False):
        self._name = name
        self._elem = _ElementHandler(elem, interior_facet, interior_facet_horiz, isreal)
        self._dtype = dtype
        self._interior_facet = interior_facet

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
    def shape(self):
        return self._elem.tensor_shape

    @property
    def node_shape(self):
        return self._elem.node_shape

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self._elem.loopy_shape)


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

    def __init__(self, elem, isreal, dtype, *, interior_facet=False, interior_facet_horiz=False):
        self._elem = _ElementHandler(elem, interior_facet, interior_facet_horiz, isreal)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._elem.tensor_shape

    @property
    def node_shape(self):
        return self._elem.node_shape

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self._elem.loopy_shape)


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

    def __init__(
        self, elem, isreal, dtype, *, interior_facet=False, diagonal=False, interior_facet_horiz=False
    ):
        self._elem = _ElementHandler(elem, interior_facet, interior_facet_horiz, isreal)
        self._dtype = dtype

        self._interior_facet = interior_facet
        self._diagonal = diagonal

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._elem.tensor_shape

    @property
    def node_shape(self):
        return self._elem.node_shape

    @property
    def loopy_arg(self):
        return lp.GlobalArg(self.name, self.dtype, shape=self._elem.loopy_shape)

    # TODO Function please
    def make_gem_exprs(self, multiindices):
        u_shape = np.array([np.prod(self._elem._elem.index_shape, dtype=int)])
        c_shape = tuple(2*u_shape) if self._interior_facet else tuple(u_shape)

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
        return gem.Indexed(gem.reshape(restricted, self._elem._elem.index_shape),
                           tuple(itertools.chain(*multiindices)))


class MatrixOutputKernelArg(RankTwoKernelArg, OutputKernelArg):

    def __init__(self, relem, risreal, celem, cisreal, dtype, *, interior_facet=False, interior_facet_horiz=False):
        self._relem = _ElementHandler(relem, interior_facet, interior_facet_horiz, risreal)
        self._celem = _ElementHandler(celem, interior_facet, interior_facet_horiz, cisreal)
        self._dtype = dtype
        self._interior_facet = interior_facet

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        rshape = self._relem.loopy_shape
        cshape = self._celem.loopy_shape
        return lp.GlobalArg(self.name, self.dtype, shape=(rshape, cshape))

    @property
    def rshape(self):
        return self._relem.tensor_shape

    @property
    def cshape(self):
        return self._celem.tensor_shape

    @property
    def rnode_shape(self):
        return self._relem.node_shape

    @property
    def cnode_shape(self):
        return self._celem.node_shape

    def make_gem_exprs(self, multiindices):
        u_shape = np.array([np.prod(elem._elem.index_shape, dtype=int)
                            for elem in [self._relem, self._celem]])
        c_shape = tuple(2*u_shape) if self._interior_facet else tuple(u_shape)

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
        return gem.Indexed(gem.reshape(restricted, self._relem._elem.index_shape, self._celem._elem.index_shape),
                           tuple(itertools.chain(*multiindices)))


class _ElementHandler:

    def __init__(self, elem, interior_facet=False, interior_facet_horiz=False, is_real_tensor_product=False):
        self._elem = elem
        self._interior_facet = interior_facet
        self._interior_facet_horiz = interior_facet_horiz
        self._is_real_tensor_product = is_real_tensor_product

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
