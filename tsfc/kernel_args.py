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

    def __init__(self, elem, fs_id, dtype, interior_facet=False):
        self._elem = _ElementHandler(elem)
        self._fs_id = fs_id
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
        shape = self._elem.node_shape
        return 2*shape if self._interior_facet else shape

    @property
    def loopy_arg(self):
        shape = np.prod([self.node_shape, *self.shape], dtype=int)
        return lp.GlobalArg(self.name, self.dtype, shape=shape)

    @property
    def function_space_id(self):
        return self._fs_id


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

    def __init__(self, name, elem, fs_id, dtype, *, interior_facet=False):
        self._name = name
        self._elem = _ElementHandler(elem)
        self._fs_id = fs_id
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
        shape = self._elem.node_shape
        return 2*shape if self._interior_facet else shape

    @property
    def loopy_arg(self):
        shape = np.prod([self.node_shape, *self.shape], dtype=int)
        return lp.GlobalArg(self.name, self.dtype, shape=shape)

    @property
    def function_space_id(self):
        return self._fs_id


class CellOrientationsKernelArg(RankOneKernelArg):

    name = "cell_orientations"
    intent = Intent.IN
    dtype = np.int32

    node_shape = 1

    def __init__(self, interior_facet=False):
        self._interior_facet = interior_facet

    @property
    def loopy_arg(self):
        shape = np.prod([self.node_shape, *self.shape], dtype=int)
        return lp.GlobalArg(self.name, self.dtype, shape=shape)

    @property
    def shape(self):
        return (2,) if self._interior_facet else (1,)


class CellSizesKernelArg(RankOneKernelArg):

    name = "cell_sizes"
    intent = Intent.IN

    def __init__(self, elem, fs_id, dtype, *, interior_facet=False):
        self._elem = _ElementHandler(elem)
        self._fs_id = fs_id
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
        shape = self._elem.node_shape
        return 2*shape if self._interior_facet else shape

    @property
    def loopy_arg(self):
        shape = np.prod([self.node_shape, *self.shape], dtype=int)
        return lp.GlobalArg(self.name, self.dtype, shape=shape)

    @property
    def function_space_id(self):
        return self._fs_id


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


# TODO Find a case where we actually need to use this.
# class TabulationKernelArg(KernelArg):

#     rank = 1
#     intent = Intent.IN

#     def __init__(self, name, shape, dtype, interior_facet=False):
#         self.name = name
#         self.shape = shape
#         self.dtype = dtype
#         self.interior_facet = interior_facet

#     @property
#     def loopy_arg(self):
#         raise NotImplementedError

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
        self, elem, fs_id, dtype, *, interior_facet=False, diagonal=False
    ):
        self._elem = _ElementHandler(elem)
        self._fs_id = fs_id
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
        shape = self._elem.node_shape
        return 2*shape if self._interior_facet else shape

    @property
    def loopy_arg(self):
        shape = np.prod([self.node_shape, *self.shape], dtype=int)
        return lp.GlobalArg(self.name, self.dtype, shape=shape)

    @property
    def function_space_id(self):
        return self._fs_id

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

    def __init__(self, relem, celem, rfs_id, cfs_id, dtype, *, interior_facet=False):
        self._relem = _ElementHandler(relem)
        self._celem = _ElementHandler(celem)
        self._rfs_id = rfs_id
        self._cfs_id = cfs_id
        self._dtype = dtype
        self._interior_facet = interior_facet

    @property
    def dtype(self):
        return self._dtype

    @property
    def loopy_arg(self):
        rshape = np.prod([self.rnode_shape, *self.rshape], dtype=int)
        cshape = np.prod([self.cnode_shape, *self.cshape], dtype=int)
        return lp.GlobalArg(self.name, self.dtype, shape=(rshape, cshape))

    @property
    def rshape(self):
        return self._relem.tensor_shape

    @property
    def cshape(self):
        return self._celem.tensor_shape

    @property
    def rnode_shape(self):
        shape = self._relem.node_shape
        return 2*shape if self._interior_facet else shape

    @property
    def cnode_shape(self):
        shape = self._celem.node_shape
        return 2*shape if self._interior_facet else shape

    @property
    def rfunction_space_id(self):
        return self._rfs_id

    @property
    def cfunction_space_id(self):
        return self._cfs_id

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

    def __init__(self, elem):
        self._elem = elem

    @property
    def node_shape(self):
        if self._is_tensor_element:
            shape = self._elem.index_shape[:-len(self.tensor_shape)]
        else:
            shape = self._elem.index_shape
        return np.prod(shape, dtype=int)

    @property
    def tensor_shape(self):
        return self._elem._shape if self._is_tensor_element else (1,)

    @property
    def _is_tensor_element(self):
        return isinstance(self._elem, finat.TensorFiniteElement)
