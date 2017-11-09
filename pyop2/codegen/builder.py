from abc import ABCMeta, abstractmethod
from collections import OrderedDict, namedtuple
import numpy

from pyop2.codegen.representation import (Index, FixedIndex, RuntimeIndex,
                                          MultiIndex, Extent, Indexed,
                                          BitShift, BitwiseNot, BitwiseAnd,
                                          Conditional, Comparison, DummyInstruction,
                                          LogicalNot, LogicalAnd, LogicalOr,
                                          Argument, Literal, NamedLiteral,
                                          Materialise, Accumulate, FunctionCall, When,
                                          Symbol, Zero, Sum, Product, view)
from pyop2.codegen.representation import (PackInst, UnpackInst, KernelInst)

from pyop2.utils import cached_property
from pyop2.datatypes import IntType
from pyop2.op2 import ON_BOTTOM, ON_TOP, ON_INTERIOR_FACETS, ALL, Subset, DecoratedMap
from pyop2.op2 import READ, INC, WRITE
from loopy.types import OpaqueType
from functools import reduce
import itertools


class PetscMat(OpaqueType):

    def __init__(self):
        super(PetscMat, self).__init__(name="Mat")


class SparseArray(namedtuple("SparseArray", ("values", "dof", "offset"))):

    @cached_property
    def nrows(self):
        extent, = self.offset.shape
        return extent


class Map(object):

    __slots__ = ("values", "offset", "boundary_masks", "interior_horizontal",
                 "variable", "vector_bc", "implicit_bcs", "layer_bounds",
                 "variable_entity_masks", "prefetch")

    def __init__(self, map_, interior_horizontal, layer_bounds, variable_entity_masks,
                 values=None, offset=None, boundary_masks=None):
        self.variable = map_.iterset._extruded and not map_.iterset.constant_layers
        self.vector_bc = map_.vector_index
        self.implicit_bcs = map_.implicit_bcs
        self.variable_entity_masks = variable_entity_masks
        self.layer_bounds = layer_bounds
        self.interior_horizontal = interior_horizontal
        self.prefetch = {}
        if values is not None:
            self.values = values
            if map_.offset is not None:
                assert offset is not None
            self.offset = offset
            if map_.boundary_masks is not None:
                assert boundary_masks is not None
            self.boundary_masks = boundary_masks
            return

        offset = map_.offset
        iterset = map_.iterset
        boundary_masks = map_.boundary_masks
        shape = (None, ) + map_.shape[1:]
        values = Argument(shape, dtype=map_.dtype, pfx="map")
        if offset is not None:
            offset = NamedLiteral(offset, name=values.name + "_offset")

        if boundary_masks is not None:
            v = {}
            for method, (section, indices, (*_, bottom, top)) in boundary_masks.items():
                if iterset.constant_layers:
                    vals = []
                    for location, p in [("bottom", bottom),
                                        ("top", top)]:
                        dof = section.getDof(p)
                        off = section.getOffset(p)
                        name = values.name + ("_%s_%s_indices" % (method, location))
                        vals.append(NamedLiteral(indices[off:off+dof], name))
                    v[method] = tuple(vals)
                else:
                    name = values.name + ("_%s" % method)
                    indices = NamedLiteral(indices, name + "_indices")
                    chart = section.getChart()
                    off = numpy.asarray(list(section.getOffset(p) for p in range(*chart)), dtype=IntType)
                    dof = numpy.asarray(list(section.getDof(p) for p in range(*chart)),
                                        dtype=IntType)
                    off = NamedLiteral(off, name + "_offset")
                    dof = NamedLiteral(dof, name + "_dof")
                    v[method] = SparseArray(indices, dof, off)
            boundary_masks = v
        self.values = values
        self.offset = offset
        self.boundary_masks = boundary_masks

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def indexed(self, multiindex, layer=None):
        n, i, f = multiindex
        if layer is not None and self.offset is not None:
            # For extruded mesh, prefetch the indirections for each map, so that they don't
            # need to be recomputed. Different f values need to be treated separately.
            key = f.extent
            if key is None:
                key = 1
            if key not in self.prefetch:
                bottom_layer, _ = self.layer_bounds
                offset_extent, = self.offset.shape
                j = Index(offset_extent)
                base = Indexed(self.values, (n, j))
                if f.extent:
                    k = Index(f.extent)
                else:
                    k = Index(1)
                offset = Sum(Sum(layer, Product(Literal(numpy.int32(-1)), bottom_layer)), k)
                offset = Product(offset, Indexed(self.offset, (j,)))
                self.prefetch[key] = Materialise(PackInst(), Sum(base, offset), MultiIndex(k, j))

            return Indexed(self.prefetch[key], (f, i)), (f, i)
        else:
            assert f.extent == 1 or f.extent is None
            base = Indexed(self.values, (n, i))
            return base, (f, i)

    def indexed_vector(self, n, shape, layer=None):
        shape = self.shape[1:] + shape
        if self.interior_horizontal:
            shape = (2, ) + shape
        else:
            shape = (1, ) + shape
        f, i, j = (Index(e) for e in shape)
        base, (f, i) = self.indexed((n, i, f), layer=layer)
        discard = Comparison("<", base, Zero((), self.dtype))
        if self.vector_bc is not None:
            # Exposition:
            # Vector-index bcs are encoded in the high bits of the map.
            # The incoming value is:
            # input := ~(row + sum_i 2**(nbit - i))
            # Where i are the components to zero
            # The actual row is then:
            # row := (~input) & (~(sum_{k<3} 2**(nbit - k)))
            # And the high bits that are non-zero tell us which
            # values to mask out.
            nbits = Literal(self.dtype.type(self.dtype.itemsize*8 - 2))
            mask = Literal(self.dtype.type(sum(2**(nbits.value - i) for i in range(3))))
            flipped = BitwiseNot(base)
            base = Conditional(discard,
                               BitwiseAnd(flipped, BitwiseNot(mask)),
                               base)
            expr = LogicalNot(BitwiseAnd(flipped, mask))
            expr = LogicalOr(expr,
                             BitwiseAnd(flipped,
                                        BitShift("<<", Literal(self.dtype.type(1)),
                                                 Sum(nbits,
                                                     Product(Literal(self.dtype.type(-1)), j)))))
            discard = LogicalAnd(discard, expr)

        init = Conditional(discard, Literal(self.dtype.type(-1)), Sum(Product(base, Literal(numpy.int32(j.extent))), j))
        pack = Materialise(PackInst(), init, MultiIndex(f, i, j))
        multiindex = tuple(Index(e) for e in pack.shape)
        return Indexed(pack, multiindex), multiindex

    def indexed_implicit(self, n, layer=None):
        if layer is None:
            raise ValueError("Implicit bcs and no layers?!")
        shape = self.shape[1:]
        if self.interior_horizontal:
            shape = (2, ) + shape
        else:
            shape = (1, ) + shape
        f, i = (Index(e) for e in shape)
        base, (f, i) = self.indexed((n, i, f), layer=layer)

        expressions = [PackInst(), base, MultiIndex(f, i)]
        if self.variable:
            for location, method in self.implicit_bcs:
                index_array = self.boundary_masks[method]
                # For facets
                if self.interior_horizontal:
                    f = Index(2)
                else:
                    f = FixedIndex(0)
                bottom_mask, top_mask = self.variable_entity_masks

                idx, = bottom_mask.multiindex
                idx = Sum(idx, f)
                if location == "bottom":
                    mask = Indexed(bottom_mask.aggregate, (idx, ))
                else:
                    mask = Indexed(top_mask.aggregate, (idx, ))

                if all(index_array.offset.value == 0):
                    # No need to do this if there are no boundary dofs
                    continue
                bit = Index(index_array.nrows)
                when = BitwiseAnd(mask, BitShift("<<", Literal(numpy.int64(1)), bit))
                off = Materialise(PackInst(), Indexed(index_array.offset, (bit, )), MultiIndex())
                dof = Materialise(PackInst(), Indexed(index_array.dof, (bit, )), MultiIndex())
                k = RuntimeIndex(off, Sum(off, dof),
                                 LogicalAnd(
                                     Comparison("<=", Zero((), numpy.int32), off),
                                     Comparison("<=", Zero((), numpy.int32), dof)))

                index = Indexed(index_array.values, (k, ))

                expr = When(when, Literal(self.dtype.type(-1)))
                indices = MultiIndex(f, index)
                expressions.append(expr)
                expressions.append(indices)
        else:
            for location, method in self.implicit_bcs:
                i = Index()
                bottom, top = self.boundary_masks[method]
                idx = FixedIndex(0)
                if location == "bottom":
                    indices = bottom
                    bound = self.layer_bounds[0]
                else:
                    indices = top
                    bound = Sum(self.layer_bounds[1], Literal(IntType.type(-1)))
                    if self.interior_horizontal:
                        idx = FixedIndex(1)

                index = Indexed(indices, (i, ))
                when = Comparison("==", layer, bound)

                expr = When(when, Literal(self.dtype.type(-1)))
                indices = MultiIndex(idx, index)
                expressions.append(expr)
                expressions.append(indices)
        pack = Materialise(*expressions)
        multiindex = tuple(Index(e) for e in pack.shape)
        return Indexed(pack, multiindex), multiindex


class Pack(metaclass=ABCMeta):

    @abstractmethod
    def kernel_arg(self, loop_indices=None):
        pass

    @abstractmethod
    def pack(self, loop_indices=None):
        pass

    @abstractmethod
    def emit_unpack_instruction(self, *,
                                loop_indices=None):
        pass


class GlobalPack(Pack):

    def __init__(self, outer, access):
        self.outer = outer
        self.access = access

    def kernel_arg(self, loop_indices=None):
        return Indexed(self.outer, (Index(e) for e in self.outer.shape))

    # TODO: do we make a temporary and zero it?
    def pack(self, loop_indices=None):
        return None

    def emit_unpack_instruction(self, *,
                                loop_indices=None):
        yield None


class DatPack(Pack):
    def __init__(self, outer, access, map_=None, interior_horizontal=False,
                 view_index=None, layer_bounds=None):
        self.outer = outer
        self.map_ = map_
        self.access = access
        self.interior_horizontal = interior_horizontal
        self.view_index = view_index
        self.layer_bounds = layer_bounds

    def _rvalue(self, multiindex, loop_indices=None):
        f, i, *j = multiindex
        try:
            n, layer = loop_indices
        except ValueError:
            n, = loop_indices
            layer = None
        if self.view_index is not None:
            j = tuple(j) + tuple(FixedIndex(i) for i in self.view_index)
        map_, (f, i) = self.map_.indexed((n, i, f), layer=layer)
        return Indexed(self.outer,
                       MultiIndex(map_, *j))

    def pack(self, loop_indices=None):
        if self.map_ is None:
            return None

        if hasattr(self, "_pack"):
            return self._pack

        if self.interior_horizontal:
            shape = (2, )
        else:
            shape = (1, )

        shape = shape + self.map_.shape[1:]
        if self.view_index is None:
            shape = shape + self.outer.shape[1:]

        if self.access in {INC, WRITE}:
            val = Zero((), self.outer.dtype)
            multiindex = MultiIndex(*(Index(e) for e in shape))
            self._pack = Materialise(PackInst(), val, multiindex)
        else:
            multiindex = MultiIndex(*(Index(e) for e in shape))
            self._pack = Materialise(PackInst(),
                                     self._rvalue(multiindex, loop_indices=loop_indices),
                                     multiindex)
        return self._pack

    def kernel_arg(self, loop_indices=None):
        if self.map_ is None:
            if loop_indices is None:
                raise ValueError("Need iteration index")
            try:
                n, layer = loop_indices
            except ValueError:
                n, = loop_indices
            # Direct dats on extruded sets never get a layer index
            # (they're defined on the "base" set, effectively).
            # FIXME: is this a bug?
            shape = self.outer.shape
            if self.view_index is None:
                multiindex = (n, ) + tuple(Index(e) for e in shape[1:])
            else:
                multiindex = (n, ) + tuple(FixedIndex(i) for i in self.view_index)
            return Indexed(self.outer, multiindex)
        else:
            pack = self.pack(loop_indices)
            shape = pack.shape
            return Indexed(pack, (Index(e) for e in shape))

    def emit_unpack_instruction(self, *,
                                loop_indices=None):
        pack = self.pack(loop_indices)
        if pack is None:
            yield None
        elif self.access is READ:
            yield None
        elif self.access is INC:
            multiindex = tuple(Index(e) for e in pack.shape)
            rvalue = self._rvalue(multiindex, loop_indices=loop_indices)
            yield Accumulate(UnpackInst(),
                             rvalue,
                             Sum(rvalue, view(pack, tuple((0, i) for i in multiindex))))
        else:
            multiindex = tuple(Index(e) for e in pack.shape)
            yield Accumulate(UnpackInst(),
                             self._rvalue(multiindex, loop_indices=loop_indices),
                             view(pack, tuple((0, i) for i in multiindex)))


class MixedDatPack(Pack):
    def __init__(self, packs, access, dtype, interior_horizontal):
        self.packs = packs
        self.access = access
        self.dtype = dtype
        self.interior_horizontal = interior_horizontal

    def pack(self, loop_indices=None):
        if hasattr(self, "_pack"):
            return self._pack

        flat_shape = numpy.sum(tuple(numpy.prod(p.map_.shape[1:] + p.outer.shape[1:]) for p in self.packs))

        if self.interior_horizontal:
            _shape = (2,)
            flat_shape *= 2
        else:
            _shape = (1,)

        if self.access in {INC, WRITE}:
            val = Zero((), self.dtype)
            multiindex = MultiIndex(Index(flat_shape))
            self._pack = Materialise(PackInst(), val, multiindex)
        else:
            multiindex = MultiIndex(Index(flat_shape))
            val = Zero((), self.dtype)
            expressions = []
            offset = 0
            for p in self.packs:
                shape = _shape + p.map_.shape[1:] + p.outer.shape[1:]
                mi = MultiIndex(*(Index(e) for e in shape))
                expr = p._rvalue(mi, loop_indices)
                extents = [numpy.prod(shape[i+1:], dtype=numpy.int32) for i in range(len(shape))]
                index = reduce(Sum, [Product(i, Literal(IntType.type(e), casting=False)) for i, e in zip(mi, extents)], Literal(IntType.type(0), casting=False))
                indices = MultiIndex(Sum(index, Literal(IntType.type(offset), casting=False)),)
                offset += numpy.prod(shape, dtype=numpy.int32)
                expressions.append(expr)
                expressions.append(indices)

            self._pack = Materialise(PackInst(), val, multiindex, *expressions)

        return self._pack

    def kernel_arg(self, loop_indices=None):
        pack = self.pack(loop_indices)
        shape = pack.shape
        return Indexed(pack, (Index(e) for e in shape))

    def emit_unpack_instruction(self, *,
                                loop_indices=None):
        pack = self.pack(loop_indices)

        if self.access is READ:
            yield None
        else:
            if self.interior_horizontal:
                _shape = (2,)
            else:
                _shape = (1,)
            offset = 0
            for p in self.packs:
                shape = _shape + p.map_.shape[1:] + p.outer.shape[1:]
                mi = MultiIndex(*(Index(e) for e in shape))
                rvalue = p._rvalue(mi, loop_indices)
                extents = [numpy.prod(shape[i+1:], dtype=numpy.int32) for i in range(len(shape))]
                index = reduce(Sum, [Product(i, Literal(IntType.type(e), casting=False)) for i, e in zip(mi, extents)], Literal(IntType.type(0), casting=False))
                indices = MultiIndex(Sum(index, Literal(IntType.type(offset), casting=False)),)
                rhs = Indexed(pack, indices)
                offset += numpy.prod(shape, dtype=numpy.int32)

                if self.access is INC:
                    rhs = Sum(rvalue, rhs)

                yield Accumulate(UnpackInst(), rvalue, rhs)


class MatPack(Pack):
    def __init__(self, outer, access, maps, dims, dtype, interior_horizontal=False):
        self.outer = outer
        self.access = access
        self.maps = maps
        self.dims = dims
        self.dtype = dtype
        self.interior_horizontal = interior_horizontal

    def pack(self, loop_indices=None):
        if hasattr(self, "_pack"):
            return self._pack
        ((rdim, cdim), ), = self.dims
        rmap, cmap = self.maps
        if self.interior_horizontal:
            shape = (2, )
        else:
            shape = (1, )
        rshape = shape + rmap.shape[1:] + (rdim, )
        cshape = shape + cmap.shape[1:] + (cdim, )
        if self.access in {WRITE, INC}:
            val = Zero((), self.dtype)
            multiindex = MultiIndex(*(Index(e) for e in (rshape + cshape)))
            pack = Materialise(PackInst(), val, multiindex)
            self._pack = pack
            return pack
        else:
            raise ValueError("Unexpected access type")

    def kernel_arg(self, loop_indices=None):
        pack = self.pack(loop_indices=loop_indices)
        return Indexed(pack, tuple(Index(e) for e in pack.shape))

    def emit_unpack_instruction(self, *,
                                loop_indices=None):
        ((rdim, cdim), ), = self.dims
        rmap, cmap = self.maps
        try:
            n, layer = loop_indices
        except ValueError:
            n, = loop_indices
            layer = None
        vector = rmap.vector_bc or cmap.vector_bc
        if vector:
            maps = [map_.indexed_vector(n, (dim, ), layer=layer)
                    for map_, dim in zip(self.maps, (rdim, cdim))]
        else:
            maps = []
            for map_ in self.maps:
                if map_.implicit_bcs:
                    maps.append(map_.indexed_implicit(n, layer=layer))
                else:
                    i = Index()
                    if self.interior_horizontal:
                        f = Index(2)
                    else:
                        f = Index(1)
                    maps.append(map_.indexed((n, i, f), layer=layer))
        (rmap, cmap), (rindices, cindices) = zip(*maps)

        pack = self.pack(loop_indices=loop_indices)
        if vector:
            # The shape of MatPack is
            # (row, cols) if it has vector BC
            # (block_rows, row_cmpt, block_cols, col_cmpt) otherwise
            free_indices = rindices + cindices
            pack = Indexed(pack, free_indices)
            name = "MatSetValuesLocal"
        else:
            free_indices = rindices + (Index(), ) + cindices + (Index(), )
            pack = Indexed(pack, free_indices)
            name = "MatSetValuesBlockedLocal"

        access = Symbol({WRITE: "INSERT_VALUES",
                         INC: "ADD_VALUES"}[self.access])

        rextent = Extent(MultiIndex(*rindices))
        cextent = Extent(MultiIndex(*cindices))

        call = FunctionCall(name,
                            UnpackInst(),
                            (self.access, READ, READ, READ, READ, READ, READ),
                            free_indices,
                            self.outer,
                            rextent,
                            rmap,
                            cextent,
                            cmap,
                            pack,
                            access)

        yield call


class WrapperBuilder(object):

    def __init__(self, *, iterset, iteration_region=None, single_cell=False,
                 pass_layer_to_kernel=False, forward_arg_types=()):
        super().__init__()
        self.arguments = []
        self.argument_accesses = []
        self.packed_args = []
        self.indices = []
        self.maps = OrderedDict()
        self.iterset = iterset
        if iteration_region is None:
            self.iteration_region = ALL
        else:
            self.iteration_region = iteration_region
        self.pass_layer_to_kernel = pass_layer_to_kernel
        self.single_cell = single_cell
        self.forward_arguments = tuple(Argument((), fa, pfx="farg") for fa in forward_arg_types)

    @property
    def subset(self):
        return isinstance(self.iterset, Subset)

    @property
    def extruded(self):
        return self.iterset._extruded

    @property
    def constant_layers(self):
        return self.extruded and self.iterset.constant_layers

    def set_kernel(self, kernel):
        self.kernel = kernel

    @cached_property
    def loop_extents(self):
        return (Argument((), IntType, name="start"),
                Argument((), IntType, name="end"))

    @cached_property
    def _loop_index(self):
        start, end = self.loop_extents
        return RuntimeIndex(start, end,
                            LogicalAnd(
                                Comparison("<=", Zero((), numpy.int32), start),
                                Comparison("<=", start, end)),
                            name="n")

    @cached_property
    def _subset_indices(self):
        return Argument(("end", ), IntType, name="subset_indices")

    @cached_property
    def loop_index(self):
        n = self._loop_index
        if self.subset:
            n = Materialise(PackInst(), Indexed(self._subset_indices, MultiIndex(n)), MultiIndex())
        return n

    @cached_property
    def _layers_array(self):
        if self.constant_layers:
            return Argument((1, 2), IntType, name="layers")
        else:
            return Argument((None, 2), IntType, name="layers")

    @cached_property
    def bottom_layer(self):
        if self.iteration_region == ON_TOP:
            return Materialise(PackInst(),
                               Indexed(self._layers_array, (self._layer_index, FixedIndex(0))),
                               MultiIndex())
        else:
            start, _ = self.layer_extents
            return start

    @cached_property
    def top_layer(self):
        if self.iteration_region == ON_BOTTOM:
            return Materialise(PackInst(),
                               Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                                   Literal(IntType.type(-1))),
                               MultiIndex())
        else:
            _, end = self.layer_extents
            return end

    @cached_property
    def variable_entity_masks(self):
        if self.extruded:
            off = Argument((None, ), IntType, name="entity_offset")
            # FIXME: this is never actually used.
            dof = Argument((None, ), IntType, name="entity_dof")
            bottom = Argument((None, ), numpy.int64, name="entity_bottom_mask")
            top = Argument((None, ), numpy.int64, name="entity_top_mask")
            return SparseArray(bottom, dof, off), SparseArray(top, dof, off)
        else:
            return None

    @cached_property
    def indexed_variable_entity_masks(self):
        if self.extruded:
            bottom, top = self.variable_entity_masks
            off = Indexed(bottom.offset, (self.loop_index, ))
            index = Sum(off, Sum(self.layer_index, Product(Literal(numpy.int32(-1)),
                                                           self.bottom_layer)))
            bottom = Indexed(bottom.values, (index, ))
            top = Indexed(top.values, (index, ))
            return bottom, top
        return None

    @cached_property
    def layer_extents(self):
        if self.iteration_region == ON_BOTTOM:
            start = Indexed(self._layers_array, (self._layer_index, FixedIndex(0)))
            end = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(0))),
                      Literal(IntType.type(1)))
        elif self.iteration_region == ON_TOP:
            start = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                        Literal(IntType.type(-2)))
            end = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                      Literal(IntType.type(-1)))
        elif self.iteration_region == ON_INTERIOR_FACETS:
            start = Indexed(self._layers_array, (self._layer_index, FixedIndex(0)))
            end = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                      Literal(IntType.type(-2)))
        elif self.iteration_region == ALL:
            start = Indexed(self._layers_array, (self._layer_index, FixedIndex(0)))
            end = Sum(Indexed(self._layers_array, (self._layer_index, FixedIndex(1))),
                      Literal(IntType.type(-1)))
        else:
            raise ValueError("Unknown iteration region")
        return (Materialise(PackInst(), start, MultiIndex()),
                Materialise(PackInst(), end, MultiIndex()))

    @cached_property
    def _layer_index(self):
        if self.constant_layers:
            return FixedIndex(0)
        if self.subset:
            return self._loop_index
        else:
            return self.loop_index

    @cached_property
    def layer_index(self):
        if self.extruded:
            start, end = self.layer_extents
            return RuntimeIndex(start, end,
                                LogicalAnd(
                                    Comparison("<=", Zero((), numpy.int32), start),
                                    Comparison("<=", start, end)),
                                name="layer")
        else:
            return None

    @property
    def loop_indices(self):
        if self.extruded:
            return (self.loop_index, self.layer_index)
        else:
            return (self.loop_index, )

    def add_argument(self, arg):
        interior_horizontal = self.iteration_region == ON_INTERIOR_FACETS
        if arg._is_dat:
            if arg._is_mixed:
                packs = []
                for a in arg:
                    shape = (None, *a.data.shape[1:])
                    argument = Argument(shape, a.data.dtype, pfx="mdat")
                    packs.append(DatPack(argument, arg.access, self.map_(a.map),
                                         interior_horizontal=interior_horizontal))
                    self.arguments.append(argument)
                pack = MixedDatPack(packs, arg.access, arg.dtype, interior_horizontal=interior_horizontal)
                self.packed_args.append(pack)
                self.argument_accesses.append(arg.access)
                return
            if arg._is_dat_view:
                view_index = arg.data.index
                data = arg.data._parent
            else:
                view_index = None
                data = arg.data
            shape = (None, *data.shape[1:])
            argument = Argument(shape,
                                arg.data.dtype,
                                pfx="dat")
            pack = DatPack(argument, arg.access, self.map_(arg.map),
                           interior_horizontal=interior_horizontal,
                           view_index=view_index)
        elif arg._is_global:
            argument = Argument(arg.data.dim,
                                arg.data.dtype,
                                pfx="glob")
            pack = GlobalPack(argument, arg.access)
        elif arg._is_mat:
            argument = Argument((), PetscMat(), pfx="mat")
            map_ = tuple(self.map_(m) for m in arg.map)
            pack = MatPack(argument, arg.access, map_,
                           arg.data.dims, arg.data.dtype,
                           interior_horizontal=interior_horizontal)
        else:
            raise ValueError("Unhandled argument type")
        self.arguments.append(argument)
        self.packed_args.append(pack)
        self.argument_accesses.append(arg.access)

    def map_(self, map_):
        if map_ is None:
            return None
        interior_horizontal = self.iteration_region == ON_INTERIOR_FACETS
        if isinstance(map_, DecoratedMap):
            key = map_.map
        else:
            key = map_
        try:
            return self.maps[key]
        except KeyError:
            map_ = Map(map_, interior_horizontal,
                       (self.bottom_layer, self.top_layer),
                       self.indexed_variable_entity_masks)
            self.maps[key] = map_
            return map_

    @property
    def kernel_args(self):
        return tuple(p.kernel_arg(self.loop_indices) for p in self.packed_args)

    @property
    def wrapper_args(self):
        # Loop extents come from here.
        args = list(self.forward_arguments)
        args.extend(self._loop_index.extents)
        if self.extruded:
            args.append(self._layers_array)
            if not self.constant_layers:
                bottom, top = self.variable_entity_masks
                assert bottom.offset == top.offset
                args.append(bottom.offset)
                args.append(bottom.values)
                args.append(top.values)
        if self.subset:
            args.append(self._subset_indices)
        # parloop args passed "as is"
        args.extend(self.arguments)
        # maps are refcounted
        for map_ in self.maps.values():
            args.append(map_.values)
        return tuple(args)

    def kernel_call(self):
        args = self.kernel_args
        access = tuple(self.argument_accesses)
        # assuming every index is free index
        free_indices = set(itertools.chain.from_iterable(arg.multiindex for arg in args))
        # remove runtime index
        free_indices = tuple(i for i in free_indices if isinstance(i, Index))
        if self.pass_layer_to_kernel:
            args = args + (self.layer_index, )
            access = access + (READ,)
        if self.forward_arguments:
            args = self.forward_arguments + args
            access = tuple([WRITE] * len(self.forward_arguments)) + access
        return FunctionCall(self.kernel.name, KernelInst(), access, free_indices, *args)

    def emit_instructions(self):
        yield DummyInstruction(PackInst(), *self.loop_indices)
        yield self.kernel_call()
        for pack in self.packed_args:
            insns = pack.emit_unpack_instruction(loop_indices=self.loop_indices)
            for insn in insns:
                if insn is not None:
                    yield insn
