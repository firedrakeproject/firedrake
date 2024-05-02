r"""This module implements parallel loops reading and writing
:class:`.Function`\s. This provides a mechanism for implementing
non-finite element operations such as slope limiters."""
import collections
import functools
from cachetools import LRUCache
from typing import Any

import finat
import loopy
import numpy as np
import pyop3 as op3
import ufl
from pyop2 import op2, READ, WRITE, RW, INC, MIN, MAX
from pyop3.axtree.tree import ExpressionEvaluator
from pyop3.array.harray import ArrayVar
from pyop3.itree.tree import compose_axes
from pyrsistent import freeze, pmap
from ufl.indexed import Indexed
from ufl.domain import join_domains

from firedrake import constant
from firedrake.cofunction import Cofunction
from firedrake.function import CoordinatelessFunction, Function
from firedrake.functionspaceimpl import WithGeometry, MixedFunctionSpace
from firedrake.matrix import Matrix
from firedrake.petsc import PETSc
from firedrake.parameters import target
from firedrake.ufl_expr import extract_domains
from firedrake.utils import IntType, assert_empty


# Set a default loopy language version (should be in __init__.py)
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


kernel_cache = LRUCache(maxsize=128)


__all__ = ['par_loop', 'direct', 'READ', 'WRITE', 'RW', 'INC', 'MIN', 'MAX']


class _DirectLoop(object):
    r"""A singleton object which can be used in a :func:`par_loop` in place
    of the measure in order to indicate that the loop is a direct loop
    over degrees of freedom."""

    def integral_type(self):
        return "direct"

    def __repr__(self):

        return "direct"


direct = _DirectLoop()
r"""A singleton object which can be used in a :func:`par_loop` in place
of the measure in order to indicate that the loop is a direct loop
over degrees of freedom."""


def indirect_measure(mesh, measure):
    return mesh.measure_set(measure.integral_type(),
                            measure.subdomain_id())


_maps = {
    'cell': {
        'nodes': lambda x: x.cell_node_map(),
        'itspace': indirect_measure
    },
    'interior_facet': {
        'nodes': lambda x: x.interior_facet_node_map(),
        'itspace': indirect_measure
    },
    'exterior_facet': {
        'nodes': lambda x: x.exterior_facet_node_map(),
        'itspace': indirect_measure
    },
    'direct': {
        'nodes': lambda x: None,
        'itspace': lambda mesh, measure: mesh
    }
}
r"""Map a measure to the correct maps."""


def _form_loopy_kernel(kernel_domains, instructions, measure, args, **kwargs):

    kargs = []

    for var, (func, intent) in args.items():
        is_input = intent in [INC, READ, RW, MAX, MIN]
        is_output = intent in [INC, RW, WRITE, MAX, MIN]
        if isinstance(func, constant.Constant):
            if intent is not READ:
                raise RuntimeError("Only READ access is allowed to Constant")
            # Constants modelled as Globals, so no need for double
            # indirection
            ndof = func.dat.cdim
            kargs.append(loopy.GlobalArg(var, dtype=func.dat.dtype, shape=(ndof,), is_input=is_input, is_output=is_output))
        else:
            # Do we have a component of a mixed function?
            if isinstance(func, Indexed):
                c, i = func.ufl_operands
                idx = i._indices[0]._value
                ndof = c.function_space()[idx].finat_element.space_dimension()
                cdim = c.dat[idx].cdim
                dtype = c.dat[idx].dtype
            else:
                if func.function_space().ufl_element().family() == "Real":
                    ndof = func.function_space().dim()  # == 1
                    kargs.append(loopy.GlobalArg(var, dtype=func.dat.dtype, shape=(ndof,), is_input=is_input, is_output=is_output))
                    continue
                else:
                    if len(func.function_space()) > 1:
                        raise NotImplementedError("Must index mixed function in par_loop.")
                    ndof = func.function_space().finat_element.space_dimension()
                    cdim = func.dat.cdim
                    dtype = func.dat.dtype
            if measure.integral_type() == 'interior_facet':
                ndof *= 2
            # FIXME: shape for facets [2][ndof]?
            kargs.append(loopy.GlobalArg(var, dtype=dtype, shape=(ndof, cdim), is_input=is_input, is_output=is_output))
        kernel_domains = kernel_domains.replace(var+".dofs", str(ndof))

    if kernel_domains == "":
        kernel_domains = "[] -> {[]}"
    try:
        key = (kernel_domains, tuple(instructions), tuple(map(tuple, kwargs.items())))
        # Add shape, dtype and intent to the cache key
        for func, intent in args.values():
            if isinstance(func, Indexed):
                for dat in func.ufl_operands[0].dat.split:
                    key += (dat.shape, dat.dtype, intent)
            else:
                key += (func.dat.shape, func.dat.dtype, intent)
        return kernel_cache[key]
    except KeyError:
        kargs.append(...)
        knl = loopy.make_function(kernel_domains, instructions, kargs, name="par_loop_kernel", target=target,
                                  seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"])
        knl = op2.Kernel(knl, "par_loop_kernel", **kwargs)
        return kernel_cache.setdefault(key, knl)


@PETSc.Log.EventDecorator()
def par_loop(kernel, measure, args, kernel_kwargs=None, **kwargs):
    r"""A :func:`par_loop` is a user-defined operation which reads and
    writes :class:`.Function`\s by looping over the mesh cells or facets
    and accessing the degrees of freedom on adjacent entities.

    :arg kernel: A 2-tuple of (domains, instructions) to create
        a loopy kernel . The domains and instructions should be specified
        in loopy kernel syntax. See the `loopy tutorial
        <https://documen.tician.de/loopy/tutorial.html>`_ for details.

    :arg measure: is a UFL :class:`~ufl.measure.Measure` which determines the
        manner in which the iteration over the mesh is to occur.
        Alternatively, you can pass :data:`direct` to designate a direct loop.
    :arg args: is a dictionary mapping variable names in the kernel to
        :class:`.Function`\s or components of mixed :class:`.Function`\s and
        indicates how these :class:`.Function`\s are to be accessed.
    :arg kernel_kwargs: keyword arguments to be passed to the
        ``pyop2.Kernel`` constructor
    :arg kwargs: additional keyword arguments are passed to the underlying
        ``pyop2.par_loop``

    :kwarg iterate: Optionally specify which region of an
                    :class:`pyop2.types.set.ExtrudedSet` to iterate over.
                    Valid values are the following objects from pyop2:

                    - ``ON_BOTTOM``: iterate over the bottom layer of cells.
                    - ``ON_TOP`` iterate over the top layer of cells.
                    - ``ALL`` iterate over all cells (the default if unspecified)
                    - ``ON_INTERIOR_FACETS`` iterate over all the layers
                      except the top layer, accessing data two adjacent (in
                      the extruded direction) cells at a time.

    **Example**

    Assume that `A` is a :class:`.Function` in CG1 and `B` is a
    :class:`.Function` in DG0. Then the following code sets each DoF in
    `A` to the maximum value that `B` attains in the cells adjacent to
    that DoF::

      A.assign(numpy.finfo(0.).min)
      domain = '{[i]: 0 <= i < A.dofs}'
      instructions = '''
      for i
          A[i] = max(A[i], B[0])
      end
      '''
      par_loop((domain, instructions), dx, {'A' : (A, RW), 'B': (B, READ)})


    **Argument definitions**

    Each item in the `args` dictionary maps a string to a tuple
    containing a :class:`.Function` or :class:`.Constant` and an
    argument intent. The string is the c language variable name by
    which this function will be accessed in the kernel. The argument
    intent indicates how the kernel will access this variable:

    `READ`
       The variable will be read but not written to.
    `WRITE`
       The variable will be written to but not read. If multiple kernel
       invocations write to the same DoF, then the order of these writes
       is undefined.
    `RW`
       The variable will be both read and written to. If multiple kernel
       invocations access the same DoF, then the order of these accesses
       is undefined, but it is guaranteed that no race will occur.
    `INC`
       The variable will be added into using +=. As before, the order in
       which the kernel invocations increment the variable is undefined,
       but there is a guarantee that no races will occur.

    .. note::

       Only `READ` intents are valid for :class:`.Constant`
       coefficients, and an error will be raised in other cases.

    **The measure**

    The measure determines the mesh entities over which the iteration
    will occur, and the size of the kernel stencil. The iteration will
    occur over the same mesh entities as if the measure had been used
    to define an integral, and the stencil will likewise be the same
    as the integral case. That is to say, if the measure is a volume
    measure, the kernel will be called once per cell and the DoFs
    accessible to the kernel will be those associated with the cell,
    its facets, edges and vertices. If the measure is a facet measure
    then the iteration will occur over the corresponding class of
    facets and the accessible DoFs will be those on the cell(s)
    adjacent to the facet, and on the facets, edges and vertices
    adjacent to those facets.

    For volume measures the DoFs are guaranteed to be in the FInAT
    local DoFs order. For facet measures, the DoFs will be in sorted
    first by the cell to which they are adjacent. Within each cell,
    they will be in FInAT order. Note that if a continuous
    :class:`.Function` is accessed via an internal facet measure, the
    DoFs on the interface between the two facets will be accessible
    twice: once via each cell. The orientation of the cell(s) relative
    to the current facet is currently arbitrary.

    A direct loop over nodes without any indirections can be specified
    by passing :data:`direct` as the measure. In this case, all of the
    arguments must be :class:`.Function`\s in the same
    :class:`.FunctionSpace`.

    **The kernel code**

    Indirect free variables referencing :class:`.Function`\s are all
    of type `double*`. For spaces with rank greater than zero (Vector
    or TensorElement), the data are laid out XYZ... XYZ... XYZ....
    With the vector/tensor component moving fastest.

    In loopy syntax, these may be addressed using 2D indexing::

       A[i, j]

    Where ``i`` runs over nodes, and ``j`` runs over components.

    In a direct :func:`par_loop`, the variables will all be of type
    `double*` with the single index being the vector component.

    :class:`.Constant`\s are always of type `double*`, both for
    indirect and direct :func:`par_loop` calls.

    """
    raise NotImplementedError("TODO pyop3")

    # catch deprecated C-string parloops
    if isinstance(kernel, str):
        raise TypeError("C-string kernels are no longer supported by Firedrake parloops")
    if "is_loopy_kernel" in kwargs:
        if kwargs.pop("is_loopy_kernel"):
            import warnings
            warnings.warn(
                "is_loopy_kernel does not need to be specified", FutureWarning)
        else:
            raise ValueError(
                "Support for C-string kernels has been dropped, firedrake.parloop "
                "will only work with loopy parloops.")

    if kernel_kwargs is None:
        kernel_kwargs = {}

    _map = _maps[measure.integral_type()]
    # Ensure that the dict args passed in are consistently ordered
    # (sorted by the string key).
    sorted_args = collections.OrderedDict()
    for k in sorted(args.keys()):
        sorted_args[k] = args[k]
    args = sorted_args

    if measure is direct:
        mesh = None
        for (func, intent) in args.values():
            if isinstance(func, Indexed):
                c, i = func.ufl_operands
                idx = i._indices[0]._value
                if mesh and c.node_set[idx] is not mesh:
                    raise ValueError("Cannot mix sets in direct loop.")
                mesh = c.node_set[idx]
            else:
                try:
                    if mesh and func.node_set is not mesh:
                        raise ValueError("Cannot mix sets in direct loop.")
                    mesh = func.node_set
                except AttributeError:
                    # Argument was a Global.
                    pass
        if not mesh:
            raise TypeError("No Functions passed to direct par_loop")
    else:
        domains = []
        for func, _ in args.values():
            domains.extend(extract_domains(func))
        domains = join_domains(domains)
        # Assume only one domain
        domain, = domains
        mesh = domain

    kernel_domains, instructions = kernel
    op2args = [_form_loopy_kernel(kernel_domains, instructions, measure, args, **kernel_kwargs)]

    op2args.append(_map['itspace'](mesh, measure))

    def mkarg(f, intent):
        if isinstance(f, Indexed):
            c, i = f.ufl_operands
            idx = i._indices[0]._value
            m = _map['nodes'](c)
            return c.dat[idx](intent, m.split[idx] if m else None)
        return f.dat(intent, _map['nodes'](f))
    op2args += [mkarg(func, intent) for (func, intent) in args.values()]

    return op2.parloop(*op2args, **kwargs)


@functools.singledispatch
def pack_tensor(tensor: Any, index: op3.LoopIndex, integral_type: str):
    raise TypeError(f"No handler defined for {type(tensor).__name__}")


@pack_tensor.register(Function)
@pack_tensor.register(Cofunction)
@pack_tensor.register(CoordinatelessFunction)
def _(func, index: op3.LoopIndex, integral_type: str):
    return pack_pyop3_tensor(
        func.dat, func.function_space(), index, integral_type
    )


@pack_tensor.register
def _(matrix: Matrix, index: op3.LoopIndex, integral_type: str):
    return pack_pyop3_tensor(
        matrix.M, *matrix.ufl_function_spaces(), index, integral_type
    )


@functools.singledispatch
def pack_pyop3_tensor(tensor: Any, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(tensor).__name__}")


@pack_pyop3_tensor.register
def _(
    array: op3.HierarchicalArray,
    V: WithGeometry,
    index: op3.LoopIndex,
    integral_type: str,
):
    plex = V.mesh().topology

    if V.ufl_element().family() == "Real":
        return array

    if integral_type == "cell":
        # TODO ideally the FIAT permutation would not need to be known
        # about by the mesh topology and instead be handled here. This
        # would probably require an oriented mesh
        # (see https://github.com/firedrakeproject/firedrake/pull/3332)
        # indexed = array.getitem(plex._fiat_closure(index), strict=True)
        pack_indices = _cell_integral_pack_indices(V, index)
    elif integral_type in {"exterior_facet", "interior_facet"}:
        pack_indices = _facet_integral_pack_indices(V, index)
    else:
        raise NotImplementedError

    indexed = array.getitem(pack_indices, strict=True)

    if plex.ufl_cell().is_simplex():
        return indexed

    if plex.ufl_cell() == ufl.hexahedron:
        perms = _entity_permutations(V)
        mytree = _orientations(plex, perms, index, integral_type)
        mytree = _with_shape_indices(V, mytree, integral_type in {"exterior_facet", "interior_facet"})

        indexed = indexed.getitem(mytree, strict=True)

    tensor_axes, ttarget_paths, tindex_exprs = _tensorify_axes(V.finat_element.product)
    tensor_axes, ttarget_paths, tindex_exprs = _with_shape_axes(V, tensor_axes, ttarget_paths, tindex_exprs, integral_type)

    # This should be cleaned up - basically we need to accumulate the target_paths
    # and index_exprs along the nodes. This is done inside index_axes in pyop3.
    from pyop3.itree.tree import _acc_target_paths
    ttarget_paths = _acc_target_paths(tensor_axes, ttarget_paths)
    tindex_exprs = _acc_target_paths(tensor_axes, tindex_exprs)

    tensor_axes = op3.IndexedAxisTree(
        tensor_axes.node_map,
        indexed.axes.unindexed,
        target_paths=ttarget_paths,
        index_exprs=tindex_exprs,
        layout_exprs={},
        outer_loops=indexed.axes.outer_loops,
    )
    composed_axes = compose_axes(tensor_axes, indexed.axes)

    return op3.HierarchicalArray(
        composed_axes,
        data=indexed.buffer,
        max_value=indexed.max_value,
        name=indexed.name,
    )


@pack_pyop3_tensor.register
def _(
    mat: op3.Mat,
    Vrow: WithGeometry,
    Vcol: WithGeometry,
    index: op3.LoopIndex,
    integral_type: str
):
    if integral_type not in {"cell", "interior_facet", "exterior_facet"}:
        raise NotImplementedError("TODO")

    plex = op3.utils.single_valued(V.mesh().topology for V in {Vrow, Vcol})

    # First collect the DoFs in the cell closure in FIAT order.
    if integral_type == "cell":
        rmap = _cell_integral_pack_indices(Vrow, index)
        cmap = _cell_integral_pack_indices(Vcol, index)
    elif integral_type in {"exterior_facet", "interior_facet"}:
        rmap = _facet_integral_pack_indices(Vrow, index)
        cmap = _facet_integral_pack_indices(Vcol, index)
    else:
        raise NotImplementedError

    # FIXME: These maps are technically context-sensitive since they depend
    # on the loop index, but it is always trivial. However, we turn the index
    # tree into a forest here because that is what pyop3 "should" be doing
    # internally.
    context = pmap({index.id: (index.source_path, index.path)})
    rmap = {context: rmap}
    cmap = {context: cmap}

    indexed = mat.getitem((rmap, cmap), strict=True)

    # Indexing an array with a loop index makes it "context sensitive" since
    # the index could be over multiple entities (e.g. all mesh points). Here
    # we know we are only looping over cells so the context is trivial.
    cf_indexed = indexed.context_free

    if plex.ufl_cell().is_simplex():
        return cf_indexed

    if plex.ufl_cell() is ufl.hexahedron:
        raise NotImplementedError
    #     _entity_permutations(Vrow)

    # TODO: shouldn't need to do this
    from pyop3.itree.tree import _acc_target_paths

    axes0, target_paths0, index_exprs0 = _tensorify_axes(Vrow.finat_element.product)
    axes0, target_paths0, index_exprs0 = _with_shape_axes(Vrow, axes0, target_paths0, index_exprs0, integral_type)
    target_paths0 = _acc_target_paths(axes0, target_paths0)
    index_exprs0 = _acc_target_paths(axes0, index_exprs0)

    axes0 = op3.IndexedAxisTree(
        axes0.node_map,
        cf_indexed.raxes.unindexed,
        target_paths=target_paths0,
        index_exprs=index_exprs0,
        layout_exprs={},
        outer_loops=cf_indexed.raxes.outer_loops,
    )
    axes0 = compose_axes(axes0, cf_indexed.raxes)

    axes1, target_paths1, index_exprs1 = _tensorify_axes(Vcol.finat_element.product)
    axes1, target_paths1, index_exprs1 = _with_shape_axes(Vcol, axes1, target_paths1, index_exprs1, integral_type)
    target_paths1 = _acc_target_paths(axes1, target_paths1)
    index_exprs1 = _acc_target_paths(axes1, index_exprs1)

    axes1 = op3.IndexedAxisTree(
        axes1.node_map,
        cf_indexed.caxes.unindexed,
        target_paths=target_paths1,
        index_exprs=index_exprs1,
        layout_exprs={},
        outer_loops=cf_indexed.caxes.outer_loops,
    )
    axes1 = compose_axes(axes1, cf_indexed.caxes)

    return op3.Mat(
        axes0,
        axes1,
        mat_type=cf_indexed.mat_type,
        mat=cf_indexed.mat,
        name=cf_indexed.name,
    )


def _cell_integral_pack_indices(V: WithGeometry, cell: op3.LoopIndex) -> op3.IndexTree:
    plex = V.mesh().topology

    indices = op3.IndexTree.from_nest({
        plex._fiat_closure(cell): [
            op3.Slice("dof", [op3.AffineSliceComponent("XXX")])
            for _ in range(plex.dimension+1)
        ]
    })
    return _with_shape_indices(V, indices)


def _facet_integral_pack_indices(V: WithGeometry, facet: op3.LoopIndex) -> op3.IndexTree:
    plex = V.ufl_domain().topology

    indices = op3.IndexTree.from_nest({
        plex._fiat_closure(plex.support(facet)): [
            op3.Slice("dof", [op3.AffineSliceComponent("XXX")])
            for _ in range(plex.dimension+1)
        ]
    })
    # don't add support as an extra axis here, done already
    return _with_shape_indices(V, indices, and_support=False)


def _with_shape_indices(V: WithGeometry, indices: op3.IndexTree, and_support=False):
    is_mixed = isinstance(V.topological, MixedFunctionSpace)

    if is_mixed:
        spaces = V.topological._spaces
        trees = (indices,) * len(spaces)
    else:
        spaces = (V.topological,)
        trees = (indices,)

    # Add tensor shape innermost, this applies to cells, edges etc equally
    trees_ = []
    for space, tree in zip(spaces, trees):
        if space.shape:
            tensor_slices = tuple(
                op3.Slice(f"dim{i}", [op3.AffineSliceComponent("XXX")])
                for i, dim in enumerate(space.shape)
            )
            tensor_indices = op3.IndexTree.from_iterable(tensor_slices)

            for leaf in tree.leaves:
                tree = tree.add_subtree(tensor_indices, *leaf, uniquify_ids=True)

        trees_.append(tree)
    trees = tuple(trees_)

    if and_support:
        # FIXME: Currently assume that facet axis is inside the mixed one, this may
        # be wrong.
        if is_mixed:
            raise NotImplementedError("Might break")

        support_indices = op3.IndexTree(
            op3.Slice(
                "support",
                [op3.AffineSliceComponent("XXX")],
            )
        )
        trees_ = []
        for subtree in trees:
            tree = support_indices.add_subtree(subtree, *support_indices.leaf)
            trees_.append(tree)
        trees = tuple(trees_)

    # outer mixed bit
    if is_mixed:
        field_indices = op3.IndexTree(
            op3.Slice(
                "field",
                [op3.AffineSliceComponent(str(i)) for i, _ in enumerate(spaces)]
            )
        )
        tree = field_indices
        for leaf, subtree in op3.utils.checked_zip(field_indices.leaves, trees):
            tree = tree.add_subtree(subtree, *leaf, uniquify_ids=True)
    else:
        tree = op3.utils.just_one(trees)

    return tree


def _with_shape_axes(V, axes, target_paths, index_exprs, integral_type):
    axes = op3.AxisTree(axes.node_map)
    new_target_paths = dict(target_paths)
    new_index_exprs = dict(index_exprs)

    is_mixed = isinstance(V.topological, MixedFunctionSpace)
    if is_mixed:
        spaces = V.topological._spaces
        trees = (axes,) * len(spaces)
    else:
        spaces = (V.topological,)
        trees = (axes,)

    # Add tensor shape innermost, this applies to cells, edges etc equally
    trees_ = []
    for space, tree in zip(spaces, trees):
        if space.shape:
            for parent, component in tree.leaves:
                axis_list = [
                    op3.Axis({"XXX": dim}, f"dim{ii}")
                    for ii, dim in enumerate(space.shape)
                ]
                tree = tree.add_subtree(
                    op3.AxisTree.from_iterable(axis_list),
                    parent=parent,
                    component=component
                )
                for axis in axis_list:
                    new_target_paths[axis.id, "XXX"] = pmap({axis.label: "XXX"})
                    new_index_exprs[axis.id, "XXX"] = pmap({axis.label: op3.AxisVariable(axis.label)})

        trees_.append(tree)
    trees = tuple(trees_)

    if integral_type in {"exterior_facet", "interior_facet"}:
        arity = {"exterior_facet": 1, "interior_facet": 2}[integral_type]
        # FIXME: Currently assume that facet axis is inside the mixed one, this may
        # be wrong.
        if is_mixed:
            raise NotImplementedError("Might break")
        else:
            assert len(trees) == 1

        root = op3.Axis({"XXX": arity}, "support")
        support_indices = op3.AxisTree(root)
        trees_ = []
        for subtree in trees:
            tree = support_indices.add_subtree(subtree, *support_indices.leaf)
            trees_.append(tree)

        new_target_paths[root.id, "XXX"] = pmap({"support": "XXX"})
        new_index_exprs[root.id, "XXX"] = pmap({"support": op3.AxisVariable("support")})
        trees = tuple(trees_)

    # outer mixed bit
    if is_mixed:
        raise NotImplementedError("Need to add extra exprs as for shape above")
        field_indices = op3.AxisTree(
            op3.Axis(
                {str(i): 1 for i, _ in enumerate(spaces)},
                "field",
            )
        )
        tree = field_indices
        for leaf, subtree in op3.utils.checked_zip(field_indices.leaves, trees):
            tree = tree.add_subtree(subtree, *leaf, uniquify_ids=True)
    else:
        tree = op3.utils.just_one(trees)

    return tree, freeze(new_target_paths), freeze(new_index_exprs)


@functools.singledispatch
def _tensorify_axes(element, suffix=""):
    """

    FIAT numbering:

        1-----7-----3     v1----e3----v3
        |           |     |           |
        |           |     |           |
        4     8     5  =  e0    c0    e1
        |           |     |           |
        |           |     |           |
        0-----6-----2     v0----e2----v2

    therefore P3 standard DoF ordering (returned by packing closure):

        1---10--11--3
        |           |
        5   13  15  7
        |           |
        4   12  14  6
        |           |
        0---8---9---2

    which would have a data layout like:

        ╔═══════════════════╦═══════════════════╦════╗
        ║ v0 │ v1 │ v2 │ v3 ║ e0 │ e1 │ e2 │ e3 ║ c0 ║
        ╚═══════════════════╩═══════════════════╩════╝
             /                 /                   \
         ╔═════╗      ╔═══════════╗     ╔═══════════════════════╗
         ║ dof ║      ║ dof │ dof ║     ║ dof │ dof │ dof │ dof ║
         ╚═════╝      ╚═══════════╝     ╚═══════════════════════╝

    But it's a tensor product of intervals:

        1---9---13--5
        |           |
        3   11  15  7
        |           |
        2   10  14  6
        |           |
        0---8---12--4

    so it actually has a data layout like:

                        ╔═════════╦════╗
                        ║ v0 │ v1 ║ e0 ║
                        ╚═════════╩════╝
                         /            \
                   ╔═════╗          ╔═══════════╗
                   ║ dof ║          ║ dof │ dof ║
                   ╚═════╝          ╚═══════════╝
                      |                   |
              ╔═════════╦════╗     ╔═════════╦════╗
              ║ v0 │ v1 ║ e0 ║     ║ v0 │ v1 ║ e0 ║
              ╚═════════╩════╝     ╚═════════╩════╝
              /           |             |         \
       ╔═════╗      ╔═══════════╗    ╔═════╗    ╔═══════════╗
       ║ dof ║      ║ dof │ dof ║    ║ dof ║    ║ dof │ dof ║
       ╚═════╝      ╚═══════════╝    ╚═════╝    ╚═══════════╝

    We therefore want a temporary that is in the tensor-product form, but we
    need to be able to fill it from the "flat" closure that DMPlex gives us.

    """
    raise TypeError(f"No handler defined for {type(element).__name__}")


@_tensorify_axes.register(finat.TensorProductElement)
def _(element, suffix=""):
    subelem_axess = []
    for i, subelem in enumerate(element.factors):
        npoints = []
        subaxess = []
        for tdim, edofs in subelem.entity_dofs().items():
            npoints.append(len(edofs))
            ndofs = op3.utils.single_valued(len(d) for d in edofs.values())
            subaxes = op3.Axis(ndofs, f"dof{i}"+suffix)
            subaxess.append(subaxes)

        subelem_axes = op3.AxisTree.from_nest({
            op3.Axis(
                {str(tdim): n for tdim, n in enumerate(npoints)}, f"points{i}"+suffix
            ): subaxess
        })
        subelem_axess.append(subelem_axes)

    tensor_axes = _build_rec(subelem_axess)

    tensor_target_paths = _build_target_paths(tensor_axes, suffix=suffix)
    tensor_index_exprs = _tensor_product_index_exprs(tensor_axes, tensor_target_paths)

    return tensor_axes, tensor_target_paths, tensor_index_exprs


@_tensorify_axes.register(finat.EnrichedElement)
def _(element, suffix=""):
    root = op3.Axis((1,) * len(element.elements))
    axes = op3.AxisTree(root)
    target_paths = {}
    index_exprs = {}
    for component, subelem in zip(root.components, element.elements):
        subaxes, subtarget_paths, subindex_exprs = _tensorify_axes(subelem, suffix=suffix)
        axes = axes.add_subtree(subaxes, root, component)
        target_paths.update(subtarget_paths)
        index_exprs.update(subindex_exprs)

        key = (root.id, component.label)
        target_paths[key] = {root.label: component.label}
        index_exprs[key] = {root.label: op3.AxisVariable(root.label)}

    return axes, freeze(target_paths), freeze(index_exprs)


@_tensorify_axes.register(finat.HDivElement)
@_tensorify_axes.register(finat.HCurlElement)
def _(element, suffix=""):
    return _tensorify_axes(element.wrappee, suffix=suffix)


def _indexify_tensor_axes(
    axes,
    target_paths,
    index_exprs,
    *,
    axis=None,
    indices=None,
    suffix="",
):
    """Convert a tensor-product axis tree into a series of slices."""
    if axis is None:
        assert indices is None
        axis = axes.root
        indices = pmap()

    index_trees = []
    for component in axis.components:
        # This is bad as it unconditionally deconstructs the innermost axis
        for ci in range(component.count):
            indices_ = indices | {axis.label: ci}

            if subaxis := axes.child(axis, component):
                subindex_trees = _indexify_tensor_axes(
                    axes,
                    target_paths,
                    index_exprs,
                    axis=subaxis,
                    indices=indices_,
                    suffix=suffix,
                )
                index_trees.extend(subindex_trees)
            else:
                target_path = target_paths[axis.id, component.label]
                index_expr = index_exprs[axis.id, component.label]

                target_indices = {}
                evaluator = ExpressionEvaluator(indices_, ())
                for ax, expr in index_expr.items():
                    target_indices[ax] = evaluator(expr)

                index_tree = (
                    op3.AffineSliceComponent(
                        target_path["closure"+suffix],
                        target_indices["closure"],
                        target_indices["closure"]+1
                    ),
                    op3.Slice(
                        "dof"+suffix,
                        [
                            op3.AffineSliceComponent(
                                target_path["dof"+suffix],
                                target_indices["dof"],
                                target_indices["dof"]+1,
                            )
                        ],
                    ),
                )
                index_trees.append(index_tree)
    return tuple(index_trees)


# this is easy, the target path is just summing things up!
def _build_target_paths(_axes, _axis=None, tdim=0, suffix=""):
    if _axis is None:
        _axis = _axes.root

    paths = {}
    for component in _axis.components:
        if not (_axis.label.startswith("dof") or _axis.label.startswith("dim")):
            tdim_ = tdim + int(component.label)
        else:
            tdim_ = tdim
        if subaxis := _axes.child(_axis, component):
            paths.update(_build_target_paths(_axes, subaxis, tdim_, suffix=suffix))
        else:
            paths[_axis.id, component.label] = {"closure"+suffix: str(tdim_), "dof"+suffix: "XXX"}
    return paths


def _build_rec(_axess):
    _axes, *_subaxess = _axess

    if _subaxess:
        for leaf in _axes.leaves:
            subtree = _build_rec(_subaxess)
            _axes = _axes.add_subtree(subtree, *leaf, uniquify_ids=True)
    return _axes


def _tensor_product_index_exprs(tensor_axes, target_paths):
    index_exprs = collections.defaultdict(dict)

    # first points
    point_axess = _flatten_tensor_axes_points_only(tensor_axes)
    ndims = max(point_axess.keys())
    leaves_per_dim = [[] for _ in range(ndims+1)]
    for leaf in tensor_axes.leaves:
        dim = int(target_paths[leaf[0].id, leaf[1]]["closure"])
        leaves_per_dim[dim].append(leaf)

    leaf_iters = [iter(lvs) for lvs in leaves_per_dim]
    for dim, point_axes in point_axess.items():
        for pleaf in point_axes.leaves:
            ppath = point_axes.path(*pleaf)
            playout = point_axes.layouts[ppath]

            leaf = next(leaf_iters[dim])
            index_exprs[leaf[0].id, leaf[1]]["closure"] = playout

    for leaf_iter in leaf_iters:
        assert_empty(leaf_iter)

    # then DoFs
    dof_axess = _flatten_tensor_axes_dofs_only(tensor_axes)
    leaf_iter = iter(tensor_axes.leaves)

    for dof_axes in dof_axess:
        dpath = dof_axes.path(*dof_axes.leaf)
        dlayout = dof_axes.layouts[dpath]

        # I think that we don't need to do the dim-catching above, this isn't
        # strictly necessary
        leaf = next(leaf_iter)
        index_exprs[leaf[0].id, leaf[1]]["dof"] = dlayout
    assert_empty(leaf_iter)

    return index_exprs


def _flatten_tensor_axes_points_only(tensor_axes, axis=None, is_point_axis=True, tdim_acc=0):
    if axis is None:
        axis = tensor_axes.root

    if is_point_axis:
        assert axis.label.startswith("point")

        tree_infos = collections.defaultdict(list)
        for component in axis.components:
            tdim_acc_ = tdim_acc + int(component.label)

            subaxis = tensor_axes.child(axis, component)
            assert subaxis is not None, "Final axis should be for DoFs"

            subtrees = _flatten_tensor_axes_points_only(
                tensor_axes, subaxis, False, tdim_acc_
            )
            for tdim, subtree in subtrees.items():
                tree_infos[tdim].append((component, subtree))

        # now combine the trees
        trees = {}
        for tdim, tree_info in tree_infos.items():
            root = op3.Axis([c for c, _ in tree_info], axis.label)
            tree = op3.AxisTree(root)
            for component, subtree in tree_info:
                tree = tree.add_subtree(subtree, root, component)
            trees[tdim] = tree
        return trees

    else:
        assert axis.label.startswith("dof")

        component = op3.utils.just_one(axis.components)
        subaxis = tensor_axes.child(axis, component)
        # nasty trick to handle shape
        if subaxis and subaxis.label.startswith("point"):
            return _flatten_tensor_axes_points_only(tensor_axes, subaxis, True, tdim_acc)
        else:
            return {tdim_acc: op3.AxisTree()}


def _flatten_tensor_axes_dofs_only(tensor_axes, axis=None, is_dof_axis=False, dof_axes_acc=()):
    """

    Return an `pyop3.AxisTree` for each leaf of ``tensor_axes``.

    """
    if axis is None:
        axis = tensor_axes.root

    if is_dof_axis:
        assert axis.label.startswith("dof")

        dof_axes_acc_ = dof_axes_acc + (axis,)

        component = op3.utils.just_one(axis.components)
        subaxis = tensor_axes.child(axis, component)
        if subaxis and subaxis.label.startswith("point"):
            return _flatten_tensor_axes_dofs_only(tensor_axes, subaxis, False, dof_axes_acc_)
        else:
            return (op3.AxisTree.from_iterable(dof_axes_acc_),)

    else:
        # assert axis.label.startswith("points")

        dof_axess = []
        for component in axis.components:
            subaxis = tensor_axes.child(axis, component)
            assert subaxis is not None, "Final axis should be for DoFs"

            # nasty trick to catch extra shape
            if subaxis.label.startswith("dof"):
                dof_axess.extend(
                    _flatten_tensor_axes_dofs_only(
                        tensor_axes, subaxis, True, dof_axes_acc
                    )
                )
        return tuple(dof_axess)


@functools.cache
def _entity_permutations(V):
    mesh = V.mesh().topology
    elem = V.finat_element

    perm_dats = []
    for dim in range(mesh.dimension+1):
        # take the zeroth entry because they are all the same
        perms = elem.entity_permutations[dim][0]
        nperms = len(perms)
        perm_size = len(perms[0])
        perms_concat = np.empty((nperms, perm_size), dtype=IntType)
        for ip, perm in perms.items():
            perms_concat[ip] = perm

        # the inner label needs to match here for codegen to resolve
        axes = op3.AxisTree.from_iterable([op3.Axis(nperms), op3.Axis({"XXX": perm_size}, "dof")])
        # dat = op3.HierarchicalArray(axes, data=perms_concat.flatten(), constant=True, prefix=)
        dat = op3.HierarchicalArray(axes, data=perms_concat.flatten(), prefix="perm")
        perm_dats.append(dat)
    return tuple(perm_dats)


def _orientations(mesh, perms, cell, integral_type):
    pkey = pmap({mesh.name: mesh.cell_label})
    closure_dats = mesh._fiat_closure.connectivity[pkey]
    orientations = mesh.entity_orientations_dat

    subsets = []
    subtrees = []
    for dim in range(mesh.dimension+1):
        perms_ = perms[dim]

        mymap = closure_dats[dim]
        subset = op3.AffineSliceComponent(mymap.target_component, label=mymap.target_component)
        subsets.append(subset)

        # Attempt to not relabel the interior axis, is this the right approach?
        all_bits = op3.Slice("closure", [op3.AffineSliceComponent(str(dim), label=str(dim))], label="closure")

        # FIXME (NEXT): If we have interior facets then this index is a facet, not a cell
        # How do we get the cell from this? Just the support?
        if integral_type == "cell":
            inner_subset = orientations[dim][cell, all_bits]
        else:
            assert integral_type in {"exterior_facet", "interior_facet"}
            inner_subset = orientations[dim][mesh.support(cell), all_bits]

        # I am struggling to index this...
        # perm = perms_[inner_subset]
        (root_label, root_clabel), (leaf_label, leaf_clabel) = op3.utils.just_one(perms_.axes.ordered_leaf_paths)

        source_path = inner_subset.axes.path_with_nodes(*inner_subset.axes.leaf)
        index_keys = [None] + [
            (axis.id, cpt) for axis, cpt in source_path.items()
        ]
        target_path = op3.utils.merge_dicts(
            inner_subset.axes.target_paths.get(key, {}) for key in index_keys
        )
        myindices = op3.utils.merge_dicts(
            inner_subset.axes.index_exprs.get(key, {}) for key in index_keys
        )
        inner_subset_var = ArrayVar(inner_subset, myindices, target_path)

        mypermindices = (
            op3.ScalarIndex(root_label, root_clabel, inner_subset_var),
            op3.Slice(leaf_label, [op3.AffineSliceComponent(leaf_clabel)]),
        )
        perm = perms_[mypermindices]

        subtree = op3.Slice("dof", [op3.Subset("XXX", perm, label="XXX")], label="dof")
        subtrees.append(subtree)

    myroot = op3.Slice("closure", subsets, label="closure")
    mychildren = {myroot.id: subtrees}
    mynodemap = {None: (myroot,)}
    mynodemap.update(mychildren)
    return op3.IndexTree(mynodemap)
