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
from pyop2.caching import serial_cache
from pyop3.expr_visitors import evaluate as eval_expr
from pyop3.itree.tree import compose_axes
from pyop3.utils import readonly, invert as invert_permutation
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
from firedrake.utils import IntType, assert_empty, tuplify


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
          A[i] = fmax(A[i], B[0])
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


@pack_pyop3_tensor.register(op3.Dat)
def _(
    array: op3.Dat,
    V: WithGeometry,
    index: op3.LoopIndex,
    integral_type: str,
):
    plex = V. mesh().topology

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


    # handle entity_dofs - this is done treating all nodes as equivalent so we have to
    # discard shape beforehand
    dof_numbering = _flatten_entity_dofs(V.finat_element.entity_dofs())
    # breakpoint()
    perm = invert_permutation(dof_numbering)

    # skip if identity
    if not np.all(perm == np.arange(perm.size, dtype=IntType)):
        perm_buffer = op3.ArrayBuffer(perm, constant=True)
        perm_dat = op3.Dat(V._packed_nodal_axes.root.copy(label="mylabel"), perm_buffer, prefix="p")
        perm_subset = op3.Slice("nodes_flat", [op3.Subset("XXX", perm_dat)], label="mylabel")
        indexed = indexed.reshape(V._packed_nodal_axes)[perm_subset]

    if plex.ufl_cell() == ufl.hexahedron:
        raise NotImplementedError
        # FUSE TODO this should happen for all cells
        perms = _entity_permutations(V)
        mytree = _orientations(plex, perms, index, integral_type)
        mytree = _with_shape_indices(V, mytree, integral_type in {"exterior_facet", "interior_facet"})

        indexed = indexed.getitem(mytree, strict=True)

    return indexed


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

    indexed = mat.getitem(rmap, cmap, strict=True)

    row_perm = _flatten_entity_dofs(Vrow.finat_element.entity_dofs())
    row_perm = invert_permutation(row_perm)
    col_perm = _flatten_entity_dofs(Vcol.finat_element.entity_dofs())
    col_perm = invert_permutation(col_perm)

    # skip if identity
    if (
        np.any(row_perm != np.arange(row_perm.size, dtype=IntType))
        or np.any(col_perm != np.arange(col_perm.size, dtype=IntType))
    ):
        # NOTE: This construction is horrible
        row_perm_buffer = op3.ArrayBuffer(row_perm, constant=True)
        row_perm_dat = op3.Dat(
            Vrow._packed_nodal_axes.root.copy(label="mylabel"),
            buffer=row_perm_buffer,
        )
        row_perm_subset = op3.Slice("nodes_flat", [op3.Subset("XXX", row_perm_dat)], label="mylabel")

        col_perm_buffer = op3.ArrayBuffer(col_perm, constant=True)
        col_perm_dat = op3.Dat(
            Vcol._packed_nodal_axes.root.copy(label="mylabel"),
            buffer=col_perm_buffer,
        )
        col_perm_subset = op3.Slice("nodes_flat", [op3.Subset("XXX", col_perm_dat)], label="mylabel")

        if Vrow.shape or Vcol.shape:
            raise NotImplementedError("Not considering any extra axes here")

        # index_tree = op3.IndexTree.from_iterable([row_perm_subset, col_perm_subset])

        if Vrow.shape or Vcol.shape:
            raise NotImplementedError("need index tree")

        indexed = indexed.reshape(Vrow._packed_nodal_axes, Vcol._packed_nodal_axes)[row_perm_subset, col_perm_subset]

    if plex.ufl_cell() is ufl.hexahedron:
        raise NotImplementedError

    return indexed

def _cell_integral_pack_indices(V: WithGeometry, cell: op3.LoopIndex) -> op3.IndexTree:
    plex = V.mesh().topology

    indices = op3.IndexTree.from_nest({
        plex._fiat_closure(cell): [
            op3.Slice(f"dof{d}", [op3.AffineSliceComponent("XXX")])
            for d in range(plex.dimension+1)
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


# TODO: This is absolutely awful - need to traverse "canonical" function space axis tree
# and build slices as appropriate
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
        for leaf, subtree in zip(field_indices.leaves, trees, strict=True):
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
        # dat = op3.Dat(axes, data=perms_concat.flatten(), constant=True, prefix=)
        dat = op3.Dat(axes, data=perms_concat.flatten(), prefix="perm")
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

        # source_path = inner_subset.axes.path_with_nodes(*inner_subset.axes.leaf)
        # index_keys = [None] + [
        #     (axis.id, cpt) for axis, cpt in source_path.items()
        # ]
        # target_path = op3.utils.merge_dicts(
        #     inner_subset.axes.target_paths.get(key, {}) for key in index_keys
        # )
        # myindices = op3.utils.merge_dicts(
        #     inner_subset.axes.index_exprs.get(key, {}) for key in index_keys
        # )
        # inner_subset_var = ArrayVar(inner_subset, myindices, target_path)
        inner_subset_var = inner_subset

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


def _entity_dofs_hashkey(entity_dofs: dict) -> tuple:
    """Provide a canonical key for FInAT ``entity_dofs``."""
    hashkey = []
    for k in sorted(entity_dofs.keys()):
        sub_key = [k]
        for sk in sorted(entity_dofs[k]):
            sub_key.append(tuple(entity_dofs[k][sk]))
        hashkey.append(tuple(sub_key))
    return tuple(hashkey)


# NOTE: This needs inverting!
@serial_cache(_entity_dofs_hashkey)
def _flatten_entity_dofs(entity_dofs):
    """Flatten FInAT element ``entity_dofs`` into an array."""
    flat_entity_dofs = []
    for dim in sorted(entity_dofs.keys()):
        num_entities = len(entity_dofs[dim])
        for entity_num in range(num_entities):
            dofs = entity_dofs[dim][entity_num]
            flat_entity_dofs.extend(dofs)
    flat_entity_dofs = np.asarray(flat_entity_dofs, dtype=IntType)
    return readonly(flat_entity_dofs)
