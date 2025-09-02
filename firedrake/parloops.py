r"""This module implements parallel loops reading and writing
:class:`.Function`\s. This provides a mechanism for implementing
non-finite element operations such as slope limiters."""
import collections
import functools
import warnings
from cachetools import LRUCache
from immutabledict import immutabledict as idict
from typing import Any

import finat
import loopy
import numpy as np
import pyop3 as op3
import ufl
from pyop2.caching import serial_cache
from pyop3 import READ, WRITE, RW, INC
from pyop3.expr.visitors import evaluate as eval_expr
from pyop3.utils import readonly, invert as invert_permutation
from pyrsistent import freeze, pmap
from ufl.indexed import Indexed
from ufl.domain import join_domains

from firedrake import constant, utils
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


__all__ = ['par_loop', 'direct']


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
        'itspace': indirect_measure
    },
    'interior_facet': {
        'itspace': indirect_measure
    },
    'exterior_facet': {
        'nodes': lambda x: x.exterior_facet_node_map(),
        'itspace': indirect_measure
    },
    'direct': {
        'itspace': lambda mesh, measure: mesh
    }
}
r"""Map a measure to the correct maps."""


def _form_loopy_kernel(kernel_domains, instructions, measure, args, **kwargs) -> op3.Function:
    intents = []
    kargs = []
    for var, (func, intent) in args.items():
        is_input = intent in [INC, READ, RW]
        is_output = intent in [INC, RW, WRITE]
        if isinstance(func, constant.Constant):
            if intent is not READ:
                raise RuntimeError("Only READ access is allowed to Constant")
            # Constants modelled as Globals, so no need for double
            # indirection
            ndof = func.function_space().value_size
            kargs.append(loopy.GlobalArg(var, dtype=func.dat.dtype, shape=(ndof,), is_input=is_input, is_output=is_output))
        else:
            # Do we have a component of a mixed function?
            if isinstance(func, Indexed):
                c, i = func.ufl_operands
                idx = i._indices[0]._value
                ndof = c.function_space()[idx].finat_element.space_dimension()
                cdim = c.function_space()[idx].value_size
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
                    cdim = func.function_space().value_size
                    dtype = func.dat.dtype
            if measure.integral_type() == 'interior_facet':
                ndof *= 2
            # FIXME: shape for facets [2][ndof]?
            kargs.append(loopy.GlobalArg(var, dtype=dtype, shape=(ndof, cdim), is_input=is_input, is_output=is_output))
        kernel_domains = kernel_domains.replace(var+".dofs", str(ndof))

        intents.append(intent)

    if kernel_domains == "":
        kernel_domains = "[] -> {[]}"
    try:
        key = (kernel_domains, tuple(instructions), tuple(map(tuple, kwargs.items())))
        # Add shape, dtype and intent to the cache key
        for func, intent in args.values():
            if isinstance(func, Indexed):
                for dat in func.ufl_operands[0].dat.split:
                    key += (dat.axes, dat.dtype, intent)
            else:
                key += (func.dat.axes, func.dat.dtype, intent)
        return kernel_cache[key]
    except KeyError:
        kargs.append(...)
        knl = loopy.make_kernel(kernel_domains, instructions, kargs, name="par_loop_kernel", target=target,
                                  seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"])
        knl = op3.Function(knl, intents)
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
    warnings.warn("par_loop is no longer necessary - prefer to use pyop3 directly", FutureWarning)

    # catch deprecated C-string parloops
    if isinstance(kernel, str):
        raise TypeError("C-string kernels are no longer supported by Firedrake parloops")
    if "is_loopy_kernel" in kwargs:
        if kwargs.pop("is_loopy_kernel"):
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
    function = _form_loopy_kernel(kernel_domains, instructions, measure, args, **kernel_kwargs)

    iterset = _map['itspace'](mesh, measure)
    index = iterset.index()

    def mkarg(f):
        if isinstance(f, Indexed):
            raise NotImplementedError("Think about this")
            c, i = f.ufl_operands
            idx = i._indices[0]._value
            m = _map['nodes'](c)
            return pack_pyop3_tensor(c.dat, index, measure.integral_type())
        return pack_tensor(f, index, measure.integral_type())
    args = tuple(mkarg(arg) for arg, _ in args.values())

    op3.do_loop(index, function(*args))


@functools.singledispatch
def pack_tensor(tensor: Any, index: op3.LoopIndex, integral_type: str, **kwargs):
    raise TypeError(f"No handler defined for {type(tensor).__name__}")


@pack_tensor.register(Function)
@pack_tensor.register(Cofunction)
@pack_tensor.register(CoordinatelessFunction)
def _(func, index: op3.LoopIndex, integral_type: str, *, target_mesh=None):
    return pack_pyop3_tensor(
        func.dat, func.function_space(), index, integral_type, target_mesh=target_mesh
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
    *,
    target_mesh=None
):
    if target_mesh is None:
        target_mesh = V.mesh()

    if V.mesh() != target_mesh:
        index = target_mesh.cell_parent_cell_map(index)

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

    indexed = array[pack_indices]

    # TODO: use maybe_permute_packed_tensor here

    plex = V.mesh().topology  # used?

    # handle entity_dofs - this is done treating all nodes as equivalent so we have to
    # discard shape beforehand
    subaxes = []
    sub_perms = []
    permutation_needed = False
    for subspace in V:
        dof_numbering = _flatten_entity_dofs(subspace.finat_element.entity_dofs())
        perm = invert_permutation(dof_numbering)

        # skip if identity
        if not np.all(perm == np.arange(perm.size, dtype=IntType)):
            permutation_needed = True

        perm_dat = op3.Dat.from_array(perm, prefix="perm", buffer_kwargs={"constant": True})
        perm_axis = perm_dat.axes.root
        perm_subset = op3.Slice(perm_axis.label, [op3.Subset(perm_axis.component.label, perm_dat)])
        sub_perms.append(perm_subset)

        indexed_axes = op3.AxisTree.from_iterable([perm_axis, *subspace.shape])
        subaxes.append(indexed_axes)

    if permutation_needed:
        if len(V) > 1:
            indexed_axes = op3.AxisTree(indexed.axes.root)
            for subspace, subtree in zip(V, subaxes, strict=True):
                indexed_axes = indexed_axes.add_subtree(idict({"field": subspace.index}), subtree)

            field_slice = op3.Slice("field", [op3.AffineSliceComponent(subspace.index, label=subspace.index) for subspace in V], label="field")
            index_tree = op3.IndexTree.from_nest({field_slice: sub_perms})

            # indexed = indexed.reshape(indexed_axes)[:, sub_perms]
            indexed = indexed.reshape(indexed_axes)[index_tree]
        else:
            # TODO: Should be able to just pass a Dat here and have it DTRT
            indexed_axes = utils.just_one(subaxes)
            perm_subset = utils.just_one(sub_perms)
            indexed = indexed.reshape(indexed_axes)[perm_subset]

    if plex.ufl_cell() == ufl.hexahedron:
        raise NotImplementedError
        perms = _entity_permutations(V)
        mytree = _orientations(plex, perms, index, integral_type)
        mytree = _with_shape_indices(V, mytree, integral_type in {"exterior_facet", "interior_facet"})

        indexed = indexed.getitem(mytree, strict=True)

    return indexed


@pack_pyop3_tensor.register(op3.Mat)
def _(
    mat: op3.Mat,
    Vrow: WithGeometry,
    Vcol: WithGeometry,
    index: op3.LoopIndex,
    integral_type: str
):
    if integral_type not in {"cell", "interior_facet", "exterior_facet"}:
        raise NotImplementedError("TODO")

    if mat.buffer.mat_type == "python":
        mat_context = mat.buffer.mat.getPythonContext()
        if isinstance(mat_context, op3.RowDatPythonMatContext):
            space = Vrow
        else:
            assert isinstance(mat_context, op3.ColumnDatPythonMatContext)
            space = Vcol
        dat = mat_context.dat
        return pack_pyop3_tensor(dat, space, index, integral_type)

    if any(fs.mesh().ufl_cell() is ufl.hexahedron for fs in {Vrow, Vcol}):
        raise NotImplementedError

    # First collect the DoFs in the cell closure in FIAT order.
    if integral_type == "cell":
        rmap = _cell_integral_pack_indices(Vrow, index)
        cmap = _cell_integral_pack_indices(Vcol, index)
    elif integral_type in {"exterior_facet", "interior_facet"}:
        rmap = _facet_integral_pack_indices(Vrow, index)
        cmap = _facet_integral_pack_indices(Vcol, index)
    else:
        raise NotImplementedError

    indexed = mat[rmap, cmap]

    return maybe_permute_packed_tensor(indexed, Vrow.finat_element, Vcol.finat_element, Vrow.shape, Vcol.shape)


@functools.singledispatch
def maybe_permute_packed_tensor(tensor: op3.Tensor, *spaces) -> op3.Tensor:
    raise TypeError


@maybe_permute_packed_tensor.register
def _(packed_dat: op3.Dat, element, shape):
    perm = _flatten_entity_dofs(element.entity_dofs())
    perm = invert_permutation(perm)
    # skip if identity
    if np.any(perm != np.arange(perm.size, dtype=IntType)):
        row_perm_dat = op3.Dat.from_array(perm, prefix="perm", buffer_kwargs={"constant": True})
        row_perm_axis = row_perm_dat.axes.root
        row_perm_subset = op3.Slice(row_perm_axis.label, [op3.Subset(row_perm_axis.component.label, row_perm_dat)])

        indexed_row_axes = op3.AxisTree.from_iterable([row_perm_axis, *shape])

        packed_dat = packed_dat.reshape(indexed_row_axes)[row_perm_subset]
    return packed_dat


@maybe_permute_packed_tensor.register
def _(packed_mat: op3.Mat, row_element, column_element, row_shape, column_shape):
    row_perm = _flatten_entity_dofs(row_element.entity_dofs())
    row_perm = invert_permutation(row_perm)
    col_perm = _flatten_entity_dofs(column_element.entity_dofs())
    col_perm = invert_permutation(col_perm)

    # skip if identity
    if (
        np.any(row_perm != np.arange(row_perm.size, dtype=IntType))
        or np.any(col_perm != np.arange(col_perm.size, dtype=IntType))
    ):
        row_perm_dat = op3.Dat.from_array(row_perm, prefix="perm", buffer_kwargs={"constant": True})
        row_perm_axis = row_perm_dat.axes.root
        row_perm_subset = op3.Slice(row_perm_axis.label, [op3.Subset(row_perm_axis.component.label, row_perm_dat)])

        col_perm_dat = op3.Dat.from_array(col_perm, prefix="perm", buffer_kwargs={"constant": True})
        col_perm_axis = col_perm_dat.axes.root
        col_perm_subset = op3.Slice(col_perm_axis.label, [op3.Subset(col_perm_axis.component.label, col_perm_dat)])

        indexed_row_axes = op3.AxisTree.from_iterable([row_perm_axis, *row_shape])
        indexed_col_axes = op3.AxisTree.from_iterable([col_perm_axis, *column_shape])

        packed_mat = packed_mat.reshape(indexed_row_axes, indexed_col_axes)[row_perm_subset, col_perm_subset]

    return packed_mat

def _cell_integral_pack_indices(V: WithGeometry, cell: op3.LoopIndex) -> op3.IndexTree:
    if len(V) > 1:
        return (slice(None), V._mesh.closure(cell))
    else:
        return V._mesh.closure(cell)


def _facet_integral_pack_indices(V: WithGeometry, facet: op3.LoopIndex) -> op3.IndexTree:
    mesh = V._mesh
    if len(V) > 1:
        return (slice(None), mesh.closure(mesh.support(facet)))
    else:
        return mesh.closure(mesh.support(facet))


@serial_cache(lambda fs: fs.finat_element)
def _entity_permutations(fs: WithGeometry):
    mesh = fs.mesh().topology
    elem = fs.finat_element

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
