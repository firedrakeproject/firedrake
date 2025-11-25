r"""This module implements parallel loops reading and writing
:class:`.Function`\s. This provides a mechanism for implementing
non-finite element operations such as slope limiters."""
import collections
import functools
import warnings
from cachetools import LRUCache
from immutabledict import immutabledict as idict
from typing import Any

import FIAT
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


# TODO: rename to pack_tensor, and return tuple of instructions
# TODO: Actually don't do that, pass indices in...
@functools.singledispatch
def pack_pyop3_tensor(tensor: Any, *args, **kwargs):
    raise TypeError(f"No handler defined for {type(tensor).__name__}")


@pack_pyop3_tensor.register(op3.Dat)
def _(
    dat: op3.Dat,
    V: WithGeometry,
    loop_index: op3.LoopIndex,
    integral_type: str,
    *,
    target_mesh=None
):
    """
    Consider:

    t0 = dat[closure(cell)]
    t1 = f(t0)
    kernel(t1)
    t2 = f^{-1}(t1)
    dat[closure(cell)] += t2

    To coordinate this we have to provide 'dat[closure(cell)]' and we need the instructions back along with t1 and t2 (not t0)
    We use the terminology of 'local' and 'global' representations of the packed data.

    """
    if target_mesh is None:
        target_mesh = V.mesh()

    if V.mesh() != target_mesh:
        index = target_mesh.cell_parent_cell_map(index)

    mesh = V.mesh()

    if len(V) > 1:
        # do a loop
        raise NotImplementedError
        # This is tricky. Consider the case where you have a mixed space with hexes and
        # each space needs a different (non-permutation) transform. That means that we
        # have to generate code like:
        #
        # t0 = dat[:, closure(cell)]
        # t1 = transform0(t0[0])  # (field 0)
        # t2 = transform1(t0[1])  # (field 1)
        # t3[0] = t1
        # t3[1] = t2
        #
        # I think that the easiest approach here is to index the full thing before passing it
        # down. We can then combine everything at the top-level

    if integral_type == "cell":
        cell = loop_index
        packed_dat = dat[mesh.closure(cell)]
        depth = 0
    elif integral_type in {"interior_facet", "exterior_facet"}:
        facet = loop_index
        cell = mesh.support(facet)
        packed_dat = dat[mesh.closure(cell)]
        depth = 1
    else:
        raise NotImplementedError

    return transform_packed_cell_closure_dat(packed_dat, V, cell, depth=depth)


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

    if Vrow.mesh() is not Vcol.mesh():
        raise NotImplementedError("Think we need to have different loop indices for row+col")

    if any(fs.mesh().ufl_cell() == ufl.hexahedron for fs in {Vrow, Vcol}):
        raise NotImplementedError

    if integral_type == "cell":
        cell = index
        depth = 0
    elif integral_type in {"interior_facet", "exterior_facet"}:
        facet = index
        cell = Vrow.mesh().support(facet)
        depth = 1
    else:
        raise NotImplementedError

    packed_mat = mat[Vrow.mesh().closure(cell), Vcol.mesh().closure(cell)]
    return transform_packed_cell_closure_mat(packed_mat, Vrow, Vcol, cell, row_depth=depth, column_depth=depth)


def transform_packed_cell_closure_dat(packed_dat: op3.Dat, space, loop_index: op3.LoopIndex, *, depth: int = 0):
    # Do this before the DoF transformations because this occurs at the level of entities, not nodes
    # TODO: Can be more fussy I think, only higher degree?
    # NOTE: This is now a special case of the fuse stuff below
    # if space.ufl_element().cell == ufl.hexahedron:
    #     perms = _entity_permutations(space)
    #     orientation_perm = _orientations(space, perms, loop_index)
    #     dat_sequence[-1] = dat_sequence[-1][*(slice(None),)*depth, orientation_perm]

    transform_in_kernel, transform_out_kernel = fuse_orientations([space])

    if packed_dat.dtype == IntType:
        warnings.warn("Int Type dats cannot be transformed using fuse transforms")
    elif transform_in_kernel and transform_out_kernel:
        orientations = space.mesh().entity_orientations_dat

        mat_work_array = op3.Dat.null(op3.AxisTree.from_iterable([packed_dat.size, packed_dat.size]), dtype=utils.ScalarType, prefix="trans")

        def transform_in(untransformed, transformed):
            return (
                transform_in_kernel(orientations[loop_index], mat_work_array, untransformed, transformed),
            )

        def transform_out(transformed, untransformed):
            return (
                transform_out_kernel(orientations[loop_index], mat_work_array, transformed, untransformed),
            )

        transform = op3.OutOfPlaceTensorTransform(packed_dat, transform_in, transform_out)
        temp = packed_dat.materialize()
        packed_dat = temp.__record_init__(_parent=transform)

        """Some old exposition:

        Consider the general case:

        t0 <- dat[f(p)]
        t1 <- g(t0)
        func(t1)
        t2 <- ginv(t1)
        dat[f(p)] <- t2

        but with READ:

            t0 <- dat[f(p)]
            t1 <- g(t0)
            func(t1)

        and INC:

            t1 <- 0
            func(t1)
            t2 <- ginv(t1)
            dat[f(p)] += t2
        """

    # /THIS IS NEW

    if _needs_static_permutation(space.finat_element):
        nodal_axis_tree, dof_perm_slice = _static_node_permutation_slice(packed_dat.axes, space, depth)
        packed_dat = packed_dat.reshape(nodal_axis_tree)[dof_perm_slice]

    return packed_dat


def transform_packed_cell_closure_mat(packed_mat: op3.Mat, row_space, column_space, cell_index: op3.Index, *, row_depth=0, column_depth=0):
    mat_sequence = [packed_mat]

    row_element = row_space.finat_element
    column_element = column_space.finat_element
    
    transform_in_kernel, transform_out_kernel = fuse_orientations([row_space, column_space])

    # Do this before the DoF transformations because this occurs at the level of entities, not nodes
    # TODO: Can be more fussy I think, only higher degree?
    if utils.single_valued(space.ufl_element().cell == ufl.hexahedron for space in {row_space, column_space}):
        row_orientation_perm = _orientations(row_space, _entity_permutations(row_space), cell_index)
        column_orientation_perm = _orientations(column_space, _entity_permutations(column_space), cell_index)
        row_perm = [*(slice(None),)*row_depth, row_orientation_perm]
        column_perm = [*(slice(None),)*column_depth, column_orientation_perm]
        mat_sequence[-1] = mat_sequence[-1][row_perm, column_perm]

    if _needs_static_permutation(row_space.finat_element) or _needs_static_permutation(column_space.finat_element):
        row_nodal_axis_tree, row_dof_perm_slice = _static_node_permutation_slice(packed_mat.row_axes, row_space, row_depth)
        column_nodal_axis_tree, column_dof_perm_slice = _static_node_permutation_slice(packed_mat.column_axes, column_space, column_depth)
        mat_sequence[-1] = mat_sequence[-1].reshape(row_nodal_axis_tree, column_nodal_axis_tree)[row_dof_perm_slice, column_dof_perm_slice]

    assert len(mat_sequence) % 2 == 1, "Must have an odd number"
    # I want to return a 'PackUnpackKernelArg' type that has information
    # about how to transform something before and after passing to a function. We can then defer
    # emitting these instructions until the intent information dicates that it is needed.
    if len(mat_sequence) > 1:
        # need to have sequential assignments I think
        raise NotImplementedError
    return mat_sequence[len(mat_sequence) // 2]


@serial_cache(lambda fs: fs.finat_element)
def _entity_permutations(fs: WithGeometry):
    mesh = fs.mesh().topology
    elem = fs.finat_element

    perm_dats = []
    for dim in range(mesh.dimension+1):
        perms = utils.single_valued(elem.entity_permutations[dim].values())
        nperms = len(perms)
        perm_size = utils.single_valued(map(len, perms.values()))
        perms_concat = np.empty((nperms, perm_size), dtype=IntType)
        for ip, perm in perms.items():
            perms_concat[ip] = perm

        # the inner label needs to match here for codegen to resolve
        axes = op3.AxisTree.from_iterable([op3.Axis(nperms), op3.Axis({"XXX": perm_size}, "dof")])
        # dat = op3.Dat(axes, data=perms_concat.flatten(), constant=True, prefix=)
        dat = op3.Dat(axes, data=perms_concat.flatten(), prefix="perm")
        perm_dats.append(dat)
    return tuple(perm_dats)


def _orientations(space, perms, cell):
    mesh = space.mesh()
    pkey = pmap({mesh.name: mesh.cell_label})
    # closure_dats = mesh._fiat_closure.connectivity[pkey]
    orientations = mesh.entity_orientations_dat

    subsets = []
    subtrees = []
    for dim in range(mesh.dimension+1):
        perms_ = perms[dim]

        # mymap = closure_dats[dim]
        subset = op3.AffineSliceComponent(dim, label=dim)
        subsets.append(subset)

        # Attempt to not relabel the interior axis, is this the right approach?
        all_bits = op3.Slice("closure", [op3.AffineSliceComponent(dim, label=dim)], label="closure")

        # the orientations for these entities
        inner_subset = orientations[cell, all_bits]

        root = perms_.axes.root
        inner_subset_really = op3.ScalarIndex(root.label, root.component.label, op3.as_linear_buffer_expression(inner_subset)),
        perm = perms_[inner_subset_really]

        perm = op3.as_linear_buffer_expression(perm)
        # assert isinstance(perm, op3.LinearDatBufferExpression)

        subtree = op3.Slice(f"dof{dim}", [op3.Subset("XXX", perm, label="XXX")], label=f"dof")  # I think the label must match 'perm'
        subtrees.append(subtree)

    myroot = op3.Slice("closure", subsets, label="closure")
    return op3.IndexTree.from_nest({myroot: subtrees})


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


def _static_node_permutation_slice(packed_axis_tree: op3.AxisTree, space: WithGeometry, depth: int) -> tuple[op3.AxisTree, tuple]:
    permutation = _node_permutation_from_element(space.finat_element)

    # TODO: Could be 'AxisTree.linear_to_depth()' or similar
    outer_axes = []
    path = idict()
    for _ in range(depth):
        outer_axis = packed_axis_tree.node_map[path]
        assert len(outer_axis.components) == 1
        outer_axes.append(outer_axis)

    nodal_axis = op3.Axis(permutation.size)
    nodal_axis_tree = op3.AxisTree.from_iterable([*outer_axes, nodal_axis, *space.shape])

    dof_perm_dat = op3.Dat(nodal_axis, data=permutation, prefix="perm", buffer_kwargs={"constant": True})
    dof_perm_slice = op3.Slice(
        nodal_axis.label,
        [op3.Subset(None, dof_perm_dat)],
    )

    return nodal_axis_tree, (*[slice(None)]*depth, dof_perm_slice)


@functools.cache
def _node_permutation_from_element(element) -> np.ndarray:
    return readonly(invert_permutation(_flatten_entity_dofs(element.entity_dofs())))


@functools.cache
def _needs_static_permutation(element) -> bool:
    perm = _node_permutation_from_element(element)
    return any(perm != np.arange(perm.size, dtype=perm.dtype))


def construct_switch_statement(self, mats, n, args, var_list):
    string = []
    string += "a = iden; \n "
    string += "\nswitch (dim) { \n"

    var_list += ["iden"]
    args += [loopy.TemporaryVariable("iden", initializer=np.identity(n), dtype=utils.ScalarType, read_only=True, address_space=loopy.AddressSpace(1))]

    closure_sizes = self._mesh._closure_sizes[self._mesh.dimension]
    closure_size_acc = 0
    indent = 0
    for dim in range(len(closure_sizes)):
        string += f"case {dim}:\n "
        indent += 1
        string += indent*"\t" + "o_val = o[i + closure_size_acc]; \n "
        #string += indent*"\t" + "o_val = 0; \n "
        string += [indent*"\t" + "switch (i) { \n"]
        indent += 1
        for i in range(closure_sizes[dim]):
            string += indent*"\t" + f"case {i}:\n "
            indent += 1
            string += indent*"\t" + "switch (o_val) { \n"
            indent += 1
            for val in sorted(mats[dim][i].keys()):
                string += indent*"\t" + f"case {val}:\n "
                indent += 1
                matname = f"mat{dim}_{i}_{val}"
                string += indent*"\t" + f"a = {matname};\n"
                string += indent*"\t" + "break;\n"
                var_list += [matname]
                mat = np.array(mats[dim][i][val], dtype=utils.ScalarType)
                args += [loopy.TemporaryVariable(matname, initializer=mat, dtype=utils.ScalarType, read_only=True, address_space=loopy.AddressSpace(1))]
                indent -= 1
            indent -= 1
            string += indent*"\t" + "default: break;}break;\n"
            indent -= 1
        string += indent*"\t" + "default: break; }\n"
        closure_size_acc += dim
        # string += indent*"\t" + "printf(\"res (before) %f, %f, %f\\n\", res[0], res[1], res[2]); \n"
        # string += indent*"\t" + "printf(\"b (before) %f, %f, %f\\n\", b[0], b[1], b[2]); \n"
        #string += indent*"\t" + "printf(\"d: %d, dim: %d, i: %d, o_val: %d \\n\", d, dim, i, o_val); \n"
        #for i in range(n):
        #    string += indent*"\t" + f"printf(\"a row {i}: {" ".join('%f' for i in range(n))}\\n\", {", ".join(f"a[{i*n + j}]" for j in range(n))}); \n"
        #string += indent*"\t" + f"printf(\"a row 1: {" ".join('%f' for i in range(n))}\\n\", {", ".join(f"a[{n + i}]" for i in range(n))}); \n"
        #string += indent*"\t" + f"printf(\"a row 2: {" ".join('%f' for i in range(n))}\\n\", {", ".join(f"a[{2*n + i}]" for i in range(n))}); \n"
        #string += indent*"\t" + "printf(\"a rows ...\\n\"); \n"
        # string += indent*"\t" + "printf(\"\\n\");\n"
        string += indent*"\t" + "\n"
        string += indent*"\t" + "break;\n"
        indent -= 1

    string += "default: break; }\n"
    return string, args, var_list

def get_utility_kernels(output_shape: int, n: int) -> tuple:
    if output_shape == 1:
        res_idx = "j"
        iter_idx = "i"
        combined = iter_idx + "," + res_idx
        res_shape = (n,)
    elif output_shape == 2:
        res_idx = "i,j"
        res_shape = (n, n)
    else:
        raise NotImplementedError("Fuse orientations cannot handle tensors")
    matmul = loopy.make_function(
        f"{{[{combined}]:0<={combined} < {n}}}",
        f"""
            res[{res_idx}] =  res[{res_idx}] + a[{combined}]*b[{iter_idx}]
        """, name="matmul", target=loopy.CWithGNULibcTarget())

    set_args = [loopy.GlobalArg("b", dtype=utils.ScalarType, shape=res_shape, is_input=True, is_output=True),
                loopy.GlobalArg("res", dtype=utils.ScalarType, shape=res_shape, is_input=True)]
    set_knl = loopy.make_function(
        f"{{[{res_idx}]:0<= {res_idx} < {n}}}",
        [f"b[{res_idx}] = res[{res_idx}]"],
        kernel_data=set_args,
        name="set", target=loopy.CWithGNULibcTarget()
    )
    return matmul, set_knl


def fuse_orientations(spaces: list[WithGeometry]):
    fuse_defined_spaces = all([hasattr(space.ufl_element(), "triple") for space in spaces])
    if not fuse_defined_spaces:
        return None, None

    # TODO case where only one has matrices? or do all fuse elements have them, check
    if fuse_defined_spaces and all([hasattr(space.ufl_element().triple, "matrices") for space in spaces]):
        output_shape = len(spaces)
        print("NEW FUSE", output_shape)
        fs = spaces[0]
        mats = fs.ufl_element().triple.matrices
        reversed_mats = fs.ufl_element().triple.reversed_matrices
        t_dim = fs.ufl_element().cell._tdim
        os = mats[t_dim][0]
        n = os[next(iter(os.keys()))].shape[0]
        closures_dict = fs._mesh._closure_sizes[fs._mesh.dimension]
        closures = [closures_dict[c] for c in sorted(closures_dict.keys())]

        args = [loopy.ValueArg("d", dtype=utils.IntType),
                loopy.ValueArg("closure_size_acc", dtype=utils.IntType),
                loopy.ValueArg("o_val", dtype=utils.IntType),
                loopy.GlobalArg("o", dtype=utils.IntType, shape=(sum(closures)), is_input=True),
                loopy.GlobalArg("a", dtype=utils.ScalarType, shape=(n, n), is_input=True, is_output=False),
                loopy.GlobalArg("b", dtype=utils.ScalarType, shape=(n, ), is_input=True, is_output=True),
                loopy.GlobalArg("res", dtype=utils.ScalarType, shape=(n,), is_input=True, is_output=True)]
        matmul, set_knl = get_utility_kernels(output_shape, n) 
        # matmul = loopy.make_function(
        #     f"{{[i, j]:0<=i, j < {n}}}",
        #     """
        #         res[j] =  res[j] + a[i, j]*b[i]
        #     """, name="matmul", target=loopy.CWithGNULibcTarget())

        # set_args = [loopy.GlobalArg("b", dtype=utils.ScalarType, shape=(n, ), is_input=True, is_output=True),
        #             loopy.GlobalArg("res", dtype=utils.ScalarType, shape=(n,), is_input=True)]
        # set_knl = loopy.make_function(
        #     f"{{[j]:0<=j < {n}}}",
        #     ["b[j] = res[j]"],
        #     kernel_data=set_args,
        #     name="set", target=loopy.CWithGNULibcTarget()
        # )

        zero_knl = loopy.make_function(
            f"{{ [i]: 0 <= i < {n}}}",
            ["res[i] = 0"],
            [loopy.GlobalArg("res", shape=(3,), dtype=int, is_input=True, is_output=True)],
            target=loopy.CWithGNULibcTarget(),
            name="zero",
        )

        var_list = ["o", "d", "i", "o_val", "dim"]
        in_string, args, var_list = construct_switch_statement(fs, mats, n, args, var_list)
        transform_in_insn = loopy.CInstruction(tuple(), "".join(in_string), assignees=("a", "o_val"), read_variables=frozenset(var_list), id="assign", depends_on="zero")

        dim_arg = [loopy.ValueArg("dim", dtype=utils.IntType)]
        in_switch = loopy.make_function(
            "{[i]:0<= i < d}",
            ["res[:] = zero(res) {id=zero, inames=i}",
             transform_in_insn, "res[:] = matmul(a, b, res) {id=matmul, dep=*, dep=assign}",
             "b[:] = set(b[:], res[:]) {id=set, dep=matmul, inames=i}"],
            name="in_switch_on_o",
            kernel_data=dim_arg + args,
            target=loopy.CWithGNULibcTarget())

        out_string, args, var_list = construct_switch_statement(fs, reversed_mats, n, args, var_list)
        transform_out_insn = loopy.CInstruction(tuple(), "".join(out_string), assignees=("a", "o_val"), read_variables=frozenset(var_list), id="assign", depends_on="zero")
        # print_insn = loopy.CInstruction(tuple(), "printf(\"res: %f, %f, %f\\n\", res[0], res[1], res[2]);", assignees=(), read_variables=frozenset([]), within_inames=frozenset(["i"]), id="print", depends_on="matmul")
        out_switch = loopy.make_function(
            "{[i]:0<= i < d}",
            ["res[:] = zero(res) {id=zero, inames=i}",
             transform_out_insn, "res[:] = matmul(a, b, res) {id=matmul, dep=*, dep=assign}",
             "b[:] = set(b[:], res[:]) {id=set, dep=matmul, inames=i}"
             ],
            name="out_switch_on_o",
            kernel_data=dim_arg + args,
            target=loopy.CWithGNULibcTarget())

        closure_arg = [loopy.TemporaryVariable("closure_sizes", initializer=np.array(closures, dtype=np.int32), dtype=utils.IntType, read_only=True, address_space=loopy.AddressSpace(1))]
        # printres_insn = loopy.CInstruction(tuple(), "printf(\"replaces res: %f\\n\", res[0]);", assignees=(), read_variables=frozenset(["res"]), depends_on="replace")

        def loop_dims(direction):
            return loopy.make_function(
                f"{{[dim]:{0} <= dim <= {fs._mesh.dimension - 1}}}",
                ["d = closure_sizes[dim] {id=closure}",
                 f"b[:], res[:] = {direction}_switch_on_o(dim, d, closure_size_acc, o_val, o[:], a[:, :], b[:], res[:]) {{id=switch, dep=*}}",
                 "closure_size_acc = closure_size_acc + d {id=replace, dep=switch}"
                 ],
                name=f"{direction}_loop_over_dims",
                kernel_data=closure_arg + args,
                target=loopy.CWithGNULibcTarget())

        print_insn = loopy.CInstruction(tuple(), f"printf(\"initial b: {" ".join('%f' for i in range(n))}\\n\", {', '.join(f"b[{j}]" for j in range(n))});printf(\"o: %d, %d, %d, %d, %d, %d, %d\\n\", o[0], o[1], o[2], o[3], o[4],o[5], o[6]);", assignees=(), read_variables=frozenset([]), id="print")
        print_insn1 = loopy.CInstruction(tuple(), f"printf(\"final res: {" ".join('%f' for i in range(n))}\\n\", {', '.join(f"res[{j}]" for j in range(n))});", assignees=(), read_variables=frozenset(["res"]), depends_on="replace")

        def overall(direction):
            if direction == "out" or direction=="in":
                return loopy.make_kernel(
                "{:}",
                [print_insn, f"b[:], res[:] = {direction}_loop_over_dims(0,0,0,o[:], a[:,:], b[:], res[:]) {{dep=print, id=loop}}",
                 "res[:] = set(res[:], b[:]) {id=replace, dep=loop}",
                 "b[:] = zero(b[:]) {dep=replace, id=zerob}",
                 print_insn1],
                name=f"{direction}_transform",
                kernel_data=args[3:7],
                target=loopy.CWithGNULibcTarget())
            return loopy.make_kernel(
                "{:}",
                [print_insn,
                 "res[:] = set(res[:], b[:]) {id=replace,dep=*, dep=print}",
                 "b[:] = zero(b[:]) {dep=replace, id=zerob}",
                 print_insn1],
                name=f"{direction}_transform",
                kernel_data=args[3:7],
                target=loopy.CWithGNULibcTarget())

        in_knl = loopy.merge([overall("in"), loop_dims("in"), in_switch, matmul, set_knl, zero_knl])
        out_knl = loopy.merge([overall("out"), loop_dims("out"), out_switch, matmul, set_knl, zero_knl])
        

        # b is modified in the transform functions but the result is written to res and therefore is not needed further.
        transform_in = op3.Function(in_knl, [op3.READ, op3.WRITE, op3.READ, op3.WRITE])
        transform_out = op3.Function(out_knl, [op3.READ, op3.WRITE, op3.READ, op3.WRITE])
        return transform_in, transform_out
    elif fuse_defined_spaces and any([hasattr(space.ufl_element(), "triple") for space in spaces]):
        raise ValueError("If a fuse space is used, all spaces must be fuse spaces")
        if space.ufl_element().family() != "Lagrange":
            n = 3
            args = [loopy.ValueArg("d", dtype=utils.IntType),
                    loopy.ValueArg("closure_size_acc", dtype=utils.IntType),
                    loopy.ValueArg("o_val", dtype=utils.IntType),
                    loopy.GlobalArg("o", dtype=utils.IntType, shape=(7,), is_input=True),
                    loopy.GlobalArg("a", dtype=utils.ScalarType, shape=(n, n), is_input=True, is_output=False),
                    loopy.GlobalArg("b", dtype=utils.ScalarType, shape=(n, ), is_input=True, is_output=True),
                    loopy.GlobalArg("res", dtype=utils.ScalarType, shape=(n,), is_input=True, is_output=True)]

            set_args = [loopy.GlobalArg("b", dtype=utils.ScalarType, shape=(n, ), is_input=True, is_output=True),
                        loopy.GlobalArg("res", dtype=utils.ScalarType, shape=(n,), is_input=True)]
            set_knl = loopy.make_function(
                f"{{[j]:0<=j < {n}}}",
                ["b[j] = res[j]"],
                kernel_data=set_args,
                name="set", target=loopy.CWithGNULibcTarget()
            )
            zero_knl = loopy.make_function(
                f"{{ [i]: 0 <= i < {n}}}",
                ["res[i] = 0"],
                [loopy.GlobalArg("res", shape=(3,), dtype=int, is_input=True, is_output=True)],
                target=loopy.CWithGNULibcTarget(),
                name="zero",
            )

            print_insn = loopy.CInstruction(tuple(), f"printf(\"initial b: {" ".join('%f' for i in range(n))}\\n\", {', '.join(f"b[{j}]" for j in range(n))});printf(\"o: %d, %d, %d, %d, %d, %d, %d\\n\", o[0], o[1], o[2], o[3], o[4],o[5], o[6]);", assignees=(), read_variables=frozenset([]), id="print")
            print_insn1 = loopy.CInstruction(tuple(), f"printf(\"final res: {" ".join('%f' for i in range(n))}\\n\", {', '.join(f"res[{j}]" for j in range(n))});", assignees=(), read_variables=frozenset(["res"]), depends_on="zerob")
            def overall(direction):
                return loopy.make_kernel(
                    "{:}",
                    [print_insn, 
                    "res[:] = zero(res[:]) {id=zero, dep=print}",
                    "res[:] = set(res[:], b[:]) {id=replace, dep=*, dep=zero}",
                    "b[:] = zero(b[:]) {dep=replace, id=zerob}",
                     print_insn1],
                    name=f"{direction}_transform",
                    kernel_data=args[3:7],
                    target=loopy.CWithGNULibcTarget())

            in_knl = loopy.merge([overall("in"), set_knl, zero_knl])
            out_knl = loopy.merge([overall("out"), set_knl, zero_knl])
            
            # b is modified in the transform functions but the result is written to res and therefore is not needed further.
            transform_in = op3.Function(in_knl, [op3.READ, op3.WRITE, op3.READ, op3.WRITE])
            transform_out = op3.Function(out_knl, [op3.READ, op3.WRITE, op3.READ, op3.WRITE])
            return transform_in, transform_out
        return None, None
        # n = 3
        # args = [loopy.GlobalArg("o", dtype=utils.IntType, shape=(7,), is_input=True),
        #         loopy.GlobalArg("a", dtype=utils.ScalarType, shape=(n, n), is_input=True, is_output=False),
        #         loopy.GlobalArg("b", dtype=utils.ScalarType, shape=(n, ), is_input=True, is_output=True),
        #         loopy.GlobalArg("res", dtype=utils.ScalarType, shape=(n,), is_input=True, is_output=True)]

        # def overall(direction):
        #     print_insn = loopy.CInstruction(tuple(), f"printf(\"NON-FUSE initial {direction} b: %f, %f, %f, %f\\n\", b[0], b[1], b[2], b[3]);", assignees=(), read_variables=frozenset([]), id="print")
        #     return loopy.make_kernel(
        #     "{:}",
        #     [print_insn],
        #     name=f"{direction}_transform",
        #     kernel_data = args,
        #     target=loopy.CWithGNULibcTarget())
        # transform_in = op3.Function(overall("in"), [op3.READ, op3.WRITE, op3.READ, op3.WRITE])
        # transform_out = op3.Function(overall("out"), [op3.READ, op3.WRITE, op3.READ, op3.WRITE])
        # return transform_in, transform_out, (n,)
        # elif len(Vs) == 2 and any(fuse_defined_spaces):
        #     raise NotImplementedError
        #     if not all(fuse_defined_spaces):
        #         raise NotImplementedError("Fuse-defined elements cannot be combined with FIAT elements")
        #     shapes = [0] * len(Vs)
        #     os = [None] * len(Vs)
        #     args = [loopy.GlobalArg("o", dtype=np.uint8, shape=(1,))]
        #     child_knls = [None] * len(Vs)
        #     for i in range(len(Vs)):
        #         fs = Vs[i]
        #         temp_os = fs.ufl_element().triple.matrices
        #         t_dim = fs.ufl_element().cell._tdim
        #         os[i] = temp_os[t_dim][0]
        #         n = os[i][next(iter(os[i].keys()))].shape[0]
        #         shapes[i] = n
        #         args += [loopy.GlobalArg(f"a{i}", dtype=utils.ScalarType, shape=(n, n))]
        #
        #     args += [loopy.GlobalArg("b", dtype=utils.ScalarType, shape=tuple(shapes)), loopy.GlobalArg("temp", dtype=utils.ScalarType, shape=tuple(shapes)), loopy.GlobalArg("res", dtype=utils.ScalarType, shape=tuple(shapes))]
        #     child_knls[0] = loopy.make_function(
        #         f"{{[i, j, k]:0<=i, k < {shapes[0]} and 0<= j < {shapes[1]}}}",
        #         """
        #             res[k, j] =  res[k, j] + a[k, i]*b[i, j]
        #         """, name=f"matmul{0}", target=loopy.CWithGNULibcTarget())
        #     child_knls[1] = loopy.make_function(
        #         f"{{[i, j, k]:0<=k < {shapes[0]} and 0<=  j, i < {shapes[1]}}}",
        #         """
        #             res[k, j] =  res[k, j] + a[k, i]*b[i, j]
        #         """, name=f"matmul{1}", target=loopy.CWithGNULibcTarget())
        #
        #     var_list = ["o"]
        #     string = "\nswitch (*o) { \n"
        #     for val in sorted(os[0].keys()):
        #         string += f"case {val}:\n"
        #         for i in range(len(Vs)):
        #             string += f"a{i} = mat{i}_{val};"
        #             var_list += [f"mat{i}_{val}"]
        #             if no_dense:
        #                 mat = np.eye(shapes[i])
        #             else:
        #                 mat = np.array(os[i][val], dtype=utils.ScalarType)
        #             args += [loopy.TemporaryVariable(f"mat{i}_{val}", initializer=mat, dtype=utils.ScalarType, read_only=True, address_space=loopy.AddressSpace(1))]
        #         string += "break;\n"
        #     string += "default:\nbreak;\n }"
        #     transform_insn = loopy.CInstruction(tuple(), "".join(string), assignees=("a0", "a1"), read_variables=frozenset(var_list), id="assign")
        #     parent_knl = loopy.make_kernel(
        #         "{:}",
        #         [transform_insn, "temp[:, :] = matmul0(a0, b, temp) {dep=assign}", "res[:, :] = matmul1(temp, a1, res) {dep=assign}"],
        #         kernel_data=args,
        #         target=loopy.CWithGNULibcTarget())
        #     knl = loopy.merge([parent_knl, child_knls[0], child_knls[1]])
        #     # print(lp.generate_code_v2(knl).device_code())
        #     # print(knl)
        #     transform = op3.Function(knl, [op3.READ, op3.WRITE, op3.WRITE, op3.READ, op3.WRITE, op3.WRITE])
        #     return transform, shapes
        # return None, tuple()

