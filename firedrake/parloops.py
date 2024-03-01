r"""This module implements parallel loops reading and writing
:class:`.Function`\s. This provides a mechanism for implementing
non-finite element operations such as slope limiters."""
import collections
import functools
from cachetools import LRUCache

import finat
import loopy
import numpy as np
import pyop3 as op3
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401
from pyop2 import op2, READ, WRITE, RW, INC, MIN, MAX
from pyrsistent import freeze, pmap
from ufl.indexed import Indexed
from ufl.domain import join_domains

from firedrake import constant
from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.matrix import Matrix
from firedrake.petsc import PETSc
from firedrake.parameters import target
from firedrake.ufl_expr import extract_domains
from firedrake.utils import IntType


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


def pack_tensor(tensor, index, intent, integral_type):
    if integral_type in {
        "cell",
        "exterior_facet_top",
        "exterior_facet_bottom",
        "interior_facet_horiz",
    }:
        return pack_tensor_cell_integral(tensor, index, intent)
    elif integral_type in {"exterior_facet", "exterior_facet_vert"}:
        return pack_tensor_exterior_facet_integral(tensor, index)
    else:
        assert integral_type in {"interior_facet", "interior_facet_vert"}
        return pack_tensor_interior_facet_integral(tensor, index)


# ah, never use intent!
@functools.singledispatch
def pack_tensor_cell_integral(tensor, index, intent):
    raise TypeError


@pack_tensor_cell_integral.register(Function)
@pack_tensor_cell_integral.register(Cofunction)
def _(tensor, cell: op3.LoopIndex, intent):
    V = tensor.function_space()
    plex = V.mesh().topology

    # First collect the DoFs in the cell closure in FIAT order.
    # TODO ideally the FIAT permutation would not need to be known
    # about by the mesh topology and instead be handled here. This
    # would probably require an oriented mesh
    # (see https://github.com/firedrakeproject/firedrake/pull/3332)
    indexed = tensor.dat[plex._fiat_closure(cell)]

    # Indexing an array with a loop index makes it "context sensitive" since
    # the index could be over multiple entities (e.g. all mesh points). Here
    # we know we are only looping over cells so the context is trivial.
    cf_indexed = op3.utils.just_one(indexed.context_map.values())

    if not isinstance(V.finat_element, finat.FlattenedDimensions):
        return cf_indexed

    ###

    # FIAT numbering:
    #
    #     1-----7-----3     v1----e3----v3
    #     |           |     |           |
    #     |           |     |           |
    #     4     8     5  =  e0    c0    e1
    #     |           |     |           |
    #     |           |     |           |
    #     0-----6-----2     v0----e2----v2
    #
    # therefore P3 standard DoF ordering (returned by packing closure):
    #
    #     1---10--11--3
    #     |           |
    #     5   13  15  7
    #     |           |
    #     4   12  14  6
    #     |           |
    #     0---8---9---2
    #
    # which would have a data layout like:
    #
    #     +----+----+----+----#----+----+----+----#----+
    #     | v0 | v1 | v2 | v3 # e0 | e1 | e2 | e3 # c0 |
    #     +----+----+----+----#----+----+----+----#----+
    #          /                 /                   \
    #      +-----+      +-----+-----+     +-----+-----+-----+-----+
    #      | dof |      | dof | dof |     | dof | dof | dof | dof |
    #      +-----+      +-----+-----+     +-----+-----+-----+-----+
    #
    # But it's a tensor product of intervals:
    #
    #     1---9---13--5
    #     |           |
    #     3   11  15  7
    #     |           |
    #     2   10  14  6
    #     |           |
    #     0---8---12--4
    #
    # so it actually has a data layout like:
    #
    #                     +----+----#----+
    #                     | v0 | v1 # e0 |
    #                     +----+----#----+
    #                      /            \
    #                +-----+          +-----+-----+
    #                | dof |          | dof | dof |
    #                +-----+          +-----+-----+
    #                   |                   |
    #           +----+----#----+     +----+----#----+
    #           | v0 | v1 # e0 |     | v0 | v1 # e0 |
    #           +----+----#----+     +----+----#----+
    #           /           |             |         \
    #    +-----+      +-----+-----+    +-----+    +-----+-----+
    #    | dof |      | dof | dof |    | dof |    | dof | dof |
    #    +-----+      +-----+-----+    +-----+    +-----+-----+
    #
    # We therefore want a temporary that is in the tensor-product form, but we
    # need to be able to fill it from the "flat" closure that DMPlex gives us.

    if V.mesh().ufl_cell().cellname() != "quadrilateral":
        raise NotImplementedError

    subelem_axess = []
    subelems = V.finat_element.product.factors
    for i, subelem in enumerate(subelems):
        npoints = []
        subaxess = []
        for tdim, edofs in subelem.entity_dofs().items():
            npoints.append(len(edofs))
            ndofs = op3.utils.single_valued(len(d) for d in edofs.values())
            subaxes = op3.Axis(ndofs, f"dof{i}")
            subaxess.append(subaxes)

        subelem_axes = op3.AxisTree.from_nest({
            op3.Axis(
                {str(tdim): n for tdim, n in enumerate(npoints)}, f"points{i}"
            ): subaxess
        })

        # hack
        subelem_axes = op3.PartialAxisTree(subelem_axes.parent_to_children)
        subelem_axess.append(subelem_axes)

    # lazy, make a separate function
    def _build_rec(_axess):
        _axes, *_subaxess = _axess

        if _subaxess:
            for leaf in _axes.leaves:
                subtree = _build_rec(_subaxess)
                _axes = _axes.add_subtree(subtree, *leaf, uniquify_ids=True)
        return _axes

    tensor_axes = _build_rec(subelem_axess)

    shape_target_paths = {None: {}}
    shape_index_exprs = {None: {}}
    if V.shape:
        shape_axes = []
        for i, s in enumerate(V.shape):
            axis_label = f"dim{i}"
            cpt_label = "XXX"
            shape_axes.append(op3.Axis({cpt_label: s}, axis_label))
            shape_target_paths[None][axis_label] = cpt_label
            shape_index_exprs[None][axis_label] = op3.AxisVariable(axis_label)

        shape_axes = op3.AxisTree.from_iterable(shape_axes)
        for leaf in tensor_axes.leaves:
            tensor_axes = tensor_axes.add_subtree(shape_axes, *leaf, uniquify_ids=True)

    tensor_axes = tensor_axes.set_up()

    # this is easy, the target path is just summing things up!
    def _build_target_paths(_axes, _axis=None, tdim=0):
        if _axis is None:
            _axis = _axes.root

        paths = {}
        for component in _axis.components:
            if not (_axis.label.startswith("dof") or _axis.label.startswith("dim")):
                tdim_ = tdim + int(component.label)
            else:
                tdim_ = tdim
            if subaxis := _axes.child(_axis, component):
                paths.update(_build_target_paths(_axes, subaxis, tdim_))
            else:
                paths[_axis.id,component.label] = {"closure": str(tdim_), "dof": "XXX"}
        return paths

    tensor_target_paths = _build_target_paths(tensor_axes)
    tensor_target_paths.update(shape_target_paths)

    # hard code for quads for now, hexes are harder and I can't yet come up with
    # a general solution/algorithm
    my_leaves = tensor_axes.leaves
    if len(my_leaves) != 4:
        raise NotImplementedError

    tensor_index_exprs = {}
    flat_mesh_axis_label = "closure"
    flat_dof_axis_label = "dof"
    for i, (leaf_axis, leaf_cpt) in enumerate(my_leaves):
        if i == 0:  # (0, 0)
            index_expr = {
                flat_mesh_axis_label: 2*op3.AxisVariable("points0") + op3.AxisVariable("points1"),
                flat_dof_axis_label: 0,  # single DoF
            }
        elif i == 1:  # (0, 1)
            index_expr = {
                flat_mesh_axis_label: op3.AxisVariable("points0"),  # e0, e1
                flat_dof_axis_label: op3.AxisVariable("dof1"),
            }
        elif i == 2:  # (1, 0)
            index_expr = {
                flat_mesh_axis_label: op3.AxisVariable("points1") + 2,  # e2, e3
                flat_dof_axis_label: op3.AxisVariable("dof0"),
            }
        else:  # (1, 1)
            assert i == 3
            if subelems[0] is not subelems[1]:
                raise NotImplementedError("Currently assume that sub elements are identical")
            dof_stride = len(subelems[0].entity_dofs()[1][0])
            index_expr = {
                flat_mesh_axis_label: 0,  # single cell
                flat_dof_axis_label: dof_stride*op3.AxisVariable("dof0") + op3.AxisVariable("dof1"),
            }
        tensor_index_exprs[leaf_axis.id, leaf_cpt] = index_expr

    tensor_index_exprs.update(shape_index_exprs)

    # !!!

    # debug
    old_target_paths = tensor_target_paths.copy()
    old_index_exprs = tensor_index_exprs.copy()

    from pyop3.itree.tree import _compose_bits
    tensor_target_paths, tensor_index_exprs, _ = _compose_bits(
        cf_indexed.axes,
        cf_indexed.target_paths,
        cf_indexed.index_exprs,
        None,
        tensor_axes,
        tensor_target_paths,
        tensor_index_exprs,
        {},
    )

    # breakpoint()
    #
    cf_indexed_tensor = op3.HierarchicalArray(
        tensor_axes,
        target_paths=freeze(tensor_target_paths),
        index_exprs=freeze(tensor_index_exprs),
        layouts=cf_indexed.layouts,
        data=cf_indexed.buffer,
        name=cf_indexed.name,
    )

    # Create the temporary that will be passed to the local kernel
    packed = op3.HierarchicalArray(
        tensor_axes,
        data=op3.NullBuffer(cf_indexed.dtype),
        prefix="T",
    )

    return op3.Pack(cf_indexed_tensor, packed)


@pack_tensor_cell_integral.register(Matrix)
def _(tensor, cell: op3.LoopIndex, intent):
    V0, V1 = tensor.ufl_function_spaces()
    return pack_petsc_mat(tensor.M, V0, V1, cell, intent)


def pack_petsc_mat(tensor, V0, V1, cell, intent):
    rplex = V0.mesh().topology
    cplex = V1.mesh().topology

    # First collect the DoFs in the cell closure in FIAT order.
    # TODO ideally the FIAT permutation would not need to be known
    # about by the mesh topology and instead be handled here. This
    # would probably require an oriented mesh
    # (see https://github.com/firedrakeproject/firedrake/pull/3332)
    rmap = rplex._fiat_closure(cell)
    cmap = cplex._fiat_closure1(cell)
    indexed = tensor[rmap, cmap]

    # Indexing an array with a loop index makes it "context sensitive" since
    # the index could be over multiple entities (e.g. all mesh points). Here
    # we know we are only looping over cells so the context is trivial.
    cf_indexed = op3.utils.just_one(indexed.context_map.values())

    # Determine the DoF layout per entity type in the cell closure. This is
    # needed because the DoFs of the element may be interleaved (e.g. for quads).
    roffs = []
    for rdim in range(rplex.dimension+1):
        roff = []  # TODO could be a numpy array
        for entity, entity_offsets in V0.finat_element.entity_dofs()[rdim].items():
            for offset in entity_offsets:
                for i in range(V0._cdim):
                    roff.append(offset*V0._cdim + i)
        roff = np.asarray(roff, dtype=IntType)
        roffs.append(roff)

    # TODO These are constant for a particular finite element, cache them
    coffs = []
    for cdim in range(cplex.dimension+1):
        coff = []  # TODO could be a numpy array
        for entity, entity_offsets in V1.finat_element.entity_dofs()[cdim].items():
            for offset in entity_offsets:
                for i in range(V1._cdim):
                    coff.append(offset*V1._cdim + i)
        coff = np.asarray(coff, dtype=IntType)
        coffs.append(coff)

    # not for vector things!
    all_coffs = sum(len(o) for o in coffs)

    layouts = {}
    for path in cf_indexed.axes.ordered_leaf_paths_with_nodes:
        # TODO explain this
        slices = tuple(
            op3.Slice(ax.label, [op3.AffineSliceComponent(cpt)])
            for ax, cpt in path
        )
        # drop extra info like the star forest
        layout_axes = op3.AxisTree(cf_indexed.axes[slices].parent_to_children)

        if layout_axes.size == 0:
            continue

        # Collect the layout data
        raxis, rcpt = op3.utils.just_one(p for p in path if p[0].label == rmap.label)
        rkey = (raxis.id, rcpt)
        rdim = int(cf_indexed.axes.target_paths[rkey][rplex.name])
        roffsets = roffs[rdim]

        caxis, ccpt = op3.utils.just_one(p for p in path if p[0].label == cmap.label)
        ckey = (caxis.id, ccpt)
        cdim = int(cf_indexed.axes.target_paths[ckey][cplex.name+"1"])
        coffsets = coffs[cdim]

        # TODO explain this better
        # build a matrix of layout entries. for roffsets = [2, 3, 4],
        # all_coffs = 3 and coffsets = [0, 1, 2] this should make
        # [ [2*3+0, 2*3+1, 2*3+2]
        #   [3*3+0, 3*3+1, 3*3+2]
        #   [4*3+0, 4*3+1, 4*3+2] ] (flattened)
        layout_data = np.tile(roffsets*all_coffs, (len(coffsets), 1)).T
        layout_data += coffsets
        # TODO check for identity, if so not needed
        layout_dat = op3.HierarchicalArray(layout_axes, data=layout_data.flatten())

        label_path = pmap({ax.label: cpt for ax, cpt in path})
        index_exprs = {ax.label: op3.AxisVariable(ax.label) for ax, _ in path}
        layouts[label_path] = op3.MultiArrayVariable(layout_dat, label_path, index_exprs)

    # Create the temporary that will be passed to the local kernel
    packed = op3.HierarchicalArray(
        op3.AxisTree(cf_indexed.axes.parent_to_children),
        data=op3.NullBuffer(cf_indexed.dtype),
        layouts=layouts,
        prefix="T",  # not really needed if temporaries are also renumbered in codegen, which they should be
    )

    return op3.Pack(cf_indexed, packed)
