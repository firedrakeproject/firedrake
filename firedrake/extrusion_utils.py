import collections
import functools
import itertools
import numpy
import islpy as isl

import finat
from pyop2 import op2
from firedrake.petsc import PETSc
from firedrake.utils import IntType, RealType, ScalarType
from tsfc.finatinterface import create_element
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401
from firedrake.parameters import target
from ufl.domain import extract_unique_domain


@PETSc.Log.EventDecorator()
def make_extruded_coords(extruded_topology, base_coords, ext_coords,
                         layer_height, extrusion_type='uniform', kernel=None):
    """
    Given either a kernel or a (fixed) layer_height, compute an
    extruded coordinate field for an extruded mesh.

    :arg extruded_topology: an :class:`~.ExtrudedMeshTopology` to extrude
         a coordinate field for.
    :arg base_coords: a :class:`~.Function` to read the base
         coordinates from.
    :arg ext_coords: a :class:`~.Function` to write the extruded
         coordinates into.
    :arg layer_height: the height for each layer.  Either a scalar,
         where layers will be equi-spaced at the specified height, or a
         1D array of variable layer heights to use through the extrusion.
    :arg extrusion_type: the type of extrusion to use.  Predefined
         options are either "uniform" (creating equi-spaced layers by
         extruding in the (n+1)dth direction), "radial" (creating
         equi-spaced layers by extruding in the outward direction from
         the origin) or "radial_hedgehog" (creating equi-spaced layers
         by extruding coordinates in the outward cell-normal
         direction, needs a P1dgxP1 coordinate field).
    :arg kernel: an optional kernel to carry out coordinate extrusion.

    The kernel signature (if provided) is::

        void kernel(double **base_coords, double **ext_coords,
                    double *layer_height, int layer)

    The kernel iterates over the cells of the mesh and receives as
    arguments the coordinates of the base cell (to read), the
    coordinates on the extruded cell (to write to), the fixed layer
    height, and the current cell layer.
    """
    _, vert_space = ext_coords.function_space().ufl_element().sub_elements[0].sub_elements
    if kernel is None and not (vert_space.degree() == 1
                               and vert_space.family() in ['Lagrange',
                                                           'Discontinuous Lagrange']):
        raise RuntimeError('Extrusion of coordinates is only possible for a P1 or P1dg interval unless a custom kernel is provided')

    layer_height = numpy.atleast_1d(numpy.array(layer_height, dtype=RealType))

    if layer_height.ndim > 1:
        raise RuntimeError('Extrusion layer height should be 1d or scalar')

    if layer_height.size > 1:
        layer_height = numpy.cumsum(numpy.concatenate(([0], layer_height)))

    layer_heights = layer_height.size
    layer_height = op2.Global(layer_heights, layer_height, dtype=RealType, comm=extruded_topology._comm)

    if kernel is not None:
        op2.ParLoop(kernel,
                    ext_coords.cell_set,
                    ext_coords.dat(op2.WRITE, ext_coords.cell_node_map()),
                    base_coords.dat(op2.READ, base_coords.cell_node_map()),
                    layer_height(op2.READ),
                    pass_layer_arg=True).compute()
        return
    ext_fe = create_element(ext_coords.ufl_element())
    ext_shape = ext_fe.index_shape
    base_fe = create_element(base_coords.ufl_element())
    base_shape = base_fe.index_shape
    data = []
    data.append(lp.GlobalArg("ext_coords", dtype=ScalarType, shape=ext_shape))
    data.append(lp.GlobalArg("base_coords", dtype=ScalarType, shape=base_shape))
    data.append(lp.GlobalArg("layer_height", dtype=RealType, shape=(layer_heights,)))
    data.append(lp.ValueArg('layer'))
    base_coord_dim = base_coords.function_space().value_size
    # Deal with tensor product cells
    adim = len(ext_shape) - 2

    # handle single or variable layer heights
    if layer_heights == 1:
        height_var = "layer_height[0] * (layer + l)"
    else:
        height_var = "layer_height[layer + l]"

    def _get_arity_axis_inames(_base):
        return tuple(_base + str(i) for i in range(adim))

    def _get_lp_domains(_inames, _extents):
        domains = []
        for idx, extent in zip(_inames, _extents):
            inames = isl.make_zero_and_vars([idx])
            domains.append(((inames[0].le_set(inames[idx])) & (inames[idx].lt_set(inames[0] + extent))))
        return domains

    if extrusion_type == 'uniform':
        domains = []
        dd = _get_arity_axis_inames('d')
        domains.extend(_get_lp_domains(dd, ext_shape[:adim]))
        domains.extend(_get_lp_domains(('c',), (base_coord_dim,)))
        if layer_heights == 1:
            domains.extend(_get_lp_domains(('l',), (2,)))
        else:
            domains.append("[layer] -> { [l] : 0 <= l <= 1 & 0 <= l + layer < %d}" % layer_heights)
        instructions = """
        ext_coords[{dd}, l, c] = base_coords[{dd}, c]
        ext_coords[{dd}, l, {base_coord_dim}] = ({hv})
        """.format(dd=', '.join(dd),
                   base_coord_dim=base_coord_dim,
                   hv=height_var)
        name = "pyop2_kernel_uniform_extrusion"
    elif extrusion_type == 'radial':
        domains = []
        dd = _get_arity_axis_inames('d')
        domains.extend(_get_lp_domains(dd, ext_shape[:adim]))
        domains.extend(_get_lp_domains(('c', 'k'), (base_coord_dim, ) * 2))
        if layer_heights == 1:
            domains.extend(_get_lp_domains(('l',), (2,)))
        else:
            domains.append("[layer] -> { [l] : 0 <= l <= 1 & 0 <= l + layer < %d}" % layer_heights)
        instructions = """
        <{RealType}> tt[{dd}] = 0
        <{RealType}> bc[{dd}] = 0
        for k
            bc[{dd}] = real(base_coords[{dd}, k])
            tt[{dd}] = tt[{dd}] + bc[{dd}] * bc[{dd}]
        end
        tt[{dd}] = sqrt(tt[{dd}])
        ext_coords[{dd}, l, c] = base_coords[{dd}, c] + base_coords[{dd}, c] * ({hv}) / tt[{dd}]
        """.format(RealType=RealType,
                   dd=', '.join(dd),
                   hv=height_var)
        name = "pyop2_kernel_radial_extrusion"
    elif extrusion_type == 'radial_hedgehog':
        # Only implemented for interval in 2D and triangle in 3D.
        # gdim != tdim already checked in ExtrudedMesh constructor.
        tdim = extract_unique_domain(base_coords).ufl_cell().topological_dimension()
        if tdim not in [1, 2]:
            raise NotImplementedError("Hedgehog extrusion not implemented for %s" % extract_unique_domain(base_coords).ufl_cell())
        # tdim == 1:
        #
        # normal is:
        # (0 -1) (x2 - x1)
        # (1  0) (y2 - y1)
        #
        # tdim == 2:
        # normal is
        # v0 x v1
        #
        #    /\
        # v0/  \
        #  /    \
        # /------\
        #    v1
        domains = []
        dd = _get_arity_axis_inames('d')
        _dd = _get_arity_axis_inames('_d')
        domains.extend(_get_lp_domains(dd, ext_shape[:adim]))
        domains.extend(_get_lp_domains(_dd, ext_shape[:adim]))
        if tdim == 1:
            domains.extend(_get_lp_domains(('c0', 'c1', 'c2', 'k', 'l'), (base_coord_dim, ) * 4 + (2, )))
        else:
            domains.extend(_get_lp_domains(('c0', 'c1', 'c2', 'c3', 'k', 'l'), (base_coord_dim, ) * 5 + (2, )))
        # Formula for normal, n
        n_1_1 = """
        n[0] = -bc[1, 1] + bc[0, 1]
        n[1] = bc[1, 0] - bc[0, 0]
        """
        n_2_1 = """
        v0[c3] = bc[1, c3] - bc[0, c3]
        v1[c3] = bc[2, c3] - bc[0, c3]
        n[0] = v0[1] * v1[2] - v0[2] * v1[1]
        n[1] = v0[2] * v1[0] - v0[0] * v1[2]
        n[2] = v0[0] * v1[1] - v0[1] * v1[0]
        """
        n_2_2 = """
        v0[c3] = bc[0, 1, c3] - bc[0, 0, c3]
        v1[c3] = bc[1, 0, c3] - bc[0, 0, c3]
        n[0] = v0[1] * v1[2] - v0[2] * v1[1]
        n[1] = v0[2] * v1[0] - v0[0] * v1[2]
        n[2] = v0[0] * v1[1] - v0[1] * v1[0]
        """
        n_dict = {1: {1: n_1_1},
                  2: {1: n_2_1,
                      2: n_2_2}}
        instructions = """
        <{RealType}> dot = 0
        <{RealType}> norm = 0
        <{RealType}> v0[c2] = 0
        <{RealType}> v1[c2] = 0
        <{RealType}> n[c2] = 0
        <{RealType}> x[c2] = 0
        <{RealType}> bc[{_dd}, c1] = real(base_coords[{_dd}, c1])
        for {_dd}
            x[c1] = x[c1] + bc[{_dd}, c1]
        end
        {ninst}
        for k
            dot = dot + x[k] * n[k]
            norm = norm + n[k] * n[k]
        end
        norm = sqrt(norm)
        norm = -norm if dot < 0 else norm
        ext_coords[{dd}, l, c0] = base_coords[{dd}, c0] + n[c0] * ({hv}) / norm
        """.format(RealType=RealType,
                   dd=', '.join(dd),
                   _dd=', '.join(_dd),
                   ninst=n_dict[tdim][adim],
                   hv=height_var)
        name = "pyop2_kernel_radial_hedgehog_extrusion"
    else:
        raise NotImplementedError('Unsupported extrusion type "%s"' % extrusion_type)

    ast = lp.make_function(domains, instructions, data, name=name, target=target,
                           seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"])
    kernel = op2.Kernel(ast, name)
    op2.ParLoop(kernel,
                ext_coords.cell_set,
                ext_coords.dat(op2.WRITE, ext_coords.cell_node_map()),
                base_coords.dat(op2.READ, base_coords.cell_node_map()),
                layer_height(op2.READ),
                pass_layer_arg=True).compute()


def flat_entity_dofs(entity_dofs):
    flat_entity_dofs = {}
    for b, v in entity_dofs:
        # v in [0, 1].  Only look at the ones, then grab the data from zeros.
        if v == 0:
            continue
        flat_entity_dofs[b] = {}
        for i in entity_dofs[(b, v)]:
            # This line is fairly magic.
            # It works because an interval has two points.
            # We pick up the DoFs from the bottom point,
            # then the DoFs from the interior of the interval,
            # then finally the DoFs from the top point.
            flat_entity_dofs[b][i] = (entity_dofs[(b, 0)][2*i]
                                      + entity_dofs[(b, 1)][i]
                                      + entity_dofs[(b, 0)][2*i+1])
    return flat_entity_dofs


def flat_entity_permutations(entity_permutations):
    flat_entity_permutations = {}
    for b in set(b for b, v in entity_permutations):
        flat_entity_permutations[b] = {}
        for eb in set(e // 2 for e in entity_permutations[(b, 0)]):
            flat_entity_permutations[b][eb] = {}
            for ob in set(ob for eo, ob, ov in entity_permutations[(b, 0)][2 * eb]):
                # eo (extrinsic orientation) is always 0 for:
                # -- quad x interval,
                # -- triangle x interval,
                # -- etc.
                # eo = {0, 1}, but only eo = 0 is relevant for:
                # -- interval x interval on dim = (1, 1).
                eo = 0
                # Orientation in the extruded direction is always 0
                ov = 0
                perm0 = entity_permutations[(b, 0)][2 * eb][(eo, ob, ov)]
                perm1 = entity_permutations[(b, 1)][eb][(eo, ob, ov)]
                n0, n1 = len(perm0), len(perm1)
                flat_entity_permutations[b][eb][ob] = \
                    list(perm0) + \
                    [n0 + p for p in perm1] + \
                    [n0 + n1 + p for p in perm0]
    return flat_entity_permutations


def entity_indices(cell):
    """Return a dict mapping topological entities on a cell to their integer index.

    This provides an iteration ordering for entities on extruded meshes.

    :arg cell: a FIAT cell.
    """
    subents, = cell.sub_entities[cell.get_dimension()].values()
    return {e: i for i, e in enumerate(sorted(subents))}


def entity_reordering(cell):
    """Return an array reordering extruded cell entities.

    If we iterate over the base cell, it is natural to then go over
    all the entities induced by the product with an interval.  This
    iteration order is not the same as the natural iteration order, so
    we need a reordering.

    :arg cell: a FIAT tensor product cell.
    """
    def points(t):
        for k in sorted(t.keys()):
            yield itertools.repeat(k, len(t[k]))

    counter = collections.Counter()

    topos = (c.get_topology() for c in cell.cells)

    indices = entity_indices(cell)
    ordering = numpy.zeros(len(indices), dtype=IntType)
    for i, ent in enumerate(itertools.product(*(itertools.chain(*points(t)) for t in topos))):
        ordering[i] = indices[ent, counter[ent]]
        counter[ent] += 1
    return ordering


def entity_closures(cell):
    """Map entities in a cell to points in the topological closure of
    the entity.

    :arg cell: a FIAT cell.
    """
    indices = entity_indices(cell)
    closure = {}
    for e, ents in cell.sub_entities.items():
        for ent, vals in ents.items():
            idx = indices[(e, ent)]
            closure[idx] = list(map(indices.get, vals))
    return closure


@functools.lru_cache()
def calculate_dof_offset(finat_element):
    """Return the offset between the neighbouring cells of a
    column for each DoF.

    :arg finat_element: A FInAT element.
    :returns: A numpy array containing the offset for each DoF.
    """
    # scalar-valued elements only
    if isinstance(finat_element, finat.TensorFiniteElement):
        finat_element = finat_element.base_element

    dof_offset = numpy.zeros(finat_element.space_dimension(), dtype=IntType)

    if is_real_tensor_product_element(finat_element):
        return dof_offset

    entity_offset = [0] * (1 + finat_element.cell.get_dimension()[0])
    for (b, v), entities in finat_element.entity_dofs().items():
        entity_offset[b] += len(entities[0])

    for (b, v), entities in finat_element.entity_dofs().items():
        for dof_indices in entities.values():
            for i in dof_indices:
                dof_offset[i] = entity_offset[b]
    return dof_offset


@functools.lru_cache()
def calculate_dof_offset_quotient(finat_element):
    """Return the offset quotient for each DoF within the base cell.

    :arg finat_element: A FInAT element.
    :returns: A numpy array containing the offset quotient for each DoF.

    offset_quotient q of each DoF (in a local cell) is defined as
    i // o, where i is the local DoF ID of the DoF on the entity and
    o is the offset of that DoF computed in ``calculate_dof_offset()``.

    Let DOF(e, l, i) represent a DoF on (base-)entity e on layer l that has local ID i
    and suppose this DoF has offset o and offset_quotient q. In periodic extrusion it
    is convenient to identify DOF(e, l, i) as DOF(e, l + q, i % o); this transformation
    allows one to always work with the "unit cell" in which i < o always holds.

    In FEA offset_quotient is 0 or 1.

    Example::

               local ID   offset     offset_quotient

               2--2--2    2--2--2    1--1--1
               |     |    |     |    |     |
        CG2    1  1  1    2  2  2    0  0  0
               |     |    |     |    |     |
               0--0--0    2--2--2    0--0--0

               +-----+    +-----+    +-----+
               | 1 3 |    | 4 4 |    | 0 0 |
        DG1    |     |    |     |    |     |
               | 0 2 |    | 4 4 |    | 0 0 |
               +-----+    +-----+    +-----+

    """
    # scalar-valued elements only
    if isinstance(finat_element, finat.TensorFiniteElement):
        finat_element = finat_element.base_element
    if is_real_tensor_product_element(finat_element):
        return None
    dof_offset_quotient = numpy.zeros(finat_element.space_dimension(), dtype=IntType)
    for (b, v), entities in finat_element.entity_dofs().items():
        for entity, dof_indices in entities.items():
            quotient = 1 if v == 0 and entity % 2 == 1 else 0
            for i in dof_indices:
                dof_offset_quotient[i] = quotient
    if (dof_offset_quotient == 0).all():
        # Avoid unnecessary codegen in pyop2/codegen/builder.
        dof_offset_quotient = None
    return dof_offset_quotient


def is_real_tensor_product_element(element):
    """Is the provided FInAT element a tensor product involving the real space?

    :arg element: A scalar FInAT element.
    """
    assert not isinstance(element, finat.TensorFiniteElement)

    if isinstance(element, finat.TensorProductElement):
        _, factor = element.factors
        return isinstance(factor, finat.Real)
    else:
        return False
