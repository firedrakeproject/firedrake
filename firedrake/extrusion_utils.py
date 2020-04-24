import collections
import itertools
import numpy

import firedrake
from pyop2 import op2
from pyop2.datatypes import IntType, RealType, ScalarType
from tsfc.finatinterface import create_element
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2


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
    :arg layer_height: an equi-spaced height for each layer.
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
    _, vert_space = ext_coords.function_space().ufl_element().sub_elements()[0].sub_elements()
    if kernel is None and not (vert_space.degree() == 1
                               and vert_space.family() in ['Lagrange',
                                                           'Discontinuous Lagrange']):
        raise RuntimeError('Extrusion of coordinates is only possible for a P1 or P1dg interval unless a custom kernel is provided')
    layer_height = op2.Global(1, layer_height, dtype=ScalarType) 
    if kernel is not None:
        op2.ParLoop(kernel,
                    ext_coords.cell_set,
                    ext_coords.dat(op2.WRITE, ext_coords.cell_node_map()),
                    base_coords.dat(op2.READ, base_coords.cell_node_map()),
                    layer_height(op2.READ),
                    pass_layer_arg=True,
                    is_loopy_kernel=True).compute()
        return
    ext_fe = create_element(ext_coords.ufl_element())
    ext_shape = ext_fe.index_shape
    base_fe = create_element(base_coords.ufl_element())
    base_shape = base_fe.index_shape
    data = []
    data.append(lp.GlobalArg("ext_coords", dtype=ScalarType, shape=ext_shape))
    data.append(lp.GlobalArg("base_coords", dtype=ScalarType, shape=base_shape))
    data.append(lp.GlobalArg("layer_height", dtype=ScalarType, shape=()))
    data.append(lp.ValueArg('layer'))
    base_coord_dim = base_coords.function_space().value_size
    adim = len(ext_shape) - 2
    # Deal with tensor product cells
    dd_list = []
    ddextents_list = []
    for j in range(3):
        # Create 3 sets of indices/index extents
        prefix = "_" * j
        dd = ""
        ddextents = ""
        for i in range(adim):
            dd += "%sd%s, " % (prefix, i)
            ddextents += "0<=%sd%s<%d and " % (prefix, i, ext_shape[i])
        dd_list.append(dd[:-len(", ")])
        ddextents_list.append(ddextents[:-len(" and ")])
    if extrusion_type == 'uniform':
        domain = "{{ [{0}, c, l]: {1} and 0<=c<{2} and 0<=l<=1}}".format(dd_list[0], ddextents_list[0], base_coord_dim)
        instructions = """
        <{1}> base_coord_dim = {2}
        ext_coords[{0}, l, c] = base_coords[{0}, c]
        ext_coords[{0}, l, base_coord_dim] = layer_height[0] * (layer + l)
        """.format(dd_list[0], IntType, base_coord_dim)
        ast = lp.make_function(domain, instructions, data, name="pyop2_kernel_uniform_extrusion", target=lp.CTarget(),
                               seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"])
    elif extrusion_type == 'radial':
        domain = "{{ [{0}, c, k, l]: {1} and 0<=c, k<{2} and 0<=l<=1 }}".format(dd_list[0], ddextents_list[0], base_coord_dim)
        instructions = """
        <{1}> tt[{0}] = 0
        <{1}> bc[{0}] = 0
        for k
            bc[{0}] = real(base_coords[{0}, k])
            tt[{0}] = tt[{0}] + bc[{0}] * bc[{0}]
        end
        tt[{0}] = sqrt(tt[{0}])
        ext_coords[{0}, l, c] = base_coords[{0}, c] + base_coords[{0}, c] * layer_height[0] * (layer+l) / tt[{0}]
        """.format(dd_list[0], RealType)
        ast = lp.make_function(domain, instructions, data, name="pyop2_kernel_radial_extrusion", target=lp.CTarget(),
                               seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"])
    elif extrusion_type == 'radial_hedgehog':
        # Only implemented for interval in 2D and triangle in 3D.
        # gdim != tdim already checked in ExtrudedMesh constructor.
        tdim = base_coords.ufl_domain().ufl_cell().topological_dimension()
        if tdim not in [1, 2]:
            raise NotImplementedError("Hedgehog extrusion not implemented for %s" % base_coords.ufl_domain().ufl_cell())
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
        #    /\\
        # v0/  \\
        #  /    \\
        # /------\\
        #    v1
        domain = "{{ [{0}, {2}, {4}, c, _c, __c, ___c, l, k]: {1} and {3} and {5} and 0<=c, _c, __c, ___c, k<{6} and 0<=l<=1 }}".format(dd_list[0], ddextents_list[0], dd_list[1], ddextents_list[1], dd_list[2], ddextents_list[2], base_coord_dim)
        # Formula for normal, n
        n_1_1 = """
        n[0] = -bc[1, 1] + bc[0, 1]
        n[1] = bc[1, 0] - bc[0, 0]
        """
        n_2_1 = """
        v0[___c] = bc[1, ___c] - bc[0, ___c]
        v1[___c] = bc[2, ___c] - bc[0, ___c]
        n[0] = v0[1] * v1[2] - v0[2] * v1[1]
        n[1] = v0[2] * v1[0] - v0[0] * v1[2]
        n[2] = v0[0] * v1[1] - v0[1] * v1[0]
        """
        n_2_2 = """
        v0[___c] = bc[0, 1, ___c] - bc[0, 0, ___c]
        v1[___c] = bc[1, 0, ___c] - bc[0, 0, ___c]
        n[0] = v0[1] * v1[2] - v0[2] * v1[1]
        n[1] = v0[2] * v1[0] - v0[0] * v1[2]
        n[2] = v0[0] * v1[1] - v0[1] * v1[0]
        """
        n_dict = {1: {1: n_1_1},
                  2: {1: n_2_1,
                      2: n_2_2}}
        instructions = """
        <{3}> v0[__c] = 0
        <{3}> v1[__c] = 0
        <{3}> n[__c] = 0
        <{3}> x[__c] = 0
        <{3}> dot = 0
        <{3}> norm = 0
        <{3}> bc[{1}, _c] = real(base_coords[{1}, _c])
        for {1}
            x[_c] = x[_c] + bc[{1}, _c]
        end
        {5}
        for k
            dot = dot + x[k] * n[k]
            norm = norm + n[k] * n[k]
        end
        norm = sqrt(norm)
        norm = -norm if dot < 0 else norm
        ext_coords[{0}, l, c] = base_coords[{0}, c] + n[c] * layer_height[0] * (layer + l) / norm
        """.format(dd_list[0], dd_list[1], dd_list[2], RealType, base_coords.ufl_domain().ufl_cell().topological_dimension(), n_dict[tdim][adim])
        ast = lp.make_function(domain, instructions, data, name="pyop2_kernel_radial_hedgehog_extrusion", target=lp.CTarget(),
                               seq_dependencies=True, silenced_warnings=["summing_if_branches_ops"])
    else:
        raise NotImplementedError('Unsupported extrusion type "%s"' % extrusion_type)

    kernel = op2.Kernel(ast, ast.name)
    op2.ParLoop(kernel,
                ext_coords.cell_set,
                ext_coords.dat(op2.WRITE, ext_coords.cell_node_map()),
                base_coords.dat(op2.READ, base_coords.cell_node_map()),
                layer_height(op2.READ),
                pass_layer_arg=True,
                is_loopy_kernel=True).compute()


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
