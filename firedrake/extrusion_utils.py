import collections
import itertools
import numpy

from pyop2 import op2
from pyop2.datatypes import IntType


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
    if kernel is None and not (vert_space.degree() == 1 and
                               vert_space.family() in ['Lagrange',
                                                       'Discontinuous Lagrange']):
        raise RuntimeError('Extrusion of coordinates is only possible for a P1 or P1dg interval unless a custom kernel is provided')
    if kernel is not None:
        pass
    elif extrusion_type == 'uniform':
        kernel = op2.Kernel("""
inline void pyop2_kernel_uniform_extrusion(double *ext_coords,
                                           const double *base_coords,
                                           const double *layer_height,
                                           int layer) {
    for ( int d = 0; d < %(base_map_arity)d; d++ ) {
        for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
            ext_coords[2*d*(%(base_coord_dim)d+1)+c] = base_coords[d*%(base_coord_dim)d+c];
            ext_coords[(2*d+1)*(%(base_coord_dim)d+1)+c] = base_coords[d*%(base_coord_dim)d+c];
        }
        ext_coords[2*d*(%(base_coord_dim)d+1)+%(base_coord_dim)d] = *layer_height * (layer);
        ext_coords[(2*d+1)*(%(base_coord_dim)d+1)+%(base_coord_dim)d] = *layer_height * (layer + 1);
    }
}""" % {'base_map_arity': base_coords.cell_node_map().arity, 'base_coord_dim': base_coords.function_space().value_size},
            "pyop2_kernel_uniform_extrusion")
    elif extrusion_type == 'radial':
        kernel = op2.Kernel("""
inline void pyop2_kernel_radial_extrusion(double *ext_coords,
                                          const double *base_coords,
                                          const double *layer_height,
                                          int layer) {
    for ( int d = 0; d < %(base_map_arity)d; d++ ) {
        double norm = 0.0;
        for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
            norm += base_coords[d*%(base_coord_dim)d+c] * base_coords[d*%(base_coord_dim)d+c];
        }
        norm = sqrt(norm);
        for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
            ext_coords[2*d*%(base_coord_dim)d+c] = base_coords[d*%(base_coord_dim)d+c] * (1 + (*layer_height * layer)/norm);
            ext_coords[(2*d+1)*%(base_coord_dim)d+c] = base_coords[d*%(base_coord_dim)d+c] * (1 + (*layer_height * (layer+1))/norm);
        }
    }
}""" % {'base_map_arity': base_coords.cell_node_map().arity, 'base_coord_dim': base_coords.function_space().value_size},
            "pyop2_kernel_radial_extrusion")
    elif extrusion_type == 'radial_hedgehog':
        # Only implemented for interval in 2D and triangle in 3D.
        # gdim != tdim already checked in ExtrudedMesh constructor.
        if base_coords.ufl_domain().ufl_cell().topological_dimension() not in [1, 2]:
            raise NotImplementedError("Hedgehog extrusion not implemented for %s" % base_coords.ufl_domain().ufl_cell())
        kernel = op2.Kernel("""
inline void pyop2_kernel_radial_hedgehog_extrusion(double *ext_coords,
                                                   const double *base_coords,
                                                   const double *layer_height,
                                                   int layer) {
    double v0[%(base_coord_dim)d];
    double v1[%(base_coord_dim)d];
    double n[%(base_coord_dim)d];
    double x[%(base_coord_dim)d] = {0};
    double dot = 0.0;
    double norm = 0.0;
    int i, c, d;
    if (%(base_coord_dim)d == 2) {
        /*
         * normal is:
         * (0 -1) (x2 - x1)
         * (1  0) (y2 - y1)
         */
        n[0] = -(base_coords[%(base_coord_dim)d+1] - base_coords[1]);
        n[1] = base_coords[%(base_coord_dim)d] - base_coords[0];
    } else if (%(base_coord_dim)d == 3) {
        /*
         * normal is
         * v0 x v1
         *
         *    /\\
         * v0/  \\
         *  /    \\
         * /------\\
         *    v1
         */
        for (i = 0; i < 3; ++i) {
            v0[i] = base_coords[%(base_coord_dim)d+i] - base_coords[i];
            v1[i] = base_coords[2*%(base_coord_dim)d+i] - base_coords[i];
        }
        n[0] = v0[1] * v1[2] - v0[2] * v1[1];
        n[1] = v0[2] * v1[0] - v0[0] * v1[2];
        n[2] = v0[0] * v1[1] - v0[1] * v1[0];
    }
    for (i = 0; i < %(base_map_arity)d; ++i) {
        for (c = 0; c < %(base_coord_dim)d; ++c) {
            x[c] += base_coords[i*%(base_coord_dim)d+c];
        }
    }
    for (i = 0; i < %(base_coord_dim)d; ++i) {
        dot += x[i] * n[i];
        norm += n[i] * n[i];
    }
    /*
     * Make inward-pointing normals point out
     */
    norm = sqrt(norm);
    norm *= (dot < 0 ? -1 : 1);
    for (d = 0; d < %(base_map_arity)d; ++d) {
        for (c = 0; c < %(base_coord_dim)d; ++c ) {
            ext_coords[2*d*%(base_coord_dim)d+c] = base_coords[d*%(base_coord_dim)d+c] + n[c] * layer_height[0] * layer / norm;
            ext_coords[(2*d+1)*%(base_coord_dim)d+c] = base_coords[d*%(base_coord_dim)d+c] + n[c] * layer_height[0] * (layer + 1)/ norm;
        }
    }
}""" % {'base_map_arity': base_coords.cell_node_map().arity, 'base_coord_dim': base_coords.function_space().value_size},
            "pyop2_kernel_radial_hedgehog_extrusion")
    else:
        raise NotImplementedError('Unsupported extrusion type "%s"' % extrusion_type)

    height = op2.Global(1, layer_height, dtype=float)
    op2.par_loop(kernel,
                 ext_coords.cell_set,
                 base_coords.dat(op2.READ, base_coords.cell_node_map()),
                 ext_coords.dat(op2.WRITE, ext_coords.cell_node_map()),
                 height(op2.READ),
                 pass_layer_arg=True)


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
            flat_entity_dofs[b][i] = (entity_dofs[(b, 0)][2*i] +
                                      entity_dofs[(b, 1)][i] +
                                      entity_dofs[(b, 0)][2*i+1])
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
