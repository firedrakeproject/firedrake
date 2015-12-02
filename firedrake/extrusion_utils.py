from __future__ import absolute_import
import numpy as np

from pyop2 import op2


def make_extruded_coords(extruded_topology, base_coords, ext_coords,
                         layer_height, extrusion_type='uniform', kernel=None):
    """
    Given either a kernel or a (fixed) layer_height, compute an
    extruded coordinate field for an extruded mesh.

    :arg extruded_topology: an :class:`ExtrudedMeshTopology` to extrude
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
                    int **layer, double *layer_height)

    The kernel iterates over the cells of the mesh and receives as
    arguments the coordinates of the base cell (to read), the
    coordinates on the extruded cell (to write to), the layer number
    of each cell and the fixed layer height.
    """
    vert_space = ext_coords.function_space().ufl_element()._B
    if kernel is None and not (vert_space.degree() == 1 and
                               vert_space.family() in ['Lagrange',
                                                       'Discontinuous Lagrange']):
        raise RuntimeError('Extrusion of coordinates is only possible for a P1 or P1dg interval unless a custom kernel is provided')
    if kernel is not None:
        pass
    elif extrusion_type == 'uniform':
        kernel = op2.Kernel("""
        void uniform_extrusion_kernel(double **base_coords,
                    double **ext_coords,
                    int **layer,
                    double *layer_height) {
            for ( int d = 0; d < %(base_map_arity)d; d++ ) {
                for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                    ext_coords[2*d][c] = base_coords[d][c];
                    ext_coords[2*d+1][c] = base_coords[d][c];
                }
                ext_coords[2*d][%(base_coord_dim)d] = *layer_height * (layer[0][0]);
                ext_coords[2*d+1][%(base_coord_dim)d] = *layer_height * (layer[0][0] + 1);
            }
        }""" % {'base_map_arity': base_coords.cell_node_map().arity,
                'base_coord_dim': base_coords.function_space().dim},
            "uniform_extrusion_kernel")
    elif extrusion_type == 'radial':
        kernel = op2.Kernel("""
        void radial_extrusion_kernel(double **base_coords,
                   double **ext_coords,
                   int **layer,
                   double *layer_height) {
            for ( int d = 0; d < %(base_map_arity)d; d++ ) {
                double norm = 0.0;
                for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                    norm += base_coords[d][c] * base_coords[d][c];
                }
                norm = sqrt(norm);
                for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                    ext_coords[2*d][c] = base_coords[d][c] * (1 + (*layer_height * layer[0][0])/norm);
                    ext_coords[2*d+1][c] = base_coords[d][c] * (1 + (*layer_height * (layer[0][0]+1))/norm);
                }
            }
        }""" % {'base_map_arity': base_coords.cell_node_map().arity,
                'base_coord_dim': base_coords.function_space().dim},
            "radial_extrusion_kernel")
    elif extrusion_type == 'radial_hedgehog':
        # Only implemented for interval in 2D and triangle in 3D.
        # gdim != tdim already checked in ExtrudedMesh constructor.
        if base_coords.cell().topological_dimension() not in [1, 2]:
            raise NotImplementedError("Hedgehog extrusion not implemented for %s" % base_coords.cell())
        kernel = op2.Kernel("""
        void radial_hedgehog_extrusion_kernel(double **base_coords,
                                              double **ext_coords,
                                              int **layer,
                                              double *layer_height) {
            double v0[%(base_coord_dim)d];
            double v1[%(base_coord_dim)d];
            double n[%(base_map_arity)d];
            double x[%(base_map_arity)d] = {0};
            double dot = 0.0;
            double norm = 0.0;
            int i, c, d;
            if (%(base_coord_dim)d == 2) {
                /*
                 * normal is:
                 * (0 -1) (x2 - x1)
                 * (1  0) (y2 - y1)
                 */
                n[0] = -(base_coords[1][1] - base_coords[0][1]);
                n[1] = base_coords[1][0] - base_coords[0][0];
            } else if (%(base_coord_dim)d == 3) {
                /*
                 * normal is
                 * v0 x v1
                 *
                 *    /\
                 * v0/  \
                 *  /    \
                 * /------\
                 *    v1
                 */
                for (i = 0; i < 3; ++i) {
                    v0[i] = base_coords[1][i] - base_coords[0][i];
                    v1[i] = base_coords[2][i] - base_coords[0][i];
                }
                n[0] = v0[1] * v1[2] - v0[2] * v1[1];
                n[1] = v0[2] * v1[0] - v0[0] * v1[2];
                n[2] = v0[0] * v1[1] - v0[1] * v1[0];
            }
            for (i = 0; i < %(base_map_arity)d; ++i) {
                for (c = 0; c < %(base_coord_dim)d; ++c) {
                    x[i] += base_coords[c][i];
                }
            }
            for (i = 0; i < %(base_map_arity)d; ++i) {
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
                    ext_coords[2*d][c] = base_coords[d][c] + n[c] * layer_height[0] * layer[0][0] / norm;
                    ext_coords[2*d+1][c] = base_coords[d][c] + n[c] * layer_height[0] * (layer[0][0] + 1)/ norm;
                }
            }
        }""" % {'base_map_arity': base_coords.cell_node_map().arity,
                'base_coord_dim': base_coords.function_space().dim},
            "radial_hedgehog_extrusion_kernel")
    else:
        raise NotImplementedError('Unsupported extrusion type "%s"' % extrusion_type)

    # Dat to hold layer number
    import firedrake.functionspace as fs
    layer_fs = fs.FunctionSpace(extruded_topology, 'DG', 0)
    layers = extruded_topology.layers
    layer = op2.Dat(layer_fs.dof_dset,
                    np.repeat(np.arange(layers-1, dtype=np.int32),
                              extruded_topology.cell_set.total_size).reshape(layers-1, extruded_topology.cell_set.total_size).T.ravel(), dtype=np.int32)
    height = op2.Global(1, layer_height, dtype=float)
    op2.par_loop(kernel,
                 ext_coords.cell_set,
                 base_coords.dat(op2.READ, base_coords.cell_node_map()),
                 ext_coords.dat(op2.WRITE, ext_coords.cell_node_map()),
                 layer(op2.READ, layer_fs.cell_node_map()),
                 height(op2.READ))
