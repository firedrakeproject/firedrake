from __future__ import absolute_import


def slope(mesh, debug=False):
    """Initialize the SLOPE library by providing information about the mesh,
    including:

        * Mesh coordinates
        * All available maps binding sets of mesh components
    """
    try:
        import slope_python
    except ImportError:
        return

    coords = mesh.coordinates.dat

    # Add coordinates
    if debug:
        slope_python.set_debug_mode('VERY_LOW', (coords.dataset.set.name,
                                                 coords.data_ro,
                                                 coords.shape[1]))

    # Add available maps describing the mesh topology
    # 1) cells to nodes map
    slope_maps = []
    maps = [mesh.coordinates.cell_node_map(),
            mesh.coordinates.interior_facet_node_map()]
    slope_maps = [(m.name, m.iterset.name, m.toset.name, m.values) for m in maps]
    slope_python.set_mesh_maps(slope_maps)
