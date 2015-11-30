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

    # Should SLOPE print out the tiled, colored meshes ?
    coordinates = None
    if debug:
        coordinates = mesh.coordinates.dat
        coordinates = (coordinates.dataset.set.name,
                       coordinates.data_ro,
                       coordinates.shape[1])
    slope_python.set_debug_mode('MINIMAL', coordinates)

    # Add available maps describing the mesh topology
    # 1) cells to nodes map
    slope_maps = []
    maps = [mesh.coordinates.cell_node_map(),
            mesh.coordinates.interior_facet_node_map()]
    slope_maps = [(m.name, m.iterset.name, m.toset.name, m.values_with_halo) for m in maps]
    slope_python.set_mesh_maps(slope_maps)
