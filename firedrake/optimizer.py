import slope_python

def slope(mesh, debug=False):
    """Initialize the SLOPE library by providing information about the mesh,
    including:

        * Mesh coordinates
        * All available maps binding sets of mesh components
    """
    # Add coordinates
    if debug:
        coords = mesh.coordinates.dat
        slope_python.set_debug_mode('VERY_LOW', (coords.dataset.set.name,
                                                 coords._data,
                                                 coords.shape[1]))
    # Add any known maps potentially useful for fusion of cell and facet integrals
    slope_python.HardFusion.add_maps([('cell_to_interior_facet',
                                       mesh.interior_facets.set.name,
                                       mesh.cell_set.name,
                                       mesh.interior_facets.facet_cell)])
