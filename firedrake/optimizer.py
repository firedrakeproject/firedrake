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
