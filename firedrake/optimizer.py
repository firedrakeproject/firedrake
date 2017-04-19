from __future__ import absolute_import, print_function, division


def slope(mesh, debug=False):
    """Initialize the SLOPE library by providing information about the mesh,
    including:

        * Mesh coordinates
        * All available maps binding sets of mesh components
    """
    try:
        from pyslope import slope
    except ImportError:
        return

    # Add coordinates
    if debug:
        coords = mesh.coordinates.dat
        slope.set_debug_mode('MINIMAL', (coords.dataset.set.name, coords._data, coords.shape[1]))
