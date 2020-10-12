from firedrake import FunctionSpace, Function, DirichletBC

__all__ = ['ExtrudedExtendFunction']

def ExtrudedExtendFunction(mesh, fcn, extend_type=None):
    """On an extruded mesh, extend a function by a constant in the extruded
    direction.  The returned function uses the degree 0 'R' space in the
    extruded direction.

    :arg mesh:           the extruded mesh

    :arg fcn:            the function to extend

    ``extend_type`` has three cases:

    ``None``
        ``fcn`` is defined on the base mesh and the returned Function has
        the same values in the extruded direction.

    ``"top"`` or ``"bottom"``
        ``fcn`` is defined on the extruded mesh.  The values of the returned
        Function come from the values of ``fcn`` on the ``"top"``/``"bottom"``
        boundary of the extruded mesh.
    """

    # FIXME need to use fcn.function_space() to get element; e.g.
    #     Q3D = FunctionSpace(mesh, fcn.function_space().ufl_element, vfamily='R', vdegree=0)
    # does not work; note the base mesh is mesh._base_mesh
    
    Q3D = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)  # FIXME
    f3D = Function(Q3D)

    if extend_type == None:
        f3D.dat.data[:] = fcn.dat.data_ro[:]
    else:
        if extend_type in {"top", "bottom"}:
            bc = DirichletBC(fcn.function_space(), 1.0, extend_type)
            f3D.dat.data[:] = fcn.dat.data_ro[bc.nodes]
        else:
            raise ValueError('unknown extend_type')
    return f3D

