try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401
    from .vtk_output import VTKFile  # noqa: F401
except ModuleNotFoundError:
    from .vtk_unavailable import VTKFile  # noqa: F401
