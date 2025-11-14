try:
    import vtkmodules.vtkCommonDataModel  # noqa: F401
    from firedrake.output.vtk_output import VTKFile  # noqa: F401
except ModuleNotFoundError:
    from firedrake.output.vtk_unavailable import VTKFile  # noqa: F401
