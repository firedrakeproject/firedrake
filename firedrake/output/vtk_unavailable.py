class VTKFile(object):
    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            "Error importing vtkmodules. Firedrake does not install VTK by default, "
            "you may need to install VTK by running\n\t"
            "pip install vtk"
        )
