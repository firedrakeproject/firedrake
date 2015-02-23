from petsc import PETSc


def update_parameters(obj, petsc_obj):
    """Update parameters on a petsc object

    :arg obj: An object with a parameters dict (mapping to petsc options).
    :arg petsc_obj: The PETSc object to set parameters on."""
    # Skip if parameters haven't changed
    if hasattr(obj, '_set_parameters') and obj.parameters == obj._set_parameters:
        return
    opts = PETSc.Options(obj._opt_prefix)
    for k, v in obj.parameters.iteritems():
        if type(v) is bool:
            if v:
                opts[k] = None
        else:
            opts[k] = v
    petsc_obj.setFromOptions()
    obj._set_parameters = obj.parameters.copy()
