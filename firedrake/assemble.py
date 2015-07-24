import numpy
import ufl

from pyop2 import op2
from pyop2.exceptions import MapValueError
from pyop2.profiling import timed_region, profile

import assembly_cache
import assemble_expressions
import ffc_interface
import function
import functionspace
import matrix
import parameters
import solving


__all__ = ["assemble"]


@profile
def assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
             inverse=False, nest=None):
    """Evaluate f.

    :arg f: a :class:`ufl.Form` or :class:`ufl.core.expr.Expr`.
    :arg tensor: an existing tensor object to place the result in
         (optional).
    :arg bcs: a list of boundary conditions to apply (optional).
    :arg form_compiler_parameters: (optional) dict of parameters to pass to
         the form compiler.  Ignored if not assembling a
         :class:`ufl.Form`.  Any parameters provided here will be
         overridden by parameters set on the :class;`ufl.Measure` in the
         form.  For example, if a :data:`quadrature_degree` of 4 is
         specified in this argument, but a degree of 3 is requested in
         the measure, the latter will be used.
    :arg inverse: (optional) if f is a 2-form, then assemble the inverse
         of the local matrices.
    :arg nest: (optional) flag indicating if a 2-form (matrix) on a
         mixed space should be assembled as a block matrix (if
         :data:`nest` is :data:`True`) or not.  The default value is
         taken from the parameters dict :data:`parameters["matnest"]`.

    If f is a :class:`ufl.Form` then this evaluates the corresponding
    integral(s) and returns a :class:`float` for 0-forms, a
    :class:`.Function` for 1-forms and a :class:`.Matrix` for 2-forms.

    If f is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    If ``bcs`` is supplied and ``f`` is a 2-form, the rows and columns
    of the resulting :class:`.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``f`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.
    """

    if isinstance(f, ufl.form.Form):
        return _assemble(f, tensor=tensor, bcs=solving._extract_bcs(bcs),
                         form_compiler_parameters=form_compiler_parameters,
                         inverse=inverse, nest=nest)
    elif isinstance(f, ufl.core.expr.Expr):
        return assemble_expressions.assemble_expression(f)
    else:
        raise TypeError("Unable to assemble: %r" % f)


def _assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
              inverse=False, nest=None):
    """Assemble the form f and return a Firedrake object representing the
    result. This will be a :class:`float` for 0-forms, a
    :class:`.Function` for 1-forms and a :class:`.Matrix` for 2-forms.

    :arg bcs: A tuple of :class`.DirichletBC`\s to be applied.
    :arg tensor: An existing tensor object into which the form should be
        assembled. If this is not supplied, a new tensor will be created for
        the purpose.
    :arg form_compiler_parameters: (optional) dict of parameters to pass to
        the form compiler.
    :arg inverse: (optional) if f is a 2-form, then assemble the inverse
         of the local matrices.
    :arg nest: (optional) flag indicating if matrices on mixed spaces
         should be built in blocks or monolithically.

    """

    if form_compiler_parameters:
        form_compiler_parameters = form_compiler_parameters.copy()
    else:
        form_compiler_parameters = {}
    form_compiler_parameters["assemble_inverse"] = inverse

    kernels = ffc_interface.compile_form(f, "form", parameters=form_compiler_parameters,
                                         inverse=inverse)
    rank = len(f.arguments())

    is_mat = rank == 2
    is_vec = rank == 1

    if any(isinstance(coeff.function_space(), functionspace.IndexedVFS)
           for coeff in f.coefficients()):
        raise NotImplementedError("Integration of subscribed VFS not yet implemented")

    if inverse and rank != 2:
        raise ValueError("Can only assemble the inverse of a 2-form")

    integrals = f.integrals()

    def get_rank(arg):
        return arg.function_space().rank

    if nest is None:
        nest = parameters.parameters["matnest"]
    # Pass this through for assembly caching purposes
    form_compiler_parameters["matnest"] = nest

    zero_tensor = lambda: None

    if is_mat:
        test, trial = f.arguments()

        map_pairs = []
        cell_domains = []
        exterior_facet_domains = []
        interior_facet_domains = []
        if tensor is None:
            # For horizontal facets of extrded meshes, the corresponding domain
            # in the base mesh is the cell domain. Hence all the maps used for top
            # bottom and interior horizontal facets will use the cell to dofs map
            # coming from the base mesh as a starting point for the actual dynamic map
            # computation.
            for integral in integrals:
                integral_type = integral.integral_type()
                if integral_type == "cell":
                    cell_domains.append(op2.ALL)
                elif integral_type == "exterior_facet":
                    exterior_facet_domains.append(op2.ALL)
                elif integral_type == "interior_facet":
                    interior_facet_domains.append(op2.ALL)
                elif integral_type == "exterior_facet_bottom":
                    cell_domains.append(op2.ON_BOTTOM)
                elif integral_type == "exterior_facet_top":
                    cell_domains.append(op2.ON_TOP)
                elif integral_type == "exterior_facet_vert":
                    exterior_facet_domains.append(op2.ALL)
                elif integral_type == "interior_facet_horiz":
                    cell_domains.append(op2.ON_INTERIOR_FACETS)
                elif integral_type == "interior_facet_vert":
                    interior_facet_domains.append(op2.ALL)
                else:
                    raise RuntimeError('Unknown integral type "%s"' % integral_type)

            # To avoid an extra check for extruded domains, the maps that are being passed in
            # are DecoratedMaps. For the non-extruded case the DecoratedMaps don't restrict the
            # space over which we iterate as the domains are dropped at Sparsity construction
            # time. In the extruded case the cell domains are used to identify the regions of the
            # mesh which require allocation in the sparsity.
            if cell_domains:
                map_pairs.append((op2.DecoratedMap(test.cell_node_map(), cell_domains),
                                  op2.DecoratedMap(trial.cell_node_map(), cell_domains)))
            if exterior_facet_domains:
                map_pairs.append((op2.DecoratedMap(test.exterior_facet_node_map(), exterior_facet_domains),
                                  op2.DecoratedMap(trial.exterior_facet_node_map(), exterior_facet_domains)))
            if interior_facet_domains:
                map_pairs.append((op2.DecoratedMap(test.interior_facet_node_map(), interior_facet_domains),
                                  op2.DecoratedMap(trial.interior_facet_node_map(), interior_facet_domains)))

            map_pairs = tuple(map_pairs)
            # Construct OP2 Mat to assemble into
            fs_names = (
                test.function_space().name, trial.function_space().name)
            sparsity = op2.Sparsity((test.function_space().dof_dset,
                                     trial.function_space().dof_dset),
                                    map_pairs,
                                    "%s_%s_sparsity" % fs_names,
                                    nest=nest)
            result_matrix = matrix.Matrix(f, bcs, sparsity, numpy.float64,
                                          "%s_%s_matrix" % fs_names)
            tensor = result_matrix._M
        else:
            result_matrix = tensor
            # Replace any bcs on the tensor we passed in
            result_matrix.bcs = bcs
            tensor = tensor._M
            zero_tensor = lambda: tensor.zero()

        def mat(testmap, trialmap, i, j):
            return tensor[i, j](op2.INC,
                                (testmap(test.function_space()[i])[op2.i[0]],
                                 trialmap(trial.function_space()[j])[op2.i[1]]),
                                flatten=True)
        result = lambda: result_matrix
    elif is_vec:
        test = f.arguments()[0]
        if tensor is None:
            result_function = function.Function(test.function_space())
            tensor = result_function.dat
        else:
            result_function = tensor
            tensor = result_function.dat
            zero_tensor = lambda: tensor.zero()

        def vec(testmap, i):
            return tensor[i](op2.INC,
                             testmap(test.function_space()[i])[op2.i[0]],
                             flatten=True)
        result = lambda: result_function
    else:
        # 0-forms are always scalar
        if tensor is None:
            tensor = op2.Global(1, [0.0])
        result = lambda: tensor.data[0]

    # Since applying boundary conditions to a matrix changes the
    # initial assembly, to support:
    #     A = assemble(a)
    #     bc.apply(A)
    #     solve(A, ...)
    # we need to defer actually assembling the matrix until just
    # before we need it (when we know if there are any bcs to be
    # applied).  To do so, we build a closure that carries out the
    # assembly and stash that on the Matrix object.  When we hit a
    # solve, we funcall the closure with any bcs the Matrix now has to
    # assemble it.
    def thunk(bcs):
        zero_tensor()
        for (i, j), integral_type, subdomain_id, coords, coefficients, needs_orientations, kernel in kernels:
            m = coords.function_space().mesh()
            if needs_orientations:
                cell_orientations = m.cell_orientations()
            # Extract block from tensor and test/trial spaces
            # FIXME Ugly variable renaming required because functions are not
            # lexical closures in Python and we're writing to these variables
            if is_mat and tensor.sparsity.shape > (1, 1):
                tsbc = []
                trbc = []
                # Unwind IndexedVFS to check for matching BCs
                for bc in bcs:
                    fs = bc.function_space()
                    if isinstance(fs, functionspace.IndexedVFS):
                        fs = fs._parent
                    if fs.index == i:
                        tsbc.append(bc)
                    if fs.index == j:
                        trbc.append(bc)
            elif is_mat:
                tsbc, trbc = bcs, bcs
            if integral_type == 'cell':
                with timed_region("Assemble cells"):
                    if is_mat:
                        tensor_arg = mat(lambda s: s.cell_node_map(tsbc),
                                         lambda s: s.cell_node_map(trbc),
                                         i, j)
                    elif is_vec:
                        tensor_arg = vec(lambda s: s.cell_node_map(), i)
                    else:
                        tensor_arg = tensor(op2.INC)

                    itspace = m.cell_set
                    args = [kernel, itspace, tensor_arg,
                            coords.dat(op2.READ, coords.cell_node_map(),
                                       flatten=True)]

                    if needs_orientations:
                        args.append(cell_orientations.dat(op2.READ,
                                                          cell_orientations.cell_node_map(),
                                                          flatten=True))
                    for c in coefficients:
                        args.append(c.dat(op2.READ, c.cell_node_map(),
                                          flatten=True))

                    try:
                        op2.par_loop(*args)
                    except MapValueError:
                        raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            elif integral_type in ['exterior_facet', 'exterior_facet_vert']:
                with timed_region("Assemble exterior facets"):
                    if is_mat:
                        tensor_arg = mat(lambda s: s.exterior_facet_node_map(tsbc),
                                         lambda s: s.exterior_facet_node_map(trbc),
                                         i, j)
                    elif is_vec:
                        tensor_arg = vec(lambda s: s.exterior_facet_node_map(), i)
                    else:
                        tensor_arg = tensor(op2.INC)
                    args = [kernel, m.exterior_facets.measure_set(integral_type,
                                                                  subdomain_id),
                            tensor_arg,
                            coords.dat(op2.READ, coords.exterior_facet_node_map(),
                                       flatten=True)]
                    if needs_orientations:
                        args.append(cell_orientations.dat(op2.READ,
                                                          cell_orientations.exterior_facet_node_map(),
                                                          flatten=True))
                    for c in coefficients:
                        args.append(c.dat(op2.READ, c.exterior_facet_node_map(),
                                          flatten=True))
                    args.append(m.exterior_facets.local_facet_dat(op2.READ))
                    try:
                        op2.par_loop(*args)
                    except MapValueError:
                        raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            elif integral_type in ['exterior_facet_top', 'exterior_facet_bottom']:
                with timed_region("Assemble exterior facets"):
                    # In the case of extruded meshes with horizontal facet integrals, two
                    # parallel loops will (potentially) get created and called based on the
                    # domain id: interior horizontal, bottom or top.

                    # Get the list of sets and globals required for parallel loop construction.
                    set_global_list = m.exterior_facets.measure_set(integral_type, subdomain_id)

                    # Iterate over the list and assemble all the args of the parallel loop
                    for (index, set) in set_global_list:
                        if is_mat:
                            tensor_arg = mat(lambda s: op2.DecoratedMap(s.cell_node_map(tsbc), index),
                                             lambda s: op2.DecoratedMap(s.cell_node_map(trbc), index),
                                             i, j)
                        elif is_vec:
                            tensor_arg = vec(lambda s: s.cell_node_map(), i)
                        else:
                            tensor_arg = tensor(op2.INC)

                        # Add the kernel, iteration set and coordinate fields to the loop args
                        args = [kernel, set, tensor_arg,
                                coords.dat(op2.READ, coords.cell_node_map(),
                                           flatten=True)]
                        if needs_orientations:
                            args.append(cell_orientations.dat(op2.READ,
                                                              cell_orientations.cell_node_map(),
                                                              flatten=True))
                        for c in coefficients:
                            args.append(c.dat(op2.READ, c.cell_node_map(),
                                              flatten=True))
                        try:
                            op2.par_loop(*args, iterate=index)
                        except MapValueError:
                            raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            elif integral_type in ['interior_facet', 'interior_facet_vert']:
                with timed_region("Assemble interior facets"):
                    if is_mat:
                        tensor_arg = mat(lambda s: s.interior_facet_node_map(tsbc),
                                         lambda s: s.interior_facet_node_map(trbc),
                                         i, j)
                    elif is_vec:
                        tensor_arg = vec(lambda s: s.interior_facet_node_map(), i)
                    else:
                        tensor_arg = tensor(op2.INC)
                    args = [kernel, m.interior_facets.set, tensor_arg,
                            coords.dat(op2.READ, coords.interior_facet_node_map(),
                                       flatten=True)]
                    if needs_orientations:
                        args.append(cell_orientations.dat(op2.READ,
                                                          cell_orientations.interior_facet_node_map(),
                                                          flatten=True))
                    for c in coefficients:
                        args.append(c.dat(op2.READ, c.interior_facet_node_map(),
                                          flatten=True))
                    args.append(m.interior_facets.local_facet_dat(op2.READ))
                    try:
                        op2.par_loop(*args)
                    except MapValueError:
                        raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            elif integral_type == 'interior_facet_horiz':
                with timed_region("Assemble interior facets"):
                    if is_mat:
                        tensor_arg = mat(lambda s: op2.DecoratedMap(s.cell_node_map(tsbc),
                                                                    op2.ON_INTERIOR_FACETS),
                                         lambda s: op2.DecoratedMap(s.cell_node_map(trbc),
                                                                    op2.ON_INTERIOR_FACETS),
                                         i, j)
                    elif is_vec:
                        tensor_arg = vec(lambda s: s.cell_node_map(), i)
                    else:
                        tensor_arg = tensor(op2.INC)

                    args = [kernel, m.interior_facets.measure_set(integral_type, subdomain_id),
                            tensor_arg,
                            coords.dat(op2.READ, coords.cell_node_map(),
                                       flatten=True)]
                    if needs_orientations:
                        args.append(cell_orientations.dat(op2.READ,
                                                          cell_orientations.cell_node_map(),
                                                          flatten=True))
                    for c in coefficients:
                        args.append(c.dat(op2.READ, c.cell_node_map(),
                                          flatten=True))
                    try:
                        op2.par_loop(*args, iterate=op2.ON_INTERIOR_FACETS)
                    except MapValueError:
                        raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

            else:
                raise RuntimeError('Unknown integral type "%s"' % integral_type)

        # Must apply bcs outside loop over kernels because we may wish
        # to apply bcs to a block which is otherwise zero, and
        # therefore does not have an associated kernel.
        if bcs is not None and is_mat:
            with timed_region('DirichletBC apply'):
                for bc in bcs:
                    fs = bc.function_space()
                    if isinstance(fs, functionspace.MixedFunctionSpace):
                        raise RuntimeError("""Cannot apply boundary conditions to full mixed space. Did you forget to index it?""")
                    shape = tensor.sparsity.shape
                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            # Set diagonal entries on bc nodes to 1 if the current
                            # block is on the matrix diagonal and its index matches the
                            # index of the function space the bc is defined on.
                            if i != j:
                                continue
                            if isinstance(fs, functionspace.IndexedFunctionSpace):
                                # Mixed, index
                                if fs.index == i:
                                    tensor[i, j].set_local_diagonal_entries(bc.nodes)
                            elif isinstance(fs, functionspace.IndexedVFS):
                                if isinstance(fs._parent, functionspace.IndexedFunctionSpace):
                                    if fs._parent.index != i:
                                        continue
                                tensor[i, j].set_local_diagonal_entries(bc.nodes, idx=fs.index)
                            elif fs.index is None:
                                tensor[i, j].set_local_diagonal_entries(bc.nodes)
                            else:
                                raise RuntimeError("Unhandled BC case")
        if bcs is not None and is_vec:
            for bc in bcs:
                bc.apply(result_function)
        if is_mat:
            # Queue up matrix assembly (after we've done all the other operations)
            tensor.assemble()
        return result()

    thunk = assembly_cache._cache_thunk(thunk, f, result(), form_compiler_parameters)

    if is_mat:
        result_matrix._assembly_callback = thunk
        return result()
    else:
        return thunk(bcs)
