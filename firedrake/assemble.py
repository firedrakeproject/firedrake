from __future__ import absolute_import
import numpy
import ufl

from pyop2 import op2
from pyop2.exceptions import MapValueError

from firedrake import assembly_cache
from firedrake import assemble_expressions
from firedrake import tsfc_interface
from firedrake import function
from firedrake import matrix
from firedrake import parameters
from firedrake import solving
from firedrake import utils


__all__ = ["assemble"]


def assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
             inverse=False, nest=None, matfree=False, state=None):
    """Evaluate f.

    :arg f: a :class:`~ufl.classes.Form` or :class:`~ufl.classes.Expr`.
    :arg tensor: an existing tensor object to place the result in
         (optional).
    :arg bcs: a list of boundary conditions to apply (optional).
    :arg form_compiler_parameters: (optional) dict of parameters to pass to
         the form compiler.  Ignored if not assembling a
         :class:`~ufl.classes.Form`.  Any parameters provided here will be
         overridden by parameters set on the :class:`~ufl.classes.Measure` in the
         form.  For example, if a ``quadrature_degree`` of 4 is
         specified in this argument, but a degree of 3 is requested in
         the measure, the latter will be used.
    :arg inverse: (optional) if f is a 2-form, then assemble the inverse
         of the local matrices.
    :arg nest: (optional) flag indicating if a 2-form (matrix) on a
         mixed space should be assembled as a block matrix (if
         ``nest`` is ``True``) or not.  The default value is
         taken from the parameters dict, ``parameters["matnest"]``.

    If f is a :class:`~ufl.classes.Form` then this evaluates the corresponding
    integral(s) and returns a :class:`float` for 0-forms, a
    :class:`.Function` for 1-forms and a :class:`.Matrix` for 2-forms.

    If f is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`.

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
                         inverse=inverse, nest=nest, matfree=matfree, state=state)
    elif isinstance(f, ufl.core.expr.Expr):
        return assemble_expressions.assemble_expression(f)
    else:
        raise TypeError("Unable to assemble: %r" % f)


@utils.known_pyop2_safe
def _assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
              inverse=False, nest=None, matfree=False, state=None):
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

    kernels = tsfc_interface.compile_form(f, "form", parameters=form_compiler_parameters,
                                          inverse=inverse)
    rank = len(f.arguments())

    is_mat = rank == 2
    is_vec = rank == 1

    if any((coeff.function_space() and coeff.function_space().component is not None)
           for coeff in f.coefficients()):
        raise NotImplementedError("Integration of subscripted VFS not yet implemented")

    if inverse and rank != 2:
        raise ValueError("Can only assemble the inverse of a 2-form")

    integrals = f.integrals()

    if nest is None:
        nest = parameters.parameters["matnest"]
    # Pass this through for assembly caching purposes
    form_compiler_parameters["matnest"] = nest

    zero_tensor = lambda: None

    if is_mat:
        if matfree:  # intercept matrix-free matrices here
            if tensor is None:
                from firedrake.matrix import ImplicitMatrix
                return ImplicitMatrix(f, bcs, state=state,
                                      fc_params=form_compiler_parameters)
            else:
                return tensor
        # Done intercepting matrix-free matrices

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
                    raise ValueError('Unknown integral type "%s"' % integral_type)

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
            fs_names = (test.function_space().name, trial.function_space().name)
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

    subdomain_data = f.subdomain_data()
    coefficients = f.coefficients()
    domains = f.ufl_domains()
    if len(domains) != 1:
        raise NotImplementedError("Assembly of forms with more than one domain not supported")
    m = domains[0]
    # Ensure mesh is "initialised" (could have got here without
    # building a functionspace (e.g. if integrating a constant)).
    m.init()
    subdomain_data = subdomain_data[m]

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
        for indices, (kernel, integral_type, needs_orientations, subdomain_id, coeff_map) in kernels:
            # Find argument space indices
            if is_mat:
                i, j = indices
            elif is_vec:
                i, = indices
            else:
                assert len(indices) == 0

            sdata = subdomain_data.get(integral_type, None)
            if integral_type != 'cell' and sdata is not None:
                raise NotImplementedError("subdomain_data only supported with cell integrals.")

            # Extract block from tensor and test/trial spaces
            # FIXME Ugly variable renaming required because functions are not
            # lexical closures in Python and we're writing to these variables
            if is_mat and tensor.sparsity.shape > (1, 1):
                tsbc = []
                trbc = []
                # Unwind ComponentFunctionSpace to check for matching BCs
                for bc in bcs:
                    fs = bc.function_space()
                    if fs.component is not None:
                        fs = fs.parent
                    if fs.index == i:
                        tsbc.append(bc)
                    if fs.index == j:
                        trbc.append(bc)
            elif is_mat:
                tsbc, trbc = bcs, bcs

            # Now build arguments for the par_loop
            kwargs = {}
            # Some integrals require non-coefficient arguments at the
            # end (facet number information).
            extra_args = []
            # Decoration for applying to matrix maps in extruded case
            decoration = None
            if integral_type == "cell":
                itspace = sdata or m.cell_subset(subdomain_id)

                if subdomain_id not in ["otherwise", "everywhere"] and \
                   sdata is not None:
                    raise ValueError("Cannot use subdomain data and subdomain_id")

                def get_map(x, bcs=None, decoration=None):
                    return x.cell_node_map(bcs)

            elif integral_type in ("exterior_facet", "exterior_facet_vert"):
                itspace = m.exterior_facets.measure_set(integral_type, subdomain_id)
                extra_args.append(m.exterior_facets.local_facet_dat(op2.READ))

                def get_map(x, bcs=None, decoration=None):
                    return x.exterior_facet_node_map(bcs)

            elif integral_type in ("exterior_facet_top", "exterior_facet_bottom"):
                # In the case of extruded meshes with horizontal facet integrals, two
                # parallel loops will (potentially) get created and called based on the
                # domain id: interior horizontal, bottom or top.
                index, itspace = m.exterior_facets.measure_set(integral_type, subdomain_id)
                decoration = index
                kwargs["iterate"] = index

                def get_map(x, bcs=None, decoration=None):
                    map_ = x.cell_node_map(bcs)
                    if decoration is not None:
                        return op2.DecoratedMap(map_, decoration)
                    return map_

            elif integral_type in ("interior_facet", "interior_facet_vert"):
                itspace = m.interior_facets.set
                extra_args.append(m.interior_facets.local_facet_dat(op2.READ))

                def get_map(x, bcs=None, decoration=None):
                    return x.interior_facet_node_map(bcs)

            elif integral_type == "interior_facet_horiz":
                itspace = m.interior_facets.measure_set(integral_type, subdomain_id)
                decoration = op2.ON_INTERIOR_FACETS
                kwargs["iterate"] = op2.ON_INTERIOR_FACETS

                def get_map(x, bcs=None, decoration=None):
                    map_ = x.cell_node_map(bcs)
                    if decoration is not None:
                        return op2.DecoratedMap(map_, decoration)
                    return map_

            else:
                raise ValueError("Unknown integral type '%s'" % integral_type)

            # Output argument
            if is_mat:
                tensor_arg = mat(lambda s: get_map(s, tsbc, decoration),
                                 lambda s: get_map(s, trbc, decoration),
                                 i, j)
            elif is_vec:
                tensor_arg = vec(lambda s: get_map(s), i)
            else:
                tensor_arg = tensor(op2.INC)

            coords = m.coordinates
            args = [kernel, itspace, tensor_arg,
                    coords.dat(op2.READ, get_map(coords), flatten=True)]
            if needs_orientations:
                o = m.cell_orientations()
                args.append(o.dat(op2.READ, get_map(o), flatten=True))
            for n in coeff_map:
                c = coefficients[n]
                for c_ in c.split():
                    args.append(c_.dat(op2.READ, get_map(c_), flatten=True))

            args.extend(extra_args)
            try:
                op2.par_loop(*args, **kwargs)
            except MapValueError:
                raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

        # Must apply bcs outside loop over kernels because we may wish
        # to apply bcs to a block which is otherwise zero, and
        # therefore does not have an associated kernel.
        if bcs is not None and is_mat:
            for bc in bcs:
                fs = bc.function_space()
                if len(fs) > 1:
                    raise RuntimeError("""Cannot apply boundary conditions to full mixed space. Did you forget to index it?""")
                shape = tensor.sparsity.shape
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        # Set diagonal entries on bc nodes to 1 if the current
                        # block is on the matrix diagonal and its index matches the
                        # index of the function space the bc is defined on.
                        if i != j:
                            continue
                        if fs.component is None and fs.index is not None:
                            # Mixed, index (no ComponentFunctionSpace)
                            if fs.index == i:
                                tensor[i, j].set_local_diagonal_entries(bc.nodes)
                        elif fs.component is not None:
                            # ComponentFunctionSpace, check parent index
                            if fs.parent.index is not None:
                                # Mixed, index doesn't match
                                if fs.parent.index != i:
                                    continue
                            # Index matches
                            tensor[i, j].set_local_diagonal_entries(bc.nodes, idx=fs.component)
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
