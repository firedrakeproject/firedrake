import ufl
from collections import defaultdict
from itertools import chain
import functools

from pyop2 import op2
from pyop2.exceptions import MapValueError, SparsityFormatError

from firedrake import assemble_expressions
from firedrake import tsfc_interface
from firedrake import function
from firedrake import matrix
from firedrake import parameters
from firedrake import solving
from firedrake import utils
from firedrake.slate import slate
from firedrake.slate import slac
from firedrake.bcs import DirichletBC, EquationBCSplit
from firedrake.utils import ScalarType


__all__ = ["assemble"]


def assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
             inverse=False, mat_type=None, sub_mat_type=None,
             appctx={}, options_prefix=None, **kwargs):
    r"""Evaluate f.

    :arg f: a :class:`~ufl.classes.Form`, :class:`~ufl.classes.Expr` or
            a :class:`~slate.TensorBase` expression.
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
    :arg mat_type: (optional) string indicating how a 2-form (matrix) should be
         assembled -- either as a monolithic matrix ('aij' or 'baij'), a block matrix
         ('nest'), or left as a :class:`.ImplicitMatrix` giving matrix-free
         actions ('matfree').  If not supplied, the default value in
         ``parameters["default_matrix_type"]`` is used.  BAIJ differs
         from AIJ in that only the block sparsity rather than the dof
         sparsity is constructed.  This can result in some memory
         savings, but does not work with all PETSc preconditioners.
         BAIJ matrices only make sense for non-mixed matrices.
    :arg sub_mat_type: (optional) string indicating the matrix type to
         use *inside* a nested block matrix.  Only makes sense if
         ``mat_type`` is ``nest``.  May be one of 'aij' or 'baij'.  If
         not supplied, defaults to ``parameters["default_sub_matrix_type"]``.
    :arg appctx: Additional information to hang on the assembled
         matrix if an implicit matrix is requested (mat_type "matfree").
    :arg options_prefix: PETSc options prefix to apply to matrices.

    If f is a :class:`~ufl.classes.Form` then this evaluates the corresponding
    integral(s) and returns a :class:`float` for 0-forms, a
    :class:`.Function` for 1-forms and a :class:`.Matrix` or :class:`.ImplicitMatrix`
    for 2-forms.

    If f is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`.

    If f is a Slate tensor expression, then it will be compiled using Slate's
    linear algebra compiler.

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    If ``bcs`` is supplied and ``f`` is a 2-form, the rows and columns
    of the resulting :class:`.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``f`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.
    """

    if "nest" in kwargs:
        nest = kwargs.pop("nest")
        from firedrake.logging import warning, RED
        warning(RED % "The 'nest' argument is deprecated, please set 'mat_type' instead")
        if nest is not None:
            mat_type = "nest" if nest else "aij"

    collect_loops = kwargs.pop("collect_loops", False)
    allocate_only = kwargs.pop("allocate_only", False)
    if len(kwargs) > 0:
        raise TypeError("Unknown keyword arguments '%s'" % ', '.join(kwargs.keys()))

    if isinstance(f, (ufl.form.Form, slate.TensorBase)):
        loops = _assemble(f, tensor=tensor, bcs=solving._extract_bcs(bcs),
                          form_compiler_parameters=form_compiler_parameters,
                          inverse=inverse, mat_type=mat_type,
                          sub_mat_type=sub_mat_type, appctx=appctx,
                          assemble_now=not collect_loops,
                          allocate_only=allocate_only,
                          options_prefix=options_prefix)
        loops = tuple(loops)
        if collect_loops and not allocate_only:
            # Will this be useful?
            return loops
        else:
            for l in loops:
                m = l()
            return m

    elif isinstance(f, ufl.core.expr.Expr):
        return assemble_expressions.assemble_expression(f)
    else:
        raise TypeError("Unable to assemble: %r" % f)


def allocate_matrix(f, bcs=None, form_compiler_parameters=None,
                    inverse=False, mat_type=None, sub_mat_type=None, appctx={},
                    options_prefix=None):
    r"""Allocate a matrix given a form.  To be used with :func:`create_assembly_callable`.

    .. warning::

       Do not use this function unless you know what you're doing.
    """
    loops = _assemble(f, bcs=bcs, form_compiler_parameters=form_compiler_parameters,
                      inverse=inverse, mat_type=mat_type, sub_mat_type=sub_mat_type,
                      appctx=appctx, allocate_only=True,
                      options_prefix=options_prefix)
    # has only one entry
    return next(loops)()


def create_assembly_callable(f, tensor=None, bcs=None, form_compiler_parameters=None,
                             inverse=False, mat_type=None, sub_mat_type=None):
    r"""Create a callable object than be used to assemble f into a tensor.

    This is really only designed to be used inside residual and
    jacobian callbacks, since it always assembles back into the
    initially provided tensor.  See also :func:`allocate_matrix`.

    .. warning::

       Really do not use this function unless you know what you're doing.
    """
    if tensor is None:
        raise ValueError("Have to provide tensor to write to")
    if mat_type == "matfree":
        return tensor.assemble
    loops = _assemble(f, tensor=tensor, bcs=bcs,
                      form_compiler_parameters=form_compiler_parameters,
                      inverse=inverse, mat_type=mat_type,
                      sub_mat_type=sub_mat_type)

    loops = tuple(loops)

    def thunk():
        for kernel in loops:
            kernel()
    return thunk


@utils.known_pyop2_safe
def _assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
              inverse=False, mat_type=None, sub_mat_type=None,
              appctx={},
              options_prefix=None,
              assemble_now=False,
              allocate_only=False,
              zero_tensor=True):
    r"""Assemble the form or Slate expression f and return a Firedrake object
    representing the result. This will be a :class:`float` for 0-forms/rank-0
    Slate tensors, a :class:`.Function` for 1-forms/rank-1 Slate tensors and
    a :class:`.Matrix` for 2-forms/rank-2 Slate tensors.

    :arg bcs: A tuple of :class`.DirichletBC`\s and/or :class`.EquationBCSplit`\s to be applied.
    :arg tensor: An existing tensor object into which the form should be
        assembled. If this is not supplied, a new tensor will be created for
        the purpose.
    :arg form_compiler_parameters: (optional) dict of parameters to pass to
        the form compiler.
    :arg inverse: (optional) if f is a 2-form, then assemble the inverse
         of the local matrices.
    :arg mat_type: (optional) type for assembled matrices, one of
        "nest", "aij", "baij", or "matfree".
    :arg sub_mat_type: (optional) type for assembled sub matrices
        inside a "nest" matrix.  One of "aij" or "baij".
    :arg appctx: Additional information to hang on the assembled
         matrix if an implicit matrix is requested (mat_type "matfree").
    :arg options_prefix: An options prefix for the PETSc matrix
        (ignored if not assembling a bilinear form).
    """
    if mat_type is None:
        mat_type = parameters.parameters["default_matrix_type"]
    if mat_type not in ["matfree", "aij", "baij", "nest"]:
        raise ValueError("Unrecognised matrix type, '%s'" % mat_type)
    if sub_mat_type is None:
        sub_mat_type = parameters.parameters["default_sub_matrix_type"]
    if sub_mat_type not in ["aij", "baij"]:
        raise ValueError("Invalid submatrix type, '%s' (not 'aij' or 'baij')", sub_mat_type)

    if form_compiler_parameters:
        form_compiler_parameters = form_compiler_parameters.copy()
    else:
        form_compiler_parameters = {}
    form_compiler_parameters["assemble_inverse"] = inverse

    topology = f.ufl_domains()[0].topology
    for m in f.ufl_domains():
        # Ensure mesh is "initialised" (could have got here without
        # building a functionspace (e.g. if integrating a constant)).
        m.init()
        if m.topology != topology:
            raise NotImplementedError("All integration domains must share a mesh topology.")

    for o in chain(f.arguments(), f.coefficients()):
        domain = o.ufl_domain()
        if domain is not None and domain.topology != topology:
            raise NotImplementedError("Assembly with multiple meshes not supported.")

    if isinstance(f, slate.TensorBase):
        kernels = slac.compile_expression(f, tsfc_parameters=form_compiler_parameters)
        integral_types = [kernel.kinfo.integral_type for kernel in kernels]
    else:
        kernels = tsfc_interface.compile_form(f, "form", parameters=form_compiler_parameters, inverse=inverse)
        integral_types = [integral.integral_type() for integral in f.integrals()]

        if bcs is not None:
            for bc in bcs:
                integral_types += [integral.integral_type() for integral in bc.integrals()]

    rank = len(f.arguments())
    is_mat = rank == 2
    is_vec = rank == 1

    if any((coeff.function_space() and coeff.function_space().component is not None)
           for coeff in f.coefficients()):
        raise NotImplementedError("Integration of subscripted VFS not yet implemented")

    if inverse and rank != 2:
        raise ValueError("Can only assemble the inverse of a 2-form")

    zero_tensor_parloop = lambda: None

    if is_mat:
        matfree = mat_type == "matfree"
        nest = mat_type == "nest"
        if nest:
            baij = sub_mat_type == "baij"
        else:
            baij = mat_type == "baij"
        # intercept matrix-free matrices here
        if matfree:
            if inverse:
                raise NotImplementedError("Inverse not implemented with matfree")
            if tensor is None:
                result_matrix = matrix.ImplicitMatrix(f, bcs,
                                                      fc_params=form_compiler_parameters,
                                                      appctx=appctx,
                                                      options_prefix=options_prefix)
                yield lambda: result_matrix
                return
            if not isinstance(tensor, matrix.ImplicitMatrix):
                raise ValueError("Expecting implicit matrix with matfree")
            tensor.assemble()
            yield lambda: tensor
            return

        test, trial = f.arguments()

        map_pairs = []
        cell_domains = []
        exterior_facet_domains = []
        interior_facet_domains = []
        if tensor is None:
            # For horizontal facets of extruded meshes, the corresponding domain
            # in the base mesh is the cell domain. Hence all the maps used for top
            # bottom and interior horizontal facets will use the cell to dofs map
            # coming from the base mesh as a starting point for the actual dynamic map
            # computation.
            for integral_type in integral_types:
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

            # Used for the sparsity construction
            iteration_regions = []
            if cell_domains:
                map_pairs.append((test.cell_node_map(), trial.cell_node_map()))
                iteration_regions.append(tuple(cell_domains))
            if exterior_facet_domains:
                map_pairs.append((test.exterior_facet_node_map(), trial.exterior_facet_node_map()))
                iteration_regions.append(tuple(exterior_facet_domains))
            if interior_facet_domains:
                map_pairs.append((test.interior_facet_node_map(), trial.interior_facet_node_map()))
                iteration_regions.append(tuple(interior_facet_domains))

            map_pairs = tuple(map_pairs)
            # Construct OP2 Mat to assemble into
            fs_names = (test.function_space().name, trial.function_space().name)

            try:
                sparsity = op2.Sparsity((test.function_space().dof_dset,
                                         trial.function_space().dof_dset),
                                        map_pairs,
                                        iteration_regions=iteration_regions,
                                        name="%s_%s_sparsity" % fs_names,
                                        nest=nest,
                                        block_sparse=baij)
            except SparsityFormatError:
                raise ValueError("Monolithic matrix assembly is not supported for systems with R-space blocks.")

            result_matrix = matrix.Matrix(f, bcs, mat_type, sparsity, ScalarType,
                                          "%s_%s_matrix" % fs_names,
                                          options_prefix=options_prefix)
            tensor = result_matrix.M
        else:
            if isinstance(tensor, matrix.ImplicitMatrix):
                raise ValueError("Expecting matfree with implicit matrix")

            result_matrix = tensor
            tensor = tensor.M
            zero_tensor_parloop = tensor.zero

        if result_matrix.block_shape != (1, 1) and mat_type == "baij":
            raise ValueError("BAIJ matrix type makes no sense for mixed spaces, use 'aij'")

        def mat(testmap, trialmap, rowbc, colbc, i, j):
            m = testmap(test.function_space()[i])
            n = trialmap(trial.function_space()[j])
            maps = (m if m else None, n if n else None)

            rlgmap, clgmap = tensor[i, j].local_to_global_maps
            V = test.function_space()[i]
            rlgmap = V.local_to_global_map(rowbc, lgmap=rlgmap)
            V = trial.function_space()[j]
            clgmap = V.local_to_global_map(colbc, lgmap=clgmap)
            if rowbc is None:
                rowbc = []
            if colbc is None:
                colbc = []
            unroll = any(bc.function_space().component is not None
                         for bc in chain(rowbc, colbc) if bc is not None)
            return tensor[i, j](op2.INC, maps, lgmaps=(rlgmap, clgmap), unroll_map=unroll)

        result = lambda: result_matrix
        if allocate_only:
            yield result
            return
    elif is_vec:
        test = f.arguments()[0]
        if tensor is None:
            result_function = function.Function(test.function_space())
            tensor = result_function.dat
        else:
            result_function = tensor
            tensor = result_function.dat
            zero_tensor_parloop = tensor.zero

        def vec(testmap, i):
            _testmap = testmap(test.function_space()[i])
            return tensor[i](op2.INC, _testmap if _testmap else None)
        result = lambda: result_function
    else:
        # 0-forms are always scalar
        if tensor is None:
            tensor = op2.Global(1, [0.0])
        else:
            raise ValueError("Can't assemble 0-form into existing tensor")
        result = lambda: tensor.data[0]

    coefficients = f.coefficients()
    domains = f.ufl_domains()

    # These will be used to correctly interpret the "otherwise"
    # subdomain
    all_integer_subdomain_ids = defaultdict(list)
    for k in kernels:
        if k.kinfo.subdomain_id != "otherwise":
            all_integer_subdomain_ids[k.kinfo.integral_type].append(k.kinfo.subdomain_id)
    for k, v in all_integer_subdomain_ids.items():
        all_integer_subdomain_ids[k] = tuple(sorted(v))

    # In collecting loops mode, we collect the loops, and assume the
    # boundary conditions provided are the ones we want.  It therefore
    # is only used inside residual and jacobian assembly.

    if zero_tensor:
        yield zero_tensor_parloop
    for indices, kinfo in kernels:
        kernel = kinfo.kernel
        integral_type = kinfo.integral_type
        domain_number = kinfo.domain_number
        subdomain_id = kinfo.subdomain_id
        coeff_map = kinfo.coefficient_map
        pass_layer_arg = kinfo.pass_layer_arg
        needs_orientations = kinfo.oriented
        needs_cell_facets = kinfo.needs_cell_facets
        needs_cell_sizes = kinfo.needs_cell_sizes

        m = domains[domain_number]
        subdomain_data = f.subdomain_data()[m]
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
        if is_mat:
            if bcs is not None:
                tsbc = list(bc for bc in chain(*bcs))
                if result_matrix.block_shape > (1, 1):
                    trbc = [bc for bc in tsbc if bc.function_space_index() == j and isinstance(bc, DirichletBC)]
                    tsbc = [bc for bc in tsbc if bc.function_space_index() == i]
                else:
                    trbc = [bc for bc in tsbc if isinstance(bc, DirichletBC)]
            else:
                tsbc = []
                trbc = []

        # Now build arguments for the par_loop
        kwargs = {}
        # Some integrals require non-coefficient arguments at the
        # end (facet number information).
        extra_args = []
        # Decoration for applying to matrix maps in extruded case
        decoration = None
        itspace = m.measure_set(integral_type, subdomain_id,
                                all_integer_subdomain_ids)
        if integral_type == "cell":
            itspace = sdata or itspace
            if subdomain_id not in ["otherwise", "everywhere"] and sdata is not None:
                raise ValueError("Cannot use subdomain data and subdomain_id")

            def get_map(x):
                return x.cell_node_map()

        elif integral_type in ("exterior_facet", "exterior_facet_vert"):
            extra_args.append(m.exterior_facets.local_facet_dat(op2.READ))

            def get_map(x):
                return x.exterior_facet_node_map()

        elif integral_type in ("exterior_facet_top", "exterior_facet_bottom"):
            # In the case of extruded meshes with horizontal facet integrals, two
            # parallel loops will (potentially) get created and called based on the
            # domain id: interior horizontal, bottom or top.
            decoration = {"exterior_facet_top": op2.ON_TOP,
                          "exterior_facet_bottom": op2.ON_BOTTOM}[integral_type]
            kwargs["iterate"] = decoration

            def get_map(x):
                return x.cell_node_map()

        elif integral_type in ("interior_facet", "interior_facet_vert"):
            extra_args.append(m.interior_facets.local_facet_dat(op2.READ))

            def get_map(x):
                return x.interior_facet_node_map()

        elif integral_type == "interior_facet_horiz":
            decoration = op2.ON_INTERIOR_FACETS
            kwargs["iterate"] = decoration

            def get_map(x):
                return x.cell_node_map()

        else:
            raise ValueError("Unknown integral type '%s'" % integral_type)

        # Output argument
        if is_mat:
            tensor_arg = mat(lambda s: get_map(s),
                             lambda s: get_map(s),
                             tsbc, trbc,
                             i, j)
        elif is_vec:
            tensor_arg = vec(lambda s: get_map(s), i)
        else:
            tensor_arg = tensor(op2.INC)

        coords = m.coordinates
        args = [kernel, itspace, tensor_arg,
                coords.dat(op2.READ, get_map(coords))]
        if needs_orientations:
            o = m.cell_orientations()
            args.append(o.dat(op2.READ, get_map(o)))
        if needs_cell_sizes:
            o = m.cell_sizes
            args.append(o.dat(op2.READ, get_map(o)))

        for n in coeff_map:
            c = coefficients[n]
            for c_ in c.split():
                m_ = get_map(c_)
                args.append(c_.dat(op2.READ, m_))
        if needs_cell_facets:
            assert integral_type == "cell"
            extra_args.append(m.cell_to_facets(op2.READ))

        args.extend(extra_args)
        kwargs["pass_layer_arg"] = pass_layer_arg
        try:
            yield op2.ParLoop(*args, **kwargs).compute
        except MapValueError:
            raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")

    # Must apply bcs outside loop over kernels because we may wish
    # to apply bcs to a block which is otherwise zero, and
    # therefore does not have an associated kernel.
    if bcs is not None and is_mat:
        for bc in bcs:
            if isinstance(bc, DirichletBC):
                fs = bc.function_space()
                # Evaluate this outwith a "collecting_loops" block,
                # since creation of the bc nodes actually can create a
                # par_loop.
                nodes = bc.nodes
                if len(fs) > 1:
                    raise RuntimeError(r"""Cannot apply boundary conditions to full mixed space. Did you forget to index it?""")
                shape = result_matrix.block_shape
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
                                yield functools.partial(tensor[i, j].set_local_diagonal_entries, nodes)
                        elif fs.component is not None:
                            # ComponentFunctionSpace, check parent index
                            if fs.parent.index is not None:
                                # Mixed, index doesn't match
                                if fs.parent.index != i:
                                    continue
                            # Index matches
                            yield functools.partial(tensor[i, j].set_local_diagonal_entries, nodes, idx=fs.component)
                        elif fs.index is None:
                            yield functools.partial(tensor[i, j].set_local_diagonal_entries, nodes)
                        else:
                            raise RuntimeError("Unhandled BC case")
            elif isinstance(bc, EquationBCSplit):
                yield from _assemble(bc.f, tensor=result_matrix, bcs=bc.bcs,
                                     form_compiler_parameters=form_compiler_parameters,
                                     inverse=inverse, mat_type=mat_type,
                                     sub_mat_type=sub_mat_type,
                                     appctx=appctx,
                                     assemble_now=assemble_now,
                                     allocate_only=False,
                                     zero_tensor=False)
            else:
                raise NotImplementedError("Undefined type of bcs class provided.")

    if bcs is not None and is_vec:
        for bc in bcs:
            if isinstance(bc, DirichletBC):
                if assemble_now:
                    yield functools.partial(bc.apply, result_function)
                else:
                    yield functools.partial(bc.zero, result_function)
            elif isinstance(bc, EquationBCSplit):
                yield functools.partial(bc.zero, result_function)
                yield from _assemble(bc.f, tensor=result_function, bcs=bc.bcs,
                                     form_compiler_parameters=form_compiler_parameters,
                                     inverse=inverse, mat_type=mat_type,
                                     sub_mat_type=sub_mat_type,
                                     appctx=appctx,
                                     assemble_now=assemble_now,
                                     allocate_only=False,
                                     zero_tensor=False)
    if zero_tensor:
        if is_mat:
            # Queue up matrix assembly (after we've done all the other operations)
            yield tensor.assemble
        if assemble_now:
            yield result
