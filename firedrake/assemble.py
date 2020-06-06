import functools
import operator
from collections import OrderedDict, defaultdict
from enum import IntEnum
from itertools import chain

import firedrake
import numpy
import ufl
from firedrake import (assemble_expressions, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint import annotate_assemble
from firedrake.bcs import DirichletBC, EquationBCSplit
from firedrake.slate import slac, slate
from firedrake.utils import ScalarType
from pyop2 import op2
from pyop2.exceptions import MapValueError, SparsityFormatError


__all__ = ("assemble", )


class AssemblyRank(IntEnum):
    SCALAR = 0
    VECTOR = 1
    MATRIX = 2


@annotate_assemble
def assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
             mat_type=None, sub_mat_type=None,
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
        raise ValueError("Can't use 'nest', set 'mat_type' instead")

    if "collect_loops" in kwargs or "allocate_only" in kwargs:
        raise RuntimeError

    diagonal = kwargs.pop("diagonal", False)
    if len(kwargs) > 0:
        raise TypeError("Unknown keyword arguments '%s'" % ', '.join(kwargs.keys()))

    if isinstance(f, (ufl.form.Form, slate.TensorBase)):
        loops = _assemble(f, tensor=tensor, bcs=solving._extract_bcs(bcs),
                          form_compiler_parameters=form_compiler_parameters,
                          mat_type=mat_type,
                          sub_mat_type=sub_mat_type, appctx=appctx,
                          diagonal=diagonal,
                          options_prefix=options_prefix)
        for l in loops:
            m = l()
        return m
    elif isinstance(f, ufl.core.expr.Expr):
        return assemble_expressions.assemble_expression(f)
    else:
        raise TypeError("Unable to assemble: %r" % f)


def get_mat_type(mat_type, sub_mat_type):
    if mat_type is None:
        mat_type = parameters.parameters["default_matrix_type"]
    if mat_type not in ["matfree", "aij", "baij", "nest", "dense"]:
        raise ValueError("Unrecognised matrix type, '%s'" % mat_type)
    if sub_mat_type is None:
        sub_mat_type = parameters.parameters["default_sub_matrix_type"]
    if sub_mat_type not in ["aij", "baij"]:
        raise ValueError("Invalid submatrix type, '%s' (not 'aij' or 'baij')", sub_mat_type)
    return mat_type, sub_mat_type


def allocate_matrix(f, bcs=(), form_compiler_parameters=None,
                    mat_type=None, sub_mat_type=None, appctx={},
                    options_prefix=None):
    r"""Allocate a matrix given a form.  To be used with :func:`create_assembly_callable`.

    .. warning::

       Do not use this function unless you know what you're doing.
    """
    _, _, result = get_matrix(f, mat_type, sub_mat_type,
                              bcs=bcs,
                              options_prefix=options_prefix,
                              appctx=appctx,
                              form_compiler_parameters=form_compiler_parameters)
    return result()


def create_assembly_callable(f, tensor=None, bcs=None, form_compiler_parameters=None,
                             mat_type=None, sub_mat_type=None,
                             diagonal=False):
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
                      mat_type=mat_type,
                      sub_mat_type=sub_mat_type,
                      diagonal=diagonal,
                      assemble_now=False)

    loops = tuple(loops)

    def thunk():
        for kernel in loops:
            kernel()
    return thunk


def get_matrix(form, mat_type, sub_mat_type, *, bcs=None,
               options_prefix=None, tensor=None, appctx=None,
               form_compiler_parameters=None):
    mat_type, sub_mat_type = get_mat_type(mat_type, sub_mat_type)
    matfree = mat_type == "matfree"
    arguments = form.arguments()
    if bcs is None:
        bcs = ()
    if tensor is not None and tensor.a.arguments() != arguments:
        raise ValueError("Form's arguments do not match provided result tensor")
    if matfree:
        if tensor is None:
            tensor = matrix.ImplicitMatrix(form, bcs,
                                           fc_params=form_compiler_parameters,
                                           appctx=appctx,
                                           options_prefix=options_prefix)
        elif not isinstance(tensor, matrix.ImplicitMatrix):
            raise ValueError("Expecting implicit matrix with matfree")
        else:
            pass
        return tensor, (), lambda: tensor

    if tensor is not None:
        return tensor, (tensor.M.zero, ), lambda: tensor

    if isinstance(form, slate.TensorBase):
        # FIXME: inherit from slate form somehow.
        integral_types = {"cell"}
    else:
        integral_types = set(i.integral_type() for i in form.integrals())
        for bc in bcs:
            integral_types.update(integral.integral_type()
                                  for integral in bc.integrals())
    nest = mat_type == "nest"
    if nest:
        baij = sub_mat_type == "baij"
    else:
        baij = mat_type == "baij"

    if any(len(a.function_space()) > 1 for a in arguments) and mat_type == "baij":
        raise ValueError("BAIJ matrix type makes no sense for mixed spaces, use 'aij'")

    get_cell_map = operator.methodcaller("cell_node_map")
    get_extf_map = operator.methodcaller("exterior_facet_node_map")
    get_intf_map = operator.methodcaller("interior_facet_node_map")
    domains = OrderedDict((k, set()) for k in (get_cell_map,
                                               get_extf_map,
                                               get_intf_map))
    mapping = {"cell": (get_cell_map, op2.ALL),
               "exterior_facet_bottom": (get_cell_map, op2.ON_BOTTOM),
               "exterior_facet_top": (get_cell_map, op2.ON_TOP),
               "interior_facet_horiz": (get_cell_map, op2.ON_INTERIOR_FACETS),
               "exterior_facet": (get_extf_map, op2.ALL),
               "exterior_facet_vert": (get_extf_map, op2.ALL),
               "interior_facet": (get_intf_map, op2.ALL),
               "interior_facet_vert": (get_intf_map, op2.ALL)}
    for integral_type in integral_types:
        try:
            get_map, region = mapping[integral_type]
        except KeyError:
            raise ValueError(f"Unknown integral type '{integral_type}'")
        domains[get_map].add(region)

    test, trial = arguments
    map_pairs, iteration_regions = zip(*(((get_map(test), get_map(trial)),
                                          tuple(sorted(regions)))
                                         for get_map, regions in domains.items()
                                         if regions))
    try:
        sparsity = op2.Sparsity((test.function_space().dof_dset,
                                 trial.function_space().dof_dset),
                                tuple(map_pairs),
                                iteration_regions=tuple(iteration_regions),
                                nest=nest,
                                block_sparse=baij)
    except SparsityFormatError:
        raise ValueError("Monolithic matrix assembly not supported for systems "
                         "with R-space blocks")

    tensor = matrix.Matrix(form, bcs, mat_type, sparsity, ScalarType,
                           options_prefix=options_prefix)
    return tensor, (), lambda: tensor


def collect_lgmaps(tensor, all_bcs, Vrow, Vcol, row, col):
    if len(Vrow) > 1:
        bcrow = tuple(bc for bc in all_bcs
                      if bc.function_space_index() == row)
    else:
        bcrow = all_bcs
    if len(Vcol) > 1:
        bccol = tuple(bc for bc in all_bcs
                      if bc.function_space_index() == col
                      and isinstance(bc, DirichletBC))
    else:
        bccol = tuple(bc for bc in all_bcs
                      if isinstance(bc, DirichletBC))
    rlgmap, clgmap = tensor.M[row, col].local_to_global_maps
    rlgmap = Vrow[row].local_to_global_map(bcrow, lgmap=rlgmap)
    clgmap = Vcol[col].local_to_global_map(bccol, lgmap=clgmap)
    unroll = any(bc.function_space().component is not None
                 for bc in chain(bcrow, bccol))
    return (rlgmap, clgmap), unroll


def matrix_arg(access, get_map, row, col, *,
               all_bcs=(), tensor=None, Vrow=None, Vcol=None):
    if row is None and col is None:
        maprow = get_map(Vrow)
        mapcol = get_map(Vcol)
        lgmaps, unroll = zip(*(collect_lgmaps(tensor, all_bcs,
                                              Vrow, Vcol, i, j)
                               for i, j in numpy.ndindex(tensor.block_shape)))
        return tensor.M(access, (maprow, mapcol), lgmaps=tuple(lgmaps),
                        unroll_map=any(unroll))
    else:
        assert row is not None and col is not None
        maprow = get_map(Vrow[row])
        mapcol = get_map(Vcol[col])
        lgmaps, unroll = collect_lgmaps(tensor, all_bcs,
                                        Vrow, Vcol, row, col)
        return tensor.M[row, col](access, (maprow, mapcol), lgmaps=(lgmaps, ),
                                  unroll_map=unroll)


def get_vector(argument, *, tensor=None):
    V = argument.function_space()
    if tensor is None:
        tensor = firedrake.Function(V)
        zero = ()
    else:
        if V != tensor.function_space():
            raise ValueError("Form's argument does not match provided result tensor")
        zero = (tensor.dat.zero, )
    return tensor, zero, lambda: tensor


def vector_arg(access, get_map, i, *, tensor=None, V=None):
    if i is None:
        map_ = get_map(V)
        return tensor.dat(access, map_)
    else:
        map_ = get_map(V[i])
        return tensor.dat[i](access, map_)


def get_scalar(arguments, *, tensor=None):
    assert arguments == ()
    if tensor is not None:
        raise ValueError("Can't assemble 0-form into existing tensor")

    tensor = op2.Global(1, [0.0])
    return tensor, (), lambda: tensor.data[0]


def apply_bcs(f, tensor, bcs, *, assembly_rank=None, form_compiler_parameters=None,
              mat_type=None, sub_mat_type=None, appctx={}, diagonal=False,
              assemble_now=True):
    dirichletbcs = tuple(bc for bc in bcs if isinstance(bc, DirichletBC))
    equationbcs = tuple(bc for bc in bcs if isinstance(bc, EquationBCSplit))
    if any(not isinstance(bc, (DirichletBC, EquationBCSplit)) for bc in bcs):
        raise NotImplementedError("Unhandled type of bc object")

    arguments = f.arguments()
    if assembly_rank == AssemblyRank.MATRIX:
        op2tensor = tensor.M
        for bc in dirichletbcs:
            V = bc.function_space()
            nodes = bc.nodes
            shape = tuple(len(a.function_space()) for a in arguments)
            for i, j in numpy.ndindex(shape):
                # Set diagonal entries on bc nodes to 1 if the current
                # block is on the matrix diagonal and its index matches the
                # index of the function space the bc is defined on.
                if i != j:
                    continue
                if V.component is None and V.index is not None:
                    # Mixed, index (no ComponentFunctionSpace)
                    if V.index == i:
                        yield functools.partial(op2tensor[i, j].set_local_diagonal_entries, nodes)
                elif V.component is not None:
                    # ComponentFunctionSpace, check parent index
                    if V.parent.index is not None:
                        # Mixed, index doesn't match
                        if V.parent.index != i:
                            continue
                        # Index matches
                    yield functools.partial(op2tensor[i, j].set_local_diagonal_entries, nodes, idx=V.component)
                elif V.index is None:
                    yield functools.partial(op2tensor[i, j].set_local_diagonal_entries, nodes)
                else:
                    raise RuntimeError("Unhandled BC case")
        for bc in equationbcs:
            yield from _assemble(bc.f, tensor=tensor, bcs=bc.bcs,
                                 form_compiler_parameters=form_compiler_parameters,
                                 mat_type=mat_type,
                                 sub_mat_type=sub_mat_type,
                                 appctx=appctx,
                                 assemble_now=assemble_now,
                                 zero_tensor=False)
    elif assembly_rank == AssemblyRank.VECTOR:
        for bc in dirichletbcs:
            if assemble_now:
                if diagonal:
                    yield functools.partial(bc.set, tensor, 1)
                else:
                    yield functools.partial(bc.apply, tensor)
            else:
                yield functools.partial(bc.zero, tensor)
        for bc in equationbcs:
            if diagonal:
                raise NotImplementedError("diagonal assembly and EquationBC not supported")
            yield functools.partial(bc.zero, tensor)
            yield from _assemble(bc.f, tensor=tensor, bcs=bc.bcs,
                                 form_compiler_parameters=form_compiler_parameters,
                                 mat_type=mat_type,
                                 sub_mat_type=sub_mat_type,
                                 appctx=appctx,
                                 assemble_now=assemble_now,
                                 zero_tensor=False)
    else:
        if len(bcs) != 0:
            raise ValueError("Not expecting boundary conditions for 0-forms")


def create_parloops(f, create_op2arg, bcs, *, assembly_rank=None, diagonal=False,
                    form_compiler_parameters=None):
    coefficients = f.coefficients()
    domains = f.ufl_domains()

    if isinstance(f, slate.TensorBase):
        if diagonal:
            raise NotImplementedError("Diagonal + slate not supported")
        kernels = slac.compile_expression(f, tsfc_parameters=form_compiler_parameters)
    else:
        kernels = tsfc_interface.compile_form(f, "form", parameters=form_compiler_parameters, diagonal=diagonal)

    # These will be used to correctly interpret the "otherwise"
    # subdomain
    all_integer_subdomain_ids = defaultdict(list)
    for k in kernels:
        if k.kinfo.subdomain_id != "otherwise":
            all_integer_subdomain_ids[k.kinfo.integral_type].append(k.kinfo.subdomain_id)
    for k, v in all_integer_subdomain_ids.items():
        all_integer_subdomain_ids[k] = tuple(sorted(v))

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
        if assembly_rank == AssemblyRank.MATRIX:
            i, j = indices
        elif assembly_rank == AssemblyRank.VECTOR:
            i, = indices
        else:
            assert len(indices) == 0

        sdata = subdomain_data.get(integral_type, None)
        if integral_type != 'cell' and sdata is not None:
            raise NotImplementedError("subdomain_data only supported with cell integrals.")

        # Now build arguments for the par_loop
        kwargs = {}
        # Some integrals require non-coefficient arguments at the
        # end (facet number information).
        extra_args = []
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
            kwargs["iterate"] = {"exterior_facet_top": op2.ON_TOP,
                                 "exterior_facet_bottom": op2.ON_BOTTOM}[integral_type]

            def get_map(x):
                return x.cell_node_map()
        elif integral_type in ("interior_facet", "interior_facet_vert"):
            extra_args.append(m.interior_facets.local_facet_dat(op2.READ))

            def get_map(x):
                return x.interior_facet_node_map()
        elif integral_type == "interior_facet_horiz":
            kwargs["iterate"] = op2.ON_INTERIOR_FACETS

            def get_map(x):
                return x.cell_node_map()
        else:
            raise ValueError("Unknown integral type '%s'" % integral_type)

        # Output argument
        if assembly_rank == AssemblyRank.MATRIX:
            tensor_arg = create_op2arg(op2.INC, get_map, i, j)
        elif assembly_rank == AssemblyRank.VECTOR:
            tensor_arg = create_op2arg(op2.INC, get_map, i)
        else:
            tensor_arg = create_op2arg(op2.INC)

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


@utils.known_pyop2_safe
def _assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
              mat_type=None, sub_mat_type=None,
              appctx={},
              options_prefix=None,
              zero_tensor=True,
              diagonal=False,
              assemble_now=True):
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
    :arg mat_type: (optional) type for assembled matrices, one of
        "nest", "aij", "baij", or "matfree".
    :arg sub_mat_type: (optional) type for assembled sub matrices
        inside a "nest" matrix.  One of "aij" or "baij".
    :arg appctx: Additional information to hang on the assembled
         matrix if an implicit matrix is requested (mat_type "matfree").
    :arg options_prefix: An options prefix for the PETSc matrix
        (ignored if not assembling a bilinear form).
    """
    mat_type, sub_mat_type = get_mat_type(mat_type, sub_mat_type)
    if form_compiler_parameters:
        form_compiler_parameters = form_compiler_parameters.copy()
    else:
        form_compiler_parameters = {}

    try:
        topology, = set(d.topology for d in f.ufl_domains())
    except ValueError:
        raise NotImplementedError("All integration domains must share a mesh topology")
    for m in f.ufl_domains():
        # Ensure mesh is "initialised" (could have got here without
        # building a functionspace (e.g. if integrating a constant)).
        m.init()

    if bcs is None:
        bcs = ()
    else:
        bcs = tuple(bcs)

    for o in chain(f.arguments(), f.coefficients()):
        domain = o.ufl_domain()
        if domain is not None and domain.topology != topology:
            raise NotImplementedError("Assembly with multiple meshes not supported.")

    rank = len(f.arguments())
    if diagonal:
        assert rank == 2
    if rank == 2 and not diagonal:
        assembly_rank = AssemblyRank.MATRIX
    elif rank == 1 or diagonal:
        assembly_rank = AssemblyRank.VECTOR
    else:
        assembly_rank = AssemblyRank.SCALAR

    if assembly_rank == AssemblyRank.MATRIX:
        test, trial = f.arguments()
        tensor, zeros, result = get_matrix(f, mat_type, sub_mat_type,
                                           bcs=bcs, options_prefix=options_prefix,
                                           tensor=tensor,
                                           appctx=appctx,
                                           form_compiler_parameters=form_compiler_parameters)
        # intercept matrix-free matrices here
        if mat_type == "matfree":
            if tensor.a.arguments() != f.arguments():
                raise ValueError("Form's arguments do not match provided result "
                                 "tensor")
            tensor.assemble()
            yield result
            return

        create_op2arg = functools.partial(matrix_arg,
                                          all_bcs=tuple(chain(*bcs)),
                                          tensor=tensor,
                                          Vrow=test.function_space(),
                                          Vcol=trial.function_space())
    elif assembly_rank == AssemblyRank.VECTOR:
        if diagonal:
            # actually a 2-form but throw away the trial space
            test, trial = f.arguments()
        else:
            test, = f.arguments()
        tensor, zeros, result = get_vector(test, tensor=tensor)

        create_op2arg = functools.partial(vector_arg, tensor=tensor,
                                          V=test.function_space())
    else:
        tensor, zeros, result = get_scalar(f.arguments(), tensor=tensor)
        create_op2arg = tensor

    if zero_tensor:
        yield from zeros

    yield from create_parloops(f, create_op2arg, bcs,
                               assembly_rank=assembly_rank,
                               diagonal=diagonal,
                               form_compiler_parameters=form_compiler_parameters)

    yield from apply_bcs(f, tensor, bcs,
                         assembly_rank=assembly_rank,
                         form_compiler_parameters=form_compiler_parameters,
                         mat_type=mat_type,
                         sub_mat_type=sub_mat_type,
                         appctx=appctx,
                         diagonal=diagonal,
                         assemble_now=assemble_now)
    if zero_tensor:
        if assembly_rank == AssemblyRank.MATRIX:
            # Queue up matrix assembly (after we've done all the other operations)
            yield tensor.M.assemble
        if assemble_now:
            yield result
