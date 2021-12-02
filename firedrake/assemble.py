import functools
import operator
from collections import OrderedDict, defaultdict, namedtuple
from enum import IntEnum, auto
from itertools import chain

import firedrake
import numpy
import ufl
from firedrake import (assemble_expressions, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint import annotate_assemble
from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
from firedrake.petsc import PETSc
from firedrake.slate import slac, slate
from firedrake.utils import ScalarType
from pyop2 import op2
from pyop2.exceptions import MapValueError, SparsityFormatError


__all__ = ("assemble",)


class _AssemblyRank(IntEnum):
    """Enum enumerating possible dimensions of the output tensor."""
    SCALAR = 0
    VECTOR = 1
    MATRIX = 2


class _AssemblyType(IntEnum):
    """Enum enumerating possible assembly types.

    See ``"assembly_type"`` from :func:`assemble` for more information.
    """
    SOLUTION = auto()
    RESIDUAL = auto()


_AssemblyOpts = namedtuple("_AssemblyOpts", ["diagonal",
                                             "assembly_type",
                                             "fc_params",
                                             "mat_type",
                                             "sub_mat_type",
                                             "appctx",
                                             "options_prefix"])
"""Container to hold immutable assembly options.

Please refer to :func:`assemble` for a description of the options.
"""


@PETSc.Log.EventDecorator()
@annotate_assemble
def assemble(expr, tensor=None, bcs=None, *,
             diagonal=False,
             assembly_type="solution",
             form_compiler_parameters=None,
             mat_type=None,
             sub_mat_type=None,
             appctx={},
             options_prefix=None):
    r"""Evaluate expr.

    :arg expr: a :class:`~ufl.classes.Form`, :class:`~ufl.classes.Expr` or
        a :class:`~slate.TensorBase` expression.
    :arg tensor: Existing tensor object to place the result in.
    :arg bcs: Iterable of boundary conditions to apply.
    :kwarg diagonal: If assembling a matrix is it diagonal?
    :kwarg assembly_type: String indicating how boundary conditions are applied
        (may be ``"solution"`` or ``"residual"``). If ``"solution"`` then the
        boundary conditions are applied as expected whereas ``"residual"`` zeros
        the selected components of the tensor.
    :kwarg form_compiler_parameters: Dictionary of parameters to pass to
        the form compiler. Ignored if not assembling a :class:`~ufl.classes.Form`.
        Any parameters provided here will be overridden by parameters set on the
        :class:`~ufl.classes.Measure` in the form. For example, if a
        ``quadrature_degree`` of 4 is specified in this argument, but a degree of
        3 is requested in the measure, the latter will be used.
    :kwarg mat_type: String indicating how a 2-form (matrix) should be
        assembled -- either as a monolithic matrix (``"aij"`` or ``"baij"``),
        a block matrix (``"nest"``), or left as a :class:`.ImplicitMatrix` giving
        matrix-free actions (``'matfree'``). If not supplied, the default value in
        ``parameters["default_matrix_type"]`` is used.  BAIJ differs
        from AIJ in that only the block sparsity rather than the dof
        sparsity is constructed.  This can result in some memory
        savings, but does not work with all PETSc preconditioners.
        BAIJ matrices only make sense for non-mixed matrices.
    :kwarg sub_mat_type: String indicating the matrix type to
        use *inside* a nested block matrix.  Only makes sense if
        ``mat_type`` is ``nest``.  May be one of ``"aij"`` or ``"baij"``.  If
        not supplied, defaults to ``parameters["default_sub_matrix_type"]``.
    :kwarg appctx: Additional information to hang on the assembled
        matrix if an implicit matrix is requested (mat_type ``"matfree"``).
    :kwarg options_prefix: PETSc options prefix to apply to matrices.

    :returns: See below.

    If expr is a :class:`~ufl.classes.Form` or Slate tensor expression then
    this evaluates the corresponding integral(s) and returns a :class:`float`
    for 0-forms, a :class:`.Function` for 1-forms and a :class:`.Matrix` or
    :class:`.ImplicitMatrix` for 2-forms. In the case of 2-forms the rows
    correspond to the test functions and the columns to the trial functions.

    If expr is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`.

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    If ``bcs`` is supplied and ``expr`` is a 2-form, the rows and columns
    of the resulting :class:`.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``expr`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.

    .. note::
        For 1-form assembly, the resulting object should in fact be a *cofunction*
        instead of a :class:`.Function`. However, since cofunctions are not
        currently supported in UFL, functions are used instead.
    """
    if isinstance(expr, (ufl.form.Form, slate.TensorBase)):
        return assemble_form(expr, tensor, bcs, diagonal, assembly_type,
                             form_compiler_parameters,
                             mat_type, sub_mat_type,
                             appctx, options_prefix)
    elif isinstance(expr, ufl.core.expr.Expr):
        return assemble_expressions.assemble_expression(expr)
    else:
        raise TypeError(f"Unable to assemble: {expr}")


@PETSc.Log.EventDecorator()
def assemble_form(expr, tensor, bcs, diagonal, assembly_type,
                  form_compiler_parameters,
                  mat_type, sub_mat_type,
                  appctx, options_prefix):
    """Assemble an expression.

    :arg expr: a :class:`~ufl.classes.Form` or a :class:`~slate.TensorBase`
        expression.

    See :func:`assemble` for a description of the possible additional arguments
    and return values.
    """
    # Do some setup of the arguments and wrap them in a namedtuple.
    bcs = solving._extract_bcs(bcs)
    if assembly_type == "solution":
        assembly_type = _AssemblyType.SOLUTION
    elif assembly_type == "residual":
        assembly_type = _AssemblyType.RESIDUAL
    else:
        raise ValueError("assembly_type must be either 'solution' or 'residual'")
    mat_type, sub_mat_type = _get_mat_type(mat_type, sub_mat_type,
                                           expr.arguments())
    opts = _AssemblyOpts(diagonal, assembly_type, form_compiler_parameters,
                         mat_type, sub_mat_type, appctx, options_prefix)

    assembly_rank = _get_assembly_rank(expr, diagonal)
    if assembly_rank == _AssemblyRank.SCALAR:
        if tensor:
            raise ValueError("Can't assemble 0-form into existing tensor")
        return _assemble_scalar(expr, bcs, opts)
    elif assembly_rank == _AssemblyRank.VECTOR:
        return _assemble_vector(expr, tensor, bcs, opts)
    elif assembly_rank == _AssemblyRank.MATRIX:
        return _assemble_matrix(expr, tensor, bcs, opts)
    else:
        raise AssertionError


@PETSc.Log.EventDecorator()
def allocate_matrix(expr, bcs=(), form_compiler_parameters=None,
                    mat_type=None, sub_mat_type=None, appctx={},
                    options_prefix=None):
    r"""Allocate a matrix given an expression.

    .. warning::

       Do not use this function unless you know what you're doing.
    """
    opts = _AssemblyOpts(diagonal=False,
                         assembly_type=None,
                         fc_params=form_compiler_parameters,
                         mat_type=mat_type,
                         sub_mat_type=sub_mat_type,
                         appctx=appctx,
                         options_prefix=options_prefix)
    return _make_matrix(expr, bcs, opts)


@PETSc.Log.EventDecorator()
def create_assembly_callable(expr, tensor=None, bcs=None, form_compiler_parameters=None,
                             mat_type=None, sub_mat_type=None, diagonal=False):
    r"""Create a callable object than be used to assemble expr into a tensor.

    This is really only designed to be used inside residual and
    jacobian callbacks, since it always assembles back into the
    initially provided tensor.  See also :func:`allocate_matrix`.

    .. warning::

        This function is now deprecated.

    .. warning::

       Really do not use this function unless you know what you're doing.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("once", DeprecationWarning)
        warnings.warn("create_assembly_callable is now deprecated. Please use assemble instead.",
                      DeprecationWarning)

    if tensor is None:
        raise ValueError("Have to provide tensor to write to")
    return functools.partial(assemble, expr,
                             tensor=tensor,
                             bcs=bcs,
                             form_compiler_parameters=form_compiler_parameters,
                             mat_type=mat_type,
                             sub_mat_type=sub_mat_type,
                             diagonal=diagonal,
                             assembly_type="residual")


def _get_assembly_rank(expr, diagonal):
    """Return the appropriate :class:`_AssemblyRank`.

    :arg expr: The expression (:class:`~ufl.classes.Form` or
        :class:`~slate.TensorBase`) being assembled.
    :arg diagonal: If assembling a matrix is it diagonal? (:class:`bool`)

    :returns: The appropriate :class:`_AssemblyRank` (e.g. ``_AssemblyRank.VECTOR``).
    """
    rank = len(expr.arguments())
    if diagonal:
        assert rank == 2
        return _AssemblyRank.VECTOR
    if rank == 0:
        return _AssemblyRank.SCALAR
    if rank == 1:
        return _AssemblyRank.VECTOR
    if rank == 2:
        return _AssemblyRank.MATRIX
    raise AssertionError


def _assemble_scalar(expr, bcs, opts):
    """Assemble a 0-form.

    :arg expr: The expression being assembled.
    :arg bcs: Iterable of boundary conditions.
    :arg opts: :class:`_AssemblyOpts` containing the assembly options.

    :returns: The resulting :class:`float`.

    This function does the scalar-specific initialisation of the output tensor
    before calling the generic function :func:`_assemble_expr`.
    """
    if len(expr.arguments()) != 0:
        raise ValueError("Can't assemble a 0-form with arguments")

    scalar = _make_scalar()
    _assemble_expr(expr, scalar, bcs, opts, _AssemblyRank.SCALAR)
    return scalar.data[0]


def _assemble_vector(expr, vector, bcs, opts):
    """Assemble either a 1-form or the diagonal of a 2-form.

    :arg expr: The expression being assembled.
    :arg vector: The vector to write to (may be ``None``).
    :arg bcs: Iterable of boundary conditions.
    :arg opts: :class:`_AssemblyOpts` containing the assembly options.

    :returns: The assembled vector (:class:`.Function`). Note that this should
        really be a cofunction instead but this is not currently supported in UFL.

    This function does the vector-specific initialisation of the output tensor
    before calling the generic function :func:`_assemble_expr`.
    """
    if opts.diagonal:
        test, trial = expr.arguments()
        if test.function_space() != trial.function_space():
            raise ValueError("Can only assemble diagonal of 2-form if functionspaces match")
    else:
        test, = expr.arguments()
    if vector:
        if test.function_space() != vector.function_space():
            raise ValueError("Form's argument does not match provided result tensor")
        vector.dat.zero()
    else:
        vector = _make_vector(test)

    # Might have gotten here without EquationBC objects preprocessed.
    if any(isinstance(bc, EquationBC) for bc in bcs):
        bcs = tuple(bc.extract_form("F") for bc in bcs)

    _assemble_expr(expr, vector, bcs, opts, _AssemblyRank.VECTOR)
    return vector


def _assemble_matrix(expr, matrix, bcs, opts):
    """Assemble a 2-form into a matrix.

    :arg expr: The expression being assembled.
    :arg matrix: The matrix to write to (may be ``None``).
    :arg bcs: Iterable of boundary conditions.
    :arg opts: :class:`_AssemblyOpts` containing the assembly options.

    :returns: The assembled :class:`.Matrix` or :class:`.ImplicitMatrix`. For
        more information about this object refer to :func:`assemble`.

    This function does the matrix-specific initialisation of the output tensor
    before calling the generic function :func:`_assemble_expr`.
    """
    if matrix:
        if opts.mat_type != "matfree":
            matrix.M.zero()
        if matrix.a.arguments() != expr.arguments():
            raise ValueError("Form's arguments do not match provided result tensor")
    else:
        matrix = _make_matrix(expr, bcs, opts)

    if opts.mat_type == "matfree":
        matrix.assemble()
    else:
        _assemble_expr(expr, matrix, bcs, opts, _AssemblyRank.MATRIX)
        matrix.M.assemble()
    return matrix


def _make_scalar():
    """Make an empty scalar.

    :returns: An empty :class:`~pyop2.op2.Global`.
    """
    return op2.Global(1, [0.0], dtype=utils.ScalarType)


def _make_vector(V):
    """Make an empty vector.

    :arg V: The :class:`.FunctionSpace` the function is defined for.

    :returns: An empty :class:`.Function`.
    """
    return firedrake.Function(V.function_space())


def _make_matrix(expr, bcs, opts):
    """Make an empty matrix.

    :arg expr: The expression being assembled.
    :arg bcs: Iterable of boundary conditions.
    :arg opts: :class:`_AssemblyOpts` containing the assembly options.

    :returns: An empty :class:`.Matrix` or :class:`.ImplicitMatrix`.
    """
    matfree = opts.mat_type == "matfree"
    arguments = expr.arguments()
    if bcs is None:
        bcs = ()
    else:
        if any(isinstance(bc, EquationBC) for bc in bcs):
            raise TypeError("EquationBC objects not expected here. "
                            "Preprocess by extracting the appropriate form with bc.extract_form('Jp') or bc.extract_form('J')")
    if matfree:
        return matrix.ImplicitMatrix(expr, bcs,
                                     fc_params=opts.fc_params,
                                     appctx=opts.appctx,
                                     options_prefix=opts.options_prefix)

    integral_types = set(i.integral_type() for i in expr.integrals())
    for bc in bcs:
        integral_types.update(integral.integral_type()
                              for integral in bc.integrals())
    nest = opts.mat_type == "nest"
    if nest:
        baij = opts.sub_mat_type == "baij"
    else:
        baij = opts.mat_type == "baij"

    if any(len(a.function_space()) > 1 for a in arguments) and opts.mat_type == "baij":
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

    return matrix.Matrix(expr, bcs, opts.mat_type, sparsity, ScalarType,
                         options_prefix=opts.options_prefix)


def _assemble_expr(expr, tensor, bcs, opts, assembly_rank):
    """Assemble an expression into the provided tensor.

    :arg expr: The expression to be assembled.
    :arg tensor: The tensor to write to.
    :arg bcs: Iterable of boundary conditions. If any are :class:`EquationBCSplit`
        objects then this function is recursively called using the expressions
        and boundary conditions defined for them.
    :arg opts: :class:`_AssemblyOpts` containing the assembly options.
    :arg assembly_rank: The appropriate :class:`_AssemblyRank`.
    """
    # We cache the parloops on the form but since parloops (currently) hold
    # references to large data structures (e.g. the output tensor) we only
    # cache a single set of parloops at any one time to prevent memory leaks.
    # This restriction does make the caching a lot simpler as we don't have to
    # worry about hashing the arguments.
    parloop_init_args = (expr, tensor, bcs, opts.diagonal, opts.fc_params, assembly_rank)
    cached_init_args, cached_parloops = expr._cache.get("parloops", (None, None))
    parloops = cached_parloops if cached_init_args == parloop_init_args else None

    if not parloops:
        parloops = _make_parloops(*parloop_init_args)
        expr._cache["parloops"] = (parloop_init_args, parloops)

    for parloop in parloops:
        parloop.compute()

    dir_bcs = tuple(bc for bc in bcs if isinstance(bc, DirichletBC))
    _apply_dirichlet_bcs(tensor, dir_bcs, opts, assembly_rank)

    eq_bcs = tuple(bc for bc in bcs if isinstance(bc, EquationBCSplit))
    if eq_bcs and opts.diagonal:
        raise NotImplementedError("Diagonal assembly and EquationBC not supported")
    for bc in eq_bcs:
        if assembly_rank == _AssemblyRank.VECTOR:
            bc.zero(tensor)
        _assemble_expr(bc.f, tensor, bc.bcs, opts, assembly_rank)


def _get_mat_type(mat_type, sub_mat_type, arguments):
    """Validate the matrix types provided by the user and set any that are
    undefined to default values.

    :arg mat_type: (:class:`str`) PETSc matrix type for the assembled matrix.
    :arg sub_mat_type: (:class:`str`) PETSc matrix type for blocks if
        ``mat_type`` is ``"nest"``.
    :arg arguments: The test and trial functions of the expression being assembled.
    :raises ValueError: On bad arguments.
    :returns: 2-:class:`tuple` of validated/default ``mat_type`` and ``sub_mat_type``.
    """
    if mat_type is None:
        mat_type = parameters.parameters["default_matrix_type"]
        if any(V.ufl_element().family() == "Real"
               for arg in arguments
               for V in arg.function_space()):
            mat_type = "nest"
    if mat_type not in {"matfree", "aij", "baij", "nest", "dense"}:
        raise ValueError(f"Unrecognised matrix type, '{mat_type}'")
    if sub_mat_type is None:
        sub_mat_type = parameters.parameters["default_sub_matrix_type"]
    if sub_mat_type not in {"aij", "baij"}:
        raise ValueError(f"Invalid submatrix type, '{sub_mat_type}' (not 'aij' or 'baij')")
    return mat_type, sub_mat_type


def _collect_lgmaps(matrix, all_bcs, Vrow, Vcol, row, col):
    """Obtain local to global maps for matrix insertion in the
    presence of boundary conditions.

    :arg matrix: the matrix.
    :arg all_bcs: all boundary conditions involved in the assembly of
        the matrix.
    :arg Vrow: function space for rows.
    :arg Vcol: function space for columns.
    :arg row: index into Vrow (by block).
    :arg col: index into Vcol (by block).
    :returns: 2-tuple ``(row_lgmap, col_lgmap), unroll``. unroll will
       indicate to the codegeneration if the lgmaps need to be
       unrolled from any blocking they contain.
    """
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
    rlgmap, clgmap = matrix.M[row, col].local_to_global_maps
    rlgmap = Vrow[row].local_to_global_map(bcrow, lgmap=rlgmap)
    clgmap = Vcol[col].local_to_global_map(bccol, lgmap=clgmap)
    unroll = any(bc.function_space().component is not None
                 for bc in chain(bcrow, bccol))
    return (rlgmap, clgmap), unroll


def _vector_arg(access, get_map, i, *, function, V):
    """Obtain an :class:`~pyop2.op2.Arg` for insertion into a given
    vector (:class:`Function`).

    :arg access: :mod:`~pyop2` access descriptor (e.g. :class:`~pyop2.op2.READ`).
    :arg get_map: Callable of one argument that obtains :class:`~pyop2.op2.Map`
        objects from :class:`FunctionSpace` objects.
    :arg i: Index of block (subspace of a mixed function), may be ``None``.
    :arg function: :class:`Function` to insert into.
    :arg V: :class:`FunctionSpace` corresponding to ``function``.

    :returns: An :class:`~pyop2.op2.Arg`.
    """
    if i is None:
        map_ = get_map(V)
        return function.dat(access, map_)
    else:
        map_ = get_map(V[i])
        return function.dat[i](access, map_)


def _matrix_arg(access, get_map, row, col, *,
                all_bcs, matrix, Vrow, Vcol):
    """Obtain an op2.Arg for insertion into the given matrix.

    :arg access: Access descriptor.
    :arg get_map: callable of one argument that obtains Maps from
        functionspaces.
    :arg row, col: row (column) of block matrix we are assembling (may be None for
        direct insertion into mixed matrices). Either both or neither
        must be None.
    :arg all_bcs: tuple of boundary conditions involved in assembly.
    :arg matrix: the matrix to obtain the argument for.
    :arg Vrow, Vcol: function spaces for the row and column space.
    :raises AssertionError: on invalid arguments
    :returns: an op2.Arg.
    """
    if row is None and col is None:
        maprow = get_map(Vrow)
        mapcol = get_map(Vcol)
        lgmaps, unroll = zip(*(_collect_lgmaps(matrix, all_bcs,
                                               Vrow, Vcol, i, j)
                               for i, j in numpy.ndindex(matrix.block_shape)))
        return matrix.M(access, (maprow, mapcol), lgmaps=tuple(lgmaps),
                        unroll_map=any(unroll))
    else:
        assert row is not None and col is not None
        maprow = get_map(Vrow[row])
        mapcol = get_map(Vcol[col])
        lgmaps, unroll = _collect_lgmaps(matrix, all_bcs,
                                         Vrow, Vcol, row, col)
        return matrix.M[row, col](access, (maprow, mapcol), lgmaps=(lgmaps, ),
                                  unroll_map=unroll)


def _apply_dirichlet_bcs(tensor, bcs, opts, assembly_rank):
    """Apply Dirichlet boundary conditions to a tensor.

    :arg tensor: The tensor.
    :arg bcs: Iterable of :class:`DirichletBC` objects.
    :arg opts: :class:`_AssemblyOpts` containing the assembly options.
    :arg assembly_rank: are we doing a scalar, vector, or matrix.
    """
    if assembly_rank == _AssemblyRank.MATRIX:
        op2tensor = tensor.M
        shape = tuple(len(a.function_space()) for a in tensor.a.arguments())
        for bc in bcs:
            V = bc.function_space()
            nodes = bc.nodes
            for i, j in numpy.ndindex(shape):
                # Set diagonal entries on bc nodes to 1 if the current
                # block is on the matrix diagonal and its index matches the
                # index of the function space the bc is defined on.
                if i != j:
                    continue
                if V.component is None and V.index is not None:
                    # Mixed, index (no ComponentFunctionSpace)
                    if V.index == i:
                        op2tensor[i, j].set_local_diagonal_entries(nodes)
                elif V.component is not None:
                    # ComponentFunctionSpace, check parent index
                    if V.parent.index is not None:
                        # Mixed, index doesn't match
                        if V.parent.index != i:
                            continue
                        # Index matches
                    op2tensor[i, j].set_local_diagonal_entries(nodes, idx=V.component)
                elif V.index is None:
                    op2tensor[i, j].set_local_diagonal_entries(nodes)
                else:
                    raise RuntimeError("Unhandled BC case")
    elif assembly_rank == _AssemblyRank.VECTOR:
        for bc in bcs:
            if opts.assembly_type == _AssemblyType.SOLUTION:
                if opts.diagonal:
                    bc.set(tensor, 1)
                else:
                    bc.apply(tensor)
            elif opts.assembly_type == _AssemblyType.RESIDUAL:
                bc.zero(tensor)
            else:
                raise AssertionError
    elif assembly_rank == _AssemblyRank.SCALAR:
        pass
    else:
        raise AssertionError


@utils.known_pyop2_safe
def _make_parloops(expr, tensor, bcs, diagonal, fc_params, assembly_rank):
    """Create parloops for the assembly of the expression.

    :arg expr: The expression to be assembled.
    :arg tensor: The tensor to write to. Depending on ``expr`` and ``diagonal``
        this will either be a scalar (:class:`~pyop2.op2.Global`),
        vector/cofunction (masquerading as a :class:`.Function`) or :class:`.Matrix`.
    :arg bcs: Iterable of boundary conditions.
    :arg diagonal: (:class:`bool`) If assembling a matrix is it diagonal?
    :arg fc_params: Dictionary of parameters to pass to the form compiler.
    :arg assembly_rank: The appropriate :class:`_AssemblyRank`.

    :returns: A tuple of the generated :class:`~pyop2..op2.ParLoop` objects.
    """
    if fc_params:
        form_compiler_parameters = fc_params.copy()
    else:
        form_compiler_parameters = {}

    try:
        topology, = set(d.topology for d in expr.ufl_domains())
    except ValueError:
        raise NotImplementedError("All integration domains must share a mesh topology")
    for m in expr.ufl_domains():
        # Ensure mesh is "initialised" (could have got here without
        # building a functionspace (e.g. if integrating a constant)).
        m.init()

    for o in chain(expr.arguments(), expr.coefficients()):
        domain = o.ufl_domain()
        if domain is not None and domain.topology != topology:
            raise NotImplementedError("Assembly with multiple meshes not supported.")

    if assembly_rank == _AssemblyRank.MATRIX:
        test, trial = expr.arguments()
        create_op2arg = functools.partial(_matrix_arg,
                                          all_bcs=tuple(chain(*bcs)),
                                          matrix=tensor,
                                          Vrow=test.function_space(),
                                          Vcol=trial.function_space())
    elif assembly_rank == _AssemblyRank.VECTOR:
        if diagonal:
            # actually a 2-form but throw away the trial space
            test, _ = expr.arguments()
        else:
            test, = expr.arguments()
        create_op2arg = functools.partial(_vector_arg, function=tensor,
                                          V=test.function_space())
    else:
        create_op2arg = tensor

    coefficients = expr.coefficients()
    domains = expr.ufl_domains()

    if isinstance(expr, slate.TensorBase):
        kernels = slac.compile_expression(expr, compiler_parameters=form_compiler_parameters, diagonal=diagonal)
    else:
        kernels = tsfc_interface.compile_form(expr, "form", parameters=form_compiler_parameters, diagonal=diagonal)

    # These will be used to correctly interpret the "otherwise"
    # subdomain
    all_integer_subdomain_ids = defaultdict(list)
    for k in kernels:
        if k.kinfo.subdomain_id != "otherwise":
            all_integer_subdomain_ids[k.kinfo.integral_type].append(k.kinfo.subdomain_id)
    for k, v in all_integer_subdomain_ids.items():
        all_integer_subdomain_ids[k] = tuple(sorted(v))

    parloops = []
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
        subdomain_data = expr.subdomain_data()[m]
        # Find argument space indices
        if assembly_rank == _AssemblyRank.MATRIX:
            i, j = indices
        elif assembly_rank == _AssemblyRank.VECTOR:
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
        if assembly_rank == _AssemblyRank.MATRIX:
            tensor_arg = create_op2arg(op2.INC, get_map, i, j)
        elif assembly_rank == _AssemblyRank.VECTOR:
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

        for n, split_map in coeff_map:
            c = coefficients[n]
            split_c = c.split()
            for c_ in (split_c[i] for i in split_map):
                m_ = get_map(c_)
                args.append(c_.dat(op2.READ, m_))

        if needs_cell_facets:
            assert integral_type == "cell"
            extra_args.append(m.cell_to_facets(op2.READ))
        if pass_layer_arg:
            c = op2.Global(1, itspace.layers-2, dtype=numpy.dtype(numpy.int32))
            o = c(op2.READ)
            extra_args.append(o)

        args.extend(extra_args)
        kwargs["pass_layer_arg"] = pass_layer_arg
        try:
            parloops.append(op2.ParLoop(*args, **kwargs))
        except MapValueError:
            raise RuntimeError("Integral measure does not match measure of all coefficients/arguments")
    return tuple(parloops)
