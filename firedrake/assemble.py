import abc
import collections
from collections import OrderedDict
import functools
import itertools
from itertools import product
import operator
from functools import cached_property

import cachetools
from pyrsistent import freeze, pmap
import finat
import loopy as lp
import firedrake
import numpy
from pyadjoint.tape import annotate_tape
from tsfc import kernel_args
from tsfc.finatinterface import create_element
from tsfc.ufl_utils import extract_firedrake_constants
import ufl
import pyop3 as op3
import finat.ufl
from firedrake import (extrusion_utils as eutils, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint_utils import annotate_assemble
from firedrake.ufl_expr import extract_unique_domain
from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
from firedrake.functionspaceimpl import WithGeometry, FunctionSpace, FiredrakeDualSpace
from firedrake.functionspacedata import entity_dofs_key, entity_permutations_key
from firedrake.petsc import PETSc
from firedrake.slate import slac, slate
from firedrake.slate.slac.kernel_builder import CellFacetKernelArg, LayerCountKernelArg
from firedrake.utils import ScalarType, assert_empty, tuplify
from pyop2 import op2
from pyop2.exceptions import MapValueError, SparsityFormatError


__all__ = "assemble",


_FORM_CACHE_KEY = "firedrake.assemble.FormAssembler"
"""Entry used in form cache to try and reuse assemblers where possible."""


@PETSc.Log.EventDecorator()
@annotate_assemble
def assemble(expr, *args, **kwargs):
    r"""Evaluate expr.

    :arg expr: a :class:`~ufl.classes.BaseForm`, :class:`~ufl.classes.Expr` or
        a :class:`~.slate.TensorBase` expression.
    :arg tensor: Existing tensor object to place the result in.
    :arg bcs: Iterable of boundary conditions to apply.
    :kwarg diagonal: If assembling a matrix is it diagonal?
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
    :kwarg zero_bc_nodes: If ``True``, set the boundary condition nodes in the
        output tensor to zero rather than to the values prescribed by the
        boundary condition. Default is ``False``.
    :kwarg weight: weight of the boundary condition, i.e. the scalar in front of the
        identity matrix corresponding to the boundary nodes.
        To discretise eigenvalue problems set the weight equal to 0.0.

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
    """
    if isinstance(expr, (ufl.form.BaseForm, slate.TensorBase)):
        return assemble_base_form(expr, *args, **kwargs)
    elif isinstance(expr, ufl.core.expr.Expr):
        return _assemble_expr(expr)
    else:
        raise TypeError(f"Unable to assemble: {expr}")


def assemble_base_form(expression, tensor=None, bcs=None,
                       diagonal=False,
                       mat_type=None,
                       sub_mat_type=None,
                       form_compiler_parameters=None,
                       appctx=None,
                       options_prefix=None,
                       zero_bc_nodes=False,
                       is_base_form_preprocessed=False,
                       weight=1.0):
    r"""Evaluate expression.

    :arg expression: a :class:`~ufl.classes.BaseForm`
    :kwarg tensor: Existing tensor object to place the result in.
    :kwarg bcs: Iterable of boundary conditions to apply.
    :kwarg diagonal: If assembling a matrix is it diagonal?
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
    :kwarg form_compiler_parameters: Dictionary of parameters to pass to
        the form compiler. Ignored if not assembling a :class:`~ufl.classes.Form`.
        Any parameters provided here will be overridden by parameters set on the
        :class:`~ufl.classes.Measure` in the form. For example, if a
        ``quadrature_degree`` of 4 is specified in this argument, but a degree of
        3 is requested in the measure, the latter will be used.
    :kwarg appctx: Additional information to hang on the assembled
        matrix if an implicit matrix is requested (mat_type ``"matfree"``).
    :kwarg options_prefix: PETSc options prefix to apply to matrices.
    :kwarg zero_bc_nodes: If ``True``, set the boundary condition nodes in the
        output tensor to zero rather than to the values prescribed by the
        boundary condition. Default is ``False``.
    :kwarg is_base_form_preprocessed: If ``True``, skip preprocessing of the form.
    :kwarg weight: weight of the boundary condition, i.e. the scalar in front of the
        identity matrix corresponding to the boundary nodes.
        To discretise eigenvalue problems set the weight equal to 0.0.

    :returns: a :class:`float` for 0-forms, a :class:`.Cofunction` or a :class:`.Function` for 1-forms,
              and a :class:`.MatrixBase` for 2-forms.

    This function assembles a :class:`~ufl.classes.BaseForm` object by traversing the corresponding DAG
    in a post-order fashion and evaluating the nodes on the fly.
    """

    # Preprocess the DAG and restructure the DAG
    if not is_base_form_preprocessed and not isinstance(expression, slate.TensorBase):
        # Preprocessing the form makes a new object -> current form caching mechanism
        # will populate `expr`'s cache which is now different than `expression`'s cache so we need
        # to transmit the cache. All of this only holds when `expression` is a `ufl.Form`
        # and therefore when `is_base_form_preprocessed` is False.
        expr = preprocess_base_form(expression, mat_type, form_compiler_parameters)
        if isinstance(expression, ufl.form.Form) and isinstance(expr, ufl.form.Form):
            expr._cache = expression._cache
    else:
        expr = expression

    # DAG assembly: traverse the DAG in a post-order fashion and evaluate the node on the fly.
    stack = [expr]
    visited = {}
    while stack:
        e = stack.pop()
        unvisted_children = []
        operands = base_form_operands(e)
        for arg in operands:
            if arg not in visited:
                unvisted_children.append(arg)

        if unvisted_children:
            stack.append(e)
            stack.extend(unvisted_children)
        else:
            visited[e] = base_form_assembly_visitor(e, tensor, bcs, diagonal,
                                                    form_compiler_parameters,
                                                    mat_type, sub_mat_type,
                                                    appctx, options_prefix,
                                                    zero_bc_nodes, weight,
                                                    *(visited[arg] for arg in operands))
    if tensor:
        update_tensor(visited[expr], tensor)
    return visited[expr]


def update_tensor(assembled_base_form, tensor):
    if isinstance(tensor, (firedrake.Function, firedrake.Cofunction)):
        assembled_base_form.dat.copy(tensor.dat)
    elif isinstance(tensor, matrix.MatrixBase):
        # Uses the PETSc copy method.
        assembled_base_form.petscmat.copy(tensor.petscmat)
    else:
        raise NotImplementedError("Cannot update tensor of type %s" % type(tensor))


def base_form_operands(expr):
    if isinstance(expr, (ufl.form.FormSum, ufl.Adjoint, ufl.Action)):
        return list(expr.ufl_operands)
    return []


def expand_derivatives_form(form, fc_params):
    """Expand derivatives of ufl.BaseForm objects
    :arg form: a :class:`~ufl.classes.BaseForm`
    :arg fc_params:: Dictionary of parameters to pass to the form compiler.

    :returns: The resulting preprocessed :class:`~ufl.classes.BaseForm`.
    This function preprocess the form, mainly by expanding the derivatives, in order to determine
    if we are dealing with a :class:`~ufl.classes.Form` or another :class:`~ufl.classes.BaseForm` object.
    This function is called in :func:`base_form_assembly_visitor`. Depending on the type of the resulting tensor,
    we may call :func:`assemble_form` or traverse the sub-DAG via :func:`assemble_base_form`.
    """
    if isinstance(form, ufl.form.Form):
        from firedrake.parameters import parameters as default_parameters
        from tsfc.parameters import is_complex

        if fc_params is None:
            fc_params = default_parameters["form_compiler"].copy()
        else:
            # Override defaults with user-specified values
            _ = fc_params
            fc_params = default_parameters["form_compiler"].copy()
            fc_params.update(_)

        complex_mode = fc_params and is_complex(fc_params.get("scalar_type"))

        return ufl.algorithms.preprocess_form(form, complex_mode)
    # We also need to expand derivatives for `ufl.BaseForm` objects that are not `ufl.Form`
    # Example: `Action(A, derivative(B, f))`, where `A` is a `ufl.BaseForm` and `B` can
    # be `ufl.BaseForm`, or even an appropriate `ufl.Expr`, since assembly of expressions
    # containing derivatives is not supported anymore but might be needed if the expression
    # in question is within a `ufl.BaseForm` object.
    return ufl.algorithms.ad.expand_derivatives(form)


def preprocess_base_form(expr, mat_type=None, form_compiler_parameters=None):
    """Preprocess ufl.BaseForm objects"""
    if mat_type != "matfree":
        # For "matfree", Form evaluation is delayed
        expr = expand_derivatives_form(expr, form_compiler_parameters)
    # Expanding derivatives may turn `ufl.BaseForm` objects into `ufl.Expr` objects that are not `ufl.BaseForm`.
    if not isinstance(expr, ufl.form.BaseForm):
        return assemble(expr)
    return expr


def base_form_assembly_visitor(expr, tensor, bcs, diagonal,
                               form_compiler_parameters,
                               mat_type, sub_mat_type,
                               appctx, options_prefix,
                               zero_bc_nodes, weight, *args):
    r"""Assemble a :class:`~ufl.classes.BaseForm` object given its assembled operands.

        This functions contains the assembly handlers corresponding to the different nodes that
        can arise in a `~ufl.classes.BaseForm` object. It is called by :func:`assemble_base_form`
        in a post-order fashion.
    """

    if isinstance(expr, (ufl.form.Form, slate.TensorBase)):
        return _assemble_form(expr, tensor=tensor, bcs=bcs,
                              diagonal=diagonal,
                              mat_type=mat_type,
                              sub_mat_type=sub_mat_type,
                              appctx=appctx,
                              options_prefix=options_prefix,
                              form_compiler_parameters=form_compiler_parameters,
                              zero_bc_nodes=zero_bc_nodes, weight=weight)

    elif isinstance(expr, ufl.Adjoint):
        if len(args) != 1:
            raise TypeError("Not enough operands for Adjoint")
        mat, = args
        petsc_mat = mat.petscmat
        petsc_mat.hermitianTranspose()
        (row, col) = mat.arguments()
        return matrix.AssembledMatrix((col, row), bcs, petsc_mat,
                                      appctx=appctx,
                                      options_prefix=options_prefix)
    elif isinstance(expr, ufl.Action):
        if (len(args) != 2):
            raise TypeError("Not enough operands for Action")
        lhs, rhs = args
        if isinstance(lhs, matrix.MatrixBase):
            if isinstance(rhs, (firedrake.Cofunction, firedrake.Function)):
                petsc_mat = lhs.petscmat
                (row, col) = lhs.arguments()
                res = firedrake.Cofunction(col.function_space().dual())

                with rhs.dat.vec_ro as v_vec:
                    with res.dat.vec as res_vec:
                        petsc_mat.mult(v_vec, res_vec)
                return firedrake.Cofunction(row.function_space().dual(), val=res.dat)
            elif isinstance(rhs, matrix.MatrixBase):
                petsc_mat = lhs.petscmat
                (row, col) = lhs.arguments()
                res = petsc_mat.matMult(rhs.petscmat)
                return matrix.AssembledMatrix(expr, bcs, res,
                                              appctx=appctx,
                                              options_prefix=options_prefix)
            else:
                raise TypeError("Incompatible RHS for Action.")
        elif isinstance(lhs, (firedrake.Cofunction, firedrake.Function)):
            if isinstance(rhs, (firedrake.Cofunction, firedrake.Function)):
                # Return scalar value
                with lhs.dat.vec_ro as x, rhs.dat.vec_ro as y:
                    res = x.dot(y)
                return res
            else:
                raise TypeError("Incompatible RHS for Action.")
        else:
            raise TypeError("Incompatible LHS for Action.")
    elif isinstance(expr, ufl.FormSum):
        if len(args) != len(expr.weights()):
            raise TypeError("Mismatching weights and operands in FormSum")
        if len(args) == 0:
            raise TypeError("Empty FormSum")
        if all(isinstance(op, float) for op in args):
            return sum(args)
        elif all(isinstance(op, firedrake.Cofunction) for op in args):
            V, = set(a.function_space() for a in args)
            res = sum([w*op.dat for (op, w) in zip(args, expr.weights())])
            return firedrake.Cofunction(V, res)
        elif all(isinstance(op, ufl.Matrix) for op in args):
            is_set = False
            for (op, w) in zip(args, expr.weights()):
                petsc_mat = op.petscmat
                petsc_mat.scale(w)
                if is_set:
                    res = res + petsc_mat
                else:
                    res = petsc_mat
                    is_set = True
            return matrix.AssembledMatrix(expr, bcs, res,
                                          appctx=appctx,
                                          options_prefix=options_prefix)
        else:
            raise TypeError("Mismatching FormSum shapes")
    elif isinstance(expr, (ufl.Cofunction, ufl.Coargument, ufl.Argument, ufl.Matrix, ufl.ZeroBaseForm)):
        return expr
    elif isinstance(expr, ufl.Coefficient):
        return expr
    else:
        raise TypeError(f"Unrecognised BaseForm instance: {expr}")


@PETSc.Log.EventDecorator()
def allocate_matrix(expr, bcs=None, *, mat_type=None, sub_mat_type=None,
                    appctx=None, form_compiler_parameters=None, options_prefix=None):
    r"""Allocate a matrix given an expression.

    .. warning::

       Do not use this function unless you know what you're doing.
    """
    bcs = bcs or ()
    appctx = appctx or {}

    matfree = mat_type == "matfree"
    arguments = expr.arguments()
    if bcs is None:
        bcs = ()
    else:
        if any(isinstance(bc, EquationBC) for bc in bcs):
            raise TypeError("EquationBC objects not expected here. "
                            "Preprocess by extracting the appropriate form with bc.extract_form('Jp') or bc.extract_form('J')")
    if matfree:
        return matrix.ImplicitMatrix(expr, bcs,
                                     appctx=appctx,
                                     fc_params=form_compiler_parameters,
                                     options_prefix=options_prefix)

    integral_types = set(i.integral_type() for i in expr.integrals())
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

    # experimental, will end up in pyop3
    mesh = expr.ufl_domain().topology
    # TODO handle different mat types
    if mat_type != "aij":
        raise NotImplementedError

    # not sure that this is strictly needed any more
    # adjacency_map = op3.transforms.compress(mesh.points, lambda p: mesh.closure(mesh.star(p)), uniquify=True)

    def adjacency(pt):
        return mesh.closure(mesh.star(pt))
        # return adjacency_map(pt)

    test_arg, trial_arg = arguments
    mymat = op3.PetscMat(
        mesh.points,
        adjacency,
        test_arg.function_space().axes,
        trial_arg.function_space().axes,
    )

    return matrix.Matrix(expr, bcs, mat_type, mymat, options_prefix=options_prefix)


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
        warnings.warn("create_assembly_callable is now deprecated. Please use assemble or FormAssembler instead.",
                      DeprecationWarning)

    if tensor is None:
        raise ValueError("Have to provide tensor to write to")

    rank = len(expr.arguments())
    if rank == 0:
        return ZeroFormAssembler(expr, tensor, form_compiler_parameters).assemble
    elif rank == 1 or (rank == 2 and diagonal):
        return OneFormAssembler(expr, tensor, bcs, diagonal=diagonal,
                                form_compiler_parameters=form_compiler_parameters).assemble
    elif rank == 2:
        return TwoFormAssembler(expr, tensor, bcs, form_compiler_parameters).assemble
    else:
        raise AssertionError


def _assemble_form(form, tensor=None, bcs=None, *,
                   diagonal=False,
                   mat_type=None,
                   sub_mat_type=None,
                   appctx=None,
                   options_prefix=None,
                   form_compiler_parameters=None,
                   zero_bc_nodes=False,
                   weight=1.0):
    """Assemble a form.

    See :func:`assemble` for a description of the arguments to this function.
    """
    bcs = solving._extract_bcs(bcs)

    _check_inputs(form, tensor, bcs, diagonal)

    if tensor is not None:
        _zero_tensor(tensor, form, diagonal)
    else:
        tensor = _make_tensor(form, bcs, diagonal=diagonal, mat_type=mat_type,
                              sub_mat_type=sub_mat_type, appctx=appctx,
                              form_compiler_parameters=form_compiler_parameters,
                              options_prefix=options_prefix)

    # It is expensive to construct new assemblers because extracting the data
    # from the form is slow. Since all of the data structures in the assembler
    # are persistent apart from the output tensor, we stash the assembler on the
    # form and swap out the tensor if needed.
    # The cache key only needs to contain the boundary conditions, diagonal and
    # form compiler parameters since all other assemble kwargs are only used for
    # creating the tensor which is handled above and has no bearing on the assembler.
    # Note: This technically creates a memory leak since bcs are 'heavy' and so
    # repeated assembly of the same form but with different boundary conditions
    # will lead to old bcs getting stored along with old tensors.

    # FIXME This only works for 1-forms at the moment
    is_cacheable = len(form.arguments()) == 1
    if is_cacheable:
        try:
            key = tuple(bcs), diagonal, tuplify(form_compiler_parameters), zero_bc_nodes
            assembler = form._cache[_FORM_CACHE_KEY][key]
            assembler.replace_tensor(tensor)
            return assembler.assemble()
        except KeyError:
            pass

    rank = len(form.arguments())
    if rank == 0:
        assembler = ZeroFormAssembler(form, tensor, form_compiler_parameters)
    elif rank == 1 or (rank == 2 and diagonal):
        assembler = OneFormAssembler(form, tensor, bcs, diagonal=diagonal,
                                     form_compiler_parameters=form_compiler_parameters,
                                     needs_zeroing=False, zero_bc_nodes=zero_bc_nodes)
    elif rank == 2:
        assembler = TwoFormAssembler(form, tensor, bcs, form_compiler_parameters,
                                     needs_zeroing=False, weight=weight)
    else:
        raise AssertionError

    if is_cacheable:
        if _FORM_CACHE_KEY not in form._cache:
            form._cache[_FORM_CACHE_KEY] = {}
        form._cache[_FORM_CACHE_KEY][key] = assembler

    return assembler.assemble()


def _assemble_expr(expr):
    """Assemble a pointwise expression.

    :arg expr: The :class:`ufl.core.expr.Expr` to be evaluated.
    :returns: A :class:`firedrake.Function` containing the result of this evaluation.
    """
    try:
        coefficients = ufl.algorithms.extract_coefficients(expr)
        V, = set(c.function_space() for c in coefficients) - {None}
    except ValueError:
        raise ValueError("Cannot deduce correct target space from pointwise expression")
    return firedrake.Function(V).assign(expr)


def _check_inputs(form, tensor, bcs, diagonal):
    # Ensure mesh is 'initialised' as we could have got here without building a
    # function space (e.g. if integrating a constant).
    for mesh in form.ufl_domains():
        mesh.init()

    if diagonal and any(isinstance(bc, EquationBCSplit) for bc in bcs):
        raise NotImplementedError("Diagonal assembly and EquationBC not supported")

    rank = len(form.arguments())
    if rank == 0:
        assert tensor is None
        assert not bcs
    elif rank == 1:
        test, = form.arguments()

        if tensor is not None and test.function_space() != tensor.function_space():
            raise ValueError("Form's argument does not match provided result tensor")
    elif rank == 2 and diagonal:
        test, trial = form.arguments()
        if test.function_space() != trial.function_space():
            raise ValueError("Can only assemble the diagonal of 2-form if the function spaces match")
    elif rank == 2:
        if tensor is not None and tensor.a.arguments() != form.arguments():
            raise ValueError("Form's arguments do not match provided result tensor")
    else:
        raise AssertionError

    if any(c.dat.dtype != ScalarType for c in form.coefficients()):
        raise ValueError("Cannot assemble a form containing coefficients where the "
                         "dtype is not the PETSc scalar type.")


def _zero_tensor(tensor, form, diagonal):
    rank = len(form.arguments())
    assert rank != 0
    if rank == 1 or (rank == 2 and diagonal):
        tensor.dat.zero()
    elif rank == 2:
        if not isinstance(tensor, matrix.ImplicitMatrix):
            tensor.M.zero()
    else:
        raise AssertionError


def _make_tensor(form, bcs, *, diagonal, mat_type, sub_mat_type, appctx,
                 form_compiler_parameters, options_prefix):
    rank = len(form.arguments())
    if rank == 0:
        # Getting the comm attribute of a form isn't straightforward
        # form.ufl_domains()[0]._comm seems the most robust method
        # revisit in a refactor
        comm = form.ufl_domains()[0]._comm

        # TODO this is more convoluted than strictly needed, add a factory method?
        sf = op3.sf.single_star(comm)
        axis = op3.Axis(1, sf=sf)
        return op3.HierarchicalArray(
            axis,
            data=numpy.asarray([0.0], dtype=utils.ScalarType),
        )
    elif rank == 1:
        test, = form.arguments()
        return firedrake.Cofunction(test.function_space().dual())
    elif rank == 2 and diagonal:
        test, _ = form.arguments()
        return firedrake.Cofunction(test.function_space().dual())
    elif rank == 2:
        mat_type, sub_mat_type = _get_mat_type(mat_type, sub_mat_type, form.arguments())
        return allocate_matrix(form, bcs, mat_type=mat_type, sub_mat_type=sub_mat_type,
                               appctx=appctx, form_compiler_parameters=form_compiler_parameters,
                               options_prefix=options_prefix)
    else:
        raise AssertionError


class FormAssembler(abc.ABC):
    """Abstract base class for assembling forms.

    :param form: The variational form to be assembled.
    :param tensor: The output tensor to store the result.
    :param bcs: Iterable of boundary conditions to apply.
    :param form_compiler_parameters: Optional parameters to pass to the
        TSFC and/or Slate compilers.
    :param needs_zeroing: Should ``tensor`` be zeroed before assembling?
    """

    def __init__(self, form, tensor, bcs=(), form_compiler_parameters=None, needs_zeroing=True, weight=1.0):
        assert tensor is not None

        bcs = solving._extract_bcs(bcs)

        self._form = form
        self._tensor = tensor
        self._bcs = bcs
        self._form_compiler_params = form_compiler_parameters or {}
        self._needs_zeroing = needs_zeroing
        self.weight = weight

        # we can swap tensor out but we need to remember the name so we can
        # swap it into parloops in the right place
        self._tensor_name = self._as_pyop3_type(tensor).name

    @property
    @abc.abstractmethod
    def result(self):
        """The result of the assembly operation."""

    @property
    @abc.abstractmethod
    def diagonal(self):
        """Are we assembling the diagonal of a 2-form?"""

    def assemble(self):
        """Perform the assembly.

        :returns: The assembled object.
        """
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit FormAssembler objects is not supported yet. "
                "Use assemble instead."
            )

        if self._needs_zeroing:
            self._as_pyop3_type(self._tensor).zero()

        for parloop in self.parloops:
            parloop(**{self._tensor_name: self._as_pyop3_type(self._tensor)})

        for bc in self._bcs:
            if isinstance(bc, EquationBC):  # can this be lifted?
                bc = bc.extract_form("F")
            self._apply_bc(bc)

        return self.result

    def replace_tensor(self, tensor):
        self._tensor = self._as_pyop3_type(tensor)

    @cached_property
    def local_kernels(self):
        """Return local kernels and their subdomain IDs.

        Returns
        -------
        tuple
            Collection of ``(local_kernel, subdomain_id)`` 2-tuples, one for
            each possible combination.
        """
        try:
            topology, = set(d.topology for d in self._form.ufl_domains())
        except ValueError:
            raise NotImplementedError("All integration domains must share a mesh topology")

        for o in itertools.chain(self._form.arguments(), self._form.coefficients()):
            domain = extract_unique_domain(o)
            if domain is not None and domain.topology != topology:
                raise NotImplementedError("Assembly with multiple meshes is not supported")

        if isinstance(self._form, ufl.Form):
            kernels = tsfc_interface.compile_form(
                self._form, "form", diagonal=self.diagonal,
                parameters=self._form_compiler_params
            )
        elif isinstance(self._form, slate.TensorBase):
            kernels = slac.compile_expression(
                self._form,
                compiler_parameters=self._form_compiler_params
            )
        else:
            raise AssertionError
        return tuple(
            (k, subdomain_id) for k in kernels for subdomain_id in k.kinfo.subdomain_id
        )

    @cached_property
    def all_integer_subdomain_ids(self):
        return tsfc_interface.gather_integer_subdomain_ids(
            {k for k, _ in self.local_kernels}
        )

    @cached_property
    def parloops(self):
        loops = []
        for local_kernel, subdomain_id in self.local_kernels:
            loops.append(
                ParloopBuilder(
                    self._form,
                    local_kernel,
                    self._tensor,
                    subdomain_id,
                    self.all_integer_subdomain_ids,
                    diagonal=self.diagonal,
                    lgmaps=self.collect_lgmaps(local_kernel, self._bcs)
                ).build()
            )
        return tuple(loops)

    def needs_unrolling(self, local_knl, bcs):
        """Do we need to address matrix elements directly rather than in
        a blocked fashion?

        This is slower but required for the application of some boundary conditions
        to 2-forms.

        :param local_knl: A :class:`tsfc_interface.SplitKernel`.
        :param bcs: Iterable of boundary conditions.
        """
        return False

    def collect_lgmaps(self, local_knl, bcs):
        """Return any local-to-global maps that need to be swapped out.

        This is only needed when applying boundary conditions to 2-forms.

        :param local_knl: A :class:`tsfc_interface.SplitKernel`.
        :param bcs: Iterable of boundary conditions.
        """
        return None

    @staticmethod
    def _as_pyop3_type(tensor):
        if isinstance(tensor, op3.HierarchicalArray):
            return tensor
        elif isinstance(tensor, firedrake.Cofunction):
            return tensor.dat
        elif isinstance(tensor, matrix.Matrix):
            return tensor.M
        else:
            raise AssertionError


class ZeroFormAssembler(FormAssembler):
    """Class for assembling a 0-form."""

    diagonal = False
    """Diagonal assembly not possible for zero forms."""

    def __init__(self, form, tensor, form_compiler_parameters=None):
        super().__init__(form, tensor, (), form_compiler_parameters)

    @property
    def result(self):
        # Must use private attribute here because the public .data_ro
        # attribute only yields the owned data, which for a global is empty
        # apart from rank 0.
        return self._tensor.buffer._data[0]


class OneFormAssembler(FormAssembler):
    """Class for assembling a 1-form.

    :param diagonal: Are we actually assembling the diagonal of a 2-form?
    :param zero_bc_nodes: If ``True``, set the boundary condition nodes in the
        output tensor to zero rather than to the values prescribed by the
        boundary condition.

    For all other arguments see :class:`FormAssembler` for more information.
    """

    def __init__(self, form, tensor, bcs=(), diagonal=False, zero_bc_nodes=False,
                 form_compiler_parameters=None, needs_zeroing=True):
        super().__init__(form, tensor, bcs, form_compiler_parameters, needs_zeroing)
        self._diagonal = diagonal
        self._zero_bc_nodes = zero_bc_nodes

    @property
    def diagonal(self):
        return self._diagonal

    @property
    def result(self):
        return self._tensor

    def _apply_bc(self, bc):
        # TODO Maybe this could be a singledispatchmethod?
        if isinstance(bc, DirichletBC):
            self._apply_dirichlet_bc(bc)
        elif isinstance(bc, EquationBCSplit):
            bc.zero(self._tensor)

            type(self)(bc.f, self._tensor, bc.bcs, self._diagonal, self._zero_bc_nodes,
                       self._form_compiler_params, needs_zeroing=False).assemble()
        else:
            raise AssertionError

    def _apply_dirichlet_bc(self, bc):
        if not self._zero_bc_nodes:
            tensor_func = self._tensor.riesz_representation(riesz_map="l2")
            if self._diagonal:
                bc.set(tensor_func, 1)
            else:
                bc.apply(tensor_func)
            self._tensor.assign(tensor_func.riesz_representation(riesz_map="l2"))
        else:
            bc.zero(self._tensor)


def TwoFormAssembler(form, tensor, *args, **kwargs):
    if isinstance(tensor, matrix.ImplicitMatrix):
        return MatrixFreeAssembler(tensor)
    else:
        return ExplicitMatrixAssembler(form, tensor, *args, **kwargs)


class ExplicitMatrixAssembler(FormAssembler):
    """Class for assembling a 2-form."""

    diagonal = False
    """Diagonal assembly not possible for two forms."""

    @property
    def test_function_space(self):
        test, _ = self._form.arguments()
        return test.function_space()

    @property
    def trial_function_space(self):
        _, trial = self._form.arguments()
        return trial.function_space()

    def get_indicess(self, knl):
        if all(i is None for i in knl.indices):
            return numpy.ndindex(self._tensor.block_shape)
        else:
            assert all(i is not None for i in knl.indices)
            return knl.indices,

    @property
    def result(self):
        self._tensor.M.assemble()
        return self._tensor

    def needs_unrolling(self, knl, bcs):
        for i, j in self.get_indicess(knl):
            for bc in itertools.chain(*self._filter_bcs(bcs, i, j)):
                if bc.function_space().component is not None:
                    return True
        return False

    def collect_lgmaps(self, knl, bcs):
        if not bcs:
            return None

        lgmaps = []
        for i, j in self.get_indicess(knl):
            row_bcs, col_bcs = self._filter_bcs(bcs, i, j)
            rlgmap, clgmap = self._tensor.M[i, j].local_to_global_maps
            rlgmap = self.test_function_space[i].local_to_global_map(row_bcs, rlgmap)
            clgmap = self.trial_function_space[j].local_to_global_map(col_bcs, clgmap)
            lgmaps.append((rlgmap, clgmap))
        return tuple(lgmaps)

    def _filter_bcs(self, bcs, row, col):
        if len(self.test_function_space) > 1:
            bcrow = tuple(bc for bc in bcs
                          if bc.function_space_index() == row)
        else:
            bcrow = bcs

        if len(self.trial_function_space) > 1:
            bccol = tuple(bc for bc in bcs
                          if bc.function_space_index() == col
                          and isinstance(bc, DirichletBC))
        else:
            bccol = tuple(bc for bc in bcs if isinstance(bc, DirichletBC))
        return bcrow, bccol

    def _apply_bc(self, bc):
        op2tensor = self._tensor.M
        spaces = tuple(a.function_space() for a in self._tensor.a.arguments())
        V = bc.function_space()
        component = V.component
        if component is not None:
            V = V.parent
        index = 0 if V.index is None else V.index
        space = V if V.parent is None else V.parent
        if isinstance(bc, DirichletBC):
            if space != spaces[0]:
                raise TypeError("bc space does not match the test function space")
            elif space != spaces[1]:
                raise TypeError("bc space does not match the trial function space")

            # Set diagonal entries on bc nodes to 1 if the current
            # block is on the matrix diagonal and its index matches the
            # index of the function space the bc is defined on.
            op2tensor[index, index].set_local_diagonal_entries(bc.nodes, idx=component, diag_val=self.weight)

            # Handle off-diagonal block involving real function space.
            # "lgmaps" is correctly constructed in _matrix_arg, but
            # is ignored by PyOP2 in this case.
            # Walk through row blocks associated with index.
            for j, s in enumerate(space):
                if j != index and s.ufl_element().family() == "Real":
                    self._apply_bcs_mat_real_block(op2tensor, index, j, component, bc.node_set)
            # Walk through col blocks associated with index.
            for i, s in enumerate(space):
                if i != index and s.ufl_element().family() == "Real":
                    self._apply_bcs_mat_real_block(op2tensor, i, index, component, bc.node_set)
        elif isinstance(bc, EquationBCSplit):
            for j, s in enumerate(spaces[1]):
                if s.ufl_element().family() == "Real":
                    self._apply_bcs_mat_real_block(op2tensor, index, j, component, bc.node_set)
            type(self)(bc.f, self._tensor, bc.bcs, self._form_compiler_params,
                       needs_zeroing=False).assemble()
        else:
            raise AssertionError

    @staticmethod
    def _apply_bcs_mat_real_block(op2tensor, i, j, component, node_set):
        dat = op2tensor[i, j].handle.getPythonContext().dat
        if component is not None:
            dat = op2.DatView(dat, component)
        dat.zero(subset=node_set)


class MatrixFreeAssembler:
    """Stub class wrapping matrix-free assembly."""

    def __init__(self, tensor):
        self._tensor = tensor

    def assemble(self):
        self._tensor.assemble()
        return self._tensor


def get_form_assembler(form, tensor, *args, **kwargs):
    """Provide the assemble method for `form`"""

    # Don't expand derivatives if `mat_type` is 'matfree'
    mat_type = kwargs.pop('mat_type', None)
    if not isinstance(form, slate.TensorBase):
        fc_params = kwargs.get('form_compiler_parameters')
        # Only pre-process `form` once beforehand to avoid pre-processing for each assembly call
        form = preprocess_base_form(form, mat_type=mat_type, form_compiler_parameters=fc_params)

    if isinstance(form, (ufl.form.Form, slate.TensorBase)) and not base_form_operands(form):
        diagonal = kwargs.pop('diagonal', False)
        if len(form.arguments()) == 1 or diagonal:
            return OneFormAssembler(form, tensor, *args, diagonal=diagonal, **kwargs).assemble
        elif len(form.arguments()) == 2:
            return TwoFormAssembler(form, tensor, *args, **kwargs).assemble
        else:
            raise ValueError('Expecting a 1-form or 2-form and not %s' % (form))
    elif isinstance(form, ufl.form.BaseForm):
        return functools.partial(assemble_base_form, form, *args, tensor=tensor, mat_type=mat_type,
                                 is_base_form_preprocessed=True, **kwargs)
    else:
        raise ValueError('Expecting a BaseForm or a slate.TensorBase object and not %s' % form)


class ParloopBuilder:
    """Class that builds a :class:`op2.Parloop`.

    :param form: The variational form.
    :param local_knl: :class:`tsfc_interface.SplitKernel` compiled by either
        TSFC or Slate.
    :param global_knl: A :class:`pyop2.GlobalKernel` instance.
    :param tensor: The output tensor to write to (cannot be ``None``).
    :param subdomain_id: The subdomain of the mesh to iterate over.
    :param all_integer_subdomain_ids: See :func:`tsfc_interface.gather_integer_subdomain_ids`.
    :param diagonal: Are we assembling the diagonal of a 2-form?
    :param lgmaps: Optional iterable of local-to-global maps needed for applying
        boundary conditions to 2-forms.
    """

    def __init__(self, form, local_knl, tensor, subdomain_id,
                 all_integer_subdomain_ids, diagonal=False, lgmaps=None):
        self._form = form
        self._local_knl = local_knl
        self._subdomain_id = subdomain_id
        self._all_integer_subdomain_ids = all_integer_subdomain_ids
        self._tensor = tensor
        self._diagonal = diagonal
        self._lgmaps = lgmaps

        self._active_coefficients = _FormHandler.iter_active_coefficients(form, local_knl.kinfo)
        self._constants = _FormHandler.iter_constants(form, local_knl.kinfo)

    def build(self):
        """Construct the parloop."""
        p = self._iterset.index()
        args = [
            self._as_parloop_arg(tsfc_arg, p)
            for tsfc_arg in self._kinfo.arguments
        ]

        kernel = op3.Function(self._kinfo.kernel.code, [op3.INC] + [op3.READ for _ in args[1:]])
        return op3.loop(p, kernel(*args))

    @property
    def _indices(self):
        return self._local_knl.indices

    @property
    def _kinfo(self):
        return self._local_knl.kinfo

    @property
    def _integral_type(self):
        return self._kinfo.integral_type

    @property
    def _indexed_function_spaces(self):
        return _FormHandler.index_function_spaces(self._form, self._indices)

    @property
    def _indexed_tensor(self):
        return _FormHandler.index_tensor(self._tensor, self._form, self._indices, self._diagonal)

    @cached_property
    def _mesh(self):
        return self._form.ufl_domains()[self._kinfo.domain_number]

    @cached_property
    def _iterset(self):
        try:
            subdomain_data = self._form.subdomain_data()[self._mesh][self._integral_type]
        except KeyError:
            subdomain_data = [None]

        subdomain_data = [sd for sd in subdomain_data if sd is not None]
        if subdomain_data:
            try:
                subdomain_data, = subdomain_data
            except ValueError:
                raise NotImplementedError("Assembly with multiple subdomain data values is not supported")
            if self._integral_type != "cell":
                raise NotImplementedError("subdomain_data only supported with cell integrals")
            if self._subdomain_id not in ["everywhere", "otherwise"]:
                raise ValueError("Cannot use subdomain data and subdomain_id")
            return subdomain_data
        else:
            return self._mesh.topology.measure_set(
                self._integral_type,
                self._subdomain_id,
                self._all_integer_subdomain_ids
            )

    def _get_map(self, V):
        """Return the appropriate PyOP2 map for a given function space."""
        assert isinstance(V, (WithGeometry, FiredrakeDualSpace, FunctionSpace))

        if self._integral_type in {"cell", "exterior_facet_top",
                                   "exterior_facet_bottom", "interior_facet_horiz"}:
            return self._mesh.topology._fiat_closure
        elif self._integral_type in {"exterior_facet", "exterior_facet_vert"}:
            raise NotImplementedError
            return V.exterior_facet_node_map()
        elif self._integral_type in {"interior_facet", "interior_facet_vert"}:
            raise NotImplementedError
            return V.interior_facet_node_map()
        else:
            raise AssertionError

    @functools.singledispatchmethod
    def _as_parloop_arg(self, tsfc_arg, index):
        """Return a :class:`op2.ParloopArg` corresponding to the provided
        :class:`tsfc.KernelArg`.
        """
        raise TypeError(f"No handler provided for {type(tsfc_arg).__name__}")

    @_as_parloop_arg.register(kernel_args.OutputKernelArg)
    def _as_parloop_arg_output(self, _, index):
        rank = len(self._form.arguments())
        tensor = self._indexed_tensor
        Vs = self._indexed_function_spaces

        if rank == 0:
            return tensor
        elif rank == 1 or rank == 2 and self._diagonal:
            V, = Vs
            if V.ufl_element().family() == "Real":
                return tensor
            else:
                return tensor[self._get_map(V)(index)]
        elif rank == 2:
            rmap, cmap = [self._get_map(V) for V in Vs]

            if all(V.ufl_element().family() == "Real" for V in Vs):
                assert rmap is None and cmap is None
                return tensor.handle.getPythonContext().global_
            elif any(V.ufl_element().family() == "Real" for V in Vs):
                m = rmap or cmap
                return tensor.handle.getPythonContext().dat[m(index)]
            else:
                if self._lgmaps:
                    raise NotImplementedError
                # return tensor[rmap(index), cmap(index)], lgmaps=self._lgmaps)
                return tensor[rmap(index), cmap(index)]
        else:
            raise AssertionError

    @_as_parloop_arg.register(kernel_args.CoordinatesKernelArg)
    def _as_parloop_arg_coordinates(self, _, index):
        func = self._mesh.coordinates
        map_ = self._get_map(func.function_space())
        return func.dat[map_(index)]

    @_as_parloop_arg.register(kernel_args.CoefficientKernelArg)
    def _as_parloop_arg_coefficient(self, arg, index):
        coeff = next(self._active_coefficients)
        if coeff.ufl_element().family() == "Real":
            return coeff.dat
        else:
            m = self._get_map(coeff.function_space())
            return coeff.dat[m(index)]

    @_as_parloop_arg.register(kernel_args.ConstantKernelArg)
    def _as_parloop_arg_constant(self, arg, index):
        const = next(self._constants)
        return const.dat

    @_as_parloop_arg.register(kernel_args.CellOrientationsKernelArg)
    def _as_parloop_arg_cell_orientations(self, _, index):
        func = self._mesh.cell_orientations()
        m = self._get_map(func.function_space())
        return func.dat[m(index)]

    @_as_parloop_arg.register(kernel_args.CellSizesKernelArg)
    def _as_parloop_arg_cell_sizes(self, _, index):
        func = self._mesh.cell_sizes
        m = self._get_map(func.function_space())
        return func.dat[m(index)]

    @_as_parloop_arg.register(kernel_args.ExteriorFacetKernelArg)
    def _as_parloop_arg_exterior_facet(self, _, index):
        return self._mesh.exterior_facets.local_facet_dat[index]

    @_as_parloop_arg.register(kernel_args.InteriorFacetKernelArg)
    def _as_parloop_arg_interior_facet(self, _, index):
        return self._mesh.interior_facets.local_facet_dat[index]

    @_as_parloop_arg.register(CellFacetKernelArg)
    def _as_parloop_arg_cell_facet(self, _, index):
        return self._mesh.cell_to_facets[index]

    @_as_parloop_arg.register(LayerCountKernelArg)
    def _as_parloop_arg_layer_count(self, _, index):
        raise NotImplementedError
        glob = op2.Global(
            (1,),
            self._iterset.layers-2,
            dtype=numpy.int32,
            comm=self._iterset.comm
        )
        return op2.GlobalParloopArg(glob)


class _FormHandler:
    """Utility class for inspecting forms and local kernels."""

    @staticmethod
    def iter_active_coefficients(form, kinfo):
        """Yield the form coefficients referenced in ``kinfo``."""
        all_coefficients = form.coefficients()
        for idx, subidxs in kinfo.coefficient_numbers:
            for subidx in subidxs:
                yield all_coefficients[idx].subfunctions[subidx]

    @staticmethod
    def iter_constants(form, kinfo):
        """Yield the form constants"""
        if isinstance(form, slate.TensorBase):
            for const in form.constants():
                yield const
        else:
            all_constants = extract_firedrake_constants(form)
            for constant_index in kinfo.constant_numbers:
                yield all_constants[constant_index]

    @staticmethod
    def index_function_spaces(form, indices):
        """Return the function spaces of the form's arguments, indexed
        if necessary.
        """
        if op3.utils.strictly_all(i is None for i in indices):
            return tuple(a.ufl_function_space() for a in form.arguments())
        else:
            return tuple(a.ufl_function_space()[i] for i, a in zip(indices, form.arguments()))

    @staticmethod
    def index_tensor(tensor, form, indices, diagonal):
        """Return the (indexed) pyop3 data structure tied to ``tensor``."""
        is_indexed = op3.utils.strictly_all(i is not None for i in indices)
        index_str = tuple(str(i) for i in indices)

        rank = len(form.arguments())
        if rank == 0:
            return tensor
        elif rank == 1 or rank == 2 and diagonal:
            is_mixed = type(tensor.ufl_element()) is finat.ufl.MixedElement
            return tensor.dat[index_str] if is_mixed and is_indexed else tensor.dat
        elif rank == 2:
            is_mixed = any(
                type(arg.ufl_element()) is finat.ufl.MixedElement
                for arg in tensor.a.arguments()
            )
            return tensor.M[index_str] if is_mixed and is_indexed else tensor.M
        else:
            raise AssertionError


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
