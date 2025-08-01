import abc
from collections import defaultdict
from collections.abc import Sequence  # noqa: F401
import functools
import itertools
from itertools import product
import numbers

import cachetools
import finat
import firedrake
import numpy
from pyadjoint.tape import annotate_tape
from tsfc import kernel_args
from finat.element_factory import create_element
from tsfc.ufl_utils import extract_firedrake_constants
import ufl
import finat.ufl
from firedrake import (extrusion_utils as eutils, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.formmanipulation import split_form
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
from pyop2.types.mat import _GlobalMatPayload, _DatMatPayload
from pyop2.utils import cached_property


__all__ = "assemble",


_FORM_CACHE_KEY = "firedrake.assemble.FormAssembler"
"""Entry used in form cache to try and reuse assemblers where possible."""


@PETSc.Log.EventDecorator()
@annotate_assemble
def assemble(expr, *args, **kwargs):
    """Assemble.

    Parameters
    ----------
    expr : ufl.classes.Expr or ufl.classes.BaseForm or slate.TensorBase
        Object to assemble.
    tensor : firedrake.function.Function or firedrake.cofunction.Cofunction or matrix.MatrixBase or None
        Existing tensor object to place the result in.
    bcs : Sequence
        Iterable of boundary conditions to apply.
    form_compiler_parameters : dict
        Dictionary of parameters to pass to
        the form compiler. Ignored if not assembling a `ufl.classes.Form`.
        Any parameters provided here will be overridden by parameters set on the
        `ufl.classes.Measure` in the form. For example, if a
        ``quadrature_degree`` of 4 is specified in this argument, but a degree of
        3 is requested in the measure, the latter will be used.
    mat_type : str
        String indicating how a 2-form (matrix) should be
        assembled -- either as a monolithic matrix (``"aij"`` or ``"baij"``),
        a block matrix (``"nest"``), or left as a `matrix.ImplicitMatrix` giving
        matrix-free actions (``'matfree'``). If not supplied, the default value in
        ``parameters["default_matrix_type"]`` is used.  BAIJ differs
        from AIJ in that only the block sparsity rather than the dof
        sparsity is constructed.  This can result in some memory
        savings, but does not work with all PETSc preconditioners.
        BAIJ matrices only make sense for non-mixed matrices.
    sub_mat_type : str
        String indicating the matrix type to
        use *inside* a nested block matrix.  Only makes sense if
        ``mat_type`` is ``nest``.  May be one of ``"aij"`` or ``"baij"``.  If
        not supplied, defaults to ``parameters["default_sub_matrix_type"]``.
    options_prefix : str
        PETSc options prefix to apply to matrices.
    appctx : dict
        Additional information to hang on the assembled
        matrix if an implicit matrix is requested (mat_type ``"matfree"``).
    zero_bc_nodes : bool
        If `True`, set the boundary condition nodes in the
        output tensor to zero rather than to the values prescribed by the
        boundary condition. Default is `True`.
    diagonal : bool
        If assembling a matrix is it diagonal?
    weight : float
        Weight of the boundary condition, i.e. the scalar in front of the
        identity matrix corresponding to the boundary nodes.
        To discretise eigenvalue problems set the weight equal to 0.0.
    allocation_integral_types : Sequence
        `Sequence` of integral types to be used when allocating the output
        `matrix.Matrix`.
    is_base_form_preprocessed : bool
        If `True`, skip preprocessing of the form.
    current_state : firedrake.function.Function or None
        If provided and ``zero_bc_nodes == False``, the boundary condition
        nodes of the output are set to the residual of the boundary conditions
        computed as ``current_state`` minus the boundary condition value.

    Returns
    -------
    float or firedrake.function.Function or firedrake.cofunction.Cofunction or matrix.MatrixBase
        Result of assembly.

    Notes
    -----
    Input arguments are all optional, except ``expr``.

    If expr is a `ufl.classes.Form` or `slate.TensorBase` then this evaluates
    the corresponding integral(s) and returns a `float` for 0-forms,
    a `firedrake.function.Function` or a `firedrake.cofunction.Cofunction`
    for 1-forms and a `matrix.Matrix` or a `matrix.ImplicitMatrix` for 2-forms.
    In the case of 2-forms the rows correspond to the test functions and the
    columns to the trial functions.

    If expr is an expression other than a form, it will be evaluated
    pointwise on the `firedrake.function.Function`s in the expression. This will
    only succeed if all the Functions are on the same
    `firedrake.functionspace.FunctionSpace`.

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    If ``bcs`` is supplied and ``expr`` is a 2-form, the rows and columns
    of the resulting `matrix.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``expr`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.

    """
    if args:
        raise RuntimeError(f"Got unexpected args: {args}")

    assemble_kwargs = {}
    for key in ("tensor", "current_state"):
        if key in kwargs:
            assemble_kwargs[key] = kwargs.pop(key, None)
    return get_assembler(expr, *args, **kwargs).assemble(**assemble_kwargs)


def get_assembler(form, *args, **kwargs):
    """Create an assembler.

    Notes
    -----
    See `assemble` for descriptions of the parameters. ``tensor`` and
    ``current_state`` should not be passed to this function.

    """
    is_base_form_preprocessed = kwargs.pop('is_base_form_preprocessed', False)
    fc_params = kwargs.get('form_compiler_parameters', None)
    if isinstance(form, ufl.form.BaseForm) and not is_base_form_preprocessed:
        mat_type = kwargs.get('mat_type', None)
        # Preprocess the DAG and restructure the DAG
        # Only pre-process `form` once beforehand to avoid pre-processing for each assembly call
        form = BaseFormAssembler.preprocess_base_form(form, mat_type=mat_type, form_compiler_parameters=fc_params)
    if isinstance(form, (ufl.form.Form, slate.TensorBase)) and not BaseFormAssembler.base_form_operands(form):
        diagonal = kwargs.pop('diagonal', False)
        if len(form.arguments()) == 0:
            return ZeroFormAssembler(form, form_compiler_parameters=fc_params)
        elif len(form.arguments()) == 1 or diagonal:
            return OneFormAssembler(form, *args,
                                    bcs=kwargs.get("bcs", None),
                                    form_compiler_parameters=fc_params,
                                    needs_zeroing=kwargs.get("needs_zeroing", True),
                                    zero_bc_nodes=kwargs.get("zero_bc_nodes", True),
                                    diagonal=diagonal,
                                    weight=kwargs.get("weight", 1.0))
        elif len(form.arguments()) == 2:
            return TwoFormAssembler(form, *args, **kwargs)
        else:
            raise ValueError('Expecting a 0-, 1-, or 2-form: got %s' % (form))
    elif isinstance(form, ufl.core.expr.Expr) and not isinstance(form, ufl.core.base_form_operator.BaseFormOperator):
        # BaseForm preprocessing can turn BaseForm into an Expr (cf. case (6) in `restructure_base_form`)
        return ExprAssembler(form)
    elif isinstance(form, ufl.form.BaseForm):
        return BaseFormAssembler(form, *args, **kwargs)
    else:
        raise ValueError(f'Expecting a BaseForm, slate.TensorBase, or Expr object: got {form}')


class ExprAssembler(object):
    """Expression assembler.

    Parameters
    ----------
    expr : ufl.core.expr.Expr
        Expression.

    """

    def __init__(self, expr):
        self._expr = expr

    def assemble(self, tensor=None, current_state=None):
        """Assemble the pointwise expression.

        Parameters
        ----------
        tensor : firedrake.function.Function or firedrake.cofunction.Cofunction or matrix.MatrixBase
            Output tensor.
        current_state : None
            Ignored by this class.

        Returns
        -------
        float or firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Result of evaluation: `float` for 0-forms, `firedrake.cofunction.Cofunction` or `firedrake.function.Function` for 1-forms, and `matrix.MatrixBase` for 2-forms.

        """
        from ufl.algorithms.analysis import extract_base_form_operators
        from ufl.checks import is_scalar_constant_expression

        assert tensor is None
        assert current_state is None
        expr = self._expr
        # Get BaseFormOperators (e.g. `Interpolate` or `ExternalOperator`)
        base_form_operators = extract_base_form_operators(expr)

        # -- Linear combination involving 2-form BaseFormOperators -- #
        # Example: a * dNdu1(u1, u2; v1, v*) + b * dNdu2(u1, u2; v2, v*)
        # with u1, u2 Functions, v1, v2, v* BaseArguments, dNdu1, dNdu2 BaseFormOperators, and a, b scalars.
        if len(base_form_operators) and any(len(e.arguments()) > 1 for e in base_form_operators):
            if isinstance(expr, ufl.algebra.Sum):
                a, b = [assemble(e) for e in expr.ufl_operands]
                # Only Expr resulting in a Matrix if assembled are BaseFormOperator
                if not all(isinstance(op, matrix.AssembledMatrix) for op in (a, b)):
                    raise TypeError('Mismatching Sum shapes')
                return assemble(ufl.FormSum((a, 1), (b, 1)), tensor=tensor)
            elif isinstance(expr, ufl.algebra.Product):
                a, b = expr.ufl_operands
                scalar = [e for e in expr.ufl_operands if is_scalar_constant_expression(e)]
                if scalar:
                    base_form = a if a is scalar else b
                    assembled_mat = assemble(base_form)
                    return assemble(ufl.FormSum((assembled_mat, scalar[0])), tensor=tensor)
                a, b = [assemble(e) for e in (a, b)]
                return assemble(ufl.action(a, b), tensor=tensor)
        # -- Linear combination of Functions and 1-form BaseFormOperators -- #
        # Example: a * u1 + b * u2 + c * N(u1; v*) + d * N(u2; v*)
        # with u1, u2 Functions, N a BaseFormOperator, and a, b, c, d scalars or 0-form BaseFormOperators.
        else:
            base_form_operators = extract_base_form_operators(expr)
            # Substitute base form operators with their output before examining the expression
            # which avoids conflict when determining function space, for example:
            # extract_coefficients(Interpolate(u, V2)) with u \in V1 will result in an output function space V1
            # instead of V2.
            if base_form_operators:
                assembled_bfops = {e: firedrake.assemble(e) for e in base_form_operators}
                expr = ufl.replace(expr, assembled_bfops)
            if tensor is None:
                try:
                    coefficients = ufl.algorithms.extract_coefficients(expr)
                    V, = set(c.function_space() for c in coefficients) - {None}
                except ValueError:
                    raise ValueError("Cannot deduce correct target space from pointwise expression")
                tensor = firedrake.Function(V)
            return tensor.assign(expr)


class AbstractFormAssembler(abc.ABC):
    """Abstract assembler class for forms.

    Parameters
    ----------
    form : ufl.form.BaseForm or slate.TensorBase
        ``form`` to assemble.
    bcs : DirichletBC or EquationBCSplit or Sequence
        Boundary conditions.
    form_compiler_parameters : dict
        ``form_compiler_parameters`` to use.

    """
    def __init__(self, form, bcs=None, form_compiler_parameters=None):
        self._form = form
        self._bcs = solving._extract_bcs(bcs)
        if any(isinstance(bc, EquationBC) for bc in self._bcs):
            raise TypeError("EquationBC objects not expected here. "
                            "Preprocess by extracting the appropriate form with bc.extract_form('Jp') or bc.extract_form('J')")
        self._form_compiler_params = form_compiler_parameters or {}

    @abc.abstractmethod
    def allocate(self):
        """Allocate memory for the output tensor."""

    @abc.abstractmethod
    def assemble(self, tensor=None, current_state=None):
        """Assemble the form.

        Parameters
        ----------
        tensor : firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Output tensor to contain the result of assembly; if `None`, a tensor of appropriate type is created.
        current_state : firedrake.function.Function or None
            If provided, the boundary condition nodes are set to the boundary condition residual
            computed as ``current_state`` minus the boundary condition value.

        Returns
        -------
        float or firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Result of assembly: `float` for 0-forms, `firedrake.cofunction.Cofunction` or `firedrake.function.Function` for 1-forms, and `matrix.MatrixBase` for 2-forms.

        """


class BaseFormAssembler(AbstractFormAssembler):
    """Base form assembler.

    Parameters
    ----------
    form : ufl.form.BaseForm
        `ufl.form.BaseForm` to assemble.

    Notes
    -----
    See `AbstractFormAssembler` and `assemble` for descriptions of the other parameters.

    """

    def __init__(self,
                 form,
                 bcs=None,
                 form_compiler_parameters=None,
                 mat_type=None,
                 sub_mat_type=None,
                 options_prefix=None,
                 appctx=None,
                 zero_bc_nodes=True,
                 diagonal=False,
                 weight=1.0,
                 allocation_integral_types=None):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters)
        self._mat_type = mat_type
        self._sub_mat_type = sub_mat_type
        self._options_prefix = options_prefix
        self._appctx = appctx
        self._zero_bc_nodes = zero_bc_nodes
        self._diagonal = diagonal
        self._weight = weight
        self._allocation_integral_types = allocation_integral_types

    def allocate(self):
        rank = len(self._form.arguments())
        if rank == 2 and not self._diagonal:
            if isinstance(self._form, matrix.MatrixBase):
                return self._form
            elif self._mat_type == "matfree":
                return MatrixFreeAssembler(self._form, bcs=self._bcs, form_compiler_parameters=self._form_compiler_params,
                                           options_prefix=self._options_prefix,
                                           appctx=self._appctx).allocate()
            else:
                test, trial = self._form.arguments()
                sparsity = ExplicitMatrixAssembler._make_sparsity(test, trial, self._mat_type, self._sub_mat_type, self.maps_and_regions)
                return matrix.Matrix(self._form, self._bcs, self._mat_type, sparsity, ScalarType, options_prefix=self._options_prefix)
        else:
            raise NotImplementedError("Only implemented for rank = 2 and diagonal = False")

    @cached_property
    def maps_and_regions(self):
        # The sparsity could be made tighter by inspecting the form DAG.
        test, trial = self._form.arguments()
        return ExplicitMatrixAssembler._make_maps_and_regions_default(test, trial, self.allocation_integral_types)

    @cached_property
    def allocation_integral_types(self):
        if self._allocation_integral_types is None:
            # Use the most conservative integration types.
            test, _ = self._form.arguments()
            if test.function_space().mesh().extruded:
                return ("interior_facet_vert", "interior_facet_horiz")
            else:
                return ("interior_facet", )
        else:
            return self._allocation_integral_types

    def assemble(self, tensor=None, current_state=None):
        """Assemble the form.

        Parameters
        ----------
        tensor : firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Output tensor to contain the result of assembly.
        current_state : firedrake.function.Function or None
            If provided, the boundary condition nodes are set to the boundary condition residual
            computed as ``current_state`` minus the boundary condition value.

        Returns
        -------
        float or firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Result of assembly: `float` for 0-forms, `firedrake.cofunction.Cofunction` or `firedrake.function.Function` for 1-forms, and `matrix.MatrixBase` for 2-forms.

        Notes
        -----
        This function assembles a `ufl.form.BaseForm` object by traversing the corresponding DAG
        in a post-order fashion and evaluating the nodes on the fly.

        """
        def visitor(e, *operands):
            t = tensor if e is self._form else None
            return self.base_form_assembly_visitor(e, t, *operands)

        # DAG assembly: traverse the DAG in a post-order fashion and evaluate the node on the fly.
        visited = {}
        result = BaseFormAssembler.base_form_postorder_traversal(self._form, visitor, visited)

        # Apply BCs after assembly
        rank = len(self._form.arguments())
        if rank == 1 and not isinstance(result, ufl.ZeroBaseForm):
            for bc in self._bcs:
                OneFormAssembler._apply_bc(self, result, bc, u=current_state)

        return result

    def base_form_assembly_visitor(self, expr, tensor, *args):
        r"""Assemble a :class:`~ufl.classes.BaseForm` object given its assembled operands.

            This functions contains the assembly handlers corresponding to the different nodes that
            can arise in a `~ufl.classes.BaseForm` object. It is called by :func:`assemble_base_form`
            in a post-order fashion.
        """
        if isinstance(expr, (ufl.form.Form, slate.TensorBase)):
            if args and self._mat_type != "matfree":
                # Retrieve the Form's children
                base_form_operators = BaseFormAssembler.base_form_operands(expr)
                # Substitute the base form operators by their output
                expr = ufl.replace(expr, dict(zip(base_form_operators, args)))
            form = expr
            rank = len(form.arguments())
            if rank == 0:
                assembler = ZeroFormAssembler(form, form_compiler_parameters=self._form_compiler_params)
            elif rank == 1 or (rank == 2 and self._diagonal):
                assembler = OneFormAssembler(form, form_compiler_parameters=self._form_compiler_params,
                                             zero_bc_nodes=self._zero_bc_nodes, diagonal=self._diagonal, weight=self._weight)
            elif rank == 2:
                assembler = TwoFormAssembler(form, bcs=self._bcs, form_compiler_parameters=self._form_compiler_params,
                                             mat_type=self._mat_type, sub_mat_type=self._sub_mat_type,
                                             options_prefix=self._options_prefix, appctx=self._appctx, weight=self._weight,
                                             allocation_integral_types=self.allocation_integral_types)
            else:
                raise AssertionError
            return assembler.assemble(tensor=tensor)
        elif isinstance(expr, ufl.Adjoint):
            if len(args) != 1:
                raise TypeError("Not enough operands for Adjoint")
            mat, = args
            res = tensor.petscmat if tensor else PETSc.Mat()
            petsc_mat = mat.petscmat
            # Out-of-place Hermitian transpose
            petsc_mat.hermitianTranspose(out=res)
            (row, col) = mat.arguments()
            return matrix.AssembledMatrix((col, row), self._bcs, res,
                                          options_prefix=self._options_prefix)
        elif isinstance(expr, ufl.Action):
            if len(args) != 2:
                raise TypeError("Not enough operands for Action")
            lhs, rhs = args
            if isinstance(lhs, matrix.MatrixBase):
                if isinstance(rhs, (firedrake.Cofunction, firedrake.Function)):
                    petsc_mat = lhs.petscmat
                    (row, col) = lhs.arguments()
                    # The matrix-vector product lives in the dual of the test space.
                    res = tensor if tensor else firedrake.Function(row.function_space().dual())
                    with rhs.dat.vec_ro as v_vec, res.dat.vec as res_vec:
                        petsc_mat.mult(v_vec, res_vec)
                    return res
                elif isinstance(rhs, matrix.MatrixBase):
                    result = tensor.petscmat if tensor else PETSc.Mat()
                    lhs.petscmat.matMult(rhs.petscmat, result=result)
                    if tensor is None:
                        tensor = self.assembled_matrix(expr, result)
                    return tensor
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
            if tensor:
                tensor.zero()
            if all(isinstance(op, numbers.Complex) for op in args):
                result = sum(weight * arg for weight, arg in zip(expr.weights(), args))
                return tensor.assign(result) if tensor else result
            elif (all(isinstance(op, firedrake.Cofunction) for op in args)
                    or all(isinstance(op, firedrake.Function) for op in args)):
                V, = set(a.function_space() for a in args)
                result = tensor if tensor else firedrake.Function(V)
                result.dat.maxpy(expr.weights(), [a.dat for a in args])
                return result
            elif all(isinstance(op, ufl.Matrix) for op in args):
                result = tensor.petscmat if tensor else PETSc.Mat()
                for (op, w) in zip(args, expr.weights()):
                    if result:
                        # If result is not void, then accumulate on it
                        result.axpy(w, op.petscmat)
                    else:
                        # If result is void, then allocate it with first term
                        op.petscmat.copy(result=result)
                        result.scale(w)
                if tensor is None:
                    tensor = self.assembled_matrix(expr, result)
                return tensor
            else:
                raise TypeError("Mismatching FormSum shapes")
        elif isinstance(expr, ufl.ExternalOperator):
            opts = {'form_compiler_parameters': self._form_compiler_params,
                    'mat_type': self._mat_type, 'sub_mat_type': self._sub_mat_type,
                    'appctx': self._appctx, 'options_prefix': self._options_prefix,
                    'diagonal': self._diagonal}
            # External operators might not have any children that needs to be assembled
            # -> e.g. N(u; v0, w) with v0 a ufl.Argument and w a ufl.Coefficient
            if args:
                # Replace base forms in the operands and argument slots of the external operator by their result
                v, *assembled_children = args
                if assembled_children:
                    _, *children = BaseFormAssembler.base_form_operands(expr)
                    # Replace assembled children by their results
                    expr = ufl.replace(expr, dict(zip(children, assembled_children)))
                # Always reconstruct the dual argument (0-slot argument) since it is a BaseForm
                # It is also convenient when we have a Form in that slot since Forms don't play well with `ufl.replace`
                expr = expr._ufl_expr_reconstruct_(*expr.ufl_operands, argument_slots=(v,) + expr.argument_slots()[1:])
            # Call the external operator assembly
            result = expr.assemble(assembly_opts=opts)
            return tensor.assign(result) if tensor else result
        elif isinstance(expr, ufl.Interpolate):
            # Replace assembled children
            _, operand = expr.argument_slots()
            v, *assembled_operand = args
            if assembled_operand:
                # Occur in situations such as Interpolate composition
                operand = assembled_operand[0]

            reconstruct_interp = expr._ufl_expr_reconstruct_
            if (v, operand) != expr.argument_slots():
                expr = reconstruct_interp(operand, v=v)

            # Different assembly procedures:
            # 1) Interpolate(Argument(V1, 1), Argument(V2.dual(), 0)) -> Jacobian (Interpolate matrix)
            # 2) Interpolate(Coefficient(...), Argument(V2.dual(), 0)) -> Operator (or Jacobian action)
            # 3) Interpolate(Argument(V1, 0), Argument(V2.dual(), 1)) -> Jacobian adjoint
            # 4) Interpolate(Argument(V1, 0), Cofunction(...)) -> Action of the Jacobian adjoint
            # This can be generalized to the case where the first slot is an arbitray expression.
            rank = len(expr.arguments())
            # If argument numbers have been swapped => Adjoint.
            arg_operand = ufl.algorithms.extract_arguments(operand)
            is_adjoint = (arg_operand and arg_operand[0].number() == 0)

            # Get the target space
            V = v.function_space().dual()

            # Dual interpolation from mixed source
            if is_adjoint and len(V) > 1:
                cur = 0
                sub_operands = []
                components = numpy.reshape(operand, (-1,))
                for Vi in V:
                    sub_operands.append(ufl.as_tensor(components[cur:cur+Vi.value_size].reshape(Vi.value_shape)))
                    cur += Vi.value_size

                # Component-split of the primal operands interpolated into the dual argument-split
                split_interp = sum(reconstruct_interp(sub_operands[i], v=vi) for (i,), vi in split_form(v))
                return assemble(split_interp, tensor=tensor)

            # Dual interpolation into mixed target
            if is_adjoint and len(arg_operand[0].function_space()) > 1 and rank == 1:
                V = arg_operand[0].function_space()
                tensor = tensor or firedrake.Cofunction(V.dual())

                # Argument-split of the Interpolate gets assembled into the corresponding sub-tensor
                for (i,), sub_interp in split_form(expr):
                    assemble(sub_interp, tensor=tensor.subfunctions[i])
                return tensor

            # Workaround: Renumber argument when needed since Interpolator assumes it takes a zero-numbered argument.
            if not is_adjoint and rank == 2:
                _, v1 = expr.arguments()
                operand = ufl.replace(operand, {v1: v1.reconstruct(number=0)})
            # Get the interpolator
            interp_data = expr.interp_data
            default_missing_val = interp_data.pop('default_missing_val', None)
            interpolator = firedrake.Interpolator(operand, V, **interp_data)
            # Assembly
            if rank == 0:
                Iu = interpolator._interpolate(default_missing_val=default_missing_val)
                return assemble(ufl.Action(v, Iu), tensor=tensor)
            elif rank == 1:
                # Assembling the action of the Jacobian adjoint.
                if is_adjoint:
                    return interpolator._interpolate(v, output=tensor, adjoint=True, default_missing_val=default_missing_val)
                # Assembling the Jacobian action.
                elif interpolator.nargs:
                    return interpolator._interpolate(operand, output=tensor, default_missing_val=default_missing_val)
                # Assembling the operator
                elif tensor is None:
                    return interpolator._interpolate(default_missing_val=default_missing_val)
                else:
                    return firedrake.Interpolator(operand, tensor, **interp_data)._interpolate(default_missing_val=default_missing_val)
            elif rank == 2:
                res = tensor.petscmat if tensor else PETSc.Mat()
                # Get the interpolation matrix
                op2_mat = interpolator.callable()
                petsc_mat = op2_mat.handle
                if is_adjoint:
                    # Out-of-place Hermitian transpose
                    petsc_mat.hermitianTranspose(out=res)
                else:
                    # Copy the interpolation matrix into the output tensor
                    petsc_mat.copy(result=res)
                if tensor is None:
                    tensor = self.assembled_matrix(expr, res)
                return tensor
            else:
                # The case rank == 0 is handled via the DAG restructuring
                raise ValueError("Incompatible number of arguments.")
        elif tensor and isinstance(expr, (firedrake.Function, firedrake.Cofunction, firedrake.MatrixBase)):
            return tensor.assign(expr)
        elif tensor and isinstance(expr, ufl.ZeroBaseForm):
            return tensor.zero()
        elif isinstance(expr, (ufl.Coefficient, ufl.Cofunction, ufl.Matrix, ufl.Argument, ufl.Coargument, ufl.ZeroBaseForm)):
            return expr
        else:
            raise TypeError(f"Unrecognised BaseForm instance: {expr}")

    def assembled_matrix(self, expr, petscmat):
        return matrix.AssembledMatrix(expr.arguments(), self._bcs, petscmat,
                                      options_prefix=self._options_prefix)

    @staticmethod
    def base_form_postorder_traversal(expr, visitor, visited={}):
        if expr in visited:
            return visited[expr]

        stack = [expr]
        while stack:
            e = stack.pop()
            unvisited_children = []
            operands = BaseFormAssembler.base_form_operands(e)
            for arg in operands:
                if arg not in visited:
                    unvisited_children.append(arg)

            if unvisited_children:
                stack.append(e)
                stack.extend(unvisited_children)
            else:
                visited[e] = visitor(e, *(visited[arg] for arg in operands))

        return visited[expr]

    @staticmethod
    def base_form_preorder_traversal(expr, visitor, visited={}):
        if expr in visited:
            return visited[expr]

        stack = [expr]
        while stack:
            e = stack.pop()
            unvisited_children = []
            operands = BaseFormAssembler.base_form_operands(e)
            for arg in operands:
                if arg not in visited:
                    unvisited_children.append(arg)

            if unvisited_children:
                stack.extend(unvisited_children)

            visited[e] = visitor(e)

        return visited[expr]

    @staticmethod
    def reconstruct_node_from_operands(expr, operands):
        if isinstance(expr, (ufl.Adjoint, ufl.Action)):
            return expr._ufl_expr_reconstruct_(*operands)
        elif isinstance(expr, ufl.FormSum):
            return ufl.FormSum(*[(op, w) for op, w in zip(operands, expr.weights())])
        return expr

    @staticmethod
    def base_form_operands(expr):
        if isinstance(expr, (ufl.FormSum, ufl.Adjoint, ufl.Action)):
            return expr.ufl_operands
        if isinstance(expr, ufl.Form):
            # Use reversed to treat base form operators
            # in the order in which they have been made.
            return list(reversed(expr.base_form_operators()))
        if isinstance(expr, ufl.core.base_form_operator.BaseFormOperator):
            # Conserve order
            children = dict.fromkeys(e for e in (expr.argument_slots() + expr.ufl_operands)
                                     if isinstance(e, ufl.form.BaseForm))
            return list(children)
        return []

    @staticmethod
    def restructure_base_form_postorder(expression, visited=None):
        visited = visited or {}

        def visitor(expr, *operands):
            # Need to reconstruct the expression with its visited operands!
            expr = BaseFormAssembler.reconstruct_node_from_operands(expr, operands)
            # Perform the DAG restructuring when needed
            return BaseFormAssembler.restructure_base_form(expr, visited)

        return BaseFormAssembler.base_form_postorder_traversal(expression, visitor, visited)

    @staticmethod
    def restructure_base_form_preorder(expression, visited=None):
        visited = visited or {}

        def visitor(expr):
            # Perform the DAG restructuring when needed
            return BaseFormAssembler.restructure_base_form(expr, visited)

        expression = BaseFormAssembler.base_form_preorder_traversal(expression, visitor, visited)
        # Need to reconstruct the expression at the end when all its operands have been visited!
        operands = [visited.get(args, args) for args in BaseFormAssembler.base_form_operands(expression)]
        return BaseFormAssembler.reconstruct_node_from_operands(expression, operands)

    @staticmethod
    def restructure_base_form(expr, visited=None):
        r"""Perform a preorder traversal to simplify and optimize the DAG.
        Example: Let's consider F(u, N(u; v*); v) with N(u; v*) a base form operator.

                 We have: dFdu = \frac{\partial F}{\partial u} + Action(dFdN, dNdu)
                 Now taking the action on a rank-1 object w (e.g. Coefficient/Cofunction) results in:

            (1) Action(Action(dFdN, dNdu), w)

                    Action                     Action
                    /    \                     /     \
                  Action  w     ----->       dFdN   Action
                  /    \                            /    \
                dFdN    dNdu                      dNdu    w

            This situations does not only arise for BaseFormOperator but also when we have a 2-form instead of dNdu!

            (2) Action(dNdu, w)

                 Action
                  /   \
                 /     w        ----->   dNdu(u; w, v*)
                /
           dNdu(u; uhat, v*)

            (3) Action(F, N)

                 Action                                       F
                  /   \         ----->   F(..., N)[v]  =      |
                F[v]   N                                      N

            (4) Adjoint(dNdu)

                 Adjoint
                    |           ----->   dNdu(u; v*, uhat)
               dNdu(u; uhat, v*)

            (5) N(u; w) (scalar valued)

                                     Action
                N(u; w)   ---->       /   \   = Action(N, w)
                                 N(u; v*)  w

        So from Action(Action(dFdN, dNdu(u; v*)), w) we get:

                 Action             Action               Action
                 /    \    (1)      /     \      (2)     /     \               (4)                                dFdN
               Action  w  ---->  dFdN   Action  ---->  dFdN   dNdu(u; w, v*)  ---->  dFdN(..., dNdu(u; w, v*)) =    |
               /    \                    /    \                                                                  dNdu(u; w, v*)
             dFdN    dNdu              dNdu    w

            (6) ufl.FormSum(dN1du(u; w, v*), dN2du(u; w, v*)) -> ufl.Sum(dN1du(u; w, v*), dN2du(u; w, v*))

              Let's consider `Action(dN1du, w) + Action(dN2du, w)`, we have:

                          FormSum                 (2)         FormSum                    (6)                     Sum
                          /     \                ---->        /     \                   ---->                    /  \
                         /       \                           /       \                                          /    \
              Action(dN1du, w)  Action(dN2du, w)    dN1du(u; w, v*) dN2du(u; w, v*)                 dN1du(u; w, v*)  dN2du(u; w, v*)

            This case arises as a consequence of (2) which turns sum of `Action`s (i.e. ufl.FormSum since Action is a BaseForm)
            into sum of `BaseFormOperator`s (i.e. ufl.Sum since BaseFormOperator is an Expr as well).

            (7) Action(w*, dNdu)

                     Action
                     /   \
                    w*    \        ----->   dNdu(u; v0, w*)
                           \
                      dNdu(u; v1, v0*)

        It uses a recursive approach to reconstruct the DAG as we traverse it, enabling to take into account
        various dag rotations/manipulations in expr.
        """
        if isinstance(expr, ufl.Action):
            left, right = expr.ufl_operands
            is_rank_1 = lambda x: isinstance(x, (firedrake.Cofunction, firedrake.Function, firedrake.Argument)) or len(x.arguments()) == 1
            is_rank_2 = lambda x: len(x.arguments()) == 2

            # -- Case (1) -- #
            # If left is Action and has a rank 2, then it is an action of a 2-form on a 2-form
            if isinstance(left, ufl.Action) and is_rank_2(left):
                return ufl.action(left.left(), ufl.action(left.right(), right))
            # -- Case (2) (except if left has only 1 argument, i.e. we have done case (5)) -- #
            if isinstance(left, ufl.core.base_form_operator.BaseFormOperator) and is_rank_1(right) and len(left.arguments()) != 1:
                # Retrieve the highest numbered argument
                arg = max(left.arguments(), key=lambda v: v.number())
                return ufl.replace(left, {arg: right})
            # -- Case (3) -- #
            if isinstance(left, ufl.Form) and is_rank_1(right):
                # 1) Replace the highest-numbered argument of left by right when needed
                #    -> e.g. if right is a BaseFormOperator with 1 argument.
                # Or
                # 2) Let expr as it is by returning `ufl.Action(left, right)`.
                return ufl.action(left, right)
            # -- Case (7) -- #
            if is_rank_1(left) and isinstance(right, ufl.core.base_form_operator.BaseFormOperator) and len(right.arguments()) != 1:
                # Action(w*, dNdu(u; v1, v*)) -> dNdu(u; v0, w*)
                # Get lowest numbered argument
                arg = min(right.arguments(), key=lambda v: v.number())
                # Need to replace lowest numbered argument of right by left
                replace_map = {arg: left}
                # Decrease number for all the other arguments since the lowest numbered argument will be replaced.
                other_args = [a for a in right.arguments() if a is not arg]
                new_args = [a.reconstruct(number=a.number()-1) for a in other_args]
                replace_map.update(dict(zip(other_args, new_args)))
                # Replace arguments
                return ufl.replace(right, replace_map)

        # -- Case (4) -- #
        if isinstance(expr, ufl.Adjoint) and isinstance(expr.form(), ufl.core.base_form_operator.BaseFormOperator):
            B = expr.form()
            u, v = B.arguments()
            # Let V1 and V2 be primal spaces, B: V1 -> V2 and B*: V2* -> V1*:
            # Adjoint(B(Argument(V1, 1), Argument(V2.dual(), 0))) = B(Argument(V1, 0), Argument(V2.dual(), 1))
            reordered_arguments = {u: u.reconstruct(number=v.number()),
                                   v: v.reconstruct(number=u.number())}
            # Replace arguments in argument slots
            return ufl.replace(B, reordered_arguments)

        # -- Case (5) -- #
        if isinstance(expr, ufl.core.base_form_operator.BaseFormOperator) and len(expr.arguments()) == 0:
            # We are assembling a BaseFormOperator of rank 0 (no arguments).
            # B(f, u*) be a BaseFormOperator with u* a Cofunction and f a Coefficient, then:
            #    B(f, u*) <=> Action(B(f, v*), f) where v* is a Coargument
            ustar, *_ = expr.argument_slots()
            vstar = firedrake.Argument(ustar.function_space(), 0)
            expr = ufl.replace(expr, {ustar: vstar})
            return ufl.action(expr, ustar)

        # -- Case (6) -- #
        if isinstance(expr, ufl.FormSum) and all(ufl.duals.is_dual(a.function_space()) for a in expr.arguments()):
            # Return ufl.Sum if we are assembling a FormSum with Coarguments (a primal expression)
            return sum(w*c for w, c in zip(expr.weights(), expr.components()))
        return expr

    @staticmethod
    def preprocess_base_form(expr, mat_type=None, form_compiler_parameters=None):
        """Preprocess ufl.BaseForm objects"""
        original_expr = expr
        if mat_type != "matfree":
            # Don't expand derivatives if `mat_type` is 'matfree'
            # For "matfree", Form evaluation is delayed
            expr = BaseFormAssembler.expand_derivatives_form(expr, form_compiler_parameters)
        if not isinstance(expr, (ufl.form.Form, slate.TensorBase)):
            # => No restructuring needed for Form and slate.TensorBase
            expr = BaseFormAssembler.restructure_base_form_preorder(expr)
            expr = BaseFormAssembler.restructure_base_form_postorder(expr)
        # Preprocessing the form makes a new object -> current form caching mechanism
        # will populate `expr`'s cache which is now different than `original_expr`'s cache so we need
        # to transmit the cache. All of this only holds when both are `ufl.Form` objects.
        if isinstance(original_expr, ufl.form.Form) and isinstance(expr, ufl.form.Form):
            expr._cache = original_expr._cache
        return expr

    @staticmethod
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


class FormAssembler(AbstractFormAssembler):
    """Form assembler.

    Parameters
    ----------
    form : ufl.Form or slate.TensorBase
        Variational form to be assembled.
    bcs : Sequence
        Iterable of boundary conditions to apply.
    form_compiler_parameters : dict
        Optional parameters to pass to the TSFC and/or Slate compilers.

    """

    def __new__(cls, *args, **kwargs):
        form = args[0]
        if not isinstance(form, (ufl.Form, slate.TensorBase)):
            raise TypeError(f"The first positional argument must be of ufl.Form or slate.TensorBase: got {type(form)} ({form})")
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
        key = cls._cache_key(*args, **kwargs)
        if key is not None:
            try:
                return form._cache[_FORM_CACHE_KEY][key]
            except KeyError:
                pass
        self = super().__new__(cls)
        self._initialised = False
        self.__init__(*args, **kwargs)
        if _FORM_CACHE_KEY not in form._cache:
            form._cache[_FORM_CACHE_KEY] = {}
        form._cache[_FORM_CACHE_KEY][key] = self
        return self

    @classmethod
    @abc.abstractmethod
    def _cache_key(cls, *args, **kwargs):
        """Return cache key."""

    @staticmethod
    def _skip_if_initialised(init):
        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            if not self._initialised:
                init(self, *args, **kwargs)
                self._initialised = True
        return wrapper

    def __init__(self, form, bcs=None, form_compiler_parameters=None):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters)
        if any(c.dat.dtype != ScalarType for c in form.coefficients()):
            raise ValueError("Cannot assemble a form containing coefficients where the "
                             "dtype is not the PETSc scalar type.")


class ParloopFormAssembler(FormAssembler):
    """FormAssembler that uses Parloops.

    Parameters
    ----------
    form : ufl.Form or slate.TensorBase
        Variational form to be assembled.
    bcs : Sequence
        Iterable of boundary conditions to apply.
    form_compiler_parameters : dict
        Optional parameters to pass to the TSFC and/or Slate compilers.
    needs_zeroing : bool
        Should ``tensor`` be zeroed before assembling?

    """
    def __init__(self, form, bcs=None, form_compiler_parameters=None, needs_zeroing=True):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters)
        self._needs_zeroing = needs_zeroing

    def assemble(self, tensor=None, current_state=None):
        """Assemble the form.

        Parameters
        ----------
        tensor : firedrake.cofunction.Cofunction or matrix.MatrixBase
            Output tensor to contain the result of assembly; if `None`, a tensor of appropriate type is created.
        current_state : firedrake.function.Function or None
            If provided, the boundary condition nodes are set to the boundary condition residual
            computed as ``current_state`` minus the boundary condition value.

        Returns
        -------
        float or firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Result of assembly: `float` for 0-forms, `firedrake.cofunction.Cofunction` or `firedrake.function.Function` for 1-forms, and `matrix.MatrixBase` for 2-forms.

        """
        if annotate_tape():
            raise NotImplementedError(
                "Taping with explicit FormAssembler objects is not supported yet. "
                "Use assemble instead."
            )

        if tensor is None:
            tensor = self.allocate()
        else:
            self._check_tensor(tensor)
            if self._needs_zeroing:
                self._as_pyop2_type(tensor).zero()

        self.execute_parloops(tensor)

        for bc in self._bcs:
            self._apply_bc(tensor, bc, u=current_state)

        return self.result(tensor)

    @abc.abstractmethod
    def _apply_bc(self, tensor, bc, u=None):
        """Apply boundary condition."""

    @abc.abstractmethod
    def _check_tensor(self, tensor):
        """Check input tensor."""

    @staticmethod
    @abc.abstractmethod
    def _as_pyop2_type(tensor, indices=None):
        """Cast a Firedrake tensor into a PyOP2 data structure, optionally indexing it."""

    def execute_parloops(self, tensor):
        for parloop in self.parloops(tensor):
            parloop()

    def parloops(self, tensor):
        if hasattr(self, "_parloops"):
            for (lknl, _), parloop in zip(self.local_kernels, self._parloops):
                data = self._as_pyop2_type(tensor, lknl.indices)
                parloop.arguments[0].data = data

        else:
            # Make parloops for one concrete output tensor and cache them.
            parloops_ = []
            for local_kernel, subdomain_id in self.local_kernels:
                parloop_builder = ParloopBuilder(
                    self._form,
                    self._bcs,
                    local_kernel,
                    subdomain_id,
                    self.all_integer_subdomain_ids[local_kernel.indices],
                    diagonal=self.diagonal,
                )
                pyop2_tensor = self._as_pyop2_type(tensor, local_kernel.indices)
                parloop = parloop_builder.build(pyop2_tensor)
                parloops_.append(parloop)
            self._parloops = tuple(parloops_)

        return self._parloops

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

    @property
    @abc.abstractmethod
    def diagonal(self):
        """Are we assembling the diagonal of a 2-form?"""

    @cached_property
    def all_integer_subdomain_ids(self):
        """Return a dict mapping local_kernel.indices to all integer subdomain ids."""
        all_indices = {k.indices for k, _ in self.local_kernels}
        return {
            i: tsfc_interface.gather_integer_subdomain_ids(
                {k for k, _ in self.local_kernels if k.indices == i}
            )
            for i in all_indices
        }

    @abc.abstractmethod
    def result(self, tensor):
        """The result of the assembly operation."""


class ZeroFormAssembler(ParloopFormAssembler):
    """Class for assembling a 0-form.

    Parameters
    ----------
    form : ufl.Form or slate.TensorBase
        0-form.

    Notes
    -----
    See `FormAssembler` and `assemble` for descriptions of the other parameters.

    """

    diagonal = False
    """Diagonal assembly not possible for zero forms."""

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        return

    @FormAssembler._skip_if_initialised
    def __init__(self, form, form_compiler_parameters=None):
        super().__init__(form, bcs=None, form_compiler_parameters=form_compiler_parameters)

    def allocate(self):
        # Getting the comm attribute of a form isn't straightforward
        # form.ufl_domains()[0]._comm seems the most robust method
        # revisit in a refactor
        return op2.Global(
            1,
            [0.0],
            dtype=utils.ScalarType,
            comm=self._form.ufl_domains()[0]._comm
        )

    def _apply_bc(self, tensor, bc, u=None):
        pass

    def _check_tensor(self, tensor):
        pass

    @staticmethod
    def _as_pyop2_type(tensor, indices=None):
        assert not indices
        return tensor

    def result(self, tensor):
        return tensor.data[0]


class OneFormAssembler(ParloopFormAssembler):
    """Class for assembling a 1-form.

    Parameters
    ----------
    form : ufl.Form or slate.TensorBase
        1-form.

    Notes
    -----
    See `FormAssembler` and `assemble` for descriptions of the other parameters.

    """

    @classmethod
    def _cache_key(cls, form, bcs=None, form_compiler_parameters=None, needs_zeroing=True,
                   zero_bc_nodes=True, diagonal=False, weight=1.0):
        bcs = solving._extract_bcs(bcs)
        return tuple(bcs), tuplify(form_compiler_parameters), needs_zeroing, zero_bc_nodes, diagonal, weight

    @FormAssembler._skip_if_initialised
    def __init__(self, form, bcs=None, form_compiler_parameters=None, needs_zeroing=True,
                 zero_bc_nodes=True, diagonal=False, weight=1.0):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters, needs_zeroing=needs_zeroing)
        self._weight = weight
        self._diagonal = diagonal
        self._zero_bc_nodes = zero_bc_nodes
        if self._diagonal and any(isinstance(bc, EquationBCSplit) for bc in self._bcs):
            raise NotImplementedError("Diagonal assembly and EquationBC not supported")
        rank = len(self._form.arguments())
        if rank == 2 and self._diagonal:
            test, trial = self._form.arguments()
            if test.function_space() != trial.function_space():
                raise ValueError("Can only assemble the diagonal of 2-form if the function spaces match")

    def allocate(self):
        rank = len(self._form.arguments())
        if rank == 1:
            test, = self._form.arguments()
            return firedrake.Function(test.function_space().dual())
        elif rank == 2 and self._diagonal:
            test, _ = self._form.arguments()
            return firedrake.Function(test.function_space().dual())
        else:
            raise RuntimeError(f"Not expected: found rank = {rank} and diagonal = {self._diagonal}")

    def _apply_bc(self, tensor, bc, u=None):
        # TODO Maybe this could be a singledispatchmethod?
        if isinstance(bc, DirichletBC):
            if self._diagonal:
                bc.set(tensor, self._weight)
            elif self._zero_bc_nodes:
                bc.zero(tensor)
            else:
                # The residual belongs to a mixed space that is dual on the boundary nodes
                # and primal on the interior nodes. Therefore, this is a type-safe operation.
                r = tensor.riesz_representation("l2")
                bc.apply(r, u=u)
        elif isinstance(bc, EquationBCSplit):
            bc.zero(tensor)
            OneFormAssembler(bc.f, bcs=bc.bcs,
                             form_compiler_parameters=self._form_compiler_params,
                             needs_zeroing=False,
                             zero_bc_nodes=self._zero_bc_nodes,
                             diagonal=self._diagonal,
                             weight=self._weight).assemble(tensor=tensor, current_state=u)
        else:
            raise AssertionError

    def _check_tensor(self, tensor):
        if tensor.function_space() != self._form.arguments()[0].function_space().dual():
            raise ValueError("Form's argument does not match provided result tensor")

    @staticmethod
    def _as_pyop2_type(tensor, indices=None):
        if indices is not None and any(index is not None for index in indices):
            i, = indices
            return tensor.dat[i]
        else:
            return tensor.dat

    def execute_parloops(self, tensor):
        # We are repeatedly incrementing into the same Dat so intermediate halo exchanges
        # can be skipped.
        with tensor.dat.frozen_halo(op2.INC):
            for parloop in self.parloops(tensor):
                parloop()

    @property
    def diagonal(self):
        return self._diagonal

    def result(self, tensor):
        return tensor


def TwoFormAssembler(form, *args, **kwargs):
    assert isinstance(form, (ufl.form.Form, slate.TensorBase))
    mat_type = kwargs.pop('mat_type', None)
    sub_mat_type = kwargs.pop('sub_mat_type', None)
    mat_type, sub_mat_type = _get_mat_type(mat_type, sub_mat_type, form.arguments())
    if mat_type == "matfree":
        kwargs.pop('needs_zeroing', None)
        kwargs.pop('weight', None)
        kwargs.pop('allocation_integral_types', None)
        return MatrixFreeAssembler(form, *args, **kwargs)
    else:
        return ExplicitMatrixAssembler(form, *args, mat_type=mat_type, sub_mat_type=sub_mat_type, **kwargs)


def _get_mat_type(mat_type, sub_mat_type, arguments):
    """Validate the matrix types provided by the user and set any that are
    undefined to default values.

    Parameters
    ----------
    mat_type : str
        PETSc matrix type for the assembled matrix.
    sub_mat_type : str
        PETSc matrix type for blocks if ``mat_type`` is ``"nest"``.
    arguments : Sequence
        The test and trial functions of the expression being assembled.

    Returns
    -------
    tuple
        Tuple of validated/default ``mat_type`` and ``sub_mat_type``.

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


class ExplicitMatrixAssembler(ParloopFormAssembler):
    """Class for assembling a matrix.

    Parameters
    ----------
    form : ufl.Form or slate.TensorBasehe
        2-form.

    Notes
    -----
    See `FormAssembler` and `assemble` for descriptions of the other parameters.

    """

    diagonal = False
    """Diagonal assembly not possible for two forms."""

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        return

    @FormAssembler._skip_if_initialised
    def __init__(self, form, bcs=None, form_compiler_parameters=None, needs_zeroing=True,
                 mat_type=None, sub_mat_type=None, options_prefix=None, appctx=None, weight=1.0,
                 allocation_integral_types=None):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters, needs_zeroing=needs_zeroing)
        self._mat_type = mat_type
        self._sub_mat_type = sub_mat_type
        self._options_prefix = options_prefix
        self._appctx = appctx
        self.weight = weight
        self._allocation_integral_types = allocation_integral_types

    def allocate(self):
        test, trial = self._form.arguments()
        sparsity = ExplicitMatrixAssembler._make_sparsity(test, trial,
                                                          self._mat_type,
                                                          self._sub_mat_type,
                                                          self._make_maps_and_regions())
        return matrix.Matrix(self._form, self._bcs, self._mat_type, sparsity, ScalarType,
                             options_prefix=self._options_prefix,
                             fc_params=self._form_compiler_params)

    @staticmethod
    def _make_sparsity(test, trial, mat_type, sub_mat_type, maps_and_regions):
        assert mat_type != "matfree"
        nest = mat_type == "nest"
        if nest:
            baij = sub_mat_type == "baij"
        else:
            baij = mat_type == "baij"
        if any(len(a.function_space()) > 1 for a in [test, trial]) and mat_type == "baij":
            raise ValueError("BAIJ matrix type makes no sense for mixed spaces, use 'aij'")
        try:
            sparsity = op2.Sparsity((test.function_space().dof_dset,
                                     trial.function_space().dof_dset),
                                    maps_and_regions,
                                    nest=nest,
                                    block_sparse=baij)
        except SparsityFormatError:
            raise ValueError("Monolithic matrix assembly not supported for systems "
                             "with R-space blocks")
        return sparsity

    def _make_maps_and_regions(self):
        test, trial = self._form.arguments()
        if self._allocation_integral_types is not None:
            return ExplicitMatrixAssembler._make_maps_and_regions_default(test, trial, self._allocation_integral_types)
        elif any(local_kernel.indices == (None, None) for assembler in self._all_assemblers for local_kernel, _ in assembler.local_kernels):
            # Handle special cases: slate or split=False
            assert all(local_kernel.indices == (None, None) for assembler in self._all_assemblers for local_kernel, _ in assembler.local_kernels)
            allocation_integral_types = set(local_kernel.kinfo.integral_type
                                            for assembler in self._all_assemblers
                                            for local_kernel, _ in assembler.local_kernels)
            return ExplicitMatrixAssembler._make_maps_and_regions_default(test, trial, allocation_integral_types)
        else:
            maps_and_regions = defaultdict(lambda: defaultdict(set))
            for assembler in self._all_assemblers:
                all_meshes = assembler._form.ufl_domains()
                for local_kernel, subdomain_id in assembler.local_kernels:
                    i, j = local_kernel.indices
                    mesh = all_meshes[local_kernel.kinfo.domain_number]  # integration domain
                    integral_type = local_kernel.kinfo.integral_type
                    all_subdomain_ids = assembler.all_integer_subdomain_ids[local_kernel.indices]
                    # Make Sparsity independent of the subdomain of integration for better reusability;
                    # subdomain_id is passed here only to determine the integration_type on the target domain
                    # (see ``entity_node_map``).
                    rmap_ = test.function_space().topological[i].entity_node_map(mesh.topology, integral_type, subdomain_id, all_subdomain_ids)
                    cmap_ = trial.function_space().topological[j].entity_node_map(mesh.topology, integral_type, subdomain_id, all_subdomain_ids)
                    region = ExplicitMatrixAssembler._integral_type_region_map[integral_type]
                    maps_and_regions[(i, j)][(rmap_, cmap_)].add(region)
            return {block_indices: [map_pair + (tuple(region_set), ) for map_pair, region_set in map_pair_to_region_set.items()]
                    for block_indices, map_pair_to_region_set in maps_and_regions.items()}

    @staticmethod
    def _make_maps_and_regions_default(test, trial, allocation_integral_types):
        # Make maps using outer-product of the component maps
        # using the given allocation_integral_types.
        if allocation_integral_types is None:
            raise ValueError("allocation_integral_types can not be None")
        maps_and_regions = defaultdict(lambda: defaultdict(set))
        # Use outer product of component maps.
        for integral_type in allocation_integral_types:
            region = ExplicitMatrixAssembler._integral_type_region_map[integral_type]
            for i, Vrow in enumerate(test.function_space()):
                for j, Vcol in enumerate(trial.function_space()):
                    mesh = Vrow.mesh()
                    rmap_ = Vrow.topological.entity_node_map(mesh.topology, integral_type, None, None)
                    cmap_ = Vcol.topological.entity_node_map(mesh.topology, integral_type, None, None)
                    maps_and_regions[(i, j)][(rmap_, cmap_)].add(region)
        return {block_indices: [map_pair + (tuple(region_set), ) for map_pair, region_set in map_pair_to_region_set.items()]
                for block_indices, map_pair_to_region_set in maps_and_regions.items()}

    _integral_type_region_map = \
        {"cell": op2.ALL,
         "exterior_facet_bottom": op2.ON_BOTTOM,
         "exterior_facet_top": op2.ON_TOP,
         "interior_facet_horiz": op2.ON_INTERIOR_FACETS,
         "exterior_facet": op2.ALL,
         "exterior_facet_vert": op2.ALL,
         "interior_facet": op2.ALL,
         "interior_facet_vert": op2.ALL}

    @cached_property
    def _all_assemblers(self):
        """Tuple of all assemblers used for sparsity construction.

        When constructing sparsity, we use all assemblers
        that are to be used in the actual assembly.
        """
        all_assemblers = [self]
        for bc in self._bcs:
            if isinstance(bc, EquationBCSplit):
                _assembler = type(self)(bc.f, bcs=bc.bcs, form_compiler_parameters=self._form_compiler_params, needs_zeroing=False)
                all_assemblers.extend(_assembler._all_assemblers)
        return tuple(all_assemblers)

    def _apply_bc(self, tensor, bc, u=None):
        assert u is None
        op2tensor = tensor.M
        spaces = tuple(a.function_space() for a in tensor.a.arguments())
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
            type(self)(bc.f, bcs=bc.bcs, form_compiler_parameters=self._form_compiler_params, needs_zeroing=False).assemble(tensor=tensor)
        else:
            raise AssertionError

    @staticmethod
    def _apply_bcs_mat_real_block(op2tensor, i, j, component, node_set):
        dat = op2tensor[i, j].handle.getPythonContext().dat
        if component is not None:
            dat = op2.DatView(dat, component)
        dat.zero(subset=node_set)

    def _check_tensor(self, tensor):
        if tensor.a.arguments() != self._form.arguments():
            raise ValueError("Form's arguments do not match provided result tensor")

    @staticmethod
    def _as_pyop2_type(tensor, indices=None):
        if indices is not None and any(index is not None for index in indices):
            i, j = indices
            mat = tensor.M[i, j]
        else:
            mat = tensor.M

        if mat.handle.getType() == "python":
            mat_context = mat.handle.getPythonContext()
            if isinstance(mat_context, _GlobalMatPayload):
                mat = mat_context.global_
            else:
                assert isinstance(mat_context, _DatMatPayload)
                mat = mat_context.dat

        return mat

    def result(self, tensor):
        tensor.M.assemble()
        return tensor


class MatrixFreeAssembler(FormAssembler):
    """Stub class wrapping matrix-free assembly.

    Parameters
    ----------
    form : ufl.Form or slate.TensorBase
        2-form.

    Notes
    -----
    See `FormAssembler` and `assemble` for descriptions of the other parameters.

    """

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        return

    @FormAssembler._skip_if_initialised
    def __init__(self, form, bcs=None, form_compiler_parameters=None,
                 options_prefix=None, appctx=None):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters)
        self._options_prefix = options_prefix
        self._appctx = appctx

    def allocate(self):
        return matrix.ImplicitMatrix(self._form, self._bcs,
                                     fc_params=self._form_compiler_params,
                                     options_prefix=self._options_prefix,
                                     appctx=self._appctx or {})

    def assemble(self, tensor=None, current_state=None):
        if tensor is None:
            tensor = self.allocate()
        else:
            self._check_tensor(tensor)
        tensor.assemble()
        return tensor

    def _check_tensor(self, tensor):
        if tensor.a.arguments() != self._form.arguments():
            raise ValueError("Form's arguments do not match provided result tensor")


def _global_kernel_cache_key(form, local_knl, subdomain_id, all_integer_subdomain_ids, **kwargs):
    # N.B. Generating the global kernel is not a collective operation so the
    # communicator does not need to be a part of this cache key.

    if isinstance(form, ufl.Form):
        sig = form.signature()
    elif isinstance(form, slate.TensorBase):
        sig = form.expression_hash

    # The form signature does not store this information. This should be accessible from
    # the UFL so we don't need this nasty hack.
    subdomain_key = []
    for val in form.subdomain_data().values():
        for k, v in val.items():
            for i, vi in enumerate(v):
                if vi is not None:
                    extruded = vi._extruded
                    constant_layers = extruded and vi.constant_layers
                    subset = isinstance(vi, op2.Subset)
                    subdomain_key.append((k, i, extruded, constant_layers, subset))
                else:
                    subdomain_key.append((k, i))

    return ((sig, subdomain_id)
            + tuple(subdomain_key)
            + tuplify(all_integer_subdomain_ids)
            + cachetools.keys.hashkey(local_knl, **kwargs))


@cachetools.cached(cache={}, key=_global_kernel_cache_key)
def _make_global_kernel(*args, **kwargs):
    return _GlobalKernelBuilder(*args, **kwargs).build()


class _GlobalKernelBuilder:
    """Class that builds a :class:`op2.GlobalKernel`.

    :param form: The variational form.
    :param local_knl: :class:`tsfc_interface.SplitKernel` compiled by either
        TSFC or Slate.
    :param subdomain_id: The subdomain of the mesh to iterate over.
    :param all_integer_subdomain_ids: See :func:`tsfc_interface.gather_integer_subdomain_ids`.
    :param diagonal: Are we assembling the diagonal of a 2-form?
    :param unroll: If ``True``, address matrix elements directly rather than in
        a blocked fashion. This is slower but required for the application of
        some boundary conditions.

    .. note::
        One should be able to generate a global kernel without needing to
        use any data structures (i.e. a stripped form should be sufficient).
    """

    def __init__(self, form, local_knl, subdomain_id, all_integer_subdomain_ids, diagonal=False, unroll=False):
        self._form = form
        self._indices, self._kinfo = local_knl
        self._subdomain_id = subdomain_id
        self._all_integer_subdomain_ids = all_integer_subdomain_ids
        self._diagonal = diagonal
        self._unroll = unroll

        self._active_coefficients = _FormHandler.iter_active_coefficients(form, local_knl.kinfo)
        self._constants = _FormHandler.iter_constants(form, local_knl.kinfo)

        self._map_arg_cache = {}
        # Cache for holding :class:`op2.MapKernelArg` instances.
        # This is required to ensure that we use the same map argument when the
        # data objects in the parloop would be using the same map. This is to avoid
        # unnecessary packing in the global kernel.

    def build(self):
        """Build the global kernel."""
        kernel_args = [self._as_global_kernel_arg(arg)
                       for arg in self._kinfo.arguments]

        # we should use up all of the coefficients and constants
        assert_empty(self._active_coefficients)
        assert_empty(self._constants)

        iteration_regions = {"exterior_facet_top": op2.ON_TOP,
                             "exterior_facet_bottom": op2.ON_BOTTOM,
                             "interior_facet_horiz": op2.ON_INTERIOR_FACETS}
        iteration_region = iteration_regions.get(self._integral_type, None)
        extruded = self._mesh.extruded
        extruded_periodic = self._mesh.extruded_periodic
        constant_layers = extruded and not self._mesh.variable_layers

        return op2.GlobalKernel(self._kinfo.kernel,
                                kernel_args,
                                iteration_region=iteration_region,
                                pass_layer_arg=self._kinfo.pass_layer_arg,
                                extruded=extruded,
                                extruded_periodic=extruded_periodic,
                                constant_layers=constant_layers,
                                subset=self._needs_subset)

    @property
    def _integral_type(self):
        return self._kinfo.integral_type

    @cached_property
    def _mesh(self):
        return self._form.ufl_domains()[self._kinfo.domain_number]

    @cached_property
    def _needs_subset(self):
        subdomain_data = self._form.subdomain_data()[self._mesh]
        if not all(sd is None for sd in subdomain_data.get(self._integral_type, [None])):
            return True

        if self._subdomain_id == "everywhere":
            return False
        elif self._subdomain_id == "otherwise":
            return self._all_integer_subdomain_ids.get(self._kinfo.integral_type, None) is not None
        else:
            return True

    @property
    def _indexed_function_spaces(self):
        return _FormHandler.index_function_spaces(self._form, self._indices)

    def _as_global_kernel_arg(self, tsfc_arg):
        # TODO Make singledispatchmethod with Python 3.8
        return _as_global_kernel_arg(tsfc_arg, self)

    def _get_dim(self, finat_element):
        if isinstance(finat_element, finat.TensorFiniteElement):
            return finat_element._shape
        else:
            return (1,)

    def _make_dat_global_kernel_arg(self, V, index=None):
        finat_element = create_element(V.ufl_element())
        map_arg = V.topological.entity_node_map(self._mesh.topology, self._integral_type, self._subdomain_id, self._all_integer_subdomain_ids)._global_kernel_arg
        if isinstance(finat_element, finat.EnrichedElement) and finat_element.is_mixed:
            assert index is None
            subargs = tuple(self._make_dat_global_kernel_arg(Vsub, index=index)
                            for Vsub in V)
            return op2.MixedDatKernelArg(subargs)
        else:
            dim = self._get_dim(finat_element)
            return op2.DatKernelArg(dim, map_arg, index)

    def _make_mat_global_kernel_arg(self, Vrow, Vcol):
        relem, celem = (create_element(V.ufl_element()) for V in [Vrow, Vcol])
        if any(isinstance(e, finat.EnrichedElement) and e.is_mixed for e in {relem, celem}):
            subargs = tuple(self._make_mat_global_kernel_arg(Vrow_sub, Vcol_sub)
                            for Vrow_sub, Vcol_sub in product(Vrow, Vcol))
            shape = len(relem.elements), len(celem.elements)
            return op2.MixedMatKernelArg(subargs, shape)
        else:
            rmap_arg, cmap_arg = (V.topological.entity_node_map(self._mesh.topology, self._integral_type, self._subdomain_id, self._all_integer_subdomain_ids)._global_kernel_arg for V in [Vrow, Vcol])
            # PyOP2 matrix objects have scalar dims so we flatten them here
            rdim = numpy.prod(self._get_dim(relem), dtype=int)
            cdim = numpy.prod(self._get_dim(celem), dtype=int)
            return op2.MatKernelArg((((rdim, cdim),),), (rmap_arg, cmap_arg), unroll=self._unroll)

    @staticmethod
    def _get_map_id(finat_element):
        """Return a key that is used to check if we reuse maps.

        This mirrors firedrake.functionspacedata.
        """
        if isinstance(finat_element, finat.TensorFiniteElement):
            finat_element = finat_element.base_element

        real_tensorproduct = eutils.is_real_tensor_product_element(finat_element)
        try:
            eperm_key = entity_permutations_key(finat_element.entity_permutations)
        except NotImplementedError:
            eperm_key = None
        return entity_dofs_key(finat_element.entity_dofs()), real_tensorproduct, eperm_key


@functools.singledispatch
def _as_global_kernel_arg(tsfc_arg, self):
    raise NotImplementedError


@_as_global_kernel_arg.register(kernel_args.OutputKernelArg)
def _as_global_kernel_arg_output(_, self):
    rank = len(self._form.arguments())
    Vs = self._indexed_function_spaces

    if rank == 0:
        return op2.GlobalKernelArg((1,))
    elif rank == 1 or rank == 2 and self._diagonal:
        V, = Vs
        if V.ufl_element().family() == "Real":
            return op2.GlobalKernelArg((1,))
        else:
            return self._make_dat_global_kernel_arg(V)
    elif rank == 2:
        if all(V.ufl_element().family() == "Real" for V in Vs):
            return op2.GlobalKernelArg((1,))
        elif Vs[0].ufl_element().family() == "Real":
            return self._make_dat_global_kernel_arg(Vs[1])
        elif Vs[1].ufl_element().family() == "Real":
            return self._make_dat_global_kernel_arg(Vs[0])
        else:
            return self._make_mat_global_kernel_arg(Vs[0], Vs[1])
    else:
        raise AssertionError


@_as_global_kernel_arg.register(kernel_args.CoordinatesKernelArg)
def _as_global_kernel_arg_coordinates(_, self):
    V = self._mesh.coordinates.function_space()
    return self._make_dat_global_kernel_arg(V)


@_as_global_kernel_arg.register(kernel_args.CoefficientKernelArg)
def _as_global_kernel_arg_coefficient(_, self):
    coeff = next(self._active_coefficients)
    V = coeff.ufl_function_space()
    if hasattr(V, "component") and V.component is not None:
        index = V.component,
        V = V.parent
    else:
        index = None

    ufl_element = V.ufl_element()
    if ufl_element.family() == "Real":
        return op2.GlobalKernelArg((V.value_size,))
    else:
        return self._make_dat_global_kernel_arg(V, index=index)


@_as_global_kernel_arg.register(kernel_args.ConstantKernelArg)
def _as_global_kernel_arg_constant(_, self):
    const = next(self._constants)
    value_size = numpy.prod(const.ufl_shape, dtype=int)
    return op2.GlobalKernelArg((value_size,))


@_as_global_kernel_arg.register(kernel_args.CellSizesKernelArg)
def _as_global_kernel_arg_cell_sizes(_, self):
    V = self._mesh.cell_sizes.function_space()
    return self._make_dat_global_kernel_arg(V)


@_as_global_kernel_arg.register(kernel_args.ExteriorFacetKernelArg)
def _as_global_kernel_arg_exterior_facet(_, self):
    return op2.DatKernelArg((1,))


@_as_global_kernel_arg.register(kernel_args.InteriorFacetKernelArg)
def _as_global_kernel_arg_interior_facet(_, self):
    return op2.DatKernelArg((2,))


@_as_global_kernel_arg.register(kernel_args.ExteriorFacetOrientationKernelArg)
def _as_global_kernel_arg_exterior_facet_orientation(_, self):
    return op2.DatKernelArg((1,))


@_as_global_kernel_arg.register(kernel_args.InteriorFacetOrientationKernelArg)
def _as_global_kernel_arg_interior_facet_orientation(_, self):
    return op2.DatKernelArg((2,))


@_as_global_kernel_arg.register(CellFacetKernelArg)
def _as_global_kernel_arg_cell_facet(_, self):
    if self._mesh.extruded:
        num_facets = self._mesh._base_mesh.ufl_cell().num_facets()
    else:
        num_facets = self._mesh.ufl_cell().num_facets()
    return op2.DatKernelArg((num_facets, 2))


@_as_global_kernel_arg.register(kernel_args.CellOrientationsKernelArg)
def _as_global_kernel_arg_cell_orientations(_, self):
    V = self._mesh.cell_orientations().function_space()
    return self._make_dat_global_kernel_arg(V)


@_as_global_kernel_arg.register(LayerCountKernelArg)
def _as_global_kernel_arg_layer_count(_, self):
    return op2.GlobalKernelArg((1,))


class ParloopBuilder:
    """Class that builds a :class:`op2.Parloop`.

    Parameters
    ----------
    form : ufl.Form or slate.TensorBase
        Variational form.
    bcs : Sequence
        Boundary conditions.
    local_knl : tsfc_interface.SplitKernel
        Kernel compiled by either TSFC or Slate.
    subdomain_id : int
        Subdomain of the mesh to iterate over.
    all_integer_subdomain_ids : Sequence
        See `tsfc_interface.gather_integer_subdomain_ids`.
    diagonal : bool
        Are we assembling the diagonal of a 2-form?

    """
    def __init__(self, form, bcs, local_knl, subdomain_id,
                 all_integer_subdomain_ids, diagonal):
        self._form = form
        self._local_knl = local_knl
        self._subdomain_id = subdomain_id
        self._all_integer_subdomain_ids = all_integer_subdomain_ids
        self._diagonal = diagonal
        self._bcs = bcs

        self._active_coefficients = _FormHandler.iter_active_coefficients(form, local_knl.kinfo)
        self._constants = _FormHandler.iter_constants(form, local_knl.kinfo)

    def build(self, tensor: op2.Global | op2.Dat | op2.Mat) -> op2.Parloop:
        """Construct the parloop.

        Parameters
        ----------
        tensor :
            The output tensor.

        """
        self._tensor = tensor
        parloop_args = [self._as_parloop_arg(tsfc_arg)
                        for tsfc_arg in self._kinfo.arguments]
        _global_knl = _make_global_kernel(
            self._form,
            self._local_knl,
            self._subdomain_id,
            self._all_integer_subdomain_ids,
            diagonal=self._diagonal,
            unroll=self.needs_unrolling()
        )
        try:
            return op2.Parloop(_global_knl, self._iterset, parloop_args)
        except MapValueError:
            raise RuntimeError("Integral measure does not match measure of all "
                               "coefficients/arguments")

    @property
    def test_function_space(self):
        assert len(self._form.arguments()) == 2 and not self._diagonal
        test, _ = self._form.arguments()
        return test.function_space()

    @property
    def trial_function_space(self):
        assert len(self._form.arguments()) == 2 and not self._diagonal
        _, trial = self._form.arguments()
        return trial.function_space()

    def get_indicess(self):
        assert len(self._form.arguments()) == 2 and not self._diagonal
        if all(i is None for i in self._local_knl.indices):
            test, trial = self._form.arguments()
            return numpy.ndindex((len(test.function_space()),
                                  len(trial.function_space())))
        else:
            assert all(i is not None for i in self._local_knl.indices)
            return self._local_knl.indices,

    def _filter_bcs(self, row, col):
        assert len(self._form.arguments()) == 2 and not self._diagonal
        if len(self.test_function_space) > 1:
            bcrow = tuple(bc for bc in self._bcs
                          if bc.function_space_index() == row)
        else:
            bcrow = self._bcs

        if len(self.trial_function_space) > 1:
            bccol = tuple(bc for bc in self._bcs
                          if bc.function_space_index() == col
                          and isinstance(bc, DirichletBC))
        else:
            bccol = tuple(bc for bc in self._bcs if isinstance(bc, DirichletBC))
        return bcrow, bccol

    def needs_unrolling(self):
        """Do we need to address matrix elements directly rather than in
        a blocked fashion?

        This is slower but required for the application of some boundary conditions
        to 2-forms.

        :param local_knl: A :class:`tsfc_interface.SplitKernel`.
        :param bcs: Iterable of boundary conditions.
        """
        if len(self._form.arguments()) == 2 and not self._diagonal:
            for i, j in self.get_indicess():
                for bc in itertools.chain(*self._filter_bcs(i, j)):
                    if bc.function_space().component is not None:
                        return True
        return False

    def collect_lgmaps(self):
        """Return any local-to-global maps that need to be swapped out.

        This is only needed when applying boundary conditions to 2-forms.

        :param local_knl: A :class:`tsfc_interface.SplitKernel`.
        :param bcs: Iterable of boundary conditions.
        """

        if len(self._form.arguments()) == 2 and not self._diagonal:
            if not self._bcs:
                return None

            if any(i is not None for i in self._local_knl.indices):
                i, j = self._local_knl.indices
                row_bcs, col_bcs = self._filter_bcs(i, j)
                # the tensor is already indexed
                rlgmap, clgmap = self._tensor.local_to_global_maps
                rlgmap = self.test_function_space[i].local_to_global_map(row_bcs, rlgmap)
                clgmap = self.trial_function_space[j].local_to_global_map(col_bcs, clgmap)
                return ((rlgmap, clgmap),)
            else:
                lgmaps = []
                for i, j in self.get_indicess():
                    row_bcs, col_bcs = self._filter_bcs(i, j)
                    rlgmap, clgmap = self._tensor[i, j].local_to_global_maps
                    rlgmap = self.test_function_space[i].local_to_global_map(row_bcs, rlgmap)
                    clgmap = self.trial_function_space[j].local_to_global_map(col_bcs, clgmap)
                    lgmaps.append((rlgmap, clgmap))
                return tuple(lgmaps)
        else:
            return None

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
            return self._mesh.measure_set(self._integral_type, self._subdomain_id,
                                          self._all_integer_subdomain_ids)

    def _get_map(self, V):
        """Return the appropriate PyOP2 map for a given function space."""
        assert isinstance(V, (WithGeometry, FiredrakeDualSpace, FunctionSpace))
        return V.topological.entity_node_map(self._mesh.topology, self._integral_type, self._subdomain_id, self._all_integer_subdomain_ids)

    def _as_parloop_arg(self, tsfc_arg):
        """Return a :class:`op2.ParloopArg` corresponding to the provided
        :class:`tsfc.KernelArg`.
        """
        # TODO Make singledispatchmethod with Python 3.8
        return _as_parloop_arg(tsfc_arg, self)


@functools.singledispatch
def _as_parloop_arg(tsfc_arg, self):
    raise NotImplementedError


@_as_parloop_arg.register(kernel_args.OutputKernelArg)
def _as_parloop_arg_output(_, self):
    rank = len(self._form.arguments())
    Vs = self._indexed_function_spaces

    if rank == 0:
        return op2.GlobalParloopArg(self._tensor)
    elif rank == 1 or rank == 2 and self._diagonal:
        V, = Vs
        if V.ufl_element().family() == "Real":
            return op2.GlobalParloopArg(self._tensor)
        else:
            return op2.DatParloopArg(self._tensor, self._get_map(V))
    elif rank == 2:
        rmap, cmap = [self._get_map(V) for V in Vs]

        if all(V.ufl_element().family() == "Real" for V in Vs):
            assert rmap is None and cmap is None
            return op2.GlobalParloopArg(self._tensor)
        elif any(V.ufl_element().family() == "Real" for V in Vs):
            m = rmap or cmap
            return op2.DatParloopArg(self._tensor, m)
        else:
            return op2.MatParloopArg(self._tensor, (rmap, cmap), lgmaps=self.collect_lgmaps())
    else:
        raise AssertionError


@_as_parloop_arg.register(kernel_args.CoordinatesKernelArg)
def _as_parloop_arg_coordinates(_, self):
    func = self._mesh.coordinates
    map_ = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, map_)


@_as_parloop_arg.register(kernel_args.CoefficientKernelArg)
def _as_parloop_arg_coefficient(arg, self):
    coeff = next(self._active_coefficients)
    if coeff.ufl_element().family() == "Real":
        return op2.GlobalParloopArg(coeff.dat)
    else:
        m = self._get_map(coeff.function_space())
        return op2.DatParloopArg(coeff.dat, m)


@_as_parloop_arg.register(kernel_args.ConstantKernelArg)
def _as_parloop_arg_constant(arg, self):
    const = next(self._constants)
    return op2.GlobalParloopArg(const.dat)


@_as_parloop_arg.register(kernel_args.CellOrientationsKernelArg)
def _as_parloop_arg_cell_orientations(_, self):
    func = self._mesh.cell_orientations()
    m = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, m)


@_as_parloop_arg.register(kernel_args.CellSizesKernelArg)
def _as_parloop_arg_cell_sizes(_, self):
    func = self._mesh.cell_sizes
    m = self._get_map(func.function_space())
    return op2.DatParloopArg(func.dat, m)


@_as_parloop_arg.register(kernel_args.ExteriorFacetKernelArg)
def _as_parloop_arg_exterior_facet(_, self):
    return op2.DatParloopArg(self._mesh.exterior_facets.local_facet_dat)


@_as_parloop_arg.register(kernel_args.InteriorFacetKernelArg)
def _as_parloop_arg_interior_facet(_, self):
    return op2.DatParloopArg(self._mesh.interior_facets.local_facet_dat)


@_as_parloop_arg.register(kernel_args.ExteriorFacetOrientationKernelArg)
def _as_parloop_arg_exterior_facet_orientation(_, self):
    return op2.DatParloopArg(self._mesh.exterior_facets.local_facet_orientation_dat)


@_as_parloop_arg.register(kernel_args.InteriorFacetOrientationKernelArg)
def _as_parloop_arg_interior_facet_orientation(_, self):
    return op2.DatParloopArg(self._mesh.interior_facets.local_facet_orientation_dat)


@_as_parloop_arg.register(CellFacetKernelArg)
def _as_parloop_arg_cell_facet(_, self):
    return op2.DatParloopArg(self._mesh.cell_to_facets)


@_as_parloop_arg.register(LayerCountKernelArg)
def _as_parloop_arg_layer_count(_, self):
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
        """Yield the form constants referenced in ``kinfo``."""
        if isinstance(form, slate.TensorBase):
            all_constants = form.constants()
        else:
            all_constants = extract_firedrake_constants(form)
        for constant_index in kinfo.constant_numbers:
            yield all_constants[constant_index]

    @staticmethod
    def index_function_spaces(form, indices):
        """Return the function spaces of the form's arguments, indexed
        if necessary.
        """
        if all(i is None for i in indices):
            return tuple(a.ufl_function_space() for a in form.arguments())
        elif all(i is not None for i in indices):
            return tuple(a.ufl_function_space()[i] for i, a in zip(indices, form.arguments()))
        else:
            raise AssertionError
