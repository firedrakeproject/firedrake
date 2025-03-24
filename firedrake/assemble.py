import abc
import contextlib
import functools
import itertools
import numbers
import operator
from collections import OrderedDict, defaultdict
from collections.abc import Sequence  # noqa: F401
from itertools import product
from functools import cached_property

import cachetools
from pyrsistent import freeze, pmap
import finat
import loopy as lp
import firedrake
import numpy
from pyadjoint.tape import annotate_tape
from tsfc import kernel_args
from finat.element_factory import create_element
from tsfc.ufl_utils import extract_firedrake_constants
import ufl
import pyop3 as op3
from firedrake import (extrusion_utils as eutils, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint_utils import annotate_assemble
from firedrake.ufl_expr import extract_unique_domain
from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
from firedrake.functionspaceimpl import WithGeometry, FunctionSpace, FiredrakeDualSpace
from firedrake.functionspacedata import entity_dofs_key, entity_permutations_key
from firedrake.parloops import pack_tensor, _with_shape_indices, _facet_integral_pack_indices, _cell_integral_pack_indices, pack_pyop3_tensor
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
    tensor = kwargs.pop("tensor", None)
    return get_assembler(expr, *args, **kwargs).assemble(tensor=tensor)


def get_assembler(form, *args, **kwargs):
    """Create an assembler.

    Notes
    -----
    See `assemble` for descriptions of the parameters. ``tensor`` should not be passed to this function.

    """
    is_base_form_preprocessed = kwargs.pop('is_base_form_preprocessed', False)
    fc_params = kwargs.get('form_compiler_parameters', None)
    pyop3_compiler_parameters = kwargs.get('pyop3_compiler_parameters', None)
    if isinstance(form, ufl.form.BaseForm) and not is_base_form_preprocessed:
        mat_type = kwargs.get('mat_type', None)
        # Preprocess the DAG and restructure the DAG
        # Only pre-process `form` once beforehand to avoid pre-processing for each assembly call
        form = BaseFormAssembler.preprocess_base_form(form, mat_type=mat_type, form_compiler_parameters=fc_params)
    if isinstance(form, (ufl.form.Form, slate.TensorBase)) and not BaseFormAssembler.base_form_operands(form):
        diagonal = kwargs.pop('diagonal', False)
        if len(form.arguments()) == 0:
            return ZeroFormAssembler(form, form_compiler_parameters=fc_params, pyop3_compiler_parameters=pyop3_compiler_parameters)
        elif len(form.arguments()) == 1 or diagonal:
            return OneFormAssembler(form, *args,
                                    bcs=kwargs.get("bcs", None),
                                    form_compiler_parameters=fc_params,
                                    pyop3_compiler_parameters=pyop3_compiler_parameters,
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

    def assemble(self, tensor=None):
        """Assemble the pointwise expression.

        Parameters
        ----------
        tensor : firedrake.function.Function or firedrake.cofunction.Cofunction or matrix.MatrixBase
            Output tensor.

        Returns
        -------
        float or firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Result of evaluation: `float` for 0-forms, `firedrake.cofunction.Cofunction` or `firedrake.function.Function` for 1-forms, and `matrix.MatrixBase` for 2-forms.

        """
        from ufl.algorithms.analysis import extract_base_form_operators
        from ufl.checks import is_scalar_constant_expression

        assert tensor is None
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
                return get_assembler(ufl.FormSum((a, 1), (b, 1))).assemble()
            elif isinstance(expr, ufl.algebra.Product):
                a, b = expr.ufl_operands
                scalar = [e for e in expr.ufl_operands if is_scalar_constant_expression(e)]
                if scalar:
                    base_form = a if a is scalar else b
                    assembled_mat = assemble(base_form)
                    return get_assembler(ufl.FormSum((assembled_mat, scalar[0]))).assemble()
                a, b = [assemble(e) for e in (a, b)]
                return get_assembler(ufl.action(a, b)).assemble()
        # -- Linear combination of Functions and 1-form BaseFormOperators -- #
        # Example: a * u1 + b * u2 + c * N(u1; v*) + d * N(u2; v*)
        # with u1, u2 Functions, N a BaseFormOperator, and a, b, c, d scalars or 0-form BaseFormOperators.
        else:
            base_form_operators = extract_base_form_operators(expr)
            assembled_bfops = [firedrake.assemble(e) for e in base_form_operators]
            # Substitute base form operators with their output before examining the expression
            # which avoids conflict when determining function space, for example:
            # extract_coefficients(Interpolate(u, V2)) with u \in V1 will result in an output function space V1
            # instead of V2.
            if base_form_operators:
                expr = ufl.replace(expr, dict(zip(base_form_operators, assembled_bfops)))
            try:
                coefficients = ufl.algorithms.extract_coefficients(expr)
                V, = set(c.function_space() for c in coefficients) - {None}
            except ValueError:
                raise ValueError("Cannot deduce correct target space from pointwise expression")
            return firedrake.Function(V).assign(expr)


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
    def __init__(self, form, bcs=None, form_compiler_parameters=None, pyop3_compiler_parameters=None):
        self._form = form
        self._bcs = solving._extract_bcs(bcs)
        if any(isinstance(bc, EquationBC) for bc in self._bcs):
            raise TypeError("EquationBC objects not expected here. "
                            "Preprocess by extracting the appropriate form with bc.extract_form('Jp') or bc.extract_form('J')")
        self._form_compiler_params = form_compiler_parameters or {}
        self._pyop3_compiler_parameters = pyop3_compiler_parameters

    @abc.abstractmethod
    def allocate(self):
        """Allocate memory for the output tensor."""

    @abc.abstractmethod
    def assemble(self, tensor=None):
        """Assemble the form.

        Parameters
        ----------
        tensor : firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Output tensor to contain the result of assembly; if `None`, a tensor of appropriate type is created.

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
                 pyop3_compiler_parameters=None,
                 mat_type=None,
                 sub_mat_type=None,
                 options_prefix=None,
                 appctx=None,
                 zero_bc_nodes=True,
                 diagonal=False,
                 weight=1.0,
                 allocation_integral_types=None):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters, pyop3_compiler_parameters=pyop3_compiler_parameters)
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
            if self._mat_type == "matfree":
                return MatrixFreeAssembler(self._form, bcs=self._bcs, form_compiler_parameters=self._form_compiler_params,
                                           options_prefix=self._options_prefix,
                                           appctx=self._appctx).allocate()
            else:
                raise NotImplementedError("Reuse ExplicitMatrixAssembler bits")
                # TODO: I'm a bit confused why this allocation needs to exist here
                topology = self._form.ufl_domain().topology

                # TODO handle different mat types
                if self._mat_type != "aij":
                    raise NotImplementedError

                def adjacency(pt):
                    return topology.closure(topology.star(pt))

                test_arg, trial_arg = self._form.arguments()
                mymat = op3.PetscMat(
                    topology.points,
                    adjacency,
                    test_arg.function_space().axes,
                    trial_arg.function_space().axes,
                )

                return matrix.Matrix(self._form, self._bcs, self._mat_type, mymat, options_prefix=self._options_prefix)
                # sparsity = ExplicitMatrixAssembler._make_sparsity(test, trial, self._mat_type, self._sub_mat_type, self.maps_and_regions)
                # return matrix.Matrix(self._form, self._bcs, self._mat_type, sparsity, ScalarType, options_prefix=self._options_prefix)
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

    def assemble(self, tensor=None):
        """Assemble the form.

        Parameters
        ----------
        tensor : firedrake.cofunction.Cofunction or firedrake.function.Function or matrix.MatrixBase
            Output tensor to contain the result of assembly.

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
                bc.zero(result)

        if tensor:
            BaseFormAssembler.update_tensor(result, tensor)
            return tensor
        else:
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
                assembler = ZeroFormAssembler(form, form_compiler_parameters=self._form_compiler_params, pyop3_compiler_parameters=self._pyop3_compiler_parameters)
            elif rank == 1 or (rank == 2 and self._diagonal):
                assembler = OneFormAssembler(form, form_compiler_parameters=self._form_compiler_params,
                                             pyop3_compiler_parameters=self._pyop3_compiler_parameters,
                                             zero_bc_nodes=self._zero_bc_nodes, diagonal=self._diagonal, weight=self._weight)
            elif rank == 2:
                assembler = TwoFormAssembler(form, bcs=self._bcs, form_compiler_parameters=self._form_compiler_params, pyop3_compiler_parameters=self._pyop3_compiler_parameters,
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
                                          appctx=self._appctx,
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
                    res = firedrake.Function(row.function_space().dual())
                    with rhs.dat.vec_ro as v_vec:
                        with res.dat.vec as res_vec:
                            petsc_mat.mult(v_vec, res_vec)
                    return res
                elif isinstance(rhs, matrix.MatrixBase):
                    petsc_mat = lhs.petscmat
                    (row, col) = lhs.arguments()
                    res = petsc_mat.matMult(rhs.petscmat)
                    return matrix.AssembledMatrix(expr, self._bcs, res,
                                                  appctx=self._appctx,
                                                  options_prefix=self._options_prefix)
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
            if all(isinstance(op, numbers.Complex) for op in args):
                return sum(weight * arg for weight, arg in zip(expr.weights(), args))
            elif all(isinstance(op, firedrake.Cofunction) for op in args):
                V, = set(a.function_space() for a in args)
                result = firedrake.Cofunction(V)
                result.dat.maxpy(expr.weights(), [a.dat for a in args])
                return result
            elif all(isinstance(op, ufl.Matrix) for op in args):
                res = tensor.petscmat if tensor else PETSc.Mat()
                is_set = False
                for (op, w) in zip(args, expr.weights()):
                    # Make a copy to avoid in-place scaling
                    petsc_mat = op.petscmat.copy()
                    petsc_mat.scale(w)
                    if is_set:
                        # Modify output tensor in-place
                        res += petsc_mat
                    else:
                        # Copy to output tensor
                        petsc_mat.copy(result=res)
                        is_set = True
                return matrix.AssembledMatrix(expr, self._bcs, res,
                                              appctx=self._appctx,
                                              options_prefix=self._options_prefix)
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
            return expr.assemble(assembly_opts=opts)
        elif isinstance(expr, ufl.Interpolate):
            # Replace assembled children
            _, expression = expr.argument_slots()
            v, *assembled_expression = args
            if assembled_expression:
                # Occur in situations such as Interpolate composition
                expression = assembled_expression[0]
            expr = expr._ufl_expr_reconstruct_(expression, v)

            # Different assembly procedures:
            # 1) Interpolate(Argument(V1, 1), Argument(V2.dual(), 0)) -> Jacobian (Interpolate matrix)
            # 2) Interpolate(Coefficient(...), Argument(V2.dual(), 0)) -> Operator (or Jacobian action)
            # 3) Interpolate(Argument(V1, 0), Argument(V2.dual(), 1)) -> Jacobian adjoint
            # 4) Interpolate(Argument(V1, 0), Cofunction(...)) -> Action of the Jacobian adjoint
            # This can be generalized to the case where the first slot is an arbitray expression.
            rank = len(expr.arguments())
            # If argument numbers have been swapped => Adjoint.
            arg_expression = ufl.algorithms.extract_arguments(expression)
            is_adjoint = (arg_expression and arg_expression[0].number() == 0)
            # Workaround: Renumber argument when needed since Interpolator assumes it takes a zero-numbered argument.
            if not is_adjoint and rank != 1:
                _, v1 = expr.arguments()
                expression = ufl.replace(expression, {v1: firedrake.Argument(v1.function_space(), number=0, part=v1.part())})
            # Get the interpolator
            interp_data = expr.interp_data
            default_missing_val = interp_data.pop('default_missing_val', None)
            interpolator = firedrake.Interpolator(expression, expr.function_space(), **interp_data)
            # Assembly
            if rank == 1:
                # Assembling the action of the Jacobian adjoint.
                if is_adjoint:
                    output = tensor or firedrake.Cofunction(arg_expression[0].function_space().dual())
                    return interpolator._interpolate(v, output=output, adjoint=True, default_missing_val=default_missing_val)
                # Assembling the Jacobian action.
                if interpolator.nargs:
                    return interpolator._interpolate(expression, output=tensor, default_missing_val=default_missing_val)
                # Assembling the operator
                if tensor is None:
                    return interpolator._interpolate(default_missing_val=default_missing_val)
                return firedrake.Interpolator(expression, tensor, **interp_data)._interpolate(default_missing_val=default_missing_val)
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
                return matrix.AssembledMatrix(expr.arguments(), self._bcs, res,
                                              appctx=self._appctx,
                                              options_prefix=self._options_prefix)
            else:
                # The case rank == 0 is handled via the DAG restructuring
                raise ValueError("Incompatible number of arguments.")
        elif isinstance(expr, (ufl.Cofunction, ufl.Coargument, ufl.Argument, ufl.Matrix, ufl.ZeroBaseForm)):
            return expr
        elif isinstance(expr, ufl.Coefficient):
            return expr
        else:
            raise TypeError(f"Unrecognised BaseForm instance: {expr}")

    @staticmethod
    def update_tensor(assembled_base_form, tensor):
        if isinstance(tensor, (firedrake.Function, firedrake.Cofunction)):
            if isinstance(assembled_base_form, ufl.ZeroBaseForm):
                tensor.dat.zero()
            else:
                assembled_base_form.dat.copy(tensor.dat)
        elif isinstance(tensor, matrix.MatrixBase):
            if isinstance(assembled_base_form, ufl.ZeroBaseForm):
                tensor.petscmat.zeroEntries()
            else:
                assembled_base_form.petscmat.copy(tensor.petscmat)
        else:
            raise NotImplementedError("Cannot update tensor of type %s" % type(tensor))

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
                new_args = [firedrake.Argument(a.function_space(), number=a.number()-1, part=a.part()) for a in other_args]
                replace_map.update(dict(zip(other_args, new_args)))
                # Replace arguments
                return ufl.replace(right, replace_map)

        # -- Case (4) -- #
        if isinstance(expr, ufl.Adjoint) and isinstance(expr.form(), ufl.core.base_form_operator.BaseFormOperator):
            B = expr.form()
            u, v = B.arguments()
            # Let V1 and V2 be primal spaces, B: V1 -> V2 and B*: V2* -> V1*:
            # Adjoint(B(Argument(V1, 1), Argument(V2.dual(), 0))) = B(Argument(V1, 0), Argument(V2.dual(), 1))
            reordered_arguments = (firedrake.Argument(u.function_space(), number=v.number(), part=v.part()),
                                   firedrake.Argument(v.function_space(), number=u.number(), part=u.part()))
            # Replace arguments in argument slots
            return ufl.replace(B, dict(zip((u, v), reordered_arguments)))

        # -- Case (5) -- #
        if isinstance(expr, ufl.core.base_form_operator.BaseFormOperator) and not expr.arguments():
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

    def __init__(self, form, bcs=None, form_compiler_parameters=None, pyop3_compiler_parameters=None):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters, pyop3_compiler_parameters=pyop3_compiler_parameters)
        # Ensure mesh is 'initialised' as we could have got here without building a
        # function space (e.g. if integrating a constant).
        for mesh in form.ufl_domains():
            mesh.init()
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
    # NOTE: I think it would be nice to pass the tensor in here as we need it for codegen. But
    # that is difficult to achieve.
    def __init__(self, form, bcs=None, form_compiler_parameters=None, needs_zeroing=True, pyop3_compiler_parameters=None):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters)
        self._needs_zeroing = needs_zeroing
        self._pyop3_compiler_parameters = pyop3_compiler_parameters or {}

    # NOTE: We could have an 'already zeroed' kwarg here to avoid zeroing newly allocated tensors
    def assemble(self, tensor=None):
        """Assemble the form.

        Parameters
        ----------
        tensor : firedrake.cofunction.Cofunction or matrix.MatrixBase
            Output tensor to contain the result of assembly; if `None`, a tensor of appropriate type is created.

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

        pyop3_compiler_parameters = {"optimize": True}
        pyop3_compiler_parameters.update(self._pyop3_compiler_parameters)

        if tensor is None:
            tensor = self.allocate()
        else:
            self._check_tensor(tensor)
            if self._needs_zeroing:
                self._as_pyop3_type(tensor).zero(eager=True)

        for (local_kernel, _), (parloop, lgmaps) in zip(self.local_kernels, self.parloops(tensor)):
            subtensor = self._as_pyop3_type(tensor, local_kernel.indices)

            if isinstance(self, ExplicitMatrixAssembler):
                with _modified_lgmaps(subtensor, lgmaps) as tensor_mod:
                    parloop(**{self._tensor_name: tensor_mod}, compiler_parameters=pyop3_compiler_parameters)
            else:
                parloop(**{self._tensor_name: subtensor}, compiler_parameters=pyop3_compiler_parameters)

        for bc in self._bcs:
            self._apply_bc(tensor, bc)

        return self.result(tensor)

    @abc.abstractmethod
    def _apply_bc(self, tensor, bc):
        """Apply boundary condition."""

    @abc.abstractmethod
    def _check_tensor(self, tensor):
        """Check input tensor."""

    @staticmethod
    @abc.abstractmethod
    def _as_pyop3_type(tensor, indices=None):
        """Cast a Firedrake tensor into a PyOP2 data structure, optionally indexing it."""

    def parloops(self, tensor):
        if hasattr(self, "_parloops"):
            assert hasattr(self, "_tensor_name")
        else:
            parloops_ = []
            for local_kernel, subdomain_id in self.local_kernels:
                parloop_builder = ParloopBuilder(
                    self._form,
                    tensor,
                    self._bcs,
                    local_kernel,
                    subdomain_id,
                    self.all_integer_subdomain_ids[local_kernel.indices],
                    diagonal=self.diagonal,
                )
                parloops_.append((parloop_builder.build(), parloop_builder.collect_lgmaps(tensor)))
            self._parloops = parloops_
            self._tensor_name = self._as_pyop3_type(tensor).name
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

    @staticmethod
    def _as_pyop3_type(tensor):
        if isinstance(tensor, op3.Dat):
            return tensor
        elif isinstance(tensor, firedrake.Cofunction):
            return tensor.dat
        elif isinstance(tensor, matrix.Matrix):
            return tensor.M
        else:
            raise AssertionError


class ZeroFormAssembler(ParloopFormAssembler):
    """Class for assembling a 0-form.

    Parameters
    ----------
    form : ufl.Form or slate.TensorBasehe
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
    def __init__(self, form, form_compiler_parameters=None, pyop3_compiler_parameters=None):
        super().__init__(form, bcs=None, form_compiler_parameters=form_compiler_parameters, pyop3_compiler_parameters=None)

    def allocate(self):
        # Getting the comm attribute of a form isn't straightforward
        # form.ufl_domains()[0]._comm seems the most robust method
        # revisit in a refactor
        comm = self._form.ufl_domains()[0]._comm

        # TODO this is more convoluted than strictly needed, add a factory method?
        sf = op3.sf.single_star(comm)
        axis = op3.Axis([op3.AxisComponent(1, sf=sf)])
        return op3.Dat(
            axis,
            data=numpy.asarray([0.0], dtype=utils.ScalarType),
        )

    def _apply_bc(self, tensor, bc):
        pass

    def _check_tensor(self, tensor):
        pass

    @staticmethod
    def _as_pyop3_type(tensor, indices=None):
        assert not indices
        return tensor

    def result(self, tensor):
        # NOTE: If we could return the tensor here then that would avoid a
        # halo exchange. That would be a very significant API change though.
        tensor.assemble(update_leaves=True)
        return op3.utils.just_one(tensor.buffer._data)


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
    def _cache_key(cls, form, bcs=None, form_compiler_parameters=None, pyop3_compiler_parameters=None, needs_zeroing=True,
                   zero_bc_nodes=True, diagonal=False, weight=1.0):
        bcs = solving._extract_bcs(bcs)
        return tuple(bcs), tuplify(form_compiler_parameters), tuplify(pyop3_compiler_parameters), needs_zeroing, zero_bc_nodes, diagonal, weight

    @FormAssembler._skip_if_initialised
    def __init__(self, form, bcs=None, form_compiler_parameters=None, pyop3_compiler_parameters=None, needs_zeroing=True,
                 zero_bc_nodes=True, diagonal=False, weight=1.0):
        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters, pyop3_compiler_parameters=pyop3_compiler_parameters, needs_zeroing=needs_zeroing)
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
            return firedrake.Cofunction(test.function_space().dual())
        elif rank == 2 and self._diagonal:
            test, _ = self._form.arguments()
            return firedrake.Cofunction(test.function_space().dual())
        else:
            raise RuntimeError(f"Not expected: found rank = {rank} and diagonal = {self._diagonal}")

    def _apply_bc(self, tensor, bc):
        # TODO Maybe this could be a singledispatchmethod?
        if isinstance(bc, DirichletBC):
            self._apply_dirichlet_bc(tensor, bc)
        elif isinstance(bc, EquationBCSplit):
            bc.zero(tensor)
            type(self)(bc.f, bcs=bc.bcs, form_compiler_parameters=self._form_compiler_params, needs_zeroing=False,
                       zero_bc_nodes=self._zero_bc_nodes, diagonal=self._diagonal, weight=self._weight).assemble(tensor=tensor)
        else:
            raise AssertionError

    def _apply_dirichlet_bc(self, tensor, bc):
        if self._diagonal:
            bc.set(tensor, self._weight)
        elif not self._zero_bc_nodes:
            # NOTE this only works if tensor is a Function and not a Cofunction
            bc.apply(tensor)
        else:
            bc.zero(tensor)

    def _check_tensor(self, tensor):
        if tensor.function_space() != self._form.arguments()[0].function_space().dual():
            raise ValueError("Form's argument does not match provided result tensor")

    @staticmethod
    def _as_pyop3_type(tensor, indices=None):
        if indices is not None and any(index is not None for index in indices):
            i, = indices
            return tensor.dat[i]
        else:
            return tensor.dat

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
        # Arguably we should crash here, as we would be passing ignored arguments through
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
    has_real_subspace = any(
        _is_real_space(V) for arg in arguments for V in arg.function_space()
    )

    if mat_type is None:
        if has_real_subspace:
            mat_type = "nest"
        else:
            mat_type = parameters.parameters["default_matrix_type"]

    if sub_mat_type is None:
        sub_mat_type = parameters.parameters["default_sub_matrix_type"]

    if has_real_subspace and mat_type not in ["nest", "matfree"]:
        raise ValueError("Matrices containing real space arguments must have type 'nest' or 'matfree'")
    if sub_mat_type not in {"aij", "baij"}:
        raise ValueError(
            f"Invalid submatrix type, '{sub_mat_type}' (not 'aij' or 'baij')"
        )

    if mat_type == "nest":
        mat_type = {}
        test, trial = arguments
        for test_subspace in test.function_space():
            for trial_subspace in trial.function_space():
                subspace_key = test_subspace.index, trial_subspace.index
                if any(_is_real_space(s) for s in {test_subspace, trial_subspace}):
                    mat_type[subspace_key] = "dat"
                else:
                    mat_type[subspace_key] = sub_mat_type

    # sub_mat_type no longer used after this
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
        # The previous API was that the user would specify mat_type and sub_mat_type, now
        # mat_type can be a dict so convert to that here.
        # NOTE: This function is called in TwoFormAssembler, should it be?
        # mat_type, sub_mat_type = _get_mat_type(mat_type, sub_mat_type, form.arguments())

        super().__init__(form, bcs=bcs, form_compiler_parameters=form_compiler_parameters, needs_zeroing=needs_zeroing)
        self._mat_type = mat_type
        self._options_prefix = options_prefix
        self._appctx = appctx
        self.weight = weight
        self._allocation_integral_types = allocation_integral_types

    def allocate(self):
        test, trial = self._form.arguments()
        sparsity = self._make_sparsity(
            test,
            trial,
            self._mat_type,
            self._make_maps_and_regions(),
        )
        mat = op3.Mat.from_sparsity(sparsity)
        return matrix.Matrix(
            self._form,
            self._bcs,
            # shouldn't be needed!
            self._mat_type,
            mat,
            options_prefix=self._options_prefix
        )

    @staticmethod
    def _make_sparsity(test, trial, mat_type, maps_and_regions):
        # Is this overly restrictive?
        if any(len(a.function_space()) > 1 for a in [test, trial]) and mat_type == "baij":
            raise ValueError("BAIJ matrix type makes no sense for mixed spaces, use 'aij'")

        sparsity = op3.Sparsity(
            test.function_space().axes,
            trial.function_space().axes,
            mat_type=mat_type,
            block_shape=test.function_space().value_size
        )

        if type(test.function_space().ufl_element()) is finat.ufl.MixedElement:
            n = len(test.function_space())
            if n == 1:
                # if vector function space revert to standard case
                diag_blocks = [(Ellipsis, Ellipsis)]
            else:
                raise NotImplementedError("Loop contexts are confusing")
                # otherwise treat each block separately
                diag_blocks = [(i, i) for i in range(n)]
        else:
            diag_blocks = [(Ellipsis, Ellipsis)]

        for rindex, cindex in diag_blocks:
            op3.do_loop(
                p := test.ufl_domain().points.index(),
                sparsity[rindex, cindex][p, p].assign(666, eager=False)
            )

        # Pretend that we are doing assembly by looping over the right
        # iteration sets and using the right maps.
        for iter_index, rmap, cmap, indices in maps_and_regions:
            rindex, cindex = indices
            if rindex is None:
                rindex = Ellipsis
            if cindex is None:
                cindex = Ellipsis

            op3.do_loop(
                iter_index,
                sparsity[rindex, cindex][rmap, cmap].assign(666, eager=False)
            )

        sparsity.assemble()
        return sparsity

    def _make_maps_and_regions(self):
        test, trial = self._form.arguments()

        if self._allocation_integral_types is not None:
            return ExplicitMatrixAssembler._make_maps_and_regions_default(
                test, trial, self._allocation_integral_types
            )

        elif op3.utils.strictly_all(
            local_kernel.indices == (None, None)
            for assembler in self._all_assemblers
            for local_kernel, _ in assembler.local_kernels
        ):
            # Handle special cases: slate or split=False
            allocation_integral_types = {
                local_kernel.kinfo.integral_type
                for assembler in self._all_assemblers
                for local_kernel, _ in assembler.local_kernels
            }
            return ExplicitMatrixAssembler._make_maps_and_regions_default(
                test, trial, allocation_integral_types
            )

        else:
            loops = []
            for assembler in self._all_assemblers:
                all_meshes = assembler._form.ufl_domains()
                for local_kernel, subdomain_id in self._all_local_kernels:
                    integral_type = local_kernel.kinfo.integral_type

                    mesh = all_meshes[local_kernel.kinfo.domain_number]  # integration domain
                    integral_type = local_kernel.kinfo.integral_type

                    Vrow = test.function_space()
                    Vcol = trial.function_space()

                    rindex, cindex = local_kernel.indices
                    if rindex is not None:
                        Vrow = Vrow[rindex]
                    if cindex is not None:
                        Vcol = Vcol[cindex]

                    if integral_type == "cell":
                        iterset = mesh.cells.owned
                        index = iterset.index()
                        rmap = _cell_integral_pack_indices(Vrow, index)
                        cmap = _cell_integral_pack_indices(Vcol, index)
                    elif integral_type in {"exterior_facet", "interior_facet"}:
                        if integral_type == "exterior_facet":
                            iterset = plex.exterior_facets.set
                        else:
                            assert integral_type == "interior_facet"
                            iterset = plex.interior_facets.set

                        index = iterset.index()
                        rmap = _facet_integral_pack_indices(Vrow, index)
                        cmap = _facet_integral_pack_indices(Vcol, index)

                    # These maps are technically context-sensitive since they depend
                    # on the loop index, but it is always trivial. However, we still
                    # need to turn our index tree into an index forest for things to
                    # make sense later on.
                    context = pmap({index.id: (index.source_path, index.path)})
                    rmap = {context: rmap}
                    cmap = {context: cmap}

                    loop = (index, rmap, cmap, local_kernel.indices)
                    loops.append(loop)
            return tuple(loops)

    @staticmethod
    def _make_maps_and_regions_default(test, trial, allocation_integral_types):
        assert allocation_integral_types is not None

        mesh = op3.utils.single_valued(a.ufl_domain() for a in {test, trial})
        plex = mesh.topology
        Vrow = test.function_space()
        Vcol = trial.function_space()

        # NOTE: We do not inspect subdomains here so the "full" sparsity is
        # allocated even when we might not use all of it. This increases
        # reusability.
        loops = []
        for integral_type in allocation_integral_types:
            if integral_type == "cell":
                iterset = plex.cells.owned
                index = iterset.index()
                rmap = _cell_integral_pack_indices(Vrow, index)
                cmap = _cell_integral_pack_indices(Vcol, index)
            elif integral_type in {"exterior_facet", "interior_facet"}:
                if integral_type == "exterior_facet":
                    iterset = plex.exterior_facets.set
                else:
                    assert integral_type == "interior_facet"
                    iterset = plex.interior_facets.set

                index = iterset.index()
                rmap = _facet_integral_pack_indices(Vrow, index)
                cmap = _facet_integral_pack_indices(Vcol, index)

            # These maps are technically context-sensitive since they depend
            # on the loop index, but it is always trivial. However, we still
            # need to turn our index tree into an index forest for things to
            # make sense later on.
            # context = pmap({index.id: (index.source_path, index.path)})
            # rmap = {context: rmap}
            # cmap = {context: cmap}

            loop = (index, rmap, cmap, (None, None))
            loops.append(loop)
        return tuple(loops)

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

    def _apply_bc(self, tensor, bc):
        mat = tensor.M
        spaces = tuple(a.function_space() for a in tensor.a.arguments())
        V = bc.function_space()
        component = V.component
        if component is not None:
            V = V.parent
        index = Ellipsis if V.index is None else V.index
        space = V if V.parent is None else V.parent
        if isinstance(bc, DirichletBC):
            # if fs.topological != self.topological:
            #     raise RuntimeError("Dirichlet BC defined on a different function space")
            if space.topological != spaces[0].topological:
                raise TypeError("bc space does not match the test function space")
            elif space.topological != spaces[1].topological:
                raise TypeError("bc space does not match the trial function space")

            # for some reason I need to do this first, is this still the case?
            mat.assemble()

            p = space.axes[index].root[bc.constrained_points].index()

            # If we constrain points of different dimension (e.g. vertices
            # and edges) then the assigned identity may have different sizes.
            # To resolve this we loop over each dimension in turn.
            for context, index_tree in op3.as_index_forest(p).items():
                dof_slice = op3.Slice("dof", [op3.AffineSliceComponent("XXX")])
                index_tree = index_tree.add_node(dof_slice, *index_tree.leaf)

                if component is not None:
                    component_slice = op3.ScalarIndex("dim0", "XXX", component)
                    index_tree = index_tree.add_node(component_slice, *index_tree.leaf)

                assignee = mat[index, index][index_tree, index_tree]

                if assignee.axes.size == 0:
                    continue

                # If setting a block then use an identity matrix
                if assignee.axes.size > 1:
                    # "materialize"
                    axes = op3.AxisTree(assignee.axes.node_map)
                    size = op3.utils.single_valued([
                        ax.size for ax in {assignee.raxes, assignee.caxes}
                    ])
                    expression = op3.Dat(
                        axes, data=numpy.eye(size, dtype=utils.ScalarType).flatten(), constant=True
                    )
                else:
                    expression = self.weight

                op3.do_loop(
                    p.with_context(context), assignee.assign(expression, eager=False)
                )

            # Handle off-diagonal block involving real function space.
            # "lgmaps" is correctly constructed in _matrix_arg, but
            # is ignored by PyOP2 in this case.
            # Walk through row blocks associated with index.
            for j, s in enumerate(space):
                if j != index and _is_real_space(s):
                    self._apply_bcs_mat_real_block(mat, index, j, component, bc.node_set)
            # Walk through col blocks associated with index.
            for i, s in enumerate(space):
                if i != index and _is_real_space(s):
                    self._apply_bcs_mat_real_block(mat, i, index, component, bc.node_set)
        elif isinstance(bc, EquationBCSplit):
            for j, s in enumerate(spaces[1]):
                if _is_real_space(s):
                    self._apply_bcs_mat_real_block(mat, index, j, component, bc.node_set)
            type(self)(bc.f, bcs=bc.bcs, form_compiler_parameters=self._form_compiler_params, needs_zeroing=False).assemble(tensor=tensor)
        else:
            raise AssertionError

    @staticmethod
    def _apply_bcs_mat_real_block(op2tensor, i, j, component, node_set):
        raise NotImplementedError
        dat = op2tensor[i, j].handle.getPythonContext().dat
        if component is not None:
            dat = op2.DatView(dat, component)
        dat.zero(subset=node_set, eager=True)

    def _check_tensor(self, tensor):
        if tensor.a.arguments() != self._form.arguments():
            raise ValueError("Form's arguments do not match provided result tensor")

    @staticmethod
    def _as_pyop3_type(tensor, indices=None):
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

    def assemble(self, tensor=None):
        if tensor is None:
            tensor = self.allocate()
        else:
            self._check_tensor(tensor)
        tensor.assemble()
        return tensor

    def _check_tensor(self, tensor):
        if tensor.a.arguments() != self._form.arguments():
            raise ValueError("Form's arguments do not match provided result tensor")


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
    def __init__(self, form, tensor, bcs, local_knl, subdomain_id,
                 all_integer_subdomain_ids, diagonal):
        self._form = form
        self._tensor = tensor
        self._local_knl = local_knl
        self._subdomain_id = subdomain_id
        self._all_integer_subdomain_ids = all_integer_subdomain_ids
        self._diagonal = diagonal
        self._bcs = bcs

        self._active_coefficients = _FormHandler.iter_active_coefficients(form, local_knl.kinfo)
        self._constants = _FormHandler.iter_constants(form, local_knl.kinfo)

    def build(self) -> op3.Loop:
        """Construct the parloop."""
        p = self._iterset.index()
        args = []
        for tsfc_arg in self._kinfo.arguments:
            arg = self._as_parloop_arg(tsfc_arg, p)
            args.append(arg)
        
        transform_kernels = self.fuse_orientations()
        print(transform_kernels)
        orientation = op3.Dat(op3.AxisTree(),data=numpy.zeros(1))
        axes = op3.AxisTree.from_nest({op3.Axis(10): op3.Axis(10)})
        a = op3.Dat(axes, name="a", data=numpy.zeros(axes.size), dtype=op3.ScalarType)
        b = op3.Dat(axes, name="b", data=numpy.zeros(axes.size), dtype=op3.ScalarType)
        res = op3.Dat(axes, name="res", data=numpy.zeros(axes.size), dtype=op3.ScalarType)
        # need to think about the b argument and how to get the orientation argument
        loop2 = op3.loop(p, transform_kernels[0](orientation, b[p], a, res))
        self._kinfo.kernel.code = self._kinfo.kernel.code.with_entrypoints({self._kinfo.kernel.name})
        kernel = op3.Function(
            self._kinfo.kernel.code, [op3.INC] + [op3.READ for _ in args[1:]]
        )
        loop = op3.loop(p, kernel(*args))
        # print(loop.loopy_code.ir)
        breakpoint()
        return loop

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
        return (self._local_knl.indices,)

        # think below is not needed anymore
        # assert len(self._form.arguments()) == 2 and not self._diagonal
        # if all(i is None for i in self._local_knl.indices):
        #     test, trial = self._form.arguments()
        #     return numpy.ndindex((len(test.function_space()),
        #                           len(trial.function_space())))
        # else:
        #     assert all(i is not None for i in self._local_knl.indices)
        #     return self._local_knl.indices,

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
                if i is None:
                    i = 0
                if j is None:
                    j = 0

                for bc in itertools.chain(*self._filter_bcs(i, j)):
                    if bc.function_space().component is not None:
                        return True
        return False

    def collect_lgmaps(self, matrix):
        """Return any local-to-global maps that need to be swapped out.

        This is only needed when applying boundary conditions to 2-forms.

        """

        if len(self._form.arguments()) == 2 and not self._diagonal:
            if matrix.M.nested:
                raise NotImplementedError("Mat nest not yet implemented")
            if not self._bcs:
                return None
            lgmaps = []
            for i, j in self.get_indicess():
                ibc = 0 if i is None else i
                jbc = 0 if j is None else j
                row_bcs, col_bcs = self._filter_bcs(ibc, jbc)

                i = Ellipsis if i is None else i
                j = Ellipsis if j is None else j

                if matrix.M.mat_type == "aij":
                    row_bsize = 1
                    col_bsize = 1
                else:
                    assert matrix.M.mat_type == "baij"
                    row_bsize = self.test_function_space[ibc].value_size
                    col_bsize = self.trial_function_space[jbc].value_size

                submat = matrix.M[i, j]
                rlgmap = self.test_function_space[ibc].mask_lgmap(row_bcs, submat.raxes, submat.row_lgmap_dat, row_bsize)
                clgmap = self.trial_function_space[jbc].mask_lgmap(col_bcs, submat.caxes, submat.column_lgmap_dat, col_bsize)
                lgmaps.append((i, j, rlgmap, clgmap))

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

    @property
    def _topology(self):
        return self._mesh.topology

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
            return self._topology.measure_set(
                self._integral_type,
                self._subdomain_id,
                self._all_integer_subdomain_ids
            )
    

    def fuse_orientations(self):
        Vs = self._indexed_function_spaces
        transform_kernels = []
        for fs in Vs:
            if hasattr(fs.ufl_element(), "triple"):
                os = fs.ufl_element().triple.matrices
                t_dim = fs.ufl_element().cell._tdim
                os = os[t_dim][0]
                n = os[next(iter(os.keys()))].shape[0]
                child_knl = lp.make_function(
                        f"{{[i, j]:0<=i, j < {n}}}",
                        """
                        res[j] =  res[j] + a[i, j]*b[i]
                        """, name="matmul",target=lp.CWithGNULibcTarget())
                args = [lp.GlobalArg("o", dtype=utils.IntType, shape=(1,)),
                        lp.GlobalArg("b", dtype=ScalarType, shape=(n, )),
                        lp.GlobalArg("res", dtype=ScalarType, shape =(n,)),
                        lp.GlobalArg("a", dtype=ScalarType, shape =(n, n))]

                var_list = ["o"]
                string = [f"\nswitch (o) {{ \n"]
                for val in sorted(os.keys()):
                    string += f"case {val}:\n a = mat{val};break;\n"
                    var_list += [f"mat{val}"]
                    mat = numpy.array(os[val], dtype=ScalarType)
                    args += [lp.TemporaryVariable(f"mat{val}", initializer=mat, dtype=ScalarType, read_only=True, address_space=lp.AddressSpace(1))]
                string += "default:\nbreak;\n }"
                transform_insn = lp.CInstruction(tuple(), "".join(string), assignees=("a"), read_variables=frozenset(var_list), id="assign")

                parent_knl = lp.make_kernel(
                        "{:}",
                        [transform_insn, "res[:] = matmul(a, b, res) {dep=assign}"],
                        kernel_data=args
                        ,target=lp.CWithGNULibcTarget())
                knl = lp.merge([parent_knl, child_knl])
                # print(lp.generate_code_v2(knl).device_code())
                # print(knl)
                transform = op3.Function(knl, [op3.READ, op3.READ, op3.WRITE, op3.WRITE])
                transform_kernels += [transform]
            else:
                transform_kernels = []
                # raise NotImplementedError("Dense orientations only needed for FUSE elements")
        return transform_kernels

    @functools.singledispatchmethod
    def _as_parloop_arg(self, tsfc_arg, index):
        """Return a :class:`op2.ParloopArg` corresponding to the provided
        :class:`tsfc.KernelArg`.
        """
        raise TypeError(f"No handler provided for {type(tsfc_arg).__name__}")

    @_as_parloop_arg.register(kernel_args.OutputKernelArg)
    def _as_parloop_arg_output(self, _, index):
        rank = len(self._form.arguments())
        tensor = self._tensor
        Vs = self._indexed_function_spaces
        if rank == 0:
            return tensor
        elif rank == 1 or rank == 2 and self._diagonal:
            V, = Vs

            return pack_pyop3_tensor(tensor.dat, V, index, self._integral_type)
        elif rank == 2:
            if all(_is_real_space(V) for V in Vs):
                just_one = op3.utils.just_one
                ridx, cidx = map(just_one, just_one(tensor.nest_labels))
                if ridx is None:
                    ridx = 0
                if cidx is None:
                    cidx = 0

                submat = tensor.mat.getNestSubMatrix(ridx, cidx)
                return submat.getPythonContext().dat
            elif any(_is_real_space(V) for V in Vs):
                just_one = op3.utils.just_one
                ridx, cidx = map(just_one, just_one(tensor.nest_labels))
                if ridx is None:
                    ridx = 0
                if cidx is None:
                    cidx = 0

                submat = tensor.mat.getNestSubMatrix(ridx, cidx)

                V, = [V for V in Vs if V.ufl_element().family() != "Real"]
                dat = submat.getPythonContext().dat
                return pack_pyop3_tensor(dat, V, index, self._integral_type)
            else:
                return pack_pyop3_tensor(tensor.M, *Vs, index, self._integral_type)
        else:
            raise AssertionError

    @_as_parloop_arg.register(kernel_args.CoordinatesKernelArg)
    def _as_parloop_arg_coordinates(self, _, index):
        return pack_tensor(self._mesh.coordinates, index, self._integral_type)

    @_as_parloop_arg.register(kernel_args.CoefficientKernelArg)
    def _as_parloop_arg_coefficient(self, arg, index):
        coeff = next(self._active_coefficients)
        return pack_tensor(coeff, index, self._integral_type)

    @_as_parloop_arg.register(kernel_args.ConstantKernelArg)
    def _as_parloop_arg_constant(self, arg, index):
        const = next(self._constants)
        return const.dat

    @_as_parloop_arg.register(kernel_args.CellOrientationsKernelArg)
    def _as_parloop_arg_cell_orientations(self, _, index):
        return pack_tensor(self._mesh.cell_orientations(), index, self._integral_type)

    @_as_parloop_arg.register(kernel_args.CellSizesKernelArg)
    def _as_parloop_arg_cell_sizes(self, _, index):
        raise NotImplementedError
        func = self._mesh.cell_sizes
        m = self._get_map(func.function_space())
        return func.dat[m(index)]

    @_as_parloop_arg.register(kernel_args.ExteriorFacetKernelArg)
    def _as_parloop_arg_exterior_facet(self, _, index):
        return self._topology.exterior_facets._local_facets[index]

    @_as_parloop_arg.register(kernel_args.InteriorFacetKernelArg)
    def _as_parloop_arg_interior_facet(self, _, index):
        return self._topology.interior_facets._local_facets[index]

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

    @_as_parloop_arg.register(kernel_args.ExteriorFacetOrientationKernelArg)
    def _as_parloop_arg_exterior_facet_orientation(_, self):
        raise NotImplementedError
        return op2.DatParloopArg(self._mesh.exterior_facets.local_facet_orientation_dat)

    @_as_parloop_arg.register(kernel_args.InteriorFacetOrientationKernelArg)
    def _as_parloop_arg_interior_facet_orientation(_, self):
        raise NotImplementedError
        return op2.DatParloopArg(self._mesh.interior_facets.local_facet_orientation_dat)


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
        spaces = []
        for index, arg in zip(indices, form.arguments()):
            space = arg.ufl_function_space()
            if index is not None:
                space = space[index]
            spaces.append(space)
        return tuple(spaces)

    @staticmethod
    def index_tensor(tensor, form, indices, diagonal):
        """Return the (indexed) pyop3 data structure tied to ``tensor``."""
        indices = tuple(i if i is not None else Ellipsis for i in indices)

        rank = len(form.arguments())
        if rank == 0:
            assert len(indices) == 0
            return tensor
        elif rank == 1 or rank == 2 and diagonal:
            index, = indices
            if index is Ellipsis:
                return tensor.dat
            else:
                return tensor.subfunctions[index].dat
        elif rank == 2:
            return tensor.M[indices]
        else:
            raise AssertionError

def fuse_orientations(self):
    Vs = self._indexed_function_spaces
    for fs in Vs:
        if hasattr(fs.ufl_element(), "triple"):
            os = fs.ufl_element().triple.matrices
            t_dim = fs.ufl_element().cell._tdim
            os = os[t_dim][0]
            n = os[next(iter(os.keys()))].shape[0]
            child_knl = lp.make_function(
                    f"{{[i, j]:0<=i, j < {n}}}",
                    """
                    res[j] =  res[j] + a[i, j]*b[i]
                    """, name="matmul",target=lp.CWithGNULibcTarget())
            args = [lp.GlobalArg("o", dtype=utils.IntType, shape=(1,)),
                    lp.GlobalArg("b", dtype=ScalarType, shape=(n, )),
                    lp.GlobalArg("res", dtype=ScalarType, shape =(n,)),
                    lp.GlobalArg("a", dtype=ScalarType, shape =(n, n))]

            var_list = ["o"]
            string = [f"\nswitch (o) {{ \n"]
            for val in sorted(os.keys()):
                string += f"case {val}:\n a = mat{val};break;\n"
                var_list += [f"mat{val}"]
                mat = numpy.array(os[val], dtype=ScalarType)
                args += [lp.TemporaryVariable(f"mat{val}", initializer=mat, dtype=ScalarType, read_only=True, address_space=lp.AddressSpace(1))]
            string += "default:\nbreak;\n }"
            transform_insn = lp.CInstruction(tuple(), "".join(string), assignees=("a"), read_variables=frozenset(var_list), id="assign")

            parent_knl = lp.make_kernel(
                    "{:}",
                    [transform_insn, "res[:] = matmul(a, b, res) {dep=assign}"],
                    kernel_data=args
                    ,target=lp.CWithGNULibcTarget())
            knl = lp.merge([parent_knl, child_knl])
            print(lp.generate_code_v2(knl).device_code())
            print(knl)
            transform = op3.Function(knl, [op3.READ, op3.READ, op3.WRITE, op3.WRITE])
            return transform
        else:
            raise NotImplementedError("Dense orientations only needed for FUSE elements")

# def construct_matrix_perm(os, fs):
#     n = os[next(iter(os.keys()))].shape[0]
#     child_knl = lp.make_function(
#             f"{{[i, j]:0<=i, j < {n}}}",
#             """
#             res[j] =  res[j] + a[i, j]*b[i]
#             """, name="matmul",target=lp.CWithGNULibcTarget())
#     args = [lp.GlobalArg("o", dtype=utils.IntType, shape=(1,)),
#             lp.GlobalArg("b", dtype=ScalarType, shape=(n, )),
#             lp.GlobalArg("res", dtype=ScalarType, shape =(n,)),
#             lp.GlobalArg("a", dtype=ScalarType, shape =(n, n))]

#     var_list = ["o"]
#     string = [f"\nswitch (o) {{ \n"]
#     for val in sorted(os.keys()):
#         string += f"case {val}:\n a = mat{val};break;\n"
#         var_list += [f"mat{val}"]
#         mat = numpy.array(os[val], dtype=ScalarType)
#         args += [lp.TemporaryVariable(f"mat{val}", initializer=mat, dtype=ScalarType, read_only=True, address_space=lp.AddressSpace(1))]
#     string += "default:\nbreak;\n }"
#     transform_insn = lp.CInstruction(tuple(), "".join(string), assignees=("a"), read_variables=frozenset(var_list), id="assign")

#     parent_knl = lp.make_kernel(
#             "{:}",
#             [transform_insn, "res[:] = matmul(a, b, res) {dep=assign}"],
#             kernel_data=args
#             ,target=lp.CWithGNULibcTarget())
#     knl = lp.merge([parent_knl, child_knl])
#     print(lp.generate_code_v2(knl).device_code())
#     print(knl)
#     transform = op3.Function(knl, [op3.READ, op3.READ, op3.WRITE, op3.WRITE])
#     return transform
    # iter_list = fs.mesh()._topology.measure_set("cell","otherwise",dict())
    # orientation = op3.Dat(op3.AxisTree(),data=numpy.zeros(1))
    # axes = op3.AxisTree.from_nest({op3.Axis(10): op3.Axis(10)})
    # a = op3.Dat(
    #     axes, name="a", data=numpy.zeros(axes.size), dtype=op3.ScalarType
    # )
    # b = op3.Dat(
    #     axes, name="a", data=numpy.zeros(axes.size), dtype=op3.ScalarType
    # )
    # res = op3.Dat(
    #     axes, name="res", data=numpy.zeros(axes.size), dtype=op3.ScalarType
    # )
    # # breakpoint()
    # # need to think about the b argument and how to get the orientation argument
    # loop = op3.loop(cell := iter_list.index(), transform(orientation, b[cell], a, res))
    # loop()
        
def construct_string(os, dofs, c):
    # for testing - makes it into a full c program
    dim = os[0].shape[0]
    string = ["#include <stdio.h> \n#include <stdlib.h>\n"]
   
    string += construct_assign(os, string)
    
    string += "int main(void){\n"
    zeros = ",".join(["0" for i in range(dim)])
    dof_vals = ",".join([str(d) for d in dofs])
    string += f"double b[{dim}] = {{ {dof_vals} }};\n"
    string += f"int i,j;int m = {dim};\ndouble res[{dim}] = {{ {zeros} }};\nassign(5, b, res);\n"
    
    string += "for(i=0;i<m;i++){\n\nprintf(\"%lf \", res[i]);\nprintf(\"\\n\");}\nreturn 0;\n"
    string += "}\n"

    print("".join(string))



def _is_real_space(space):
    return space.ufl_element().family() == "Real"


@contextlib.contextmanager
def _modified_lgmaps(mat, lgmaps):
    if lgmaps is None:
        yield mat
        return

    if mat.nested:
        raise NotImplementedError("Think about extracting the right sub-blocks")

    # TODO: Fixup API so this isn't needed
    (i, j, row_lgmap, column_lgmap), = lgmaps
    orig_rlgmap, orig_clgmap = mat.mat.getLGMap()
    mat.mat.setLGMap(row_lgmap, column_lgmap)

    yield mat

    mat.mat.setLGMap(orig_rlgmap, orig_clgmap)
