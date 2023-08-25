import abc
from collections import OrderedDict
import functools
import itertools
from itertools import product
import operator

import cachetools
import finat
import firedrake
import numpy
from pyadjoint.tape import annotate_tape
from tsfc import kernel_args
from tsfc.finatinterface import create_element
import ufl
from ufl.domain import extract_unique_domain
from firedrake import (extrusion_utils as eutils, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint_utils import annotate_assemble
from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
from firedrake.functionspaceimpl import WithGeometry, FunctionSpace, FiredrakeDualSpace
from firedrake.functionspacedata import entity_dofs_key, entity_permutations_key
from firedrake.petsc import PETSc
from firedrake.slate import slac, slate
from firedrake.slate.slac.kernel_builder import CellFacetKernelArg, LayerCountKernelArg
from firedrake.utils import ScalarType, tuplify
from pyop2 import op2
from pyop2.exceptions import MapValueError, SparsityFormatError
from pyop2.utils import cached_property


__all__ = "assemble",


_FORM_CACHE_KEY = "firedrake.assemble.FormAssembler"
"""Entry used in form cache to try and reuse assemblers where possible."""


@PETSc.Log.EventDecorator()
@annotate_assemble
def assemble(expr, *args, **kwargs):
    r"""Evaluate expr.

    :arg expr: a :class:`~ufl.classes.Form`, :class:`~ufl.classes.Expr` or
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
                       visited=None,
                       weight=1.0):

    # Prepr ocess and restructure the DAG
    if not is_base_form_preprocessed:
        # Preprocessing the form makes a new object -> current form caching mechanism
        # will populate `expr`'s cache which is now different than `expression`'s cache so we need
        # to transmit the cache. All of this only holds when `expression` if a ufl.Form
        # and therefore when `is_base_form_preprocessed` is False.
        expr = preprocess_base_form(expression, mat_type, form_compiler_parameters)
        if isinstance(expression, ufl.form.Form) and isinstance(expr, ufl.form.Form):
            expr._cache = expression._cache
        # BaseForm preprocessing can turn BaseForm into an Expr (cf. case (6) in `restructure_base_form`)
        if isinstance(expr, ufl.core.expr.Expr) and not isinstance(expr, ufl.core.base_form_operator.BaseFormOperator):
            # FIXME: Directly call assemble_expressions once sum of BaseFormOperator has been lifted
            return assemble(expr)
    else:
        expr = expression

    # DAG assembly: traverse the DAG in a post-order fashion and evaluate the node as we go.
    stack = [expr]
    visited = visited or {}
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
            t = tensor if e is expr else None
            visited[e] = base_form_assembly_visitor(e, t, bcs, diagonal,
                                                    form_compiler_parameters,
                                                    mat_type, sub_mat_type,
                                                    appctx, options_prefix,
                                                    zero_bc_nodes, weight,
                                                    *(visited[arg] for arg in operands))

    # Update tensor with the assembled result value
    # assembled_base_form = visited[expr]
    # Doesn't need to update `tensor` with `assembled_base_form`
    # for assembled 1-form (Cofunction) because both underlying
    # Dat objects are the same (they automatically update).
    # if tensor and isinstance(assembled_base_form, matrix.MatrixBase):
    #     Uses the PETSc copy method.
    #    assembled_base_form.petscmat.copy(tensor.petscmat)

    # What about cases where expanding derivatives produce a non-Form object ?
    # if isinstance(expression, ufl.form.Form) and isinstance(expr, ufl.form.Form):
    #    expression._cache = expr._cache
    # return assembled_base_form

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


def restructure_base_form(expr, visited=None):
    r"""Perform a preorder traversal to simplify and optimize the DAG.
    Example: Let's consider F(u, N(u; v*); v) with N(u; v*) an external operator.

             We have: dFdu = \frac{\partial F}{\partial u} + Action(dFdN, dNdu)
             Now taking the action on a rank-1 object w (e.g. Coefficient/Cofunction) results in:

        (1) Action(Action(dFdN, dNdu), w)

                Action                     Action
                /    \                     /     \
              Action  w     ----->       dFdN   Action
              /    \                            /    \
            dFdN    dNdu                      dNdu    w

        This situations does not only arise for ExternalOperator but also when we have a 2-form instead of dNdu!

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
            # 1) Replace the highest numbered argument of left by right when needed
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
    if isinstance(expr, ufl.FormSum) and all(isinstance(c, ufl.core.base_form_operator.BaseFormOperator) for c in expr.components()):
        # Return ufl.Sum
        return sum([c for c in expr.components()])
    return expr


def restructure_base_form_postorder(expr, visited=None):
    visited = visited or {}
    if expr in visited:
        return visited[expr]

    # Visit/update the children
    operands = base_form_operands(expr)
    operands = list(restructure_base_form_postorder(op, visited) for op in operands)
    # Need to reconstruct the DAG as we traverse it!
    expr = reconstruct_node_from_operands(expr, operands)
    # Perform the DAG rotation when needed
    visited[expr] = restructure_base_form(expr, visited)
    return visited[expr]


def restructure_base_form_preorder(expr, visited=None):
    visited = visited or {}
    if expr in visited:
        return visited[expr]

    # Perform the DAG rotation when needed
    expr = restructure_base_form(expr, visited)
    # Visit/update the children
    operands = base_form_operands(expr)
    operands = list(restructure_base_form_preorder(op, visited) for op in operands)
    # Need to reconstruct the DAG as we traverse it!
    visited[expr] = reconstruct_node_from_operands(expr, operands)
    return visited[expr]


def reconstruct_node_from_operands(expr, operands):
    if isinstance(expr, (ufl.Adjoint, ufl.Action)):
        return expr._ufl_expr_reconstruct_(*operands)
    elif isinstance(expr, ufl.FormSum):
        return ufl.FormSum(*[(op, w) for op, w in zip(operands, expr.weights())])
    return expr


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


def preprocess_form(form, fc_params):
    """Preprocess ufl.Form objects

    :arg form: a :class:`~ufl.classes.Form`
    :arg fc_params:: Dictionary of parameters to pass to the form compiler.

    :returns: The resulting preprocessed :class:`~ufl.classes.Form`.

    This function preprocess the form, mainly by expanding the derivatives, in order to determine
    if we are dealing with a :class:`~ufl.classes.Form` or another :class:`~ufl.classes.BaseForm` object.
    This function is called in :func:`base_form_assembly_visitor`. Depending on the type of the resulting tensor,
    we may call :func:`assemble_form` or traverse the sub-DAG via :func:`assemble_base_form`.
    """
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


def preprocess_base_form(expr, mat_type=None, form_compiler_parameters=None):
    if isinstance(expr, (ufl.form.Form, ufl.core.base_form_operator.BaseFormOperator)) and mat_type != "matfree":
        # For "matfree", Form evaluation is delayed
        expr = preprocess_form(expr, form_compiler_parameters)
    if not isinstance(expr, (ufl.form.Form, slate.TensorBase)):
        # => No restructuration needed for Form and slate.TensorBase
        expr = restructure_base_form_preorder(expr)
        expr = restructure_base_form_postorder(expr)
    return expr


def base_form_assembly_visitor(expr, tensor, bcs, diagonal,
                               form_compiler_parameters,
                               mat_type, sub_mat_type,
                               appctx, options_prefix,
                               zero_bc_nodes, weight, *args):
    if isinstance(expr, (ufl.form.Form, slate.TensorBase)):

        if args and mat_type != "matfree":
            # Retrieve the Form's children
            base_form_operators = base_form_operands(expr)
            # Substitute the base form operators by their output
            expr = ufl.replace(expr, dict(zip(base_form_operators, args)))

        return _assemble_form(expr, tensor=tensor, bcs=bcs,
                              diagonal=diagonal,
                              mat_type=mat_type,
                              sub_mat_type=sub_mat_type,
                              appctx=appctx,
                              options_prefix=options_prefix,
                              form_compiler_parameters=form_compiler_parameters,
                              zero_bc_nodes=zero_bc_nodes, weight=weight)

        # if isinstance(res, firedrake.Function):
        #   # TODO: Remove once MatrixImplicitContext is Cofunction safe.
        #   res = firedrake.Cofunction(res.function_space().dual(), val=res.vector())
        # return res

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
                res = PETSc.Mat().create()

                # TODO Figure out what goes here
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
        if all([isinstance(op, float) for op in args]):
            return sum(args)
        elif all([isinstance(op, firedrake.Cofunction) for op in args]):
            # TODO check all are in same function space
            res = sum([w*op.dat for (op, w) in zip(args, expr.weights())])
            return firedrake.Cofunction(args[0].function_space(), res)
        elif all([isinstance(op, ufl.Matrix) for op in args]):
            res = PETSc.Mat().create()
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
    elif isinstance(expr, ufl.ExternalOperator):
        opts = {'form_compiler_parameters': form_compiler_parameters,
                'mat_type': mat_type, 'sub_mat_type': sub_mat_type,
                'appctx': appctx, 'options_prefix': options_prefix,
                'diagonal': diagonal}
        # Replace base forms in the operands and argument slots of the external operator by their result
        v, *assembled_children = args
        if assembled_children:
            _, *children = base_form_operands(expr)
            # Replace assembled children by their results
            expr = ufl.replace(expr, dict(zip(children, assembled_children)))
        # Always reconstruct the dual argument (0-slot argument) since it is a BaseForm
        # It is also convenient when we have a Form in that slot since Forms don't play well with `ufl.replace`
        expr = expr._ufl_expr_reconstruct_(*expr.ufl_operands, argument_slots=(v,) + expr.argument_slots()[1:])
        # Call the external operator assembly
        return expr.assemble(assembly_opts=opts)
    elif isinstance(expr, ufl.Interp):
        # Replace assembled children
        _, expression = expr.argument_slots()
        v, *assembled_expression = args
        if assembled_expression:
            # Occur in situations such as Interp composition
            expression = assembled_expression[0]
        expr = expr._ufl_expr_reconstruct_(expression, v)

        # Different assembly procedures:
        # 1) Interp(Argument(V1, 1), Argument(V2.dual(), 0)) -> Jacobian (Interp matrix)
        # 2) Interp(Coefficient(...), Argument(V2.dual(), 0)) -> Operator (or Jacobian action)
        # 3) Interp(Argument(V1, 0), Argument(V2.dual(), 1)) -> Jacobian adjoint
        # 4) Interp(Argument(V1, 0), Cofunction(...)) -> Action of the Jacobian adjoint
        # This can be generalized to the case where the first slot is an arbitray expression.
        rank = len(expr.arguments())
        # If argument numbers have been swapped => Adjoint.
        arg_expression = ufl.algorithms.extract_arguments(expression)
        is_adjoint = (arg_expression and arg_expression[0].number() == 0)
        # Workaround: Renumber argument when needed since Interpolator assumes it takes a zero-numbered argument.
        if not is_adjoint and rank != 1:
            _, v1 = expr.arguments()
            expression = ufl.replace(expression, {v1: firedrake.Argument(v1.function_space(), number=0, part=v1.part())})
        # Should we use `freeze_expr` to cache the interpolation ? (e.g. if timestepping loop)
        interpolator = firedrake.Interpolator(expression, expr.function_space(), **expr.interp_data)
        # Assembly
        if rank == 1:
            # Assembling the action of the Jacobian adjoint.
            if is_adjoint:
                output = tensor or firedrake.Cofunction(arg_expression[0].function_space().dual())
                return interpolator._interpolate(v, output=output, transpose=True)
            # Assembling the operator, or its Jacobian action.
            return interpolator._interpolate(output=tensor)
        elif rank == 2:
            # Return the interpolation matrix
            op2_mat = interpolator.callable()
            petsc_mat = op2_mat.handle
            if is_adjoint:
                petsc_mat.hermitianTranspose()
            return matrix.AssembledMatrix(expr.arguments(), bcs, petsc_mat,
                                          appctx=appctx,
                                          options_prefix=options_prefix)
        else:
            # The case rank == 0 is handled via the DAG restructuration
            raise ValueError("Incompatible number of arguments.")
    elif isinstance(expr, (ufl.Cofunction, ufl.Coargument, ufl.Matrix, ufl.ZeroBaseForm)):
        return expr
    elif isinstance(expr, ufl.Coefficient):
        return expr
    else:
        raise TypeError(f"Unrecognised BaseForm instance: {expr}")


@PETSc.Log.EventDecorator()
def allocate_matrix(expr, bcs=None, *, mat_type=None, sub_mat_type=None,
                    appctx=None, form_compiler_parameters=None,
                    integral_types=None, options_prefix=None):
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

    integral_types = integral_types or set(i.integral_type() for i in expr.integrals())
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

    return matrix.Matrix(expr, bcs, mat_type, sparsity, ScalarType,
                         options_prefix=options_prefix)


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
    from ufl.algorithms.analysis import extract_base_form_operators
    from ufl.checks import is_scalar_constant_expression

    # Get BaseFormOperators (`Interp` or `ExternalOperator`)
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
            return assemble_base_form(ufl.FormSum((a, 1), (b, 1)))
        elif isinstance(expr, ufl.algebra.Product):
            a, b = expr.ufl_operands
            scalar = [e for e in expr.ufl_operands if is_scalar_constant_expression(e)]
            if scalar:
                base_form = a if a is scalar else b
                assembled_mat = assemble(base_form)
                return assemble_base_form(ufl.FormSum((assembled_mat, scalar[0])))
            a, b = [assemble(e) for e in (a, b)]
            return assemble_base_form(ufl.action(a, b))
    # -- Linear combination of Functions and 1-form BaseFormOperators -- #
    # Example: a * u1 + b * u2 + c * N(u1; v*) + d * N(u2; v*)
    # with u1, u2 Functions, N a BaseFormOperator, and a, b, c, d scalars or 0-form BaseFormOperators.
    else:
        base_form_operators = extract_base_form_operators(expr)
        assembled_bfops = [firedrake.assemble(e) for e in base_form_operators]
        # Substitute base form operators with their output before examining the expression
        # which avoids conflict when determining function space, for example:
        # extract_coefficients(Interp(u, V2)) with u \in V1 will result in an output function space V1
        # instead of V2.
        if base_form_operators:
            expr = ufl.replace(expr, dict(zip(base_form_operators, assembled_bfops)))
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
        return op2.Global(
            1,
            [0.0],
            dtype=utils.ScalarType,
            comm=form.ufl_domains()[0]._comm
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
            self._as_pyop2_type(self._tensor).zero()

        self.execute_parloops()

        for bc in self._bcs:
            if isinstance(bc, EquationBC):  # can this be lifted?
                bc = bc.extract_form("F")
            self._apply_bc(bc)

        return self.result

    def replace_tensor(self, tensor):
        if tensor is self._tensor:
            return

        # TODO We should have some proper checks here
        for lknl, parloop in zip(self.local_kernels, self.parloops):
            data = _FormHandler.index_tensor(tensor, self._form, lknl.indices, self.diagonal)
            parloop.arguments[0].data = data
        self._tensor = tensor

    def execute_parloops(self):
        for parloop in self.parloops:
            parloop()

    @cached_property
    def local_kernels(self):
        try:
            topology, = set(d.topology for d in self._form.ufl_domains())
        except ValueError:
            raise NotImplementedError("All integration domains must share a mesh topology")

        for o in itertools.chain(self._form.arguments(), self._form.coefficients()):
            domain = extract_unique_domain(o)
            if domain is not None and domain.topology != topology:
                raise NotImplementedError("Assembly with multiple meshes is not supported")

        if isinstance(self._form, ufl.Form):
            return tsfc_interface.compile_form(self._form, "form", diagonal=self.diagonal,
                                               parameters=self._form_compiler_params)
        elif isinstance(self._form, slate.TensorBase):
            return slac.compile_expression(self._form, compiler_parameters=self._form_compiler_params)
        else:
            raise AssertionError

    @cached_property
    def all_integer_subdomain_ids(self):
        return tsfc_interface.gather_integer_subdomain_ids(self.local_kernels)

    @cached_property
    def global_kernels(self):
        return tuple(_make_global_kernel(self._form, tsfc_knl, self.all_integer_subdomain_ids,
                                         diagonal=self.diagonal,
                                         unroll=self.needs_unrolling(tsfc_knl, self._bcs))
                     for tsfc_knl in self.local_kernels)

    @cached_property
    def parloops(self):
        return tuple(ParloopBuilder(self._form, lknl, gknl, self._tensor,
                                    self.all_integer_subdomain_ids, diagonal=self.diagonal,
                                    lgmaps=self.collect_lgmaps(lknl, self._bcs)).build()
                     for lknl, gknl in zip(self.local_kernels, self.global_kernels))

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
    def _as_pyop2_type(tensor):
        if isinstance(tensor, op2.Global):
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
        return self._tensor.data[0]


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

    def execute_parloops(self):
        # We are repeatedly incrementing into the same Dat so intermediate halo exchanges
        # can be skipped.
        with self._tensor.dat.frozen_halo(op2.INC):
            for parloop in self.parloops:
                parloop()

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
            if self._diagonal:
                bc.set(self._tensor, 1)
            else:
                bc.apply(self._tensor)
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

    # Don't expand derivatives if mat_type is 'matfree'
    mat_type = kwargs.pop('mat_type', None)
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
        return functools.partial(assemble_base_form, form, *args, tensor=tensor,
                                 mat_type=mat_type,
                                 is_base_form_preprocessed=True, **kwargs)
    else:
        raise ValueError('Expecting a BaseForm or a slate.TensorBase object and not %s' % form)


def _global_kernel_cache_key(form, local_knl, all_integer_subdomain_ids, **kwargs):
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

    return ((sig,)
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
    :param all_integer_subdomain_ids: See :func:`tsfc_interface.gather_integer_subdomain_ids`.
    :param diagonal: Are we assembling the diagonal of a 2-form?
    :param unroll: If ``True``, address matrix elements directly rather than in
        a blocked fashion. This is slower but required for the application of
        some boundary conditions.

    .. note::
        One should be able to generate a global kernel without needing to
        use any data structures (i.e. a stripped form should be sufficient).
    """

    def __init__(self, form, local_knl, all_integer_subdomain_ids, diagonal=False, unroll=False):
        self._form = form
        self._indices, self._kinfo = local_knl
        self._all_integer_subdomain_ids = all_integer_subdomain_ids.get(self._kinfo.integral_type, None)
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

        if self._kinfo.subdomain_id == "everywhere":
            return False
        elif self._kinfo.subdomain_id == "otherwise":
            return self._all_integer_subdomain_ids is not None
        else:
            return True

    @property
    def _indexed_function_spaces(self):
        return _FormHandler.index_function_spaces(self._form, self._indices)

    def _as_global_kernel_arg(self, tsfc_arg):
        # TODO Make singledispatchmethod with Python 3.8
        return _as_global_kernel_arg(tsfc_arg, self)

    def _get_map_arg(self, finat_element):
        """Get the appropriate map argument for the given FInAT element.

        :arg finat_element: A FInAT element.
        :returns: A :class:`op2.MapKernelArg` instance corresponding to
            the given FInAT element. This function uses a cache to ensure
            that PyOP2 knows when it can reuse maps.
        """
        key = self._get_map_id(finat_element)

        try:
            return self._map_arg_cache[key]
        except KeyError:
            pass

        shape = finat_element.index_shape
        if isinstance(finat_element, finat.TensorFiniteElement):
            shape = shape[:-len(finat_element._shape)]
        arity = numpy.prod(shape, dtype=int)
        if self._integral_type in {"interior_facet", "interior_facet_vert"}:
            arity *= 2

        if self._mesh.extruded:
            offset = tuple(eutils.calculate_dof_offset(finat_element))
            # for interior facet integrals we double the size of the offset array
            if self._integral_type in {"interior_facet", "interior_facet_vert"}:
                offset += offset
        else:
            offset = None
        if self._mesh.extruded_periodic:
            offset_quotient = eutils.calculate_dof_offset_quotient(finat_element)
            if offset_quotient is not None:
                offset_quotient = tuple(offset_quotient)
                if self._integral_type in {"interior_facet", "interior_facet_vert"}:
                    offset_quotient += offset_quotient
        else:
            offset_quotient = None

        map_arg = op2.MapKernelArg(arity, offset, offset_quotient)
        self._map_arg_cache[key] = map_arg
        return map_arg

    def _get_dim(self, finat_element):
        if isinstance(finat_element, finat.TensorFiniteElement):
            return finat_element._shape
        else:
            return (1,)

    def _make_dat_global_kernel_arg(self, finat_element, index=None):
        if isinstance(finat_element, finat.EnrichedElement) and finat_element.is_mixed:
            assert index is None
            subargs = tuple(self._make_dat_global_kernel_arg(subelem.element)
                            for subelem in finat_element.elements)
            return op2.MixedDatKernelArg(subargs)
        else:
            dim = self._get_dim(finat_element)
            map_arg = self._get_map_arg(finat_element)
            return op2.DatKernelArg(dim, map_arg, index)

    def _make_mat_global_kernel_arg(self, relem, celem):
        if any(isinstance(e, finat.EnrichedElement) and e.is_mixed for e in {relem, celem}):
            subargs = tuple(self._make_mat_global_kernel_arg(rel.element, cel.element)
                            for rel, cel in product(relem.elements, celem.elements))
            shape = len(relem.elements), len(celem.elements)
            return op2.MixedMatKernelArg(subargs, shape)
        else:
            # PyOP2 matrix objects have scalar dims so we flatten them here
            rdim = numpy.prod(self._get_dim(relem), dtype=int)
            cdim = numpy.prod(self._get_dim(celem), dtype=int)
            map_args = self._get_map_arg(relem), self._get_map_arg(celem)
            return op2.MatKernelArg((((rdim, cdim),),), map_args, unroll=self._unroll)

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
            return self._make_dat_global_kernel_arg(create_element(V.ufl_element()))
    elif rank == 2:
        if all(V.ufl_element().family() == "Real" for V in Vs):
            return op2.GlobalKernelArg((1,))
        elif any(V.ufl_element().family() == "Real" for V in Vs):
            el, = (create_element(V.ufl_element()) for V in Vs
                   if V.ufl_element().family() != "Real")
            return self._make_dat_global_kernel_arg(el)
        else:
            rel, cel = (create_element(V.ufl_element()) for V in Vs)
            return self._make_mat_global_kernel_arg(rel, cel)
    else:
        raise AssertionError


@_as_global_kernel_arg.register(kernel_args.CoordinatesKernelArg)
def _as_global_kernel_arg_coordinates(_, self):
    finat_element = create_element(self._mesh.ufl_coordinate_element())
    return self._make_dat_global_kernel_arg(finat_element)


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
        return op2.GlobalKernelArg((ufl_element.value_size(),))
    else:
        finat_element = create_element(ufl_element)
        return self._make_dat_global_kernel_arg(finat_element, index)


@_as_global_kernel_arg.register(kernel_args.ConstantKernelArg)
def _as_global_kernel_arg_constant(_, self):
    const = next(self._constants)
    value_size = numpy.prod(const.ufl_shape, dtype=int)
    return op2.GlobalKernelArg((value_size,))


@_as_global_kernel_arg.register(kernel_args.CellSizesKernelArg)
def _as_global_kernel_arg_cell_sizes(_, self):
    # this mirrors tsfc.kernel_interface.firedrake_loopy.KernelBuilder.set_cell_sizes
    ufl_element = ufl.FiniteElement("P", self._mesh.ufl_cell(), 1)
    finat_element = create_element(ufl_element)
    return self._make_dat_global_kernel_arg(finat_element)


@_as_global_kernel_arg.register(kernel_args.ExteriorFacetKernelArg)
def _as_global_kernel_arg_exterior_facet(_, self):
    return op2.DatKernelArg((1,))


@_as_global_kernel_arg.register(kernel_args.InteriorFacetKernelArg)
def _as_global_kernel_arg_interior_facet(_, self):
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
    # this mirrors firedrake.mesh.MeshGeometry.init_cell_orientations
    ufl_element = ufl.FiniteElement("DG", cell=self._form.ufl_domain().ufl_cell(), degree=0)
    finat_element = create_element(ufl_element)
    return self._make_dat_global_kernel_arg(finat_element)


@_as_global_kernel_arg.register(LayerCountKernelArg)
def _as_global_kernel_arg_layer_count(_, self):
    return op2.GlobalKernelArg((1,))


class ParloopBuilder:
    """Class that builds a :class:`op2.Parloop`.

    :param form: The variational form.
    :param local_knl: :class:`tsfc_interface.SplitKernel` compiled by either
        TSFC or Slate.
    :param global_knl: A :class:`pyop2.GlobalKernel` instance.
    :param tensor: The output tensor to write to (cannot be ``None``).
    :param all_integer_subdomain_ids: See :func:`tsfc_interface.gather_integer_subdomain_ids`.
    :param diagonal: Are we assembling the diagonal of a 2-form?
    :param lgmaps: Optional iterable of local-to-global maps needed for applying
        boundary conditions to 2-forms.
    """

    def __init__(self, form, local_knl, global_knl, tensor,
                 all_integer_subdomain_ids, diagonal=False, lgmaps=None):
        self._form = form
        self._local_knl = local_knl
        self._global_knl = global_knl
        self._all_integer_subdomain_ids = all_integer_subdomain_ids
        self._tensor = tensor
        self._diagonal = diagonal
        self._lgmaps = lgmaps

        self._active_coefficients = _FormHandler.iter_active_coefficients(form, local_knl.kinfo)
        self._constants = _FormHandler.iter_constants(form, local_knl.kinfo)

    def build(self):
        """Construct the parloop."""
        parloop_args = [self._as_parloop_arg(tsfc_arg)
                        for tsfc_arg in self._kinfo.arguments]
        try:
            return op2.Parloop(self._global_knl, self._iterset, parloop_args)
        except MapValueError:
            raise RuntimeError("Integral measure does not match measure of all "
                               "coefficients/arguments")

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
            if self._kinfo.subdomain_id not in ["everywhere", "otherwise"]:
                raise ValueError("Cannot use subdomain data and subdomain_id")
            return subdomain_data
        else:
            return self._mesh.measure_set(self._integral_type, self._kinfo.subdomain_id,
                                          self._all_integer_subdomain_ids)

    def _get_map(self, V):
        """Return the appropriate PyOP2 map for a given function space."""
        assert isinstance(V, (WithGeometry, FiredrakeDualSpace, FunctionSpace))

        if self._integral_type in {"cell", "exterior_facet_top",
                                   "exterior_facet_bottom", "interior_facet_horiz"}:
            return V.cell_node_map()
        elif self._integral_type in {"exterior_facet", "exterior_facet_vert"}:
            return V.exterior_facet_node_map()
        elif self._integral_type in {"interior_facet", "interior_facet_vert"}:
            return V.interior_facet_node_map()
        else:
            raise AssertionError

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
    tensor = self._indexed_tensor
    Vs = self._indexed_function_spaces

    if rank == 0:
        return op2.GlobalParloopArg(tensor)
    elif rank == 1 or rank == 2 and self._diagonal:
        V, = Vs
        if V.ufl_element().family() == "Real":
            return op2.GlobalParloopArg(tensor)
        else:
            return op2.DatParloopArg(tensor, self._get_map(V))
    elif rank == 2:
        rmap, cmap = [self._get_map(V) for V in Vs]

        if all(V.ufl_element().family() == "Real" for V in Vs):
            assert rmap is None and cmap is None
            return op2.GlobalParloopArg(tensor.handle.getPythonContext().global_)
        elif any(V.ufl_element().family() == "Real" for V in Vs):
            m = rmap or cmap
            return op2.DatParloopArg(tensor.handle.getPythonContext().dat, m)
        else:
            return op2.MatParloopArg(tensor, (rmap, cmap), lgmaps=self._lgmaps)
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
        for idx, subidxs in kinfo.coefficient_map:
            for subidx in subidxs:
                yield form.coefficients()[idx].subfunctions[subidx]

    @staticmethod
    def iter_constants(form, kinfo):
        """Yield the form constants"""
        # Is kinfo really needed?
        from tsfc.ufl_utils import extract_firedrake_constants
        if isinstance(form, slate.TensorBase):
            for const in form.constants():
                yield const
        else:
            for const in extract_firedrake_constants(form):
                yield const

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

    @staticmethod
    def index_tensor(tensor, form, indices, diagonal):
        """Return the PyOP2 data structure tied to ``tensor``, indexed
        if necessary.
        """
        rank = len(form.arguments())
        is_indexed = any(i is not None for i in indices)

        if rank == 0:
            return tensor
        elif rank == 1 or rank == 2 and diagonal:
            i, = indices
            return tensor.dat[i] if is_indexed else tensor.dat
        elif rank == 2:
            i, j = indices
            return tensor.M[i, j] if is_indexed else tensor.M
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
