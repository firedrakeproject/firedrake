
import numpy
import collections
import functools

from ufl import as_tensor, as_vector, split
from ufl.classes import (
    Form, Zero, ListTensor, ZeroBaseForm, BaseForm, Action, Adjoint,
    Expr, CoefficientDerivative, MultiIndex, Argument, Matrix,
    Interpolate, FormSum,
)
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.analysis import has_type
from ufl.algorithms import expand_derivatives
from ufl.corealg.dag_traverser import DAGTraverser

from pyop2 import MixedDat
from pyop2.utils import as_tuple

from firedrake.petsc import PETSc
from firedrake.functionspace import MixedFunctionSpace
from firedrake.functionspaceimpl import WithGeometry
from firedrake.cofunction import Cofunction
from firedrake.slate.slate import (
    Add, AssembledVector, Block, BlockAssembledVector, DiagonalTensor,
    Factorization, Inverse, Mul, Reciprocal, ScalarMul, Solve, Tensor,
    TensorBase, Transpose, as_slate,
)
from firedrake.ufl_expr import Coargument


def subspace(V, indices):
    """Construct a collapsed subspace using components from V."""
    if len(indices) == 1:
        W = V[indices[0]]
    else:
        W = MixedFunctionSpace([V[i] for i in indices])
    return W.collapse()


class ExtractSubBlock(DAGTraverser):

    """Extract a sub-block from a form."""

    def _subspace_argument(self, a, blocks):
        indices = self.select_block(blocks, a.number())
        if indices is None or len(a.function_space()) == 1:
            return a
        return Argument.reconstruct(a, function_space=subspace(a.function_space(), indices))

    @staticmethod
    def select_block(blocks: tuple, number: int) -> tuple | None:
        return blocks[number] if number < len(blocks) else None

    @PETSc.Log.EventDecorator()
    def split(self, form: BaseForm, argument_indices: tuple) -> BaseForm:
        """Split a form.

        Parameters
        ----------
        form
            The form to split.
        argument_indices
            Indices of test and trial spaces to extract.
            This should be 0-, 1-, or 2-tuple (whose length is the
            same as the number of arguments as the ``form``) whose
            entries are either an integer index, or else an iterable
            of indices.

        Returns
        -------
        ufl.classes.BaseForm
            Form on the selected subspaces.
        """
        args = form.arguments()
        if len(args) == 0:
            # Functional can't be split
            return form
        if all(len(a.function_space()) == 1 for a in args):
            return form

        if isinstance(form, FormSum) and has_type(form, TensorBase):
            # Convert a UFL sum that contains Slate components into Slate.
            form = as_slate(form)

        return self(form, blocks=tuple(as_tuple(i) for i in argument_indices))

    @functools.singledispatchmethod
    def process(self, o, blocks):
        return super().process(o, blocks=blocks)

    @process.register(Expr)
    def _(self, o, blocks):
        return self.reuse_if_untouched(o, blocks=blocks)

    @process.register(Form)
    def _(self, o, blocks):
        form = map_integrands(functools.partial(self._expression_splitter, blocks=blocks), o)
        # TODO find a way to distinguish empty Forms avoiding expand_derivatives
        if expand_derivatives(form).empty():
            return self(ZeroBaseForm(o.arguments()), blocks=blocks)
        return form

    @process.register(Adjoint)
    def _(self, o, blocks):
        # Adjoint: swap rows and columns before splitting the operand
        rows = self.select_block(blocks, 0)
        cols = self.select_block(blocks, 1)
        operand = self(o.form(), blocks=(cols, rows))
        if operand == 0:
            return self(ZeroBaseForm(o.arguments()), blocks=blocks)
        return Adjoint(operand)

    @process.register(Action)
    def _(self, o, blocks):
        # Action: preserve the contracted argument before splitting the operand
        left, right = o.ufl_operands
        if isinstance(left, BaseForm):
            contracted_arg_num = left.arguments()[-1].number()
            fields = tuple(None if i == contracted_arg_num else field for i, field in enumerate(blocks))
            left = self(left, blocks=fields)
        if isinstance(right, BaseForm):
            contracted_arg_num = right.arguments()[0].number()
            fields = tuple(None if i == contracted_arg_num else field for i, field in enumerate(blocks))
            right = self(right, blocks=fields)
        return Action(left, right)

    @process.register(FormSum)
    @DAGTraverser.postorder
    def _(self, o, *components, blocks):
        return FormSum(*zip(components, o.weights()))

    @process.register(MultiIndex)
    def _(self, o, blocks):
        return o

    @process.register(CoefficientDerivative)
    @DAGTraverser.postorder
    def _(self, o, expr, coefficients, arguments, cds, blocks):
        argument, = arguments
        if (isinstance(argument, Zero)
            or (isinstance(argument, ListTensor)
                and all(isinstance(a, Zero) for a in argument.ufl_operands))):
            # If we're only taking a derivative wrt part of an argument in
            # a mixed space other bits might come back as zero. We want to
            # propagate a zero in that case.
            return Zero(o.ufl_shape, o.ufl_free_indices, o.ufl_index_dimensions)
        else:
            return self.reuse_if_untouched(o, blocks=blocks)

    @process.register(Argument)
    @PETSc.Log.EventDecorator()
    def _(self, o, blocks):
        V = o.function_space()

        indices = self.select_block(blocks, o.number())
        if indices is None or len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        a = self._subspace_argument(o, blocks)
        asplit = (a, ) if len(indices) == 1 else split(a)

        args = []
        for i in range(len(V)):
            if i in indices:
                asub = asplit[indices.index(i)]
                args.extend(asub[j] for j in numpy.ndindex(asub.ufl_shape))
            else:
                args.extend(Zero() for j in numpy.ndindex(V[i].value_shape))
        return as_vector(args)

    @process.register(Coargument)
    def _(self, o, blocks):
        V = o.function_space()

        indices = self.select_block(blocks, o.number())
        if indices is None or len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        W = subspace(V, indices)
        return Coargument(W, number=o.number(), part=o.part())

    @process.register(Cofunction)
    def _(self, o, blocks):
        V = o.function_space()

        indices = self.select_block(blocks, 0)
        if indices is None or len(V) == 1:
            # Not on a mixed space, just return ourselves.
            return o

        # We only need the test space for Cofunction
        W = subspace(V, indices)
        if len(W) == 1:
            return Cofunction(W, val=o.dat[indices[0]])
        else:
            return Cofunction(W, val=MixedDat(o.dat[i] for i in indices))

    @process.register(Matrix)
    def _(self, o, blocks):
        from firedrake.bcs import DirichletBC, EquationBCSplit
        from firedrake.matrix import AssembledMatrix

        ises = []
        args = []
        argument_indices = []
        for a in o.arguments():
            V = a.function_space()
            iset = PETSc.IS()
            fields = self.select_block(blocks, a.number())
            if fields is not None:
                asplit = self._subspace_argument(a, blocks)
                for f in fields:
                    fset = V.dof_dset.field_ises[f]
                    iset = iset.expand(fset)
            else:
                fields = tuple(range(len(V)))
                asplit = a
                for fset in V.dof_dset.field_ises:
                    iset = iset.expand(fset)

            ises.append(iset)
            args.append(asplit)
            argument_indices.append(fields)

        if isinstance(o.a, Form):
            form = self.split(o.a, argument_indices=argument_indices)
            if isinstance(form, ZeroBaseForm):
                return form
        else:
            form = None

        submat = o.petscmat.createSubMatrix(*ises)

        bcs = []
        spaces = [a.function_space() for a in o.arguments()]
        for bc in o.bcs:
            W = bc.function_space()
            while W.parent is not None:
                W = W.parent

            number = spaces.index(W)
            field = argument_indices[number]
            V = args[number].function_space()
            if isinstance(bc, DirichletBC):
                bc_temp = bc.reconstruct(field=field, V=V, g=bc.function_arg, use_split=True)
            elif isinstance(bc, EquationBCSplit):
                row_field, col_field = argument_indices
                bc_temp = bc.reconstruct(field=field, V=V, row_field=row_field, col_field=col_field, use_split=True)
            if bc_temp is not None:
                bcs.append(bc_temp)

        return AssembledMatrix(form or tuple(args), submat, tuple(bcs))

    @process.register(ZeroBaseForm)
    def _(self, o, blocks):
        return ZeroBaseForm(self._subspace_argument(a, blocks) for a in o.arguments())

    @functools.cached_property
    def _expression_splitter(self) -> "ExtractSubBlock":
        """Return the splitter for the expressions inside a form."""
        return ExtractSubExpression()

    @staticmethod
    def _select_components(V: WithGeometry, indices: tuple, operand: Expr) -> list:
        """Select the flattened components for the requested subspaces.

        Mixed function spaces store the values of their subspaces
        consecutively. This selects those components before they are reshaped
        to the value shape of the collapsed target space.
        """
        components = []
        cur = 0
        for i, Vi in enumerate(V):
            if i in indices:
                components.extend(operand[k] for k in range(cur, cur + Vi.value_size))
            cur += Vi.value_size
        return components

    @process.register(Interpolate)
    def _(self, o, blocks):
        # The target of an Interpolate that is a form is an argument of that
        # form. Interpolation onto a mixed space acts on each subspace
        # separately, so the block onto some target subspaces interpolates
        # only the operand components that belong to those subspaces.
        dual_arg, operand = o.argument_slots()
        operand = self._expression_splitter(operand, blocks=blocks)
        indices = self.select_block(blocks, dual_arg.number()) if isinstance(dual_arg, Coargument) else None
        if indices is not None and len(dual_arg.function_space()) > 1:
            V = dual_arg.function_space()
            dual_arg = self(dual_arg, blocks=blocks)
            components = self._select_components(V, indices, operand)
            operand = as_tensor(numpy.reshape(components, dual_arg.function_space().value_shape))
        if isinstance(operand, Zero):
            return self(ZeroBaseForm(o.arguments()), blocks=blocks)
        return o._ufl_expr_reconstruct_(operand, dual_arg)


class ExtractSubExpression(ExtractSubBlock):

    """Extract a sub-block from an expression inside a form."""

    @functools.singledispatchmethod
    def process(self, o, blocks):
        return super().process(o, blocks=blocks)

    @process.register(Interpolate)
    @DAGTraverser.postorder
    def _(self, o, operand, blocks):
        # The target of an Interpolate inside an expression is contracted
        # there, so it is not an argument of the form.
        if isinstance(operand, Zero):
            return Zero(o.ufl_shape)
        return o._ufl_expr_reconstruct_(operand)

    @process.register(Transpose)
    def _(self, o, blocks):
        if not blocks:
            return o
        return Transpose(*(self(child, blocks=blocks[::-1]) for child in o.children))

    @process.register(Add)
    @process.register(DiagonalTensor)
    @process.register(Reciprocal)
    @process.register(ScalarMul)
    def _(self, o, blocks):
        return self.reuse_if_untouched(o, blocks=blocks)

    @process.register(Factorization)
    @process.register(Inverse)
    @process.register(Solve)
    def _(self, o, blocks):
        # These operators are not distributive over block extraction.
        expression = self.reuse_if_untouched(o, blocks=())
        return Block(expression, blocks) if blocks else expression

    @process.register(Mul)
    def _(self, o, blocks):
        if blocks and len(blocks) == 2 and all(operand.rank == 2 for operand in o.operands):
            A, B = o.operands
            row, col = blocks
            full_col_A = tuple(range(len(A.arguments()[1].function_space())))
            full_row_B = tuple(range(len(B.arguments()[0].function_space())))
            A = self(A, blocks=(row, full_col_A))
            B = self(B, blocks=(full_row_B, col))
            return type(o)(A, B)

        # A non-matrix product cannot distribute the outer block across its operands.
        expression = self.reuse_if_untouched(o, blocks=())
        return Block(expression, blocks) if blocks else expression

    @process.register(Tensor)
    def _(self, o, blocks):
        return Tensor(Block(o, blocks).form) if blocks else o

    @process.register(AssembledVector)
    def _(self, o, blocks):
        return BlockAssembledVector(o._function, o, blocks) if blocks else o

    @process.register(Block)
    def _(self, o, blocks):
        if blocks:
            reindexed = tuple(
                tuple(big[i] for i in small)
                for big, small in zip(o._indices, blocks)
            )
        else:
            reindexed = o._indices
        child, = o.children
        return self(child, blocks=reindexed)


SplitForm = collections.namedtuple("SplitForm", ["indices", "form"])


@PETSc.Log.EventDecorator()
def split_form(form, diagonal=False):
    """Split a form into blocks over the component spaces of its arguments.

    Parameters
    ----------
    form : ufl.BaseForm
        The form to split.
    diagonal : bool
        If ``True``, return only the diagonal blocks of a two-argument form.

    Returns
    -------
    tuple of SplitForm
        Each entry contains the component indices and the corresponding form
        on those component spaces.

    Examples
    --------
    Consider the following form:

    .. code-block:: python

        V = FunctionSpace(m, 'CG', 1)
        W = V*V*V
        u, v, w = TrialFunctions(W)
        p, q, r = TestFunctions(W)
        a = q*u*dx + p*w*dx

    Splitting ``a`` returns two nonzero blocks:

    .. code-block:: python

        ((1, 0), q*u*dx)
        ((0, 2), p*w*dx)

    Due to the limited amount of simplification that UFL does, some of
    the returned forms may evaluate to zero. The form compiler removes those
    forms during its later simplification stages.
    """
    splitter = ExtractSubBlock()
    args = form.arguments()
    shape = tuple(len(a.function_space()) for a in args)
    forms = []
    rank = len(shape)
    if diagonal:
        assert rank == 2
        rank = 1
    for idx in numpy.ndindex(shape):
        if diagonal:
            i, j = idx
            if i != j:
                continue
        f = splitter.split(form, idx)
        forms.append(SplitForm(indices=idx[:rank], form=f))
    return tuple(forms)
