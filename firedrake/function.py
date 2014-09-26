import numpy as np
import FIAT
import ufl

import coffee.base as ast

from pyop2 import op2

import assemble_expressions
import expression as expression_t
import functionspace
import projection
import utils
import vector


__all__ = ['Function']


valuetype = np.float64


class Function(ufl.Coefficient):
    """A :class:`Function` represents a discretised field over the
    domain defined by the underlying :class:`.Mesh`. Functions are
    represented as sums of basis functions:

    .. math::

            f = \\sum_i f_i \phi_i(x)

    The :class:`Function` class provides storage for the coefficients
    :math:`f_i` and associates them with a :class:`FunctionSpace` object
    which provides the basis functions :math:`\\phi_i(x)`.

    Note that the coefficients are always scalars: if the
    :class:`Function` is vector-valued then this is specified in
    the :class:`FunctionSpace`.
    """

    def __init__(self, function_space, val=None, name=None, dtype=valuetype):
        """
        :param function_space: the :class:`.FunctionSpace`, :class:`.VectorFunctionSpace`
            or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
            Alternatively, another :class:`Function` may be passed here and its function space
            will be used to build this :class:`Function`.
        :param val: NumPy array-like (or :class:`op2.Dat`) providing initial values (optional).
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to :data:`valuetype`).
        """

        if isinstance(function_space, Function):
            self._function_space = function_space._function_space
        elif isinstance(function_space, functionspace.FunctionSpaceBase):
            self._function_space = function_space
        else:
            raise NotImplementedError("Can't make a Function defined on a "
                                      + str(type(function_space)))

        ufl.Coefficient.__init__(self, self._function_space.ufl_element())

        self._label = "a function"
        self.uid = utils._new_uid()
        self._name = name or 'function_%d' % self.uid

        if isinstance(val, op2.Dat):
            self.dat = val
        else:
            self.dat = self._function_space.make_dat(val, dtype,
                                                     self._name, uid=self.uid)

        self._repr = None
        self._split = None

        if isinstance(function_space, Function):
            self.assign(function_space)

    def split(self):
        """Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`FunctionSpace`."""
        if self._split is None:
            self._split = tuple(Function(fs, dat) for fs, dat in zip(self._function_space, self.dat))
        return self._split

    def sub(self, i):
        """Extract the ith sub :class:`Function` of this :class:`Function`.

        :arg i: the index to extract

        See also :meth:`split`"""
        return self.split()[i]

    @property
    def cell_set(self):
        """The :class:`pyop2.Set` of cells for the mesh on which this
        :class:`Function` is defined."""
        return self._function_space._mesh.cell_set

    @property
    def node_set(self):
        """A :class:`pyop2.Set` containing the nodes of this
        :class:`Function`. One or (for
        :class:`.VectorFunctionSpace`\s) more degrees of freedom are
        stored at each node.
        """
        return self._function_space.node_set

    @property
    def dof_dset(self):
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`Function`."""
        return self._function_space.dof_dset

    def cell_node_map(self, bcs=None):
        return self._function_space.cell_node_map(bcs)
    cell_node_map.__doc__ = functionspace.FunctionSpace.cell_node_map.__doc__

    def interior_facet_node_map(self, bcs=None):
        return self._function_space.interior_facet_node_map(bcs)
    interior_facet_node_map.__doc__ = functionspace.FunctionSpace.interior_facet_node_map.__doc__

    def exterior_facet_node_map(self, bcs=None):
        return self._function_space.exterior_facet_node_map(bcs)
    exterior_facet_node_map.__doc__ = functionspace.FunctionSpace.exterior_facet_node_map.__doc__

    def project(self, b, *args, **kwargs):
        """Project ``b`` onto ``self``. ``b`` must be a :class:`Function` or an
        :class:`Expression`.

        This is equivalent to ``project(b, self)``.
        Any of the additional arguments to :func:`~firedrake.projection.project`
        may also be passed, and they will have their usual effect.
        """
        return projection.project(b, self, *args, **kwargs)

    def vector(self):
        """Return a :class:`.Vector` wrapping the data in this :class:`Function`"""
        return vector.Vector(self.dat)

    def function_space(self):
        """Return the :class:`.FunctionSpace`, :class:`.VectorFunctionSpace`
            or :class:`.MixedFunctionSpace` on which this :class:`Function` is defined."""
        return self._function_space

    def name(self):
        """Return the name of this :class:`Function`"""
        return self._name

    def label(self):
        """Return the label (a description) of this :class:`Function`"""
        return self._label

    def rename(self, name=None, label=None):
        """Set the name and or label of this :class:`Function`

        :arg name: The new name of the `Function` (if not `None`)
        :arg label: The new label for the `Function` (if not `None`)
        """
        if name is not None:
            self._name = name
        if label is not None:
            self._label = label

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return super(Function, self).__str__()

    def interpolate(self, expression, subset=None):
        """Interpolate an expression onto this :class:`Function`.

        :param expression: :class:`.Expression` to interpolate
        :returns: this :class:`Function` object"""

        # Make sure we have an expression of the right length i.e. a value for
        # each component in the value shape of each function space
        dims = [np.prod(fs.ufl_element().value_shape(), dtype=int)
                for fs in self.function_space()]
        if np.prod(expression.value_shape(), dtype=int) != sum(dims):
            raise RuntimeError('Expression of length %d required, got length %d'
                               % (sum(dims), np.prod(expression.value_shape(), dtype=int)))

        if expression.code:
            # Slice the expression and pass in the right number of values for
            # each component function space of this function
            d = 0
            for fs, dat, dim in zip(self.function_space(), self.dat, dims):
                idx = d if fs.rank == 0 else slice(d, d+dim)
                self._interpolate(fs, dat,
                                  expression_t.Expression(expression.code[idx],
                                                          **expression._kwargs),
                                  subset)
                d += dim
        else:
            if isinstance(fs, functionspace.MixedFunctionSpace):
                raise NotImplementedError(
                    "Python expressions for mixed functions are not yet supported.")
            self._interpolate(fs, self.dat, expression, subset)
        return self

    def _interpolate(self, fs, dat, expression, subset):
        """Interpolate expression onto a :class:`FunctionSpace`.

        :param fs: :class:`FunctionSpace`
        :param dat: :class:`pyop2.Dat`
        :param expression: :class:`.Expression`
        """
        to_element = fs.fiat_element
        to_pts = []

        # TODO very soon: look at the mapping associated with the UFL element;
        # this needs to be "identity" (updated from "affine")
        if to_element.mapping()[0] != "affine":
            raise NotImplementedError("Can only interpolate onto elements \
                with affine mapping. Try projecting instead")

        for dual in to_element.dual_basis():
            if not isinstance(dual, FIAT.functional.PointEvaluation):
                raise NotImplementedError("Can only interpolate onto point \
                    evaluation operators. Try projecting instead")
            to_pts.append(dual.pt_dict.keys()[0])

        if expression.rank() != len(fs.ufl_element().value_shape()):
            raise RuntimeError('Rank mismatch: Expression rank %d, FunctionSpace rank %d'
                               % (expression.rank(), len(fs.ufl_element().value_shape())))

        if expression.value_shape() != fs.ufl_element().value_shape():
            raise RuntimeError('Shape mismatch: Expression shape %r, FunctionSpace shape %r'
                               % (expression.value_shape(), fs.ufl_element().value_shape()))

        coords = fs.mesh().coordinates

        if expression.code:
            kernel = self._interpolate_c_kernel(expression,
                                                to_pts, to_element, fs, coords)
            args = [kernel, subset or self.cell_set,
                    dat(op2.WRITE, fs.cell_node_map()[op2.i[0]]),
                    coords.dat(op2.READ, coords.cell_node_map())]
        elif hasattr(expression, "eval"):
            kernel = self._interpolate_python_kernel(expression,
                                                     to_pts, to_element, fs, coords)
            args = [kernel, subset or self.cell_set,
                    dat(op2.WRITE, fs.cell_node_map()),
                    coords.dat(op2.READ, coords.cell_node_map())]
        else:
            raise RuntimeError(
                "Attempting to evaluate an Expression which has no value.")

        for _, arg in expression._user_args:
            args.append(arg(op2.READ))
        op2.par_loop(*args)

    def _interpolate_python_kernel(self, expression, to_pts, to_element, fs, coords):
        """Produce a :class:`PyOP2.Kernel` wrapping the eval method on the
        function provided."""

        coords_space = coords.function_space()
        coords_element = coords_space.fiat_element

        X_remap = coords_element.tabulate(0, to_pts).values()[0]

        # The par_loop will just pass us arguments, since it doesn't
        # know about keyword args at all so unpack into a dict that we
        # can pass to the user's eval method.
        def kernel(output, x, *args):
            kwargs = {}
            for (slot, _), arg in zip(expression._user_args, args):
                kwargs[slot] = arg
            X = np.dot(X_remap.T, x)

            for i in range(len(output)):
                # Pass a slice for the scalar case but just the
                # current vector in the VFS case. This ensures the
                # eval method has a Dolfin compatible API.
                expression.eval(output[i:i+1, ...] if np.rank(output) == 1 else output[i, ...],
                                X[i:i+1, ...] if np.rank(X) == 1 else X[i, ...], **kwargs)

        return kernel

    def _interpolate_c_kernel(self, expression, to_pts, to_element, fs, coords):
        """Produce a :class:`PyOP2.Kernel` from the c expression provided."""

        coords_space = coords.function_space()
        coords_element = coords_space.fiat_element

        names = {v[0] for v in expression._user_args}

        X = coords_element.tabulate(0, to_pts).values()[0]

        # Produce C array notation of X.
        X_str = "{{"+"},\n{".join([",".join(map(str, x)) for x in X.T])+"}}"

        A = utils.unique_name("A", names)
        X = utils.unique_name("X", names)
        x_ = utils.unique_name("x_", names)
        k = utils.unique_name("k", names)
        d = utils.unique_name("d", names)
        i_ = utils.unique_name("i", names)
        # x is a reserved name.
        x = "x"
        if "x" in names:
            raise ValueError("cannot use 'x' as a user-defined Expression variable")
        ass_exp = [ast.Assign(ast.Symbol(A, (k,), ((len(expression.code), i),)),
                              ast.FlatBlock("%s" % code))
                   for i, code in enumerate(expression.code)]
        vals = {
            "X": X,
            "x": x,
            "x_": x_,
            "k": k,
            "d": d,
            "i": i_,
            "x_array": X_str,
            "dim": coords_space.dim,
            "xndof": coords_element.space_dimension(),
            # FS will always either be a functionspace or
            # vectorfunctionspace, so just accessing dim here is safe
            # (we don't need to go through ufl_element.value_shape())
            "nfdof": to_element.space_dimension() * fs.dim,
            "ndof": to_element.space_dimension(),
            "assign_dim": np.prod(expression.value_shape(), dtype=int)
        }
        init = ast.FlatBlock("""
const double %(X)s[%(ndof)d][%(xndof)d] = %(x_array)s;

double %(x)s[%(dim)d];
const double pi = 3.141592653589793;

""" % vals)
        block = ast.FlatBlock("""
for (unsigned int %(d)s=0; %(d)s < %(dim)d; %(d)s++) {
  %(x)s[%(d)s] = 0;
  for (unsigned int %(i)s=0; %(i)s < %(xndof)d; %(i)s++) {
        %(x)s[%(d)s] += %(X)s[%(k)s][%(i)s] * %(x_)s[%(i)s][%(d)s];
  };
};

""" % vals)
        loop = ast.c_for(k, "%(ndof)d" % vals, ast.Block([block] + ass_exp,
                                                         open_scope=True))
        user_args = []
        user_init = []
        for _, arg in expression._user_args:
            if arg.shape == (1, ):
                user_args.append(ast.Decl("double *", "%s_" % arg.name))
                user_init.append(ast.FlatBlock("const double %s = *%s_;" %
                                               (arg.name, arg.name)))
            else:
                user_args.append(ast.Decl("double *", arg.name))
        kernel_code = ast.FunDecl("void", "expression_kernel",
                                  [ast.Decl("double", ast.Symbol(A, (int("%(nfdof)d" % vals),))),
                                   ast.Decl("double**", x_)] + user_args,
                                  ast.Block(user_init + [init, loop],
                                            open_scope=False))
        return op2.Kernel(kernel_code, "expression_kernel")

    def assign(self, expr, subset=None):
        """Set the :class:`Function` value to the pointwise value of
        expr. expr may only contain :class:`Function`\s on the same
        :class:`.FunctionSpace` as the :class:`Function` being assigned to.

        Similar functionality is available for the augmented assignment
        operators `+=`, `-=`, `*=` and `/=`. For example, if `f` and `g` are
        both Functions on the same :class:`FunctionSpace` then::

          f += 2 * g

        will add twice `g` to `f`.

        If present, subset must be an :class:`pyop2.Subset` of
        :attr:`node_set`. The expression will then only be assigned
        to the nodes on that subset.
        """

        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            expr.dat.copy(self.dat, subset=subset)
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.Assign(self, expr), subset)

        return self

    def __iadd__(self, expr):

        if np.isscalar(expr):
            self.dat += expr
            return self
        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            self.dat += expr.dat
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.IAdd(self, expr))

        return self

    def __isub__(self, expr):

        if np.isscalar(expr):
            self.dat -= expr
            return self
        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            self.dat -= expr.dat
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.ISub(self, expr))

        return self

    def __imul__(self, expr):

        if np.isscalar(expr):
            self.dat *= expr
            return self
        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            self.dat *= expr.dat
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.IMul(self, expr))

        return self

    def __idiv__(self, expr):

        if np.isscalar(expr):
            self.dat /= expr
            return self
        if isinstance(expr, Function) and \
                expr._function_space == self._function_space:
            self.dat /= expr.dat
            return self

        assemble_expressions.evaluate_expression(
            assemble_expressions.IDiv(self, expr))

        return self
