from __future__ import absolute_import, print_function, division
from six.moves import map

import collections
import numpy
import sympy
from pyop2.datatypes import IntType, as_cstr

from coffee import base as ast


def operands_and_reconstruct(expr):
    if isinstance(expr, sympy.Expr):
        return (expr.args,
                lambda children: expr.func(*children))
    else:
        # e.g. floating-point numbers
        return (), None


class SSATransformer(object):
    def __init__(self, prefix=None):
        self._regs = {}
        self._code = collections.OrderedDict()
        self._prefix = prefix or "r"

    def _new_reg(self):
        return sympy.Symbol('%s%d' % (self._prefix, len(self._regs)))

    def __call__(self, e):
        ops, reconstruct = operands_and_reconstruct(e)
        if len(ops) == 0:
            return e
        elif e in self._regs:
            return self._regs[e]
        else:
            s = reconstruct(list(map(self, ops)))
            r = self._new_reg()
            self._regs[e] = r
            self._code[r] = s
            return r

    @property
    def code(self):
        return self._code.items()


def rounding(expr):
    from firedrake.pointquery_utils import format
    eps = format["epsilon"]

    if isinstance(expr, (float, sympy.numbers.Float)):
        v = float(expr)
        if abs(v - round(v, 1)) < eps:
            return round(v, 1)
    elif isinstance(expr, sympy.Expr):
        if expr.args:
            return expr.func(*map(rounding, expr.args))

    return expr


def ssa_arrays(args, prefix=None):
    transformer = SSATransformer(prefix=prefix)

    refs = []
    for arg in args:
        ref = numpy.zeros_like(arg, dtype=object)
        arg_flat = arg.reshape(-1)
        ref_flat = ref.reshape(-1)
        for i, e in enumerate(arg_flat):
            ref_flat[i] = transformer(rounding(e))
        refs.append(ref)

    return transformer.code, refs


class _CPrinter(sympy.printing.StrPrinter):
    """sympy.printing.StrPrinter uses a Pythonic syntax which is invalid in C.
    This subclass replaces the printing of power with C compatible code."""

    def _print_Pow(self, expr, rational=False):
        # WARNING: Code mostly copied from sympy source code!
        from sympy.core import S
        from sympy.printing.precedence import precedence

        PREC = precedence(expr)

        if expr.exp is S.Half and not rational:
            return "sqrt(%s)" % self._print(expr.base)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # Note: Don't test "expr.exp == -S.Half" here, because that will
                # match -0.5, which we don't want.
                return "1/sqrt(%s)" % self._print(expr.base)
            if expr.exp is -S.One:
                # Similarly to the S.Half case, don't test with "==" here.
                return '1/%s' % self.parenthesize(expr.base, PREC)

        e = self.parenthesize(expr.exp, PREC)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and expr.exp.q != 1:
            # the parenthesized exp should be '(Rational(a, b))' so strip parens,
            # but just check to be sure.
            if e.startswith('(Rational'):
                e = e[1:-1]

        # Changes below this line!
        if e == "2":
            return '{0}*{0}'.format(self.parenthesize(expr.base, PREC))
        elif e == "3":
            return '{0}*{0}*{0}'.format(self.parenthesize(expr.base, PREC))
        else:
            return 'pow(%s,%s)' % (self.parenthesize(expr.base, PREC), e)


def c_print(expr):
    printer = _CPrinter(dict(order=None))
    return printer.doprint(expr)


def compile_element(expression):
    """Generates C code for point evaluations.

    :arg ufl_element: UFL expression
    :returns: C code as string
    """
    import collections
    from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks
    from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
    from ufl.algorithms.apply_derivatives import apply_derivatives
    from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
    from ufl.algorithms import extract_arguments, extract_coefficients
    from gem import gem, impero_utils
    from tsfc import coffee, kernel_interface, pointeval, ufl_utils

    # Imitate the compute_form_data processing pipeline
    #
    # Unfortunately, we cannot call compute_form_data here, since
    # we only have an expression, not a form
    expression = apply_algebra_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_function_pullbacks(expression)
    expression = apply_geometry_lowering(expression)
    expression = apply_derivatives(expression)
    expression = apply_geometry_lowering(expression)
    expression = apply_derivatives(expression)

    if extract_arguments(expression):
        return ValueError("Cannot interpolate UFL expression with Arguments!")

    # Prepare Coefficients
    arglist = []
    coefficient_map = {}

    coefficient, = extract_coefficients(expression)
    funarg, prepare_, variable = kernel_interface.prepare_coefficient(coefficient, "f")
    arglist.append(funarg)
    assert not prepare_
    coefficient_map[coefficient] = variable

    # Replace coordinates (if any)
    domain = expression.ufl_domain()
    assert domain
    coordinate_coefficient = ufl_utils.coordinate_coefficient(domain)
    expression = ufl_utils.replace_coordinates(expression, coordinate_coefficient)
    funarg, prepare_, variable = kernel_interface.prepare_coefficient(coordinate_coefficient, "x")
    arglist.insert(0, funarg)
    assert not prepare_
    coefficient_map[coordinate_coefficient] = variable

    X = gem.Variable('X', (domain.ufl_cell().topological_dimension(),))
    arglist.insert(0, ast.Decl("double", ast.Symbol('X', rank=(domain.ufl_cell().topological_dimension(),))))

    tabman = pointeval.TabulationManager()
    ic = collections.defaultdict(gem.Index)
    result = pointeval.process(expression, tabman, X, coefficient_map, ic)

    tensor_indices = ()
    if expression.ufl_shape:
        tensor_indices = tuple(gem.Index() for s in expression.ufl_shape)
        retvar = gem.Indexed(gem.Variable('R', expression.ufl_shape), tensor_indices)
        R_sym = ast.Symbol('R', rank=expression.ufl_shape)
        result = gem.Indexed(result, tensor_indices)
    else:
        R_sym = ast.Symbol('R', rank=(1,))
        retvar = gem.Indexed(gem.Variable('R', (1,)), (0,))

    impero_c = impero_utils.compile_gem([retvar], [result], ())
    body = coffee.generate(impero_c, [])

    # Build kernel
    arglist.insert(0, ast.Decl("double", R_sym))
    kernel_code = ast.FunDecl("void", "evaluate_kernel", arglist, body, pred=["static", "inline"])

    from ufl import TensorProductCell

    # Create FIAT element
    cell = domain.ufl_cell()
    extruded = isinstance(cell, TensorProductCell)

    code = {
        "geometric_dimension": cell.geometric_dimension(),
        "extruded_arg": ", %s nlayers" % as_cstr(IntType) if extruded else "",
        "nlayers": ", f->n_layers" if extruded else "",
        "IntType": as_cstr(IntType),
    }

    evaluate_template_c = """static inline void wrap_evaluate(double *result, double *X, double *coords, %(IntType)s *coords_map, double *f, %(IntType)s *f_map%(extruded_arg)s, %(IntType)s cell);

int evaluate(struct Function *f, double *x, double *result)
{
    struct ReferenceCoords reference_coords;
    %(IntType)s cell = locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &reference_coords);
    if (cell == -1) {
        return -1;
    }

    if (!result) {
        return 0;
    }

    wrap_evaluate(result, reference_coords.X, f->coords, f->coords_map, f->f, f->f_map%(nlayers)s, cell);
    return 0;
}
"""

    return (evaluate_template_c % code) + kernel_code.gencode()
