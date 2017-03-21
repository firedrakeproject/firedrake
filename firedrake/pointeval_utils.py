from __future__ import absolute_import, print_function, division
from six.moves import map, range

import collections
import numpy
import sympy
from pyop2.datatypes import IntType, as_cstr


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


def compile_element(ufl_element, cdim):
    """Generates C code for point evaluations.

    :arg ufl_element: UFL element of the function space
    :arg cdim: ``cdim`` of the function space
    :returns: C code as string
    """
    from tsfc import default_parameters
    from firedrake.pointquery_utils import set_float_formatting, format
    from tsfc.fiatinterface import create_element
    from FIAT.reference_element import TensorProductCell as two_product_cell
    import sympy as sp
    import numpy as np

    # Set code generation parameters
    set_float_formatting(default_parameters()["precision"])

    def calculate_basisvalues(ufl_cell, fiat_element):
        f_component = format["component"]
        f_decl = format["declaration"]
        f_float_decl = format["float declaration"]
        f_tensor = format["tabulate tensor"]
        f_new_line = format["new line"]

        tdim = ufl_cell.topological_dimension()
        gdim = ufl_cell.geometric_dimension()

        code = []

        # Symbolic tabulation
        tabs = fiat_element.tabulate(0, np.array([[sp.Symbol("reference_coords.X[%d]" % i)
                                                   for i in range(tdim)]]))
        tabs = tabs[(0,) * tdim]
        tabs = tabs.reshape(tabs.shape[:-1])

        # Generate code for intermediate values
        s_code, (theta,) = ssa_arrays([tabs])
        for name, value in s_code:
            code += [f_decl(f_float_decl, name, c_print(value))]

        # Prepare Jacobian, Jacobian inverse and determinant
        s_detJ = sp.Symbol('detJ')
        s_J = np.array([[sp.Symbol("J[{i}*{tdim} + {j}]".format(i=i, j=j, tdim=tdim))
                         for j in range(tdim)]
                        for i in range(gdim)])
        s_Jinv = np.array([[sp.Symbol("K[{i}*{gdim} + {j}]".format(i=i, j=j, gdim=gdim))
                            for j in range(gdim)]
                           for i in range(tdim)])

        # Apply transformations
        phi = []
        for i, val in enumerate(theta):
            mapping = fiat_element.mapping()[i]
            if mapping == "affine":
                phi.append(val)
            elif mapping == "contravariant piola":
                phi.append(s_J.dot(val) / s_detJ)
            elif mapping == "covariant piola":
                phi.append(s_Jinv.transpose().dot(val))
            else:
                raise ValueError("Unknown mapping: %s" % mapping)
        phi = np.asarray(phi, dtype=object)

        # Dump tables of basis values
        code += ["", "\t// Values of basis functions"]
        code += [f_decl("double", f_component("phi", phi.shape),
                        f_new_line + f_tensor(phi))]

        shape = phi.shape
        if len(shape) <= 1:
            vdim = 1
        elif len(shape) == 2:
            vdim = shape[1]
        return "\n".join(code), vdim

    # Create FIAT element
    element = create_element(ufl_element, vector_is_mixed=False)
    cell = ufl_element.cell()

    calculate_basisvalues, vdim = calculate_basisvalues(cell, element)
    extruded = isinstance(element.get_reference_element(), two_product_cell)

    code = {
        "cdim": cdim,
        "vdim": vdim,
        "geometric_dimension": cell.geometric_dimension(),
        "ndofs": element.space_dimension(),
        "calculate_basisvalues": calculate_basisvalues,
        "extruded_arg": ", %s nlayers" % as_cstr(IntType) if extruded else "",
        "nlayers": ", f->n_layers" if extruded else "",
        "IntType": as_cstr(IntType),
    }

    evaluate_template_c = """static inline void evaluate_kernel(double *result, double *phi_, double **F)
{
    const int ndofs = %(ndofs)d;
    const int cdim = %(cdim)d;
    const int vdim = %(vdim)d;

    double (*phi)[vdim] = (double (*)[vdim]) phi_;

    // F: ndofs x cdim
    // phi: ndofs x vdim
    // result = F' * phi: cdim x vdim
    //
    // Usually cdim == 1 or vdim == 1.

    for (int q = 0; q < cdim * vdim; q++) {
        result[q] = 0.0;
    }
    for (int i = 0; i < ndofs; i++) {
        for (int c = 0; c < cdim; c++) {
            for (int v = 0; v < vdim; v++) {
                result[c*vdim + v] += F[i][c] * phi[i][v];
            }
        }
    }
}

static inline void wrap_evaluate(double *result, double *phi, double *data, %(IntType)s *map%(extruded_arg)s, %(IntType)s cell);

int evaluate(struct Function *f, double *x, double *result)
{
    struct ReferenceCoords reference_coords;
    int cell = locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &reference_coords);
    if (cell == -1) {
        return -1;
    }

    if (!result) {
        return 0;
    }

    double *J = reference_coords.J;
    double *K = reference_coords.K;
    double detJ = reference_coords.detJ;

%(calculate_basisvalues)s

    wrap_evaluate(result, (double *)phi, f->f, f->f_map%(nlayers)s, cell);
    return 0;
}
"""

    return evaluate_template_c % code
