import pytest
import gem.gem as gem
from gem.flop_count import count_flops
from gem.impero_utils import preprocess_gem
from gem.impero_utils import compile_gem


def test_count_flops(expression):
    expr, expected = expression
    flops = count_flops(expr)
    assert flops == expected


@pytest.fixture(params=("expr1", "expr2", "expr3", "expr4"))
def expression(request):
    if request.param == "expr1":
        expr = gem.Sum(gem.Product(gem.Variable("a", ()), gem.Literal(2)),
                       gem.Division(gem.Literal(3), gem.Variable("b", ())))
        C = gem.Variable("C", (1,))
        i, = gem.indices(1)
        Ci = C[i]
        expr, = preprocess_gem([expr])
        assignments = [(Ci, expr)]
        expr = compile_gem(assignments, (i,))
        # C += a*2 + 3/b
        expected = 1 + 3
    elif request.param == "expr2":
        expr = gem.Comparison(">=", gem.MaxValue(gem.Literal(1), gem.Literal(2)),
                              gem.MinValue(gem.Literal(3), gem.Literal(1)))
        C = gem.Variable("C", (1,))
        i, = gem.indices(1)
        Ci = C[i]
        expr, = preprocess_gem([expr])
        assignments = [(Ci, expr)]
        expr = compile_gem(assignments, (i,))
        # C += max(1, 2) >= min(3, 1)
        expected = 1 + 3
    elif request.param == "expr3":
        expr = gem.Solve(gem.Identity(3), gem.Inverse(gem.Identity(3)))
        C = gem.Variable("C", (3, 3))
        i, j = gem.indices(2)
        Cij = C[i, j]
        expr, = preprocess_gem([expr[i, j]])
        assignments = [(Cij, expr)]
        expr = compile_gem(assignments, (i, j))
        # C += solve(Id(3x3), Id(3x3)^{-1})
        expected = 9 + 18 + 54 + 54
    elif request.param == "expr4":
        A = gem.Variable("A", (10, 15))
        B = gem.Variable("B", (8, 10))
        i, j, k = gem.indices(3)
        Aij = A[i, j]
        Bki = B[k, i]
        Cjk = gem.IndexSum(Aij * Bki, (i,))
        expr = Cjk
        expr, = preprocess_gem([expr])
        assignments = [(gem.Variable("C", (15, 8))[j, k], expr)]
        expr = compile_gem(assignments, (j, k))
        # Cjk += \sum_i Aij * Bki
        expected = 2 * 10 * 8 * 15

    else:
        raise ValueError("Unexpected expression")
    return expr, expected
