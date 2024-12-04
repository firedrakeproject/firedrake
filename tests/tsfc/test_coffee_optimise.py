import pytest

from gem.gem import Index, Indexed, Product, Variable, Division, Literal, Sum
from gem.optimise import replace_division
from tsfc.coffee_mode import optimise_expressions


def test_replace_div():
    i = Index()
    A = Variable('A', ())
    B = Variable('B', (6,))
    Bi = Indexed(B, (i,))
    d = Division(Bi, A)
    result, = replace_division([d])
    expected = Product(Bi, Division(Literal(1.0), A))

    assert result == expected


def test_loop_optimise():
    I = 20
    J = K = 10
    i = Index('i', I)
    j = Index('j', J)
    k = Index('k', K)

    A1 = Variable('a1', (I,))
    A2 = Variable('a2', (I,))
    A3 = Variable('a3', (I,))
    A1i = Indexed(A1, (i,))
    A2i = Indexed(A2, (i,))
    A3i = Indexed(A3, (i,))

    B = Variable('b', (J,))
    C = Variable('c', (J,))
    Bj = Indexed(B, (j,))
    Cj = Indexed(C, (j,))

    E = Variable('e', (K,))
    F = Variable('f', (K,))
    G = Variable('g', (K,))
    Ek = Indexed(E, (k,))
    Fk = Indexed(F, (k,))
    Gk = Indexed(G, (k,))

    Z = Variable('z', ())

    # Bj*Ek + Bj*Fk => (Ek + Fk)*Bj
    expr = Sum(Product(Bj, Ek), Product(Bj, Fk))
    result, = optimise_expressions([expr], (j, k))
    expected = Product(Sum(Ek, Fk), Bj)
    assert result == expected

    # Bj*Ek + Bj*Fk + Bj*Gk + Cj*Ek + Cj*Fk =>
    # (Ek + Fk + Gk)*Bj + (Ek+Fk)*Cj
    expr = Sum(Sum(Sum(Sum(Product(Bj, Ek), Product(Bj, Fk)), Product(Bj, Gk)),
                   Product(Cj, Ek)), Product(Cj, Fk))
    result, = optimise_expressions([expr], (j, k))
    expected = Sum(Product(Sum(Sum(Ek, Fk), Gk), Bj), Product(Sum(Ek, Fk), Cj))
    assert result == expected

    # Z*A1i*Bj*Ek + Z*A2i*Bj*Ek + A3i*Bj*Ek + Z*A1i*Bj*Fk =>
    # Bj*(Ek*(Z*A1i + Z*A2i) + A3i) + Z*A1i*Fk)

    expr = Sum(Sum(Sum(Product(Z, Product(A1i, Product(Bj, Ek))),
                       Product(Z, Product(A2i, Product(Bj, Ek)))),
                   Product(A3i, Product(Bj, Ek))),
               Product(Z, Product(A1i, Product(Bj, Fk))))
    result, = optimise_expressions([expr], (j, k))
    expected = Product(Sum(Product(Ek, Sum(Sum(Product(Z, A1i), Product(Z, A2i)), A3i)),
                           Product(Fk, Product(Z, A1i))), Bj)
    assert result == expected


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
