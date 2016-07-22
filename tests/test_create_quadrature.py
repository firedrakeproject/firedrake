from tsfc import quadrature as q
import pytest


def test_invalid_quadrature_rule():
    with pytest.raises(ValueError):
        q.QuadratureRule([[0.5, 0.5]], [0.5, 0.5, 0.5])


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])
