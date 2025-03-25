from firedrake import *
from finat.finiteelementbase import entity_support_dofs
import pytest


@pytest.fixture(scope='module')
def hex_mesh():
    return ExtrudedMesh(UnitSquareMesh(1, 1, quadrilateral=True), 1)


@pytest.mark.parametrize(('args', 'kwargs', 'horiz_expected', 'vert_expected'),
                         [(("DQ", 0), dict(),
                           {0: [0], 1: [0]},
                           {0: [0], 1: [0], 2: [0], 3: [0]}),
                          (("DQ", 1), dict(),
                           {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]},
                           {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [0, 1, 4, 5], 3: [2, 3, 6, 7]}),
                          (("Q", 1), dict(),
                           {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]},
                           {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [0, 1, 4, 5], 3: [2, 3, 6, 7]}),
                          (("DQ", 0), dict(vfamily="CG", vdegree=1),
                           {0: [0], 1: [1]},
                           {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1]}),
                          (("Q", 1), dict(vfamily="DG", vdegree=0),
                           {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]},
                           {0: [0, 1], 1: [2, 3], 2: [0, 2], 3: [1, 3]}),
                          (("NCF", 1), dict(),
                           {0: [0, 1, 2, 3, 4], 1: [0, 1, 2, 3, 5]},
                           {0: [0, 2, 3, 4, 5],
                            1: [1, 2, 3, 4, 5],
                            2: [0, 1, 2, 4, 5],
                            3: [0, 1, 3, 4, 5]}),
                          (("NCE", 1), dict(),
                           {0: [0, 1, 2, 3, 4, 6, 8, 10], 1: [0, 1, 2, 3, 5, 7, 9, 11]},
                           {0: [0, 1, 4, 5, 8, 9, 10, 11],
                            1: [2, 3, 6, 7, 8, 9, 10, 11],
                            2: [0, 2, 4, 5, 6, 7, 8, 9],
                            3: [1, 3, 4, 5, 6, 7, 10, 11]})])
def test_hex(hex_mesh, args, kwargs, horiz_expected, vert_expected):
    if not kwargs:
        fe = FiniteElement(args[0], hex_mesh.ufl_cell(), args[1], variant='equispaced')
    else:
        A, B = hex_mesh.ufl_cell().sub_cells()
        hfe = FiniteElement(args[0], A, args[1], variant='equispaced')
        vfe = FiniteElement(kwargs["vfamily"], B, kwargs['vdegree'], variant='equispaced')
        fe = TensorProductElement(hfe, vfe)
    V = FunctionSpace(hex_mesh, fe, **kwargs)
    assert horiz_expected == entity_support_dofs(V.finat_element, (2, 0))
    assert vert_expected == entity_support_dofs(V.finat_element, (1, 1))
