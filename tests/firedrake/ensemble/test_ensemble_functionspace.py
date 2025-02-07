import firedrake as fd
import pytest


scalar_elements = {
    'CG': fd.FiniteElement('CG', cell=fd.triangle, degree=1),
    'BDM': fd.FiniteElement('BDM', cell=fd.triangle, degree=2),
    'V-RT': fd.VectorElement('RT', cell=fd.triangle, degree=1, dim=2),
    'T-DG': fd.TensorElement('DG', cell=fd.triangle, degree=1, shape=(2, 3))
}

# Test EnsembleFunctionSpace with 8 subfunctions with the elements below, distributed over 1 or more processors.
# This element sequence below hits a variety of cases:
# - scalar, vector-valued, vector, tensor, mixed elements
# - mixed elements with scalar, vector-valued, vector and tensor components
# - repeated adjacent components (6, 7)
# - repeated non-adjacent components (2, 6)
# - mixed element with a single component (8)
# - mixed element with repeated component (5)
# - mixed element where the first component matches the previous element (2, 3)

elements = [
    scalar_elements['CG'],  # 1
    scalar_elements['BDM'],  # 2
    fd.MixedElement([scalar_elements[e] for e in ('BDM', 'CG')]),  # 3
    scalar_elements['T-DG'],  # 4
    fd.MixedElement([scalar_elements[e] for e in ('V-RT', 'CG', 'CG')]),  # 5
    scalar_elements['BDM'],  # 6
    scalar_elements['BDM'],  # 7
    fd.MixedElement([scalar_elements[e] for e in ('T-DG',)])  # 8
]


@pytest.fixture
def ensemble():
    if fd.COMM_WORLD.size == 1:
        return fd.Ensemble(fd.COMM_WORLD, 1)
    return fd.Ensemble(fd.COMM_WORLD, 2)


@pytest.fixture
def mesh(ensemble):
    return fd.UnitSquareMesh(2, 2, comm=ensemble.comm)

# rejects spaces on different meshes

# rejects mix of primal and dual spaces

# rejects non function space arguments

# DualSpace rejects primal spaces

# FunctionSpace returns DualSpace if given dual local spaces

# local spaces are correct
#   a) number
#   b) type

# number of global spaces is correct

# nlocal_dofs is correct

# nglobal_dofs is correct

# __eq__ is correct
#   a) True:
#   b) False:
#       - different type
#       - same subspaces but different ensemble
#       - same subspaces but dual
#       - different subspaces, same length
#       - different length but matching subspaces (when they exist)
#       - False only on some ensemble members

# .dual() returns
#   a) correct type
#   b) correct value
#   c) V.dual().dual() == V
