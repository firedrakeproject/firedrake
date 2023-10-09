# The perp routine should come from UFL when it is fully implemented there
from gusto import perp
from gusto.fml import subject, replace_subject, all_terms
from firedrake import (UnitSquareMesh, FunctionSpace, MixedFunctionSpace,
                       TestFunctions, Function, split, inner, dx, errornorm,
                       SpatialCoordinate, as_vector, TrialFunctions, solve)


def test_replace_perp():

    # The test checks that if the perp operator is applied to the
    # subject of a labelled form, the perp of the subject is found and
    # replaced by the replace_subject function. This gave particular problems
    # before the perp operator was defined

    #  set up mesh and function spaces - the subject is defined on a
    #  mixed function space because the problem didn't occur otherwise
    Nx = 5
    mesh = UnitSquareMesh(Nx, Nx)
    spaces = [FunctionSpace(mesh, "BDM", 1), FunctionSpace(mesh, "DG", 1)]
    W = MixedFunctionSpace(spaces)

    #  set up labelled form with subject u
    w, p = TestFunctions(W)
    U0 = Function(W)
    u0, _ = split(U0)
    form = subject(inner(perp(u0), w)*dx, U0)

    # make a function to replace the subject with and give it some values
    U1 = Function(W)
    u1, _ = U1.split()
    x, y = SpatialCoordinate(mesh)
    u1.interpolate(as_vector([1, 2]))

    u, D = TrialFunctions(W)
    a = inner(u, w)*dx + D*p*dx
    L = form.label_map(all_terms, replace_subject(U1, old_idx=0, new_idx=0))
    U2 = Function(W)
    solve(a == L.form, U2)

    u2, _ = U2.split()
    U3 = Function(W)
    u3, _ = U3.split()
    u3.interpolate(as_vector([-2, 1]))

    assert errornorm(u2, u3) < 1e-14
