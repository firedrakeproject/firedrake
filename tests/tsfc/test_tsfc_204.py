from tsfc import compile_form
from ufl import (Coefficient, FacetNormal,
                 FunctionSpace, Mesh, as_matrix,
                 dot, dS, ds, dx, facet, grad, inner, outer, split, triangle)
from finat.ufl import BrokenElement, FiniteElement, MixedElement, VectorElement


def test_physically_mapped_facet():
    mesh = Mesh(VectorElement("P", triangle, 1))

    # set up variational problem
    U = FiniteElement("Morley", mesh.ufl_cell(), 2)
    V = FiniteElement("P", mesh.ufl_cell(), 1)
    R = FiniteElement("P", mesh.ufl_cell(), 1)
    Vv = VectorElement(BrokenElement(V))
    Qhat = VectorElement(BrokenElement(V[facet]), dim=2)
    Vhat = VectorElement(V[facet], dim=2)
    Z = FunctionSpace(mesh, MixedElement(U, Vv, Qhat, Vhat, R))

    z = Coefficient(Z)
    u, d, qhat, dhat, lam = split(z)

    s = FacetNormal(mesh)
    trans = as_matrix([[1, 0], [0, 1]])
    mat = trans*grad(grad(u))*trans + outer(d, d) * u
    J = (u**2*dx
         + u**3*dx
         + u**4*dx
         + inner(mat, mat)*dx
         + inner(grad(d), grad(d))*dx
         + dot(s, d)**2*ds)
    L_match = inner(qhat, dhat - d)
    L = J + inner(lam, inner(d, d)-1)*dx + (L_match('+') + L_match('-'))*dS + L_match*ds
    compile_form(L)
