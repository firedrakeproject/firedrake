"""This module provides methods for constructing the nullspaces
for the statically condensed systems produced by the Slate-based
preconditioners.
"""
import ufl

from firedrake.parloops import par_loop, READ, WRITE
from firedrake.petsc import PETSc
from firedrake.slate import AssembledVector

import numpy as np


__all__ = ['create_sc_nullspace', 'create_trace_nullspace']


def create_sc_nullspace(P, V, V_facet, comm):
    """Gets the nullspace vectors corresponding to the Schur complement
    system.

    :arg P: The H1 operator from the ImplicitMatrixContext.
    :arg V: The H1 finite element space.
    :arg V_facet: The finite element space of H1 basis functions
                  restricted to the mesh skeleton.

    Returns: A nullspace (if there is one) for the Schur-complement system.
    """
    from firedrake import Function

    nullspace = P.getNullSpace()
    if nullspace.handle == 0:
        # No nullspace
        return None

    vecs = nullspace.getVecs()
    tmp = Function(V)
    scsp_tmp = Function(V_facet)
    new_vecs = []

    # Transfer the trace bit (the nullspace vector restricted
    # to facet nodes is the nullspace for the condensed system)
    kernel = """
        for (int i=0; i<%d; ++i){
            for (int j=0; j<%d; ++j){
                x_facet[i][j] = x_h[i][j];
            }
        }""" % (V_facet.finat_element.space_dimension(),
                np.prod(V_facet.shape))
    for v in vecs:
        with tmp.dat.vec_wo as t:
            v.copy(t)

        par_loop(kernel, ufl.dx, {"x_facet": (scsp_tmp, WRITE),
                                  "x_h": (tmp, READ)})

        # Map vecs to the facet space
        with scsp_tmp.dat.vec_ro as v:
            new_vecs.append(v.copy())

    # Normalize
    for v in new_vecs:
        v.normalize()
    sc_nullspace = PETSc.NullSpace().create(vectors=new_vecs, comm=comm)

    return sc_nullspace


def create_trace_nullspace(P, forward, V, V_d, TraceSpace, comm):
    """Gets the nullspace vectors corresponding to the Schur complement
    system for the Lagrange multipliers in hybridized methods.

    :arg P: The mixed operator from the ImplicitMatrixContext.
    :arg forward: A Slate expression denoting the forward elimination
                  operator.
    :arg V: The original "unbroken" space.
    :arg V_d: The broken space.
    :arg TraceSpace: The space of approximate traces.

    Returns: A nullspace (if there is one) for the multiplier system.
    """
    from firedrake import assemble, Function, project

    nullspace = P.getNullSpace()
    if nullspace.handle == 0:
        # No nullspace
        return None

    vecs = nullspace.getVecs()
    tmp = Function(V)
    tmp_b = Function(V_d)
    tnsp_tmp = Function(TraceSpace)
    forward_action = forward * AssembledVector(tmp_b)
    new_vecs = []
    for v in vecs:
        with tmp.dat.vec_wo as t:
            v.copy(t)

        project(tmp, tmp_b)
        assemble(forward_action, tensor=tnsp_tmp)
        with tnsp_tmp.dat.vec_ro as v:
            new_vecs.append(v.copy())

    # Normalize
    for v in new_vecs:
        v.normalize()
    trace_nullspace = PETSc.NullSpace().create(vectors=new_vecs, comm=comm)

    return trace_nullspace
