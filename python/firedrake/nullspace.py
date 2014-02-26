from petsc4py import PETSc


class VectorSpaceBasis(object):
    """Build a basis for a vector space.

    You can use this basis to express the null space of a singular operator.

    :arg vecs: a list of :class:`.Vector`\s or :class:`.Functions`
         spanning the space.  Note that these must be orthonormal
    :arg constant: does the null space include the constant vector?
         If you pass ``constant=True`` you should not also include the
         constant vector in the list of ``vecs`` you supply.

    .. warning::

       The vectors you pass in to this object are *not* copied.  You
       should therefore not modify them after instantiation since the
       basis will then be incorrect.
    """
    def __init__(self, vecs=None, constant=False):
        if vecs is None and not constant:
            raise RuntimeError("Must either provide a list of null space vectors, or constant keyword (or both)")
        self._vecs = vecs or []
        self._petsc_vecs = []
        for v in self._vecs:
            with v.dat.vec_ro as v_:
                self._petsc_vecs.append(v_)
        if not self.is_orthonormal():
            raise RuntimeError("Provided vectors must be orthonormal")
        self._nullspace = PETSc.NullSpace().create(constant=constant,
                                                   vectors=self._petsc_vecs)

    @property
    def nullspace(self):
        """The PETSc NullSpace object for this :class:`.VectorSpaceBasis`"""
        return self._nullspace

    def orthogonalize(self, b):
        """Orthogonalize ``b`` with respect to this :class:`.VectorSpaceBasis`.

        :arg b: a :class:`.Function`"""
        raise NotImplementedError

    def is_orthonormal(self):
        """Is this vector space basis orthonormal?"""
        for i, iv in enumerate(self._petsc_vecs):
            for j, jv in enumerate(self._petsc_vecs):
                dij = 1 if i == j else 0
                if abs(iv.dot(jv) - dij) > 1e-14:
                    return False
        return True

    def is_orthogonal(self):
        """Is this vector space basis orthogonal?"""
        for i, iv in enumerate(self._petsc_vecs):
            for j, jv in enumerate(self._petsc_vecs):
                if i == j:
                    continue
                if abs(iv.dot(jv)) > 1e-14:
                    return False
        return True
