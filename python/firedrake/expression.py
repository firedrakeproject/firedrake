import ufl
import numpy as np


class Expression(ufl.Coefficient):

    def __init__(self, code=None, element=None, cell=None, degree=None, **kwargs):

        shape = np.array(code).shape
        self._rank = len(shape)
        self._shape = shape
        if self._rank == 0:
            # Make code slot iterable even for scalar expressions
            self.code = [code]
        else:
            self.code = code
        self.cell = cell
        self.degree = degree
        # These attributes are required by ufl.Coefficient to render the repr
        # of an Expression. Since we don't call the ufl.Coefficient constructor
        # (since we don't yet know the element) we need to set them ourselves
        self._element = element
        self._repr = None
        self._count = 0

    def rank(self):
        return self._rank

    def shape(self):
        return self._shape
