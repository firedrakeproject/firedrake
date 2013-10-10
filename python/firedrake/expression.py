import ufl
import numpy as np


class Expression(ufl.Coefficient):

    def __init__(self, code=None, element=None, cell=None, degree=None, **kwargs):

        tmp = np.array(code)
        shape = tmp.shape
        self._rank = len(shape)
        self._shape = shape
        if self._rank == 0:
            # Make code slot iterable even for scalar expressions
            self.code = [code]
        else:
            self.code = code
        self.element = element
        self.cell = cell
        self.degree = degree

    def rank(self):
        return self._rank

    def shape(self):
        return self._shape
